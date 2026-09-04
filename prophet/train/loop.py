"""The training loop.

Built around one assumption: **the process will be killed without warning**. A Colab
A100 session ends on its own schedule, so every part of the loop that holds state — the
model, the optimisers, the schedule position, and the data cursor — is checkpointed
together and restored together. A resumed run must be indistinguishable from an
uninterrupted one, and :mod:`tests.test_training` asserts exactly that.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from prophet.config import ProphetConfig
from prophet.data.streaming import LoaderState, StreamingLoader
from prophet.train.checkpoint import CheckpointManager
from prophet.train.loss import compute_loss
from prophet.train.optim import build_optimizers
from prophet.train.schedule import WSDSchedule

__all__ = ["TrainConfig", "Trainer", "TrainMetrics"]

_TRAINER_STATE_VERSION = 2


@dataclass
class TrainConfig:
    total_steps: int = 1000
    batch_size: int = 8
    seq_len: int = 512
    grad_accum_steps: int = 1

    peak_lr_muon: float = 0.02
    peak_lr_adamw: float = 3e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.02
    decay_frac: float = 0.18
    grad_clip: float = 1.0

    mtp_weight: float = 0.3
    z_loss_weight: float = 1e-4
    ponder_weight: float = 0.0
    """Weight on the halting objective. Zero disables it; the model config's
    ``recurrent.halting_loss_weight`` is the value to mirror here when halting is on."""
    ponder_target_steps: float = 4.0

    checkpoint_every: int = 200
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    keep_milestones: tuple[int, ...] = ()

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class TrainMetrics:
    step: int
    loss: float
    lr: float
    phase: str
    tokens: int
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        parts = [
            f"step {self.step:>7d}",
            f"loss {self.loss:7.4f}",
            f"lr {self.lr:.2e}",
            f"{self.phase:<7s}",
            f"{self.tokens / 1e6:8.2f}M tok",
            f"{self.seconds:6.2f}s",
        ]
        for k, v in sorted(self.extra.items()):
            parts.append(f"{k}={v:.4g}")
        return "  ".join(parts)


class Trainer:
    """Single-device training with bulletproof resume."""

    def __init__(
        self,
        model: nn.Module,
        loader: StreamingLoader,
        cfg: TrainConfig,
        *,
        model_config: ProphetConfig | None = None,
        on_log: Callable[[TrainMetrics], None] | None = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.model_config = model_config
        self.on_log = on_log or (lambda m: print(m.format()))

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.optimizers, self.param_counts = build_optimizers(
            model,
            muon_lr=cfg.peak_lr_muon,
            adamw_lr=cfg.peak_lr_adamw,
            weight_decay=cfg.weight_decay,
            mup_base_width=model_config.mup_base_width if model_config else None,
            d_model=model_config.d_model if model_config else None,
        )
        self.schedule = WSDSchedule(
            peak_lr=1.0,  # the schedule returns a multiplier; per-optimiser peaks scale it
            total_steps=cfg.total_steps,
            warmup_frac=cfg.warmup_frac,
            decay_frac=cfg.decay_frac,
        )
        self._peak_lrs = [
            cfg.peak_lr_muon if type(o).__name__ == "Muon" else cfg.peak_lr_adamw
            for o in self.optimizers
        ]

        # Halting is only trained if the objective is actually weighted. Taking the
        # weight from the model config when the trainer leaves it at zero prevents the
        # silent failure where halting is enabled architecturally, receives no gradient,
        # and produces a depth distribution that is pure noise.
        if model_config is not None and model_config.recurrent.halting == "ponder":
            if cfg.ponder_weight == 0.0:
                cfg.ponder_weight = model_config.recurrent.halting_loss_weight
            cfg.ponder_target_steps = model_config.recurrent.halting_target_steps

        self.ckpt = CheckpointManager(cfg.checkpoint_dir, keep_milestones=cfg.keep_milestones)
        self.step = 0
        self.tokens_seen = 0
        self.history: list[TrainMetrics] = []

    # -- schedule ----------------------------------------------------------------------

    def _apply_lr(self) -> float:
        multiplier = self.schedule.lr_at(self.step)
        for opt, peak in zip(self.optimizers, self._peak_lrs, strict=True):
            for group in opt.param_groups:
                group["lr"] = peak * multiplier
        return self._peak_lrs[0] * multiplier if self._peak_lrs else 0.0

    # -- state -------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "trainer_state_version": _TRAINER_STATE_VERSION,
            "model": self.model.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loader": self.loader.state().to_dict(),
            "torch_rng": torch.get_rng_state(),
            "config": self.model_config.to_dict() if self.model_config else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = state.get("trainer_state_version")
        if version is not None and (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version not in (1, _TRAINER_STATE_VERSION)
        ):
            raise ValueError(f"unsupported trainer state version: {version!r}")

        current_config = self.model_config.to_dict() if self.model_config else None
        has_saved_config = "config" in state and state["config"] is not None
        if has_saved_config:
            raw_saved_config = state["config"]
            if not isinstance(raw_saved_config, Mapping):
                raise ValueError("checkpoint model config must be a mapping")
            # A current-format state was written by ``state_dict`` above and must contain
            # the complete config. Legacy states are normalised through ``from_dict`` so
            # fields added since they were written receive their historical defaults.
            saved_config = (
                dict(raw_saved_config)
                if version == _TRAINER_STATE_VERSION
                else ProphetConfig.from_dict(dict(raw_saved_config)).to_dict()
            )
            if current_config is None or saved_config != current_config:
                raise ValueError(
                    "checkpoint model config does not match the current trainer config"
                )
        elif version == _TRAINER_STATE_VERSION:
            if current_config is not None or "config" not in state:
                raise ValueError(
                    "checkpoint does not contain a model config compatible with the "
                    "current trainer"
                )
        elif current_config is not None:
            warnings.warn(
                "loading a legacy checkpoint without model config metadata; "
                "compatibility cannot be verified",
                RuntimeWarning,
                stacklevel=2,
            )

        optimizer_states = state.get("optimizers")
        if not isinstance(optimizer_states, (list, tuple)):
            raise ValueError("checkpoint optimizers must be a list of optimizer states")
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "checkpoint optimizer count does not match the current trainer: "
                f"{len(optimizer_states)} != {len(self.optimizers)}"
            )

        # Validate stream identity and cursor bounds before changing model or optimizer
        # state. A seed/corpus/packing mismatch is otherwise detected only after a
        # partial restore, leaving the Trainer unusable for a clean fallback.
        loader_state = self.loader.validate_state(LoaderState.from_dict(state["loader"]))

        self.model.load_state_dict(state["model"])
        for opt, opt_state in zip(self.optimizers, optimizer_states, strict=True):
            opt.load_state_dict(opt_state)
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])
        self.loader.load_state(loader_state)
        if "torch_rng" in state and state["torch_rng"] is not None:
            torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))

    def maybe_resume(self) -> bool:
        """Restore the newest intact checkpoint, if any. Returns whether it resumed."""
        if not self.ckpt.has_checkpoint():
            return False
        state, meta = self.ckpt.load_latest(map_location=self.device)
        self.load_state_dict(state)
        return True

    # -- training ----------------------------------------------------------------------

    def _batch(self) -> Tensor:
        rows = next(iter(self.loader.batches(1)))
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def train(self, *, max_steps: int | None = None) -> list[TrainMetrics]:
        """Run until ``total_steps`` (or ``max_steps`` more, whichever comes first)."""
        stop_at = self.cfg.total_steps
        if max_steps is not None:
            stop_at = min(stop_at, self.step + max_steps)

        self.model.train()
        while self.step < stop_at:
            start = time.time()
            lr = self._apply_lr()

            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            accumulated = 0.0
            extra: dict[str, float] = {}
            for _ in range(self.cfg.grad_accum_steps):
                batch = self._batch()
                output = self.model(batch)
                terms = compute_loss(
                    output,
                    batch,
                    mtp_weight=self.cfg.mtp_weight,
                    z_loss_weight=self.cfg.z_loss_weight,
                    ponder_weight=self.cfg.ponder_weight,
                    ponder_target_steps=self.cfg.ponder_target_steps,
                    project=getattr(self.model, "_project", None),
                )
                (terms.total / self.cfg.grad_accum_steps).backward()
                accumulated += terms.lm.item() / self.cfg.grad_accum_steps
                extra = {
                    k: v for k, v in terms.metrics.items()
                    if k.startswith(("router/", "ponder/")) or k == "loss/mtp"
                }
                self.tokens_seen += batch.numel()

            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            for opt in self.optimizers:
                opt.step()

            self.step += 1
            metrics = TrainMetrics(
                step=self.step,
                loss=accumulated,
                lr=lr,
                phase=self.schedule.phase_at(self.step),
                tokens=self.tokens_seen,
                seconds=time.time() - start,
                extra=extra,
            )
            self.history.append(metrics)

            if self.step % self.cfg.log_every == 0:
                self.on_log(metrics)
            if self.cfg.checkpoint_every and self.step % self.cfg.checkpoint_every == 0:
                self.ckpt.save(self.state_dict(), self.step,
                               extra={"loss": accumulated, "lr": lr})

        return self.history
