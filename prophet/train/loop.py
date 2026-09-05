"""The training loop.

Built around one assumption: **the process will be killed without warning**. A Colab
A100 session ends on its own schedule, so every part of the loop that holds state — the
model, the optimisers, the schedule position, and the data cursor — is checkpointed
together and restored together. A resumed run must be indistinguishable from an
uninterrupted one, and :mod:`tests.test_training` asserts exactly that.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from prophet.config import ProphetConfig
from prophet.data.streaming import StreamingLoader
from prophet.data.tokenizer import N_BYTES, SPECIAL_TOKENS
from prophet.modeling.action import build_action_targets
from prophet.modeling.moe import apply_router_updates
from prophet.train.checkpoint import CheckpointManager
from prophet.train.loss import compute_loss
from prophet.train.optim import build_optimizers
from prophet.train.schedule import WSDSchedule

TOOL_ID = N_BYTES + SPECIAL_TOKENS.index("<|tool|>")
"""Id of ``<|tool|>``: opens an observation span in the trainer's depth ceilings."""

__all__ = ["TrainConfig", "Trainer", "TrainMetrics"]


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

    mtp_weight: float | None = None
    """Weight on the multi-token-prediction loss. ``None`` takes the value from the model
    config's ``heads.mtp_loss_weight``, so the two cannot silently disagree -- they did,
    and the config field was dead."""
    confidence_weight: float | None = None
    """Same rule, from ``heads.confidence_loss_weight``."""
    z_loss_weight: float = 1e-4
    ponder_weight: float = 0.0
    sel_weight: float | None = None
    ptr_weight: float | None = None
    gate_weight: float | None = None
    jumped_lm_weight: float | None = None
    """Action-head terms (A3); ``None`` takes the model config's ``heads`` weights."""
    """Weight on the halting objective. Zero disables it; the model config's
    ``recurrent.halting_loss_weight`` is the value to mirror here when halting is on."""
    ponder_target_steps: float = 4.0

    checkpoint_every: int = 200
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    keep_milestones: tuple[int, ...] = ()

    seed: int = 0
    device: str = "cpu"
    dtype: str = "bfloat16"
    """Autocast dtype for the forward and loss on CUDA. Parameters stay fp32 -- they
    *are* the master copy -- and matmuls run in bf16. "float32" disables autocast."""
    activation_checkpointing: bool = True
    """Recompute block activations in the backward pass instead of storing them. Set on
    the model as ``gradient_checkpointing``; honoured by ProphetBlock."""
    allow_tf32: bool = True


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
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.model_config = model_config
        self.tokenizer = tokenizer
        self._action = bool(model_config is not None and model_config.heads.action_head)
        if self._action and tokenizer is None:
            raise ValueError(
                "heads.action_head derives its targets from the token stream and needs "
                "the tokenizer: Trainer(..., tokenizer=ProphetTokenizer.load(...))"
            )
        self.on_log = on_log or (lambda m: print(m.format()))

        actual_seq_len = int(getattr(loader, "seq_len", cfg.seq_len))
        if model_config is not None and actual_seq_len > model_config.max_seq_len:
            raise ValueError(
                f"seq_len={actual_seq_len} exceeds the model's max_seq_len="
                f"{model_config.max_seq_len}; raise the config field deliberately (it "
                "is what the rotary table and the budget were sized for)"
            )
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.model.gradient_checkpointing = cfg.activation_checkpointing
        if self.device.type == "cuda" and cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._autocast_dtype = {
            "bfloat16": torch.bfloat16, "float16": torch.float16,
        }.get(cfg.dtype)
        self.stop_requested = False
        """Set by a signal handler; checked once per step so a polite stop checkpoints."""

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
        # The auxiliary-head weights live on the model config, next to the heads they
        # weight. The trainer only overrides them when told to explicitly.
        if cfg.mtp_weight is None:
            cfg.mtp_weight = model_config.heads.mtp_loss_weight if model_config else 0.3
        if cfg.confidence_weight is None:
            cfg.confidence_weight = (
                model_config.heads.confidence_loss_weight if model_config else 0.0
            )
        heads = model_config.heads if model_config else None
        if cfg.sel_weight is None:
            cfg.sel_weight = heads.sel_loss_weight if heads else 0.0
        if cfg.ptr_weight is None:
            cfg.ptr_weight = heads.ptr_loss_weight if heads else 0.0
        if cfg.gate_weight is None:
            cfg.gate_weight = heads.gate_loss_weight if heads else 0.0
        if cfg.jumped_lm_weight is None:
            cfg.jumped_lm_weight = heads.jumped_token_lm_weight if heads else 1.0

        self.ckpt = CheckpointManager(cfg.checkpoint_dir, keep_milestones=cfg.keep_milestones)
        self.step = 0
        self.tokens_seen = 0
        self.history: list[TrainMetrics] = []

    # -- schedule ----------------------------------------------------------------------

    def _apply_lr(self) -> float:
        multiplier = self.schedule.lr_at(self.step)
        for opt, peak in zip(self.optimizers, self._peak_lrs):
            for group in opt.param_groups:
                group["lr"] = peak * multiplier
        return self._peak_lrs[0] * multiplier if self._peak_lrs else 0.0

    # -- state -------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loader": self.loader.state().to_dict(),
            "torch_rng": torch.get_rng_state(),
            # The recurrent state init draws on the model's device. Without the CUDA
            # generator, resume was bit-identical on CPU -- where the test runs -- and
            # not on the A100, where it matters.
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.model_config.to_dict() if self.model_config else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        for opt, opt_state in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])
        self.loader.restore(state["loader"])
        if "torch_rng" in state and state["torch_rng"] is not None:
            torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in state["cuda_rng"]])

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

    def token_depth(self, batch: Tensor, k: int) -> Tensor:
        """Per-token recurrence ceilings for one micro-batch (``recurrent.token_depth``).

        Every token starts at the sampled depth ``k``. Tool-observation spans -- from a
        ``<|tool|>`` control token up to the next control token -- drop to
        ``ingest_depth``, which is how the agent loop will feed them. With probability
        ``token_depth_random_spans`` a row also gets one random contiguous span at
        ``ingest_depth``, so that plain text teaches the shallow-then-deep transition
        too. Draws come from the global generator, which the checkpoint carries, so a
        resumed run builds the same ceilings.
        """
        r = self.model_config.recurrent  # type: ignore[union-attr]
        b, s = batch.shape
        depth = torch.full_like(batch, k)
        shallow = min(r.ingest_depth, k)
        if shallow < k and s:
            control = (batch >= N_BYTES) & (batch < N_BYTES + len(SPECIAL_TOKENS))
            idx = torch.arange(s, device=batch.device).expand(b, s)
            last_control = torch.cummax(torch.where(control, idx, -1), dim=1).values
            opened_by = batch.gather(1, last_control.clamp_min(0))
            in_tool_span = (last_control >= 0) & (opened_by == TOOL_ID)
            depth = torch.where(in_tool_span, torch.full_like(depth, shallow), depth)
            if r.token_depth_random_spans > 0:
                draw = torch.rand(b, device=batch.device) < r.token_depth_random_spans
                start = torch.randint(0, s, (b,), device=batch.device)
                length = torch.randint(1, max(2, s // 2), (b,), device=batch.device)
                span = (idx >= start[:, None]) & (idx < (start + length)[:, None])
                depth = torch.where(span & draw[:, None], torch.full_like(depth, shallow), depth)
        return depth

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
                use_autocast = self.device.type == "cuda" and self._autocast_dtype is not None
                forward_kw: dict[str, Any] = {}
                if self.model_config is not None and self.model_config.recurrent.token_depth:
                    k = self.model.sample_loop_k()
                    forward_kw = dict(loop_k=k, token_depth=self.token_depth(batch, k))
                action_targets = None
                if self._action:
                    action_targets = build_action_targets(batch, self.tokenizer)
                    forward_kw.update(action_targets.forward_kwargs())
                with torch.autocast(
                    device_type="cuda", dtype=self._autocast_dtype or torch.bfloat16,
                    enabled=use_autocast,
                ):
                    output = self.model(batch, **forward_kw)
                terms = compute_loss(
                    output,
                    batch,
                    mtp_weight=self.cfg.mtp_weight,
                    z_loss_weight=self.cfg.z_loss_weight,
                    ponder_weight=self.cfg.ponder_weight,
                    ponder_target_steps=self.cfg.ponder_target_steps,
                    project=getattr(self.model, "_project", None),
                    action_targets=action_targets,
                    sel_weight=self.cfg.sel_weight or 0.0,
                    ptr_weight=self.cfg.ptr_weight or 0.0,
                    gate_weight=self.cfg.gate_weight or 0.0,
                    jumped_lm_weight=1.0 if self.cfg.jumped_lm_weight is None else self.cfg.jumped_lm_weight,
                )
                (terms.total / self.cfg.grad_accum_steps).backward()
                # Loss-free MoE balancing moves the router biases *after* backward, so
                # a checkpointed block recomputes the routing it saved.
                apply_router_updates(getattr(output, "router_stats", ()))
                accumulated += terms.lm.item() / self.cfg.grad_accum_steps
                extra = {
                    k: v for k, v in terms.metrics.items()
                    if k.startswith(("router/", "ponder/", "action/")) or k in ("loss/mtp", "loss/action")
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
            if self.stop_requested:
                break

        return self.history
