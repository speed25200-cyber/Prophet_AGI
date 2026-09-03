"""Search the Prophet design space under hard constraints.

The research tracks disagree on model size: R04's recurrent design assumes ~9.5B total
parameters, while R07's memory arithmetic says a 10B-total MoE needs ~97 GiB of static
training state against ~77 GiB usable. Rather than split the difference by argument,
this script enumerates configurations and keeps only those that satisfy every constraint
simultaneously.

Constraints applied:
  C1  Trains on one A100 80GB with an 8-bit optimiser and room for activations.
  C2  Buys enough tokens to be worth training (>= 20 tokens per active parameter).
  C3  Prophet-mini fits an iPhone's realistic app budget at int4.
  C4  Prophet-main fits an RTX 5090 at int4 with 128k context.
  C5  No component misallocates the parameter budget.

Usage::

    python scripts/design_search.py --a100-hours 300
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.budget import (  # noqa: E402
    allocation_warnings,
    count_parameters,
    inference_profile,
    tokens_affordable,
    training_memory,
)
from prophet.config import (  # noqa: E402
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)

TRAIN_HEADROOM_GB = 12.0
"""Reserved on top of the estimate: fragmentation, the eval pass, and the fact that a
Colab A100 is never entirely ours."""


@dataclass
class Candidate:
    cfg: ProphetConfig
    total: int
    active: int
    tokens: float
    tokens_per_active: float
    train_gb: float
    device_gb: dict[str, float]
    warnings: list[str]

    def ok(self, *, device_limits: dict[str, float], min_tokens_per_active: float) -> bool:
        if self.train_gb + TRAIN_HEADROOM_GB > 79.0:
            return False
        if self.tokens_per_active < min_tokens_per_active:
            return False
        for dev, limit in device_limits.items():
            if self.device_gb.get(dev, 1e9) > limit:
                return False
        return not self.warnings


def build(
    *,
    d_model: int,
    prelude: int,
    core: int,
    coda: int,
    loop_k: int,
    n_experts: int,
    top_k: int,
    vocab: int = 32768,
    n_heads: int = 16,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    name: str = "candidate",
) -> ProphetConfig:
    """One point in the design space, following the R02 layer recipe.

    Attention lives in the prelude and coda (applied once, so its KV cache does not
    scale with the loop depth); the looped core is pure gated-delta recurrence.
    """
    ffn = (
        FeedForwardConfig(
            kind="moe", n_experts=n_experts, n_experts_per_token=top_k,
            n_shared_experts=1, expert_hidden_mult=0.5, moe_first_dense_layers=0,
        )
        if n_experts
        else FeedForwardConfig(kind="dense", hidden_mult=4.0)
    )
    return ProphetConfig(
        name=name,
        d_model=d_model,
        frontend=FrontendConfig(mode="bpe", vocab_size=vocab, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"],
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            qk_norm=True, sliding_window=2048, attention_sink_tokens=1,
            linear_heads=max(d_model // head_dim, 1), linear_head_dim=head_dim,
            linear_expand=2.0, nope_layers=(1,),
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=prelude, core_layers=core, coda_layers=coda,
            core_pattern=["gdn"], default_loop_k=loop_k,
            train_loop_min=1, train_loop_max=max(loop_k * 2, 2),
            truncated_backprop_steps=3,
        ),
        ffn=ffn,
        heads=HeadsConfig(n_multi_token_predict=1, confidence_head=True),
    )


def evaluate(cfg: ProphetConfig, a100_hours: float) -> Candidate:
    p = count_parameters(cfg)
    tb = tokens_affordable(cfg, a100_hours=a100_hours)
    tm = training_memory(cfg, batch_tokens=16384, optimizer_bytes_per_param=2.0)
    devices = {
        "rtx5090": inference_profile(cfg, device="rtx5090", context_len=131072).total_gb,
        "iphone17pro": inference_profile(
            cfg, device="iphone17pro", context_len=32768, loop_k=2
        ).total_gb,
    }
    active = tb["effective_active_params"]
    return Candidate(
        cfg=cfg,
        total=p.total,
        active=p.active_per_token,
        tokens=tb["tokens"],
        tokens_per_active=tb["tokens"] / max(active, 1),
        train_gb=tm.total_gb,
        device_gb=devices,
        warnings=allocation_warnings(cfg),
    )


def search(a100_hours: float, *, target: str) -> list[Candidate]:
    if target == "mini":
        grid = itertools.product(
            [768, 1024, 1280],          # d_model
            [(2, 3, 2), (2, 4, 2), (3, 4, 3)],  # prelude/core/coda
            [2, 4],                     # default loop k
            [0],                        # dense only: MoE routing is not ANE-friendly
        )
        device_limits = {"iphone17pro": 3.0}
        min_tpa = 40.0
    else:
        grid = itertools.product(
            [1536, 2048],
            [(3, 4, 3), (4, 4, 4), (4, 6, 4)],
            [4, 8],
            [0, 32, 64, 128],
        )
        device_limits = {"rtx5090": 24.0}
        min_tpa = 20.0

    out: list[Candidate] = []
    for d_model, (pre, core, coda), k, n_exp in grid:
        top_k = max(1, n_exp // 16) if n_exp else 0
        head_dim = 128 if d_model >= 1024 else 64
        n_heads = max(d_model // head_dim, 4)
        cfg = build(
            d_model=d_model, prelude=pre, core=core, coda=coda, loop_k=k,
            n_experts=n_exp, top_k=top_k, head_dim=head_dim, n_heads=n_heads,
            n_kv_heads=max(n_heads // 8, 1),
            name=f"prophet-{target}-d{d_model}-p{pre}c{core}x{k}o{coda}-e{n_exp}",
        )
        cand = evaluate(cfg, a100_hours)
        if cand.ok(device_limits=device_limits, min_tokens_per_active=min_tpa):
            out.append(cand)
    # Prefer the most total capacity, then the most tokens per active parameter.
    out.sort(key=lambda c: (-c.total, -c.tokens_per_active))
    return out


def _fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a100-hours", type=float, default=300.0)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    for target in ("main", "mini"):
        found = search(args.a100_hours, target=target)
        print(f"\n## Prophet-{target}: {len(found)} feasible configurations "
              f"at {args.a100_hours:.0f} A100-hours\n")
        if not found:
            print("  none — constraints are jointly unsatisfiable at this budget\n")
            continue
        print("| Config | Total | Active | Eff. depth | Tokens | Tok/active | Train GB | Device GB |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in found[: args.top]:
            dev = "rtx5090" if target == "main" else "iphone17pro"
            print(
                f"| {c.cfg.name} | {_fmt(c.total)} | {_fmt(c.active)} | "
                f"{c.cfg.effective_depth()} | {_fmt(c.tokens)} | {c.tokens_per_active:.0f} | "
                f"{c.train_gb:.1f} | {c.device_gb[dev]:.2f} |"
            )


if __name__ == "__main__":
    main()
