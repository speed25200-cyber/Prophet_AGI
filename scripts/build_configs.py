#!/usr/bin/env python3
"""Generate the shipped model configurations.

Configurations are generated rather than hand-written because a hand-written one already
went wrong in a way nothing caught: ``prophet_500m_probe.json`` omitted ``core_pattern``,
so the global attention pattern applied inside the looped core. The only full-attention
layer ended up *in* the loop -- duplicating its KV cache per iteration, which is exactly
the invariant decision D1 exists to protect -- while the prelude and coda were pure
recurrence. It validated, it trained, and it would have confounded every ablation built
on it.

:meth:`ProphetConfig.design_warnings` now catches that class of mistake, and this script
refuses to write a configuration that trips it.

    python scripts/build_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.budget import count_parameters  # noqa: E402
from prophet.config import (  # noqa: E402
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MemoryConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)

ROOT = Path(__file__).resolve().parent.parent


def build(
    name: str,
    *,
    d_model: int,
    prelude: int,
    core: int,
    coda: int,
    loop_k: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int = 128,
    vocab_size: int = 32_768,
    moe: dict | None = None,
    loop: bool = True,
    global_memory: str = "none",
    memory: dict | None = None,
) -> ProphetConfig:
    """One configuration, with every architectural invariant made explicit.

    ``loop=False`` is the plain-depth arm of the R04 gate: the same blocks run once
    (``train_loop_min = train_loop_max = 1``, no halting), so the comparison is at equal
    FLOPs per token and unequal parameters -- which is the bet. ``global_memory`` is the
    D3b switch on the global attention layers; ``memory`` the R03 output-mounted memory.
    """
    ffn = (
        FeedForwardConfig(kind="moe", **moe)
        if moe
        else FeedForwardConfig(kind="dense", hidden_mult=4.0)
    )
    return ProphetConfig(
        name=name,
        d_model=d_model,
        frontend=FrontendConfig(mode="bpe", vocab_size=vocab_size, tie_word_embeddings=True),
        mixer=MixerConfig(
            # Prelude and coda alternate windowed and global attention. The core
            # overrides this with pure recurrence below.
            pattern=["swa", "full_attn"],
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=True,
            sliding_window=2048,
            attention_sink_tokens=1,
            linear_heads=max(d_model // head_dim, 1),
            linear_head_dim=head_dim,
            linear_expand=2.0,
            linear_beta_max=2.0,   # negative transition eigenvalues; parity needs them
            nope_layers=(1,),      # the global layer runs position-free
            global_memory=global_memory,
            global_window=4096,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=prelude,
            core_layers=core,
            coda_layers=coda,
            core_pattern=["gdn"],  # D1: no attention inside the loop
            default_loop_k=loop_k if loop else 1,
            train_loop_min=1,
            train_loop_max=max(2 * loop_k, 2) if loop else 1,
            truncated_backprop_steps=3 if loop else 1,
            halting="ponder" if loop else "none",  # input-dependent depth; a constant k buys no class
            halting_loss_weight=0.05 if loop else 0.0,
            halting_target_steps=float(loop_k),
        ),
        ffn=ffn,
        heads=HeadsConfig(n_multi_token_predict=1, confidence_head=True),
        **({"memory": MemoryConfig(**memory)} if memory else {}),
    )


CONFIGS: dict[str, ProphetConfig] = {
    "prophet_main.json": build(
        "prophet-main", d_model=1536, prelude=4, core=4, coda=4, loop_k=4,
        n_heads=12, n_kv_heads=2,
        moe=dict(n_experts=128, n_experts_per_token=8, n_shared_experts=1,
                 expert_hidden_mult=0.5, moe_first_dense_layers=0),
    ),
    "prophet_mini.json": build(
        "prophet-mini", d_model=1280, prelude=3, core=4, coda=3, loop_k=2,
        n_heads=10, n_kv_heads=2,
    ),
    "prophet_500m_probe.json": build(
        "prophet-500m-probe", d_model=1536, prelude=2, core=4, coda=2, loop_k=4,
        n_heads=12, n_kv_heads=3,
    ),
    # A CPU-sized instance of the same stack: what a session without a GPU can train
    # on a real corpus end to end (scripts/first_run_cpu.py). Not a research point;
    # the first weights the pipeline has produced.
    "prophet_cpu_first_run.json": build(
        "prophet-cpu-first-run", d_model=256, prelude=2, core=2, coda=2, loop_k=2,
        n_heads=4, n_kv_heads=2, head_dim=64, vocab_size=4096,
    ),
    # --- the plan's first A100 runs, as matched pairs (prophet.plan) ------------------
    # R04 gate (24 h): looped depth against plain depth at equal FLOPs per token. Both
    # arms run 20 blocks per token; the looped one carries 221M parameters, the plain
    # one 498M. If the loop does not beat the plain stack here, the central bet is dead.
    "prophet_r04_loop.json": build(
        "prophet-r04-loop", d_model=1280, prelude=2, core=4, coda=2, loop_k=4,
        n_heads=10, n_kv_heads=2,
    ),
    "prophet_r04_plain.json": build(
        "prophet-r04-plain", d_model=1280, prelude=2, core=16, coda=2, loop_k=1,
        n_heads=10, n_kv_heads=2, loop=False,
    ),
    # D3b (6 h): ledger attention against exact global attention on real text, ~86M,
    # identical but for the switch. Failure criterion: held-out BPB worse by 0.5% or
    # recall at chance beyond the 4k window, and the ledger stays "none".
    "prophet_d3b_ledger.json": build(
        "prophet-d3b-ledger", d_model=768, prelude=3, core=2, coda=3, loop_k=2,
        n_heads=6, n_kv_heads=2, global_memory="ledger",
    ),
    "prophet_d3b_none.json": build(
        "prophet-d3b-none", d_model=768, prelude=3, core=2, coda=3, loop_k=2,
        n_heads=6, n_kv_heads=2,
    ),
    # R03 (20 h): the output-mounted product-key memory on the mini stack; the control
    # is prophet_mini.json itself.
    "prophet_r03_memory.json": build(
        "prophet-r03-memory", d_model=1280, prelude=3, core=4, coda=3, loop_k=2,
        n_heads=10, n_kv_heads=2,
        memory=dict(enabled=True, kind="product_key", mount="output"),
    ),
}


def main() -> int:
    failed = False
    for filename, cfg in CONFIGS.items():
        cfg.validate()
        warnings = cfg.design_warnings()
        params = count_parameters(cfg)

        print(
            f"{cfg.name:20s} {params.total / 1e9:5.2f}B total / "
            f"{params.active_per_token / 1e6:6.1f}M active   "
            f"depth {cfg.parameterised_depth()} blocks -> {cfg.effective_depth()} effective"
        )
        for warning in warnings:
            print(f"    ! {warning}")
            failed = True
        if not warnings:
            cfg.to_json(ROOT / "configs" / filename)

    if failed:
        print("\nRefusing to write configurations that trip a design invariant.",
              file=sys.stderr)
        return 1

    example = ProphetConfig.from_json(ROOT / "configs" / "prophet_main.json")
    kinds = {
        kind: sum(1 for *_, k in example.cache_slots() if k == kind)
        for kind in ("gdn", "swa", "full_attn")
    }
    print(f"\nprophet-main layout: {example.section_layout()}")
    print(f"cache slots at k={example.recurrent.default_loop_k}: {kinds}")
    print("Attention slot count is independent of k; only the bounded state scales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
