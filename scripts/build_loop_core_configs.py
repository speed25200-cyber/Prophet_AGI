#!/usr/bin/env python3
"""Generate the three arms of the loop-core programme (docs/29_LOOP_CORE_PREREG.md).

Three configurations, differing by what sits inside the looped core and nothing else:

* ``lc_gdn``   -- the Prophet default: four gated-delta blocks looped, bounded state.
* ``lc_attn``  -- the A-KV ablation: four full-attention blocks looped, KV cache per
                  iteration. Its feed-forward width is widened so that the resident
                  parameter count matches ``lc_gdn`` within a stated tolerance.
* ``lc_plain`` -- the unshared reference: sixteen gated-delta blocks run once.

Both looped arms train with a uniform depth in [2, 6] and gradients through every
visited pass; every arm executes twenty blocks per token at k = 4. Halting, auxiliary
heads and the input adapter are off.

``scripts/build_configs.py`` refuses to write a configuration that trips a design
warning. The attention arm trips exactly one -- decision D1's invariant -- on purpose.
This script accepts that single warning explicitly, refuses any other, and records what
it accepted in ``configs/loop_core/summary.json`` so the pre-registration can freeze it.

    python scripts/build_loop_core_configs.py [--out configs/loop_core]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prophet.budget import count_parameters  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from scripts.build_configs import build  # noqa: E402

D_MODEL, PRELUDE, CORE, CODA, LOOP_K = 1792, 2, 4, 2, 4
N_HEADS, N_KV_HEADS = 14, 2
TRAIN_LOOP_MIN, TRAIN_LOOP_MAX = 2, 6
PARAM_MATCH_TOLERANCE = 0.005
ACCEPTED_WARNING_PREFIX = "attention inside the looped core"


def _plain_r04_policy(cfg: ProphetConfig) -> ProphetConfig:
    """The R04 policy: no halting, no auxiliary heads, gradient through every pass."""
    r = cfg.recurrent
    r.halting = "none"
    r.halting_loss_weight = 0.0
    r.input_adapter = "none"
    cfg.heads.n_multi_token_predict = 0
    cfg.heads.confidence_head = False
    return cfg


def _elastic(cfg: ProphetConfig) -> ProphetConfig:
    r = cfg.recurrent
    r.train_loop_dist = "uniform"
    r.train_loop_min, r.train_loop_max = TRAIN_LOOP_MIN, TRAIN_LOOP_MAX
    r.default_loop_k = LOOP_K
    r.truncated_backprop_steps = TRAIN_LOOP_MAX
    return _plain_r04_policy(cfg)


def build_gdn() -> ProphetConfig:
    return _elastic(
        build(
            "prophet-lc-gdn",
            d_model=D_MODEL,
            prelude=PRELUDE,
            core=CORE,
            coda=CODA,
            loop_k=LOOP_K,
            n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS,
        )
    )


def build_attn(target_params: int) -> tuple[ProphetConfig, dict]:
    """Attention in the core, FFN widened until the resident count matches the target.

    The search is over the shared feed-forward width only: the same widening applies to
    every unique block, so the arms differ by the core mixer and nothing structural.
    """
    best: tuple[int, float] | None = None
    for hundredths in range(400, 520):
        mult = hundredths / 100
        cfg = build(
            "prophet-lc-attn",
            d_model=D_MODEL,
            prelude=PRELUDE,
            core=CORE,
            coda=CODA,
            loop_k=LOOP_K,
            n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS,
        )
        cfg.recurrent.core_pattern = ["full_attn"]
        cfg.ffn.hidden_mult = mult
        cfg = _elastic(cfg)
        gap = abs(count_parameters(cfg).total - target_params)
        if best is None or gap < best[0]:
            best = (gap, mult)
    assert best is not None
    cfg = build(
        "prophet-lc-attn",
        d_model=D_MODEL,
        prelude=PRELUDE,
        core=CORE,
        coda=CODA,
        loop_k=LOOP_K,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
    )
    cfg.recurrent.core_pattern = ["full_attn"]
    cfg.ffn.hidden_mult = best[1]
    cfg = _elastic(cfg)
    total = count_parameters(cfg).total
    match = {
        "target_params": target_params,
        "params": total,
        "ffn_hidden_mult": best[1],
        "relative_gap": (total - target_params) / target_params,
    }
    if abs(match["relative_gap"]) > PARAM_MATCH_TOLERANCE:
        raise ValueError(f"cannot match parameters within {PARAM_MATCH_TOLERANCE}: {match}")
    return cfg, match


def build_plain() -> ProphetConfig:
    cfg = build(
        "prophet-lc-plain",
        d_model=D_MODEL,
        prelude=PRELUDE,
        core=CORE * LOOP_K,
        coda=CODA,
        loop_k=1,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        loop=False,
    )
    r = cfg.recurrent
    r.train_loop_min = r.train_loop_max = r.default_loop_k = 1
    r.truncated_backprop_steps = 1
    return _plain_r04_policy(cfg)


def build_all() -> tuple[dict[str, ProphetConfig], dict]:
    gdn = build_gdn()
    attn, match = build_attn(count_parameters(gdn).total)
    plain = build_plain()
    return {"lc_gdn": gdn, "lc_attn": attn, "lc_plain": plain}, match


def describe(name: str, cfg: ProphetConfig) -> dict:
    cfg.validate()
    warnings = cfg.design_warnings()
    accepted = [w for w in warnings if w.startswith(ACCEPTED_WARNING_PREFIX)]
    unexpected = [w for w in warnings if not w.startswith(ACCEPTED_WARNING_PREFIX)]
    if unexpected:
        raise ValueError(f"{name}: unexpected design warning(s): {unexpected}")
    if accepted and name != "lc_attn":
        raise ValueError(f"{name}: attention inside the core is only the lc_attn ablation")
    if name == "lc_attn" and not accepted:
        raise ValueError("lc_attn must trip decision D1's warning; the core is not attention")
    params = count_parameters(cfg)
    return {
        "name": cfg.name,
        "params_resident": params.total,
        "params_active_per_token": params.active_per_token,
        "parameterised_depth": cfg.parameterised_depth(),
        "effective_depth": cfg.effective_depth(),
        "blocks_per_token_at_k4": PRELUDE + CORE * LOOP_K + CODA,
        "core_pattern": cfg.recurrent.core_pattern,
        "train_loop": [
            cfg.recurrent.train_loop_min,
            cfg.recurrent.train_loop_max,
            cfg.recurrent.train_loop_dist,
        ],
        "truncated_backprop_steps": cfg.recurrent.truncated_backprop_steps,
        "ffn_hidden_mult": cfg.ffn.hidden_mult,
        "accepted_design_warnings": accepted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "loop_core")
    args = ap.parse_args()
    configs, match = build_all()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"arms": {}, "attention_parameter_match": match}
    for name, cfg in configs.items():
        info = describe(name, cfg)
        path = args.out / f"{name}.json"
        cfg.to_json(path)
        info["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary["arms"][name] = info
        print(
            f"{name:9s} {info['params_resident'] / 1e6:7.2f}M resident  "
            f"depth {info['parameterised_depth']} -> {info['effective_depth']}  "
            f"core={info['core_pattern']}  loops={info['train_loop']}"
        )
        for warning in info["accepted_design_warnings"]:
            print(f"    accepted: {warning}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
