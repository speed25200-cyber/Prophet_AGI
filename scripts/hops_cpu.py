#!/usr/bin/env python3
"""Depth as a dial, in miniature: does looping the core compose table look-ups, and does
tying the loop count to the number of hops carry past the hops seen in training?

    python scripts/hops_cpu.py --out /tmp/prophet-hops

The CPU analogue of programme 1's H4 (docs/29) and of W2-A5/A6 (docs/research/W2), on the
programme's own layout: an attention prelude and coda around a looped core that is either
a bounded-state GDN block or an attention block. A task is a shuffled table of ``TABLE``
pairs ``a b`` (a permutation of ``TABLE`` nodes drawn from ``NODES``), then ``Q start H
=``; the answer is the node reached after ``h`` applications of the table. One hop is
retrieval; ``h`` hops need ``h`` retrievals in sequence, which an attention core can do
once per pass and a bounded state cannot hold (the prediction of docs/29 H4).

Arms (same parameter count within a core type, same data, same steps):

- ``<core>-fixed``: trained and measured at ``k = FIXED_K`` passes, whatever ``h`` is;
- ``<core>-tied``:  trained and measured at ``k = h`` -- the loop count set by the declared
  difficulty, the length-tied schedule W2-A6 proposes before any learned halting.

Training draws ``h`` in ``TRAIN_HOPS``; the measure sweeps ``h`` in ``1..MAX_HOPS``, so hops
4 to 6 are beyond anything trained. Chance is ``1 / TABLE``. Nothing here is a capability
claim: 10^5 parameters, synthetic, one seed unless told otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import (  # noqa: E402
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.modeling.model import ProphetModel  # noqa: E402

PAD, NODES, TABLE, MAX_HOPS = 0, 16, 8, 6
SEP, Q, EQ = NODES + 1, NODES + 2, NODES + 3
HOP0 = NODES + 4  # HOP0 + h - 1 is the token that declares h hops
VOCAB = HOP0 + MAX_HOPS
TRAIN_HOPS = (1, 2, 3)
SMALL_WIDTH = {"init_std": 0.06, "tied": False, "kv_heads": 4}  # docs/38
FIXED_K = 4
LENGTH = 3 * TABLE + 4


def make_example(rng: random.Random, *, hops: int) -> tuple[list[int], int]:
    """One table and one question; returns (ids, answer). Node ids are 1..NODES."""
    if not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"hops in 1..{MAX_HOPS}")
    nodes = rng.sample(range(1, NODES + 1), TABLE)
    image = nodes[:]
    rng.shuffle(image)
    table = dict(zip(nodes, image, strict=True))
    pairs = list(table.items())
    rng.shuffle(pairs)
    ids: list[int] = []
    for a, b in pairs:
        ids += [a, b, SEP]
    start = rng.choice(nodes)
    answer = start
    for _ in range(hops):
        answer = table[answer]
    ids += [Q, start, HOP0 + hops - 1, EQ]
    return ids, answer


def batch(rng: random.Random, *, n: int, hops: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [make_example(rng, hops=hops) for _ in range(n)]
    return torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])


def config(
    core: str, *, init_std: float = 0.02, tied: bool = True, kv_heads: int = 2
) -> ProphetConfig:
    """The programme 1 layout at toy width: attention prelude (1 block), looped core (1
    block, GDN or full attention), attention coda (2 blocks). The defaults are the first
    recipe; docs/36 amendment 3 runs the small-width one of docs/38 (init 0.06, untied
    embeddings, no GQA), under which attention learns a look-up at this width."""
    return ProphetConfig(
        name=f"hops-{core}",
        d_model=64,
        n_layers=4,
        max_seq_len=64,
        init_std=init_std,
        frontend=FrontendConfig(vocab_size=VOCAB, tie_word_embeddings=tied),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"],
            n_heads=4,
            n_kv_heads=kv_heads,
            qk_norm=False,  # at head_dim 16 it caps the logit at 4 (docs/10 §1)
            sliding_window=LENGTH,
            attention_sink_tokens=1,
            nope_layers=(1,),
            linear_heads=2,
            linear_head_dim=16,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=1,
            core_layers=1,
            coda_layers=2,
            train_loop_min=1,
            train_loop_max=MAX_HOPS,
            default_loop_k=FIXED_K,
            halting="none",
            core_pattern=["full_attn"] if core == "attn" else ["gdn"],
            coda_pattern=["swa", "full_attn"],
            truncated_backprop_steps=MAX_HOPS,
        ),
        heads=HeadsConfig(n_multi_token_predict=0),
    )


def loop_k(schedule: str, hops: int) -> int:
    return hops if schedule == "tied" else FIXED_K


def lr_at(step: int, *, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(
    core: str,
    schedule: str,
    *,
    steps: int,
    minutes: float,
    seed: int,
    lr: float,
    warmup: int,
    batch_size: int = 32,
    warm_hops1: int = 0,
    recipe: dict | None = None,
    log=print,
) -> tuple[ProphetModel, dict]:
    torch.manual_seed(seed)
    cfg = config(core, **(recipe or {}))
    cfg.validate()
    model = ProphetModel(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    started, losses = time.time(), []
    for step in range(steps):
        # One hop count per batch (k is per batch); the first ``warm_hops1`` steps see one
        # hop only, where answering the start node earns nothing and only the look-up pays
        # (docs/36 amendment 2).
        hops = 1 if step < warm_hops1 else TRAIN_HOPS[step % len(TRAIN_HOPS)]
        ids, answers = batch(rng, n=batch_size, hops=hops)
        logits = model(ids, loop_k=loop_k(schedule, hops)).logits[:, -1]
        loss = torch.nn.functional.cross_entropy(logits.float(), answers)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps=steps, peak=lr, warmup=warmup)
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0:
            log(
                f"[{core}-{schedule}] step {step} loss {sum(losses[-200:]) / len(losses[-200:]):.3f}"
            )
        if (time.time() - started) / 60 > minutes:
            log(f"[{core}-{schedule}] wall-clock budget reached at step {step}")
            break
    return model, {
        "steps": len(losses),
        "loss_last200": sum(losses[-200:]) / max(len(losses[-200:]), 1),
        "minutes": (time.time() - started) / 60,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def accuracy(model: ProphetModel, *, hops: int, k: int, n: int, seed: int) -> float:
    model.eval()
    rng = random.Random(f"eval-{seed}-{hops}")
    correct = done = 0
    while done < n:
        m = min(64, n - done)
        ids, answers = batch(rng, n=m, hops=hops)
        pred = model(ids, loop_k=k).logits[:, -1].argmax(-1)
        correct += int((pred == answers).sum())
        done += m
    return correct / n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="gdn-fixed,gdn-tied,attn-fixed,attn-tied")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=15.0, help="per model")
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--seeds", default="0,1", help="comma-separated; every arm per seed")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument(
        "--small-width-recipe",
        action="store_true",
        help="init_std 0.06, untied embeddings, 4 KV heads (docs/38; docs/36 amendment 3)",
    )
    ap.add_argument(
        "--warm-hops1",
        type=int,
        default=0,
        help="first steps on one hop only, before the mixed hops (docs/36 amendment 2)",
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            **vars(args),
            "table": TABLE,
            "nodes": NODES,
            "train_hops": TRAIN_HOPS,
            "max_hops": MAX_HOPS,
            "fixed_k": FIXED_K,
            "chance": 1 / TABLE,
        },
        "arms": {},
    }
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        core, _, schedule = arm.partition("-")
        if core not in ("gdn", "attn") or schedule not in ("fixed", "tied"):
            ap.error(f"unknown arm {arm!r}")
    for seed in [int(x) for x in args.seeds.split(",") if x]:
        for arm in arms:
            core, _, schedule = arm.partition("-")
            model, stats = train(
                core,
                schedule,
                steps=args.steps,
                minutes=args.minutes,
                seed=seed,
                lr=args.lr,
                warmup=args.warmup,
                warm_hops1=args.warm_hops1,
                recipe=SMALL_WIDTH if args.small_width_recipe else None,
            )
            by_hops = {
                h: accuracy(model, hops=h, k=loop_k(schedule, h), n=args.eval_n, seed=seed)
                for h in range(1, MAX_HOPS + 1)
            }
            # The same weights at every k, for the depth curve (H4's acc(k) at fixed hops).
            sweep = {
                h: {
                    k: accuracy(model, hops=h, k=k, n=args.eval_n // 2, seed=seed)
                    for k in range(1, MAX_HOPS + 1)
                }
                for h in (2, 3, 4)
            }
            report["arms"][f"{arm}-seed{seed}"] = {
                "train": stats,
                "accuracy_by_hops": by_hops,
                "k_sweep": sweep,
            }
            print(
                "ARM",
                f"{arm}-seed{seed}",
                json.dumps({"by_hops": {h: round(a, 3) for h, a in by_hops.items()}}),
                flush=True,
            )
            (out / "report.json").write_text(json.dumps(report, indent=2))
    print("HOPS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
