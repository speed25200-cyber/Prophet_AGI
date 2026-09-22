#!/usr/bin/env python3
"""Recall against depth, in miniature: can looping a bounded-state core stand in for state
size when many keys must be remembered at once?

    python scripts/recall_cpu.py --out /tmp/prophet-recall

The CPU analogue of W2-A4 (docs/research/W2 §9, prediction F4): multi-query associative
recall (MQAR). A sequence holds ``m`` key-value pairs, then ``QUERIES`` of the keys; the
model predicts each key's value right after it. The prediction under test: in a model
whose only sequence mixer is a bounded state (gated delta rule), accuracy falls past a
knee set by the state size (``linear_head_dim``), and the number of core passes ``k``
does not move it -- recall is state-size-bound, not depth-bound. A model with the
programme's attention prelude and coda is the control: attention retrieves any pair.

Arms: ``state-d<dk>-k<k>`` (every block a GDN block; prelude, looped core, coda) for each
``--dk`` and ``--k``, and ``layout-k<k>`` (the programme 1 layout, attention outside the
loop). Each model trains on ``m`` drawn from ``--pairs`` and is measured at every ``m``.
Chance is ``1 / VALUES``. A mechanism probe at 10^5 parameters, not a capability claim.
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

PAD, KEYS, VALUES, QUERIES = 0, 64, 16, 8
KEY0, VALUE0 = 1, 1 + KEYS
SEP = VALUE0 + VALUES
VOCAB = SEP + 1


def make_example(rng: random.Random, *, pairs: int) -> tuple[list[int], list[int], list[int]]:
    """(ids, query positions, answers): ``pairs`` distinct keys with values, a separator,
    then ``QUERIES`` keys drawn from them, each followed by its value (the target sits at
    the key's position, so the model predicts the value right after reading the key)."""
    if not 1 <= pairs <= KEYS:
        raise ValueError(f"pairs in 1..{KEYS}")
    keys = rng.sample(range(KEYS), pairs)
    values = {k: rng.randrange(VALUES) for k in keys}
    ids: list[int] = []
    for k in keys:
        ids += [KEY0 + k, VALUE0 + values[k]]
    ids.append(SEP)
    positions, answers = [], []
    for k in (rng.choice(keys) for _ in range(QUERIES)):
        ids.append(KEY0 + k)
        positions.append(len(ids) - 1)
        answers.append(VALUE0 + values[k])
        ids.append(VALUE0 + values[k])
    return ids, positions, answers


def batch(rng: random.Random, *, n: int, pairs: int):
    rows = [make_example(rng, pairs=pairs) for _ in range(n)]
    return (
        torch.tensor([r[0] for r in rows]),
        torch.tensor([r[1] for r in rows]),
        torch.tensor([r[2] for r in rows]),
    )


def config(kind: str, *, dk: int, max_pairs: int) -> ProphetConfig:
    """``state``: every block a GDN block (no attention anywhere). ``layout``: programme 1's
    attention prelude and coda around a GDN core."""
    length = 2 * max_pairs + 1 + 2 * QUERIES
    attention = kind == "layout"
    return ProphetConfig(
        name=f"recall-{kind}-d{dk}",
        d_model=64,
        n_layers=4,
        max_seq_len=max(64, length),
        frontend=FrontendConfig(vocab_size=VOCAB, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"] if attention else ["gdn"],
            n_heads=4,
            n_kv_heads=2,
            qk_norm=False,  # at head_dim 16 it caps the logit at 4 (docs/10 §1)
            sliding_window=length,
            attention_sink_tokens=1 if attention else 0,
            nope_layers=(1,) if attention else (),
            linear_heads=2,
            linear_head_dim=dk,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=1,
            core_layers=1,
            coda_layers=2,
            train_loop_min=1,
            train_loop_max=8,
            default_loop_k=1,
            halting="none",
            core_pattern=["gdn"],
            coda_pattern=["swa", "full_attn"] if attention else ["gdn", "gdn"],
            truncated_backprop_steps=8,
        ),
        heads=HeadsConfig(n_multi_token_predict=0),
    )


def lr_at(step: int, *, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(
    kind: str,
    *,
    dk: int,
    k: int,
    pairs: tuple[int, ...],
    steps: int,
    minutes: float,
    seed: int,
    lr: float,
    warmup: int,
    batch_size: int = 32,
    log=print,
):
    torch.manual_seed(seed)
    cfg = config(kind, dk=dk, max_pairs=max(pairs))
    cfg.validate()
    model = ProphetModel(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    started, losses = time.time(), []
    rows = torch.arange(batch_size).unsqueeze(1)
    for step in range(steps):
        ids, positions, answers = batch(rng, n=batch_size, pairs=pairs[step % len(pairs)])
        logits = model(ids, loop_k=k).logits[rows, positions]
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), answers.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps=steps, peak=lr, warmup=warmup)
        opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            log(
                f"[{kind}-d{dk}-k{k}] step {step} loss {sum(losses[-500:]) / len(losses[-500:]):.3f}"
            )
        if (time.time() - started) / 60 > minutes:
            log(f"[{kind}-d{dk}-k{k}] wall-clock budget reached at step {step}")
            break
    return model, {
        "steps": len(losses),
        "loss_last500": sum(losses[-500:]) / max(len(losses[-500:]), 1),
        "minutes": (time.time() - started) / 60,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def accuracy(model, *, pairs: int, k: int, n: int, seed: int) -> float:
    model.eval()
    rng = random.Random(f"eval-{seed}-{pairs}")
    correct = total = 0
    while total < n * QUERIES:
        m = min(32, n - total // QUERIES)
        ids, positions, answers = batch(rng, n=m, pairs=pairs)
        logits = model(ids, loop_k=k).logits[torch.arange(m).unsqueeze(1), positions]
        correct += int((logits.argmax(-1) == answers).sum())
        total += m * QUERIES
    return correct / total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--dk", default="8,16,32", help="linear_head_dim of the state arms")
    ap.add_argument("--k", default="1,4", help="core passes, train and measure")
    ap.add_argument("--pairs", default="4,8,16,32")
    ap.add_argument("--layout", action="store_true", help="also the attention-layout control")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--minutes", type=float, default=15.0, help="per model")
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args(argv)
    pairs = tuple(int(x) for x in args.pairs.split(","))
    arms = [("state", int(d), int(k)) for d in args.dk.split(",") for k in args.k.split(",")]
    if args.layout:
        arms += [("layout", 16, int(k)) for k in args.k.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            **vars(args),
            "keys": KEYS,
            "values": VALUES,
            "queries": QUERIES,
            "chance": 1 / VALUES,
        },
        "arms": {},
    }
    for kind, dk, k in arms:
        name = f"{kind}-d{dk}-k{k}"
        model, stats = train(
            kind,
            dk=dk,
            k=k,
            pairs=pairs,
            steps=args.steps,
            minutes=args.minutes,
            seed=args.seed,
            lr=args.lr,
            warmup=args.warmup,
        )
        by_pairs = {m: accuracy(model, pairs=m, k=k, n=args.eval_n, seed=args.seed) for m in pairs}
        report["arms"][name] = {"train": stats, "accuracy_by_pairs": by_pairs}
        print("ARM", name, json.dumps({m: round(a, 3) for m, a in by_pairs.items()}), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2))
    print("RECALL_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
