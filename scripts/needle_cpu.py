#!/usr/bin/env python3
"""Recall beyond the window: the number "infinite context" has to earn.

    python scripts/needle_cpu.py --out /tmp/prophet-needle --minutes 12

A synthetic task with one dial, distance. Each sequence lists pairs ``k=v`` (keys and
values are single tokens from disjoint vocabularies), then filler, then asks ``k=?`` for
one of the keys; the model must emit ``v``. Two models of identical size are trained on
it, both with a global window of ``window`` tokens:

- ``full``   -- the global layer sees the whole sequence: the learnability control. If
               this arm does not learn within the budget, the experiment says nothing.
- ``none``   -- the global layer is replaced by a RoPE sliding-window layer: a bounded
               stack without the mechanism, but not the same host layer.
- ``closed`` -- the ledger layer itself with its recall gate pinned shut and frozen: the
               clean control, since it differs from ``ledger`` by the read term only.
- ``ledger`` -- the global layer writes evicted keys and values into a bounded ledger
               and reads it back (``mixer.global_memory``).

Accuracy is then measured **by distance** between the asked pair and the question,
inside and beyond the window. The unwindowed answer is known: a full-attention model
solves this at any distance its training covered. The claim under test is narrower
and exact: with memory bounded at the window plus the ledger, does recall survive past
the window at all, and how does it decay with distance? Numbers are reported per
distance bucket; a ``ledger`` column that equals ``none`` beyond the window means the
mechanism bought nothing at this scale.

Tiny by design (a few hundred thousand parameters, minutes on CPU). Not a claim about
any real corpus; the ablation on real text is in ``prophet.plan``.
"""

from __future__ import annotations

import argparse
import json
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

N_KEYS, N_VALUES = 16, 16
FILLER = 128  # one filler token id band
KEY0, VAL0, FILL0, EQ, QMARK, SEP = 1, 1 + N_KEYS, 1 + N_KEYS + N_VALUES, 200, 201, 202
VOCAB = 256


def make_example(rng: random.Random, *, n_pairs: int, max_gap: int) -> tuple[list[int], int, int]:
    """Returns (ids, answer position, distance from the asked pair to the question)."""
    keys = rng.sample(range(N_KEYS), n_pairs)
    vals = [rng.randrange(N_VALUES) for _ in keys]
    asked = rng.randrange(n_pairs)
    ids: list[int] = []
    pair_pos = []
    for k, v in zip(keys, vals, strict=True):
        pair_pos.append(len(ids))
        ids += [KEY0 + k, EQ, VAL0 + v, SEP]
    gap = rng.randrange(0, max_gap + 1)
    ids += [FILL0 + rng.randrange(8) for _ in range(gap)]
    ids += [KEY0 + keys[asked], EQ]
    answer_pos = len(ids)  # the model predicts the value at this position
    ids.append(VAL0 + vals[asked])
    distance = answer_pos - pair_pos[asked]
    return ids, answer_pos, distance


def batch(rng: random.Random, *, n: int, n_pairs: int, max_gap: int, length: int):
    rows, targets, positions, distances = [], [], [], []
    for _ in range(n):
        ids, pos, dist = make_example(rng, n_pairs=n_pairs, max_gap=max_gap)
        ids = ids[:length] + [0] * (length - len(ids))
        rows.append(ids)
        targets.append(ids[pos])
        positions.append(pos - 1)  # logits at pos-1 predict the token at pos
        distances.append(dist)
    return (
        torch.tensor(rows), torch.tensor(targets), torch.tensor(positions), torch.tensor(distances),
    )


def config(arm: str, *, window: int, slots: int, length: int) -> ProphetConfig:
    """``full``: exact global attention (window = the whole sequence); ``none``: the
    global layer windowed; ``ledger``: windowed plus the ledger."""
    memory = "ledger" if arm in ("ledger", "closed") else "none"
    global_window = length if arm == "full" else window
    # "none": the global layer becomes a plain sliding-window layer (same window as the
    # ledger arm, no ledger), which is what a bounded stack without the mechanism is.
    pattern = ["swa", "swa"] if arm == "none" else ["swa", "full_attn"]
    return ProphetConfig(
        name=f"needle-{arm}", d_model=64, n_layers=4, max_seq_len=1024,
        frontend=FrontendConfig(vocab_size=VOCAB, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=pattern, n_heads=4, n_kv_heads=2, sliding_window=global_window,
            attention_sink_tokens=1, nope_layers=(1,), global_memory=memory,
            global_window=global_window, global_ledger_slots=slots, global_ledger_top_k=8,
            linear_heads=2, linear_head_dim=16,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(enabled=True, prelude_layers=1, core_layers=1, coda_layers=2,
                                      train_loop_min=1, train_loop_max=1, default_loop_k=1,
                                      halting="none", core_pattern=["gdn"], coda_pattern=pattern),
        heads=HeadsConfig(n_multi_token_predict=0),
    )


def accuracy_by_distance(model, rng, *, n: int, n_pairs: int, max_gap: int, length: int, window: int) -> dict:
    model.eval()
    buckets: dict[str, list[int]] = {}
    with torch.no_grad():
        for _ in range(n // 32):
            ids, targets, positions, distances = batch(rng, n=32, n_pairs=n_pairs, max_gap=max_gap, length=length)
            logits = model(ids, loop_k=1).logits
            pred = logits[torch.arange(32), positions].argmax(-1)
            for ok, d in zip((pred == targets).tolist(), distances.tolist(), strict=True):
                key = "inside" if d <= window else ("1-2x" if d <= 2 * window else ">2x")
                buckets.setdefault(key, []).append(int(ok))
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(buckets.items())}


def train(memory: str, *, window: int, slots: int, steps: int, minutes: float, seed: int, length: int,
          n_pairs: int, max_gap: int, log) -> tuple[ProphetModel, dict]:
    torch.manual_seed(seed)
    cfg = config(memory, window=window, slots=slots, length=length)
    cfg.validate()
    model = ProphetModel(cfg).train()
    if memory == "closed":
        from prophet.modeling.layers import LedgerAttention

        for layer in model.modules():
            if isinstance(layer, LedgerAttention):
                with torch.no_grad():
                    layer.gate.fill_(-1e4)  # sigmoid = 0: the read never enters
                layer.gate.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.time()
    losses = []
    for step in range(steps):
        ids, targets, positions, _ = batch(rng, n=32, n_pairs=n_pairs, max_gap=max_gap, length=length)
        logits = model(ids, loop_k=1).logits
        picked = logits[torch.arange(32), positions]
        loss = torch.nn.functional.cross_entropy(picked.float(), targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 50 == 0:
            log(f"[{memory}] step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.3f}")
        if (time.time() - started) / 60 > minutes:
            log(f"[{memory}] wall-clock budget reached at step {step}")
            break
    return model, {"steps": len(losses), "loss_last50": sum(losses[-50:]) / max(len(losses[-50:]), 1),
                   "minutes": (time.time() - started) / 60, "parameters": sum(p.numel() for p in model.parameters())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--max-gap", type=int, default=96)
    ap.add_argument("--length", type=int, default=160)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--arms", default="full,none,closed,ledger")
    ap.add_argument("--minutes", type=float, default=12.0, help="per model")
    ap.add_argument("--eval-n", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    report: dict = {"window": args.window, "slots": args.slots, "pairs": args.pairs, "max_gap": args.max_gap}
    for memory in [a for a in args.arms.split(",") if a]:
        model, stats = train(memory, window=args.window, slots=args.slots, steps=args.steps, minutes=args.minutes,
                             seed=args.seed, length=args.length, n_pairs=args.pairs, max_gap=args.max_gap, log=log)
        acc = accuracy_by_distance(model, random.Random(args.seed + 100), n=args.eval_n, n_pairs=args.pairs,
                                   max_gap=args.max_gap, length=args.length, window=args.window)
        report[memory] = {"train": stats, "accuracy_by_distance": {k: {"accuracy": a, "n": n} for k, (a, n) in acc.items()}}
        log(f"[{memory}] {stats} accuracy {acc}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    arms = [a for a in args.arms.split(",") if a]
    print("\n| Distance de la paire à la question | " + " | ".join(arms) + " |\n|---|" + "---:|" * len(arms))
    for key in ("inside", "1-2x", ">2x"):
        label = {"inside": "≤ fenêtre", "1-2x": "1 à 2 fenêtres", ">2x": "> 2 fenêtres"}[key]
        cells = [f"{report[a]['accuracy_by_distance'].get(key, {}).get('accuracy', float('nan')):.1%}" for a in arms]
        print(f"| {label} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
