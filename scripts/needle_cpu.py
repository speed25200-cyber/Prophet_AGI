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
- ``closed-rope`` / ``ledger-rope`` -- the same two on a RoPE host: the ledger layer
               rotates at attention time and addresses the ledger with unrotated keys
               (``mixer.global_ledger_rope``).

Accuracy is then measured **by distance** between the asked pair and the question,
inside and beyond the window. The unwindowed answer is known: a full-attention model
solves this at any distance its training covered. The claim under test is narrower
and exact: with memory bounded at the window plus the ledger, does recall survive past
the window at all, and how does it decay with distance? Numbers are reported per
distance bucket; a ``ledger`` column that equals ``none`` beyond the window means the
mechanism bought nothing at this scale.

The first protocol (``--form eq --inside-fraction -1``: two-hop pairs, uniform gap) was
insensitive -- the full-attention control itself reached 24% where chance is 6%, so the
arms could not be told apart. The default is now one hop and half the examples inside
the window: the learnability control must pass before any column means anything.

Tiny by design (a few hundred thousand parameters, minutes on CPU). Not a claim about
any real corpus; the ablation on real text is in ``prophet.plan``.
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

N_KEYS, N_VALUES = 16, 16
KEY0, VAL0, FILL0, EQ, QMARK, SEP = 1, 1 + N_KEYS, 1 + N_KEYS + N_VALUES, 200, 201, 202
VOCAB = 256


def set_vocab(n_keys: int, n_values: int) -> None:
    """Rebind the key and value bands (``--keys``, ``--values``); the filler band and the
    control ids follow. More keys than a bounded state can hold is what separates a
    ledger from the core's own recall."""
    global N_KEYS, N_VALUES, KEY0, VAL0, FILL0
    if n_keys + n_values + 8 > EQ:
        raise ValueError(f"keys + values must leave room below {EQ}")
    N_KEYS, N_VALUES = n_keys, n_values
    KEY0, VAL0, FILL0 = 1, 1 + N_KEYS, 1 + N_KEYS + N_VALUES


def make_example(rng: random.Random, *, n_pairs: int, max_gap: int, window: int | None = None,
                 inside_fraction: float = -1.0, form: str = "plain",
                 n_ask: int = 1) -> tuple[list[int], list[int], list[int]]:
    """Returns (ids, answer positions, distances from each asked pair to its question).

    ``n_ask`` keys are asked one after another at the end (each answer then sits in
    the context of the next question): ``n_ask`` supervised positions per sequence
    instead of one, the same mechanism to learn.

    ``form="plain"`` lists ``k v |`` and asks ``k``: the value is predicted at the key
    itself, one hop (an induction head). ``form="eq"`` is the first protocol, ``k = v |``
    asked as ``k =``, two hops. ``inside_fraction`` in [0, 1] draws that share of the
    examples with the asked pair inside ``window`` and the rest beyond it, so the host
    learns the skill on the answerable half instead of drowning it in the unanswerable
    one; negative means the gap is uniform, the first protocol.
    """
    keys = rng.sample(range(N_KEYS), n_pairs)
    vals = [rng.randrange(N_VALUES) for _ in keys]
    order = rng.sample(range(n_pairs), min(n_ask, n_pairs))
    asked = order[0]
    ids: list[int] = []
    pair_pos = []
    for k, v in zip(keys, vals, strict=True):
        pair_pos.append(len(ids))
        ids += [KEY0 + k, EQ, VAL0 + v, SEP] if form == "eq" else [KEY0 + k, VAL0 + v, SEP]
    question = [KEY0 + keys[asked], EQ] if form == "eq" else [KEY0 + keys[asked]]
    # distance = answer_pos - pair_pos[asked] = len(ids) + gap + len(question) - pair_pos[asked]
    base = len(ids) + len(question) - pair_pos[asked]
    if inside_fraction >= 0 and window is not None:
        widest_inside = window - base  # the largest gap that keeps the pair inside
        if rng.random() < inside_fraction and widest_inside >= 0:
            gap = rng.randrange(0, min(widest_inside, max_gap) + 1)
        else:
            gap = rng.randrange(max(widest_inside + 1, 0), max_gap + 1) if widest_inside < max_gap else max_gap
    else:
        gap = rng.randrange(0, max_gap + 1)
    ids += [FILL0 + rng.randrange(8) for _ in range(gap)]
    positions, distances = [], []
    for asked in order:
        ids += [KEY0 + keys[asked], EQ] if form == "eq" else [KEY0 + keys[asked]]
        positions.append(len(ids))  # the model predicts the value at this position
        distances.append(len(ids) - pair_pos[asked])
        ids.append(VAL0 + vals[asked])
    return ids, positions, distances


def batch(rng: random.Random, *, n: int, n_pairs: int, max_gap: int, length: int, window: int | None = None,
          inside_fraction: float = -1.0, form: str = "plain", n_ask: int = 1):
    """``targets``, ``positions`` and ``distances`` are ``(n, n_ask)``."""
    rows, targets, positions, distances = [], [], [], []
    for _ in range(n):
        ids, pos, dist = make_example(rng, n_pairs=n_pairs, max_gap=max_gap, window=window,
                                      inside_fraction=inside_fraction, form=form, n_ask=n_ask)
        if len(ids) > length:
            raise ValueError(f"sequence of {len(ids)} tokens exceeds --length {length}")
        ids = ids + [0] * (length - len(ids))
        rows.append(ids)
        targets.append([ids[p] for p in pos])
        positions.append([p - 1 for p in pos])  # logits at pos-1 predict the token at pos
        distances.append(dist)
    return (
        torch.tensor(rows), torch.tensor(targets), torch.tensor(positions), torch.tensor(distances),
    )


def config(arm: str, *, window: int, slots: int, length: int, qk_norm: bool = False,
           linear_heads: int = 2, linear_head_dim: int = 16) -> ProphetConfig:
    """``full``: exact global attention (window = the whole sequence); ``none``: the
    global layer windowed; ``ledger``: windowed plus the ledger.

    ``qk_norm`` is off by default here: with normalised queries and keys the attention
    logit is bounded by sqrt(head_dim) times the learned gains -- 4 at head_dim 16 --
    so one key among 160 can take at most e^4 / (e^4 + 159) = 26% of the mass. That
    is the plateau the first three protocols hit (24-25% with full attention).

    ``linear_heads`` and ``linear_head_dim`` size the delta core's state: an associative
    memory of roughly ``head_dim`` pairs per head. With 6 pairs in the default 2 x 16 it
    recalls beyond the window as well as full attention does, so to see what a ledger
    adds the pairs must exceed the state -- and shrinking the state keeps the control
    learnable, where 64 keys and 24 pairs were not in 4 000 steps.
    """
    rope = arm.endswith("-rope")  # the ledger layer keeps RoPE (mixer.global_ledger_rope)
    base = arm[: -len("-rope")] if rope else arm
    memory = "ledger" if base in ("ledger", "closed") else "none"
    global_window = length if base == "full" else window
    # "none": the global layer becomes a plain sliding-window layer (same window as the
    # ledger arm, no ledger), which is what a bounded stack without the mechanism is.
    pattern = ["swa", "swa"] if base == "none" else ["swa", "full_attn"]
    return ProphetConfig(
        name=f"needle-{arm}", d_model=64, n_layers=4, max_seq_len=1024,
        frontend=FrontendConfig(vocab_size=VOCAB, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=pattern, n_heads=4, n_kv_heads=2, sliding_window=global_window, qk_norm=qk_norm,
            attention_sink_tokens=1, nope_layers=() if rope else (1,), global_memory=memory,
            global_ledger_rope=rope,
            global_window=global_window, global_ledger_slots=slots, global_ledger_top_k=8,
            linear_heads=linear_heads, linear_head_dim=linear_head_dim,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(enabled=True, prelude_layers=1, core_layers=1, coda_layers=2,
                                      train_loop_min=1, train_loop_max=1, default_loop_k=1,
                                      halting="none", core_pattern=["gdn"], coda_pattern=pattern),
        heads=HeadsConfig(n_multi_token_predict=0),
    )


def accuracy_by_distance(model, rng, *, n: int, n_pairs: int, max_gap: int, length: int, window: int,
                         inside_fraction: float = -1.0, form: str = "plain", n_ask: int = 1) -> dict:
    model.eval()
    buckets: dict[str, list[int]] = {}
    with torch.no_grad():
        for _ in range(n // 32):
            ids, targets, positions, distances = batch(rng, n=32, n_pairs=n_pairs, max_gap=max_gap, length=length,
                                                       window=window, inside_fraction=inside_fraction, form=form,
                                                       n_ask=n_ask)
            logits = model(ids, loop_k=1).logits
            pred = logits[torch.arange(32).unsqueeze(1), positions].argmax(-1)  # (32, n_ask)
            for ok, d in zip((pred == targets).flatten().tolist(), distances.flatten().tolist(), strict=True):
                key = "inside" if d <= window else ("1-2x" if d <= 2 * window else ">2x")
                buckets.setdefault(key, []).append(int(ok))
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(buckets.items())}


def lr_at(step: int, *, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(memory: str, *, window: int, slots: int, steps: int, minutes: float, seed: int, length: int,
          n_pairs: int, max_gap: int, log, lr: float = 1e-3, warmup: int = 100,
          inside_fraction: float = -1.0, form: str = "plain", n_ask: int = 1,
          qk_norm: bool = False, linear_heads: int = 2,
          linear_head_dim: int = 16) -> tuple[ProphetModel, dict]:
    torch.manual_seed(seed)
    cfg = config(memory, window=window, slots=slots, length=length, qk_norm=qk_norm,
                 linear_heads=linear_heads, linear_head_dim=linear_head_dim)
    cfg.validate()
    model = ProphetModel(cfg).train()
    if memory.startswith("closed"):
        from prophet.modeling.layers import LedgerAttention

        for layer in model.modules():
            if isinstance(layer, LedgerAttention):
                with torch.no_grad():
                    layer.gate.fill_(-1e4)  # sigmoid = 0: the read never enters
                layer.gate.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.time()
    losses = []
    for step in range(steps):
        ids, targets, positions, _ = batch(rng, n=32, n_pairs=n_pairs, max_gap=max_gap, length=length,
                                           window=window, inside_fraction=inside_fraction, form=form, n_ask=n_ask)
        logits = model(ids, loop_k=1).logits
        picked = logits[torch.arange(32).unsqueeze(1), positions]  # (32, n_ask, vocab)
        loss = torch.nn.functional.cross_entropy(picked.float().flatten(0, 1), targets.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps=steps, peak=lr, warmup=warmup)
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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--inside-fraction", type=float, default=0.5,
                    help="share of examples with the asked pair inside the window (negative: uniform gap)")
    ap.add_argument("--form", choices=["plain", "eq"], default="plain",
                    help="'plain': k v | ... k -> v (one hop); 'eq': k = v | ... k = -> v (two hops)")
    ap.add_argument("--questions", type=int, default=None,
                    help="keys asked at the end of each sequence (default: all the pairs)")
    ap.add_argument("--qk-norm", action="store_true", help="normalise queries and keys (bounds the logit)")
    ap.add_argument("--keys", type=int, default=16, help="size of the key vocabulary (pairs are drawn from it)")
    ap.add_argument("--values", type=int, default=16, help="size of the value vocabulary")
    ap.add_argument("--state-heads", type=int, default=2, help="delta-core heads (the state is ~head_dim pairs per head)")
    ap.add_argument("--state-dim", type=int, default=16, help="delta-core head dimension")
    args = ap.parse_args()
    set_vocab(args.keys, args.values)
    if args.pairs > args.keys:
        raise SystemExit("--pairs cannot exceed --keys (keys are distinct within a sequence)")
    n_ask = args.pairs if args.questions is None else args.questions
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    report: dict = {"window": args.window, "slots": args.slots, "pairs": args.pairs, "max_gap": args.max_gap,
                    "lr": args.lr, "warmup": args.warmup, "inside_fraction": args.inside_fraction, "form": args.form,
                    "steps": args.steps, "questions": n_ask, "qk_norm": args.qk_norm, "keys": args.keys,
                    "values": args.values, "length": args.length, "state_heads": args.state_heads,
                    "state_dim": args.state_dim}
    for memory in [a for a in args.arms.split(",") if a]:
        model, stats = train(memory, window=args.window, slots=args.slots, steps=args.steps, minutes=args.minutes,
                             seed=args.seed, length=args.length, n_pairs=args.pairs, max_gap=args.max_gap, log=log,
                             lr=args.lr, warmup=args.warmup, inside_fraction=args.inside_fraction, form=args.form,
                             n_ask=n_ask, qk_norm=args.qk_norm, linear_heads=args.state_heads,
                             linear_head_dim=args.state_dim)
        acc = accuracy_by_distance(model, random.Random(args.seed + 100), n=args.eval_n, n_pairs=args.pairs,
                                   max_gap=args.max_gap, length=args.length, window=args.window,
                                   inside_fraction=args.inside_fraction, form=args.form, n_ask=n_ask)
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
