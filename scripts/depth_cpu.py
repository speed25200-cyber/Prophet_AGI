#!/usr/bin/env python3
"""Latent depth versus chain-of-thought tokens: the token-efficiency number of the loop.

    python scripts/depth_cpu.py --out /tmp/prophet-depth --ops 6

A synthetic task whose answer needs sequential computation: ``d0 op d1 op d2 ... =``
evaluated left to right modulo 10 (digits, ``+ - *``). A transformer solves it with
depth proportional to the number of operations (the transformation monoid is not
solvable, so no logarithmic shortcut exists to learn), which makes it the cleanest
place to ask what the recurrent loop buys: **the same weights** run at ``k`` passes of
the core, answering directly, against the same weights at one pass emitting the running
values as chain-of-thought tokens.

Arms (identical parameter count, same data, same steps):

- ``k1``       -- one core pass, the answer digit predicted right after ``=``.
- ``k2``/``k4`` -- two or four passes of the shared core, answer predicted directly.
- ``k1-cot``   -- one pass, but after ``=`` the model emits the running value after each
                  operation and the last one is the answer: the chain of thought, one
                  token per step, decoded greedily at evaluation.
- ``kN-sup``   -- N passes of the core, **one token emitted**, and each iteration read
                  out (``recurrent.iteration_readout``) and trained on the running value
                  after the matching operation: the chain's per-step signal moved into
                  the depth dimension. ``kops`` / ``kops-sup`` set N to the number of
                  operations of the block.

Reported per arm: accuracy, tokens emitted per answer (1, or ``ops``), and core passes
per emitted token. The claim under test is exact: does latent depth (``k4``, one token)
reach the accuracy chain-of-thought buys with ``ops`` tokens? Tiny and synthetic by
design; the 100M+ question is in ``prophet.plan``.
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

DIGIT0, PLUS, MINUS, TIMES, EQ, PAD = 1, 11, 12, 13, 14, 0
OPS = {PLUS: lambda a, b: (a + b) % 10, MINUS: lambda a, b: (a - b) % 10, TIMES: lambda a, b: (a * b) % 10}
VOCAB = 16


def make_example(rng: random.Random, *, ops: int) -> tuple[list[int], list[int]]:
    """Returns (prompt ids ending in ``=``, running values after each operation)."""
    value = rng.randrange(10)
    ids = [DIGIT0 + value]
    running = []
    for _ in range(ops):
        op = rng.choice(list(OPS))
        digit = rng.randrange(10)
        value = OPS[op](value, digit)
        ids += [op, DIGIT0 + digit]
        running.append(value)
    ids.append(EQ)
    return ids, running


def batch(rng: random.Random, *, n: int, ops: int, cot: bool, sup: bool = False):
    """Rows padded to one length; ``targets`` holds the ids to predict at ``positions``
    (one answer, or the ``ops`` running values). ``sup`` keeps the direct form (no chain
    in the input) but returns every running value: the per-iteration targets, all read
    at the single ``=`` position."""
    rows, targets, positions = [], [], []
    for _ in range(n):
        ids, running = make_example(rng, ops=ops)
        prompt_len = len(ids)
        if cot:
            full = ids + [DIGIT0 + r for r in running]
            targets.append([DIGIT0 + r for r in running])
            positions.append(list(range(prompt_len - 1, prompt_len - 1 + ops)))
        elif sup:
            full = ids + [DIGIT0 + running[-1]]
            targets.append([DIGIT0 + r for r in running])
            positions.append([prompt_len - 1])
        else:
            full = ids + [DIGIT0 + running[-1]]
            targets.append([DIGIT0 + running[-1]])
            positions.append([prompt_len - 1])
        rows.append(full)
    length = max(len(r) for r in rows)
    rows = [r + [PAD] * (length - len(r)) for r in rows]
    return torch.tensor(rows), torch.tensor(targets), torch.tensor(positions)


def config(k: int, *, length: int, qk_norm: bool = False, readout: bool = False) -> ProphetConfig:
    """``qk_norm`` off by default: at head_dim 16 it caps the attention logit at 4 (see
    ``scripts/needle_cpu.py`` and ``ProphetConfig.design_warnings``). ``readout`` turns on
    the per-iteration read-out the ``-sup`` arms train on."""
    return ProphetConfig(
        name=f"depth-k{k}", d_model=64, n_layers=4, max_seq_len=256,
        frontend=FrontendConfig(vocab_size=VOCAB, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=4, n_kv_heads=2, sliding_window=length, qk_norm=qk_norm,
            attention_sink_tokens=1, nope_layers=(1,), linear_heads=2, linear_head_dim=16,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(enabled=True, prelude_layers=1, core_layers=1, coda_layers=2,
                                      train_loop_min=k, train_loop_max=k, default_loop_k=k,
                                      halting="none", core_pattern=["gdn"], coda_pattern=["swa", "full_attn"],
                                      truncated_backprop_steps=k, iteration_readout=readout),
        heads=HeadsConfig(n_multi_token_predict=0),
    )


def lr_at(step: int, *, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(k: int, *, cot: bool, ops: int, steps: int, minutes: float, seed: int, lr: float, warmup: int,
          log, qk_norm: bool = False, sup: bool = False) -> tuple[ProphetModel, dict]:
    torch.manual_seed(seed)
    cfg = config(k, length=2 * ops + 2 + ops, qk_norm=qk_norm, readout=sup)
    cfg.validate()
    model = ProphetModel(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.time()
    losses = []
    for step in range(steps):
        ids, targets, positions = batch(rng, n=32, ops=ops, cot=cot, sup=sup)
        out = model(ids, loop_k=k)
        logits = out.logits
        if sup:
            # Iteration i answers the running value after operation min(i, ops-1); the
            # last iteration always answers the final one, which is also what the real
            # read-out predicts (the last probe equals the output, tested).
            pos = positions[:, 0]
            rows = torch.arange(32)
            loss = torch.nn.functional.cross_entropy(logits[rows, pos].float(), targets[:, -1])
            for i, h in enumerate(out.hidden_per_step):
                target_index = ops - 1 if i == k - 1 else min(i, ops - 1)
                step_logits = model.lm_head(h[rows, pos])
                loss = loss + torch.nn.functional.cross_entropy(step_logits.float(), targets[:, target_index])
            loss = loss / (len(out.hidden_per_step) + 1)
        else:
            picked = logits[torch.arange(32).unsqueeze(1), positions]  # (32, m, vocab)
            loss = torch.nn.functional.cross_entropy(picked.float().flatten(0, 1), targets.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps=steps, peak=lr, warmup=warmup)
        opt.step()
        losses.append(float(loss))
        if step % 100 == 0:
            log(f"[k{k}{'-cot' if cot else ''}] step {step} loss {sum(losses[-100:]) / len(losses[-100:]):.3f}")
        if (time.time() - started) / 60 > minutes:
            log(f"[k{k}] wall-clock budget reached at step {step}")
            break
    return model, {"steps": len(losses), "loss_last100": sum(losses[-100:]) / max(len(losses[-100:]), 1),
                   "minutes": (time.time() - started) / 60, "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def accuracy(model, k: int, rng: random.Random, *, cot: bool, ops: int, n: int) -> dict:
    model.eval()
    correct = chain_exact = 0
    for _ in range(n // 32):
        ids, targets, positions = batch(rng, n=32, ops=ops, cot=cot)
        if not cot:
            pred = model(ids, loop_k=k).logits[torch.arange(32), positions[:, 0]].argmax(-1)
            correct += int((pred == targets[:, 0]).sum())
            continue
        # Greedy chain: the prompt up to '=' is the same length for every row.
        prompt = ids[:, : positions[0, 0] + 1]
        for _ in range(ops):
            nxt = model(prompt, loop_k=k).logits[:, -1].argmax(-1)
            prompt = torch.cat([prompt, nxt.unsqueeze(1)], dim=1)
        chain = prompt[:, -ops:]
        correct += int((chain[:, -1] == targets[:, -1]).sum())
        chain_exact += int((chain == targets).all(-1).sum())
    n_done = (n // 32) * 32
    return {"accuracy": correct / n_done, "chain_exact": (chain_exact / n_done) if cot else None, "n": n_done,
            "tokens_per_answer": ops if cot else 1, "core_passes_per_token": k}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", default="6", help="operations per expression; a comma-separated list sweeps")
    ap.add_argument("--qk-norm", action="store_true", help="normalise queries and keys (bounds the logit)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--arms", default="k1,k2,k4,k1-cot")
    ap.add_argument("--minutes", type=float, default=10.0, help="per model")
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    ops_list = [int(o) for o in args.ops.split(",") if o]
    report: dict = {"ops": ops_list, "steps": args.steps, "lr": args.lr, "chance": 0.1, "qk_norm": args.qk_norm}
    arms = [a for a in args.arms.split(",") if a]
    for ops in ops_list:
        report[str(ops)] = {}
        for arm in arms:
            head = arm.split("-")[0]
            k = ops if head == "kops" else int(head[1:])
            cot = arm.endswith("-cot")
            sup = arm.endswith("-sup")
            model, stats = train(k, cot=cot, ops=ops, steps=args.steps, minutes=args.minutes, seed=args.seed,
                                 lr=args.lr, warmup=args.warmup, log=log, qk_norm=args.qk_norm, sup=sup)
            acc = accuracy(model, k, random.Random(args.seed + 100), cot=cot, ops=ops, n=args.eval_n)
            acc["core_passes_per_token"] = k
            report[str(ops)][arm] = {"train": stats, **acc}
            log(f"[ops={ops} {arm}] {stats} {acc}")
        (out / "report.json").write_text(json.dumps(report, indent=2))
    print("\n(tokens émis par réponse : 1 pour les bras directs, un par opération pour la chaîne)")
    print("| Opérations | " + " | ".join(arms) + " |\n|---:|" + "---:|" * len(arms))
    for ops in ops_list:
        cells = [f"{report[str(ops)][a]['accuracy']:.1%}" for a in arms]
        print(f"| {ops} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
