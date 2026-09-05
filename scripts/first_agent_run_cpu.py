#!/usr/bin/env python3
"""The first agentic number: fine-tune the CPU model on rendered episodes, then bench.

    python scripts/first_agent_run_cpu.py --work /tmp/prophet-first-run --minutes 20

Everything the agent pillar built is exercised on real weights, in one loop:

1. ``episodes``  perfect trajectories for the file-task family of
                 ``prophet.eval.agent_bench`` (grep the word, note the file, finish),
                 rendered by ``prophet.agent.render`` into the control-id stream --
                 the same path a promoted quarantine episode takes.
2. ``train``     the first-run checkpoint (``docs/09_FIRST_RUN.md``), with the typed
                 action heads switched on, trained on those episodes: LM loss plus
                 selection, pointer and gate terms whose targets are read off the
                 stream. One episode per row, so every call has its schemas in context.
3. ``bench``     success on unseen tasks with the executable verifier deciding, before
                 and after; the selection head's accuracy at ``<|call|>``; how many
                 argument values the copy pointer filled.

At seven million parameters nothing here is a capability claim. What it measures is
whether the mechanics -- anchors, grammar, selection, copy, verifier gates -- let a
small model learn a task family from its own verified episodes, which is the claim the
agentic pillar makes and the one a bigger run must beat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses  # noqa: E402

import torch  # noqa: E402

from prophet.agent.loop import AgentConfig  # noqa: E402
from prophet.agent.render import render_episode  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.streaming import StreamingLoader, sources_from_iterables  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.agent_bench import file_tools, make_tasks, run_bench  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.checkpoint import CheckpointManager  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "prophet_cpu_first_run.json"


def perfect_trajectory(task) -> list[dict]:
    word = task.goal.split("word ")[1].split("?")[0]
    tools = file_tools(task)
    from prophet.agent.actions import Action

    observation = tools.run(Action("grep", {"word": word}))
    return [
        {"step": 0, "think": "", "action": {"name": "grep", "args": {"word": word}},
         "p_correct": None, "tier": None, "observation": observation},
        # The note is the bare file name: a value the copy pointer can lift from the
        # grep observation verbatim. "answer: <file>" is not a context span.
        {"step": 1, "think": "", "action": {"name": "note", "args": {"text": task.answer}},
         "p_correct": None, "tier": None, "observation": ""},
        {"step": 2, "think": "", "action": {"name": "done", "args": {}},
         "p_correct": None, "tier": None, "observation": ""},
    ]


def build_rows(tokenizer: ProphetTokenizer, n: int, *, seed: int, seq_len: int) -> tuple[list[list[int]], dict]:
    rows, longest, truncated = [], 0, 0
    pad = tokenizer.pad_id
    for task in make_tasks(n, seed=seed):
        text = render_episode(task.goal, file_tools(task), perfect_trajectory(task))
        ids = tokenizer.encode(text, parse_special=True)
        longest = max(longest, len(ids))
        if len(ids) > seq_len:
            truncated += 1
            ids = ids[:seq_len]
        rows.append(ids + [pad] * (seq_len - len(ids)))
    return rows, {"episodes": n, "longest": longest, "truncated": truncated}


def agent_config(cfg: ProphetConfig) -> ProphetConfig:
    return dataclasses.replace(cfg, heads=dataclasses.replace(cfg.heads, action_head=True, action_dk=32))


def bench(model, tokenizer, *, n_tasks: int, seed: int) -> dict:
    cfg = AgentConfig(max_steps=4, think_budget=4, action_budget=64, halt_threshold=None,
                      k_decide=2, tau_done=0.0, tau_act=0.0, tau_ask=0.0)
    report = run_bench(model, tokenizer, make_tasks(n_tasks, seed=seed), cfg)
    steps = [s for e in report.episodes for s in getattr(e, "_records", [])]
    return {
        "tasks": report.n,
        "success_rate": report.success_rate,
        "mean_steps": report.mean_steps,
        "malformed_rate": report.malformed_rate,
        "copied_values": sum(e.copied for e in report.episodes),
        "tool_calls": sum(e.tool_calls for e in report.episodes),
        "curve": report.learning_curve(block=10),
        "summary": report.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the first run's work directory (tokenizer, checkpoint)")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bench-tasks", type=int, default=40)
    ap.add_argument("--bench-before-tasks", type=int, default=20)
    ap.add_argument("--from-scratch", action="store_true", help="do not start from the first-run checkpoint")
    ap.add_argument("--stage", choices=["train", "bench", "all"], default="all")
    args = ap.parse_args()
    work = Path(args.work)
    out_dir = work / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    cfg = agent_config(ProphetConfig.from_json(CONFIG))
    cfg.validate()
    torch.manual_seed(0)
    model = ProphetModel(cfg)
    report: dict = {"config": cfg.name, "parameters": sum(p.numel() for p in model.parameters())}

    if args.stage in ("train", "all"):
        if not args.from_scratch:
            state, meta = CheckpointManager(work / "checkpoints").load_latest()
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            report["init"] = {"from_step": meta.step, "fresh_params": sorted(missing)[:6], "unexpected": len(unexpected)}
        else:
            report["init"] = "scratch"
        rows, data_stats = build_rows(tokenizer, args.episodes, seed=1, seq_len=args.seq_len)
        report["episodes"] = data_stats
        report["bench_before"] = bench(model, tokenizer, n_tasks=args.bench_before_tasks, seed=7)
        print("before:", report["bench_before"]["summary"], flush=True)
        loader = StreamingLoader(
            sources_from_iterables({"episodes": (1.0, rows)}), seq_len=args.seq_len,
            batch_size=args.batch_size, seed=0,
        )
        tc = TrainConfig(
            total_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
            peak_lr_muon=0.01, peak_lr_adamw=2e-3, warmup_frac=0.05, decay_frac=0.3,
            checkpoint_dir=str(out_dir / "checkpoints"), checkpoint_every=100, log_every=25,
            device="cpu", max_wall_seconds=args.minutes * 60.0, mtp_weight=0.0,
        )
        trainer = Trainer(model, loader, tc, model_config=cfg, tokenizer=tokenizer)
        started = time.time()
        history = trainer.train()
        trainer.ckpt.save(trainer.state_dict(), trainer.step)
        last = history[-1]
        report["train"] = {
            "steps": trainer.step, "minutes": (time.time() - started) / 60,
            "loss_first": history[0].loss, "loss_last": last.loss,
            "sel_accuracy_last": last.extra.get("action/sel_accuracy"),
            "loss_action_last": last.extra.get("loss/action"),
            "skipped_nonfinite": trainer.skipped_nonfinite,
        }
        print("train:", report["train"], flush=True)
    else:
        state, _ = CheckpointManager(out_dir / "checkpoints").load_latest()
        model.load_state_dict(state["model"])

    if args.stage in ("bench", "all"):
        model.eval()
        report["bench_after"] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=7)
        print("after:", report["bench_after"]["summary"], flush=True)
        report["bench_after_unseen_seed"] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=11)
        print("after (other seed):", report["bench_after_unseen_seed"]["summary"], flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
