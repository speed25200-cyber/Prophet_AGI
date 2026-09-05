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
                 argument values the copy pointer filled. ``--carry-bench`` adds the
                 same bench with the recurrent state carried across episodes.

``--episodes-per-row N`` is the *sequences of episodes* recipe: N consecutive episodes
share a training row, so an episode learns to start from the state the previous one
left -- what a carried session state is at inference. One episode per row never shows
the model such a state, and the carried bench then measures an undefined input.

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

from prophet.agent import tasks as task_families  # noqa: E402
from prophet.agent.loop import AgentConfig  # noqa: E402
from prophet.agent.render import render_episode  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader, sources_from_iterables  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.agent_bench import file_tools, make_tasks, run_bench  # noqa: E402
from prophet.eval.harness import evaluate_bpb  # noqa: E402
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


def build_rows(tokenizer: ProphetTokenizer, n: int, *, seed: int, seq_len: int,
               families: list[str] | None = None, per_row: int = 1,
               related: bool = False) -> tuple[list[list[int]], dict]:
    """Rendered perfect episodes. ``families=None`` is the original file family of the
    benchmark; otherwise ``n`` episodes of each named family of ``prophet.agent.tasks``,
    interleaved.

    ``per_row`` consecutive episodes share one row, each starting at its ``<|bos|>``
    exactly as the loop feeds them: the recurrent state an episode starts from is then
    the one the previous episode left, which is the distribution a *carried* session
    state is drawn from at inference (``run_bench(carry_session=True)``). With one
    episode per row that state is never seen in training, and the carried bench
    measures an undefined input, not the mechanism.

    ``related`` builds the rows from ``prophet.agent.tasks.make_related_tasks``: runs of
    ``per_row`` lookup episodes where a file already read by the previous episode is
    answered without reading it again. ``n`` is then the number of episodes overall.
    """
    if per_row < 1:
        raise ValueError("per_row must be at least 1")
    rows, longest, truncated = [], 0, 0
    pad = tokenizer.pad_id
    if related:
        episodes = [
            (t.goal, task_families.tools_for(t), task_families.perfect_trajectory(t))
            for t in task_families.make_related_tasks(max(n // per_row, 1), size=per_row, seed=seed)
        ]
    elif families is None:
        episodes = [(task.goal, file_tools(task), perfect_trajectory(task)) for task in make_tasks(n, seed=seed)]
    else:
        per_family = [task_families.make_tasks(n, family=f, seed=seed) for f in families]
        episodes = [
            (t.goal, task_families.tools_for(t), task_families.perfect_trajectory(t))
            for group in zip(*per_family, strict=True) for t in group
        ]
    encoded = []
    for goal, tools, trajectory in episodes:
        text = render_episode(goal, tools, trajectory)
        ids = [tokenizer.bos_id] + tokenizer.encode(text, parse_special=True)
        longest = max(longest, len(ids))
        encoded.append(ids)
    for start in range(0, len(encoded), per_row):
        ids = [t for episode in encoded[start : start + per_row] for t in episode]
        if len(ids) > seq_len:
            truncated += 1
            ids = ids[:seq_len]
        rows.append(ids + [pad] * (seq_len - len(ids)))
    return rows, {"episodes": len(encoded), "rows": len(rows), "per_row": per_row,
                  "longest": longest, "truncated": truncated}


def replay_source(work: Path, tokenizer: ProphetTokenizer, weight: float) -> TokenisedSource:
    """The first run's prose and code as one language-modelling source, so the agentic
    fine-tune keeps seeing what the base model was trained on."""
    class _Both:
        # A class body does not see the enclosing function's names; bind them in __init__.
        def __init__(self) -> None:
            self.name = "replay"
            self.weight = weight
            self.parts = [LocalTextSource.from_root(work / "corpus", n, 1.0) for n in ("prose", "code")]
        def n_documents(self) -> int:
            return sum(p.n_documents() for p in self.parts)
        def open(self, start: int):
            for part in self.parts:
                n = part.n_documents()
                if start >= n:
                    start -= n
                    continue
                yield from part.open(start)
                start = 0
    return TokenisedSource(_Both(), tokenizer, max_epochs=None)


def heldout_bpb(work: Path, model, tokenizer: ProphetTokenizer, *, seq_len: int = 256, max_docs: int = 400) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from first_run_cpu import _batches

    docs = [json.loads(line)["text"] for line in (work / "benchmarks" / "heldout.jsonl").read_text().splitlines() if line.strip()][:max_docs]
    model.eval()
    with torch.no_grad():
        r = evaluate_bpb(model, _batches(tokenizer, docs, seq_len=seq_len, batch_size=8))
    return {"bpb": r.bits_per_byte, "nats": r.nats_per_token, "docs": len(docs)}


def agent_config(cfg: ProphetConfig) -> ProphetConfig:
    return dataclasses.replace(cfg, heads=dataclasses.replace(cfg.heads, action_head=True, action_dk=32))


def bench(model, tokenizer, *, n_tasks: int, seed: int, family: str | None = None,
          carry: bool = False, related_size: int = 0) -> dict:
    """``carry`` starts every episode from the recurrent state the previous one left:
    the session-carry measurement of ``docs/10_NEXT_ARCHITECTURE.md``. ``related_size``
    benches the related lookup sequences instead (runs of that many episodes) and
    reports seen and unseen files apart: tokens, success and how many reads the model
    still made on a file the previous episode read."""
    cfg = AgentConfig(max_steps=4, think_budget=4, action_budget=64, halt_threshold=None,
                      k_decide=2, tau_done=0.0, tau_act=0.0, tau_ask=0.0)
    tasks = None
    if related_size:
        tasks = task_families.make_related_tasks(max(n_tasks // related_size, 1), size=related_size, seed=seed)
        report = run_bench(model, tokenizer, tasks, cfg, tools_for=task_families.tools_for,
                           verifier_for_task=task_families.verifier_for, carry_session=carry)
    elif family is None:
        report = run_bench(model, tokenizer, make_tasks(n_tasks, seed=seed), cfg, carry_session=carry)
    else:
        report = run_bench(
            model, tokenizer, task_families.make_tasks(n_tasks, family=family, seed=seed), cfg,
            tools_for=task_families.tools_for, verifier_for_task=task_families.verifier_for,
            carry_session=carry,
        )
    by_seen = None
    if tasks is not None:
        by_seen = {}
        for flag, label in ((False, "unseen"), (True, "seen")):
            eps = [e for t, e in zip(tasks, report.episodes, strict=True) if t.extra["seen"] == flag]
            by_seen[label] = {
                "n": len(eps),
                "success_rate": sum(e.verified for e in eps) / max(len(eps), 1),
                "mean_tokens": sum(e.tokens for e in eps) / max(len(eps), 1),
                "reads_per_episode": sum(e.tool_calls for e in eps) / max(len(eps), 1),
            }
    return {
        "by_seen": by_seen,
        "tasks": report.n,
        "success_rate": report.success_rate,
        "mean_steps": report.mean_steps,
        "malformed_rate": report.malformed_rate,
        "copied_values": sum(e.copied for e in report.episodes),
        "mean_tokens": report.mean_tokens,
        "tokens_per_success": report.tokens_per_success,
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
    ap.add_argument("--replay-fraction", type=float, default=0.0,
                    help="share of training rows drawn from the base corpus (prose + code); the "
                         "continual-learning mitigation, measured on held-out BPB after the run")
    ap.add_argument("--families", default=None,
                    help="comma-separated task families of prophet.agent.tasks (default: the "
                         "benchmark's file family only); --episodes is then per family")
    ap.add_argument("--bpb", action="store_true", help="measure held-out BPB before and after (what the fine-tune erases)")
    ap.add_argument("--tag", default="agent", help="output directory under --work")
    ap.add_argument("--bpb-docs", type=int, default=400, help="held-out documents scored by --bpb")
    ap.add_argument("--episodes-per-row", type=int, default=1,
                    help="consecutive episodes per training row (each at its <|bos|>): above 1, the "
                         "recurrent state an episode starts from is the previous episode's, the "
                         "distribution a carried session state comes from")
    ap.add_argument("--carry-bench", action="store_true",
                    help="also bench with the session state carried from episode to episode")
    ap.add_argument("--segment-attention", action="store_true",
                    help="mask attention at every <|bos|> of a training row, so the rows of "
                         "--episodes-per-row are seen exactly as the loop sees a carried session")
    ap.add_argument("--related", action="store_true",
                    help="train and bench on related lookup sequences (a file read by the previous "
                         "episode is answered without reading it again); rows hold --episodes-per-row "
                         "episodes and the bench is run fresh and carried")
    ap.add_argument("--stage", choices=["train", "bench", "all"], default="all")
    args = ap.parse_args()
    work = Path(args.work)
    out_dir = work / args.tag
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
        families = [f for f in args.families.split(",") if f] if args.families else None
        rows, data_stats = build_rows(tokenizer, args.episodes, seed=1, seq_len=args.seq_len, families=families,
                                      per_row=args.episodes_per_row, related=args.related)
        report["episodes"] = data_stats
        report["families"] = families
        if args.bpb:
            report["bpb_before"] = heldout_bpb(work, model, tokenizer, seq_len=min(args.seq_len, 256), max_docs=args.bpb_docs)
            print("bpb before:", report["bpb_before"], flush=True)
        report["bench_before"] = bench(model, tokenizer, n_tasks=args.bench_before_tasks, seed=7)
        print("before:", report["bench_before"]["summary"], flush=True)
        sources = sources_from_iterables({"episodes": (1.0 - args.replay_fraction, rows)})
        if args.replay_fraction > 0:
            sources.append(replay_source(work, tokenizer, args.replay_fraction))
        report["replay_fraction"] = args.replay_fraction
        loader = StreamingLoader(sources, seq_len=args.seq_len, batch_size=args.batch_size, seed=0)
        tc = TrainConfig(
            total_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
            peak_lr_muon=0.01, peak_lr_adamw=2e-3, warmup_frac=0.05, decay_frac=0.3,
            checkpoint_dir=str(out_dir / "checkpoints"), checkpoint_every=100, log_every=25,
            device="cpu", max_wall_seconds=args.minutes * 60.0, mtp_weight=0.0,
            segment_by_bos=args.segment_attention,
        )
        report["segment_attention"] = args.segment_attention
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
        if args.bpb:
            report["bpb_after"] = heldout_bpb(work, model, tokenizer, seq_len=min(args.seq_len, 256), max_docs=args.bpb_docs)
            print("bpb after:", report["bpb_after"], flush=True)
    else:
        state, _ = CheckpointManager(out_dir / "checkpoints").load_latest()
        model.load_state_dict(state["model"])

    if args.stage in ("bench", "all"):
        model.eval()
        families = [f for f in args.families.split(",") if f] if args.families else [None]
        if args.related:
            families = []
            for seed, carry, name in ((7, False, "bench_after_related"), (7, True, "bench_after_related_carried"),
                                      (11, False, "bench_after_unseen_seed_related"),
                                      (11, True, "bench_after_unseen_seed_related_carried")):
                report[name] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=seed, carry=carry,
                                     related_size=args.episodes_per_row)
                print(f"{name}:", report[name]["summary"], report[name]["by_seen"], flush=True)
        for family in families:
            key = "" if family is None else f"_{family}"
            report[f"bench_after{key}"] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=7, family=family)
            print(f"after{key}:", report[f"bench_after{key}"]["summary"], flush=True)
            report[f"bench_after_unseen_seed{key}"] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=11, family=family)
            print(f"after (other seed){key}:", report[f"bench_after_unseen_seed{key}"]["summary"], flush=True)
            if args.carry_bench:
                for seed, name in ((7, "bench_after_carried"), (11, "bench_after_unseen_seed_carried")):
                    report[f"{name}{key}"] = bench(model, tokenizer, n_tasks=args.bench_tasks, seed=seed,
                                                   family=family, carry=True)
                    print(f"{name}{key}:", report[f"{name}{key}"]["summary"], flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
