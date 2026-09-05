#!/usr/bin/env python3
"""Prophet training entry point.

Designed for a preemptible Colab A100: launch it, let it be killed, launch it again with
the same arguments. It resumes from the newest intact checkpoint automatically, and a
resumed run is asserted (in ``tests/test_training.py``) to produce identical weights to an
uninterrupted one.

    # dry run: no data downloaded, synthetic corpus, verifies the whole path
    python scripts/train.py --config configs/prophet_500m_probe.json --smoke

    # real run
    python scripts/train.py --config configs/prophet_mini.json \
        --steps 40000 --checkpoint-dir /content/drive/MyDrive/prophet/ckpt

Re-running the exact same command after a session dies is the intended workflow; there
is no separate resume flag.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.budget import allocation_warnings, count_parameters, training_memory  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.corpus import build_loader  # noqa: E402
from prophet.data.decontaminate import Decontaminator  # noqa: E402
from prophet.data.recipes import prophet_v1_mixture  # noqa: E402
from prophet.data.streaming import StreamingLoader, sources_from_iterables  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402

_TRAINER = None


def _handle_signal(signum, frame) -> None:
    """Checkpoint and exit cleanly if the platform gives us any warning at all.

    Colab usually does not, which is why the checkpoint rotation has to survive being
    killed mid-write regardless. This just makes the polite case tidy. The flag lives on
    the trainer and is read once per step; the first version set a module global that
    nothing read, which also swallowed KeyboardInterrupt -- Ctrl-C did nothing.
    """
    print(f"\n[signal {signum}] finishing the current step, then checkpointing.", flush=True)
    if _TRAINER is not None:
        _TRAINER.stop_requested = True
    else:
        raise KeyboardInterrupt


def synthetic_sources(vocab_size: int, *, n_docs: int = 4000, doc_len: int = 256):
    """A structured synthetic corpus for smoke runs.

    Deliberately learnable — repeating arithmetic progressions — so that a failure to fit
    it points at the training code rather than at the data.
    """
    docs = [
        [(i * 7 + j * 3) % vocab_size for j in range(doc_len)]
        for i in range(n_docs)
    ]
    return sources_from_iterables({"synthetic": (1.0, docs)})


def build_real_loader(args, cfg: ProphetConfig):
    """The mixture as a phased loader over local corpora and, if allowed, the Hub.

    Returns the loader and the number of steps it is sized for. Refuses to start without
    a tokenizer artifact, and without a benchmark set to decontaminate against unless
    the caller passes ``--benchmarks ''`` deliberately.
    """
    if not args.tokenizer:
        print("\nA real run needs a trained tokenizer: scripts/train_tokenizer.py --data-root "
              "... --out tokenizer.json, then --tokenizer tokenizer.json.", file=sys.stderr)
        raise SystemExit(2)
    if args.data_root is None and not args.hub:
        print("\nNo corpus: pass --data-root DIR (local files) and/or --hub (stream from "
              "the Hub). Every dataset id must have passed scripts/verify_datasets.py "
              "first (docs/02_DATA.md, 'Statut de vérification').", file=sys.stderr)
        raise SystemExit(2)
    if args.benchmarks is None:
        print("\nNo --benchmarks directory: every evaluation number would be suspect. Pass "
              "--benchmarks DIR, or --benchmarks '' to run undecontaminated on purpose.",
              file=sys.stderr)
        raise SystemExit(2)

    tokenizer = ProphetTokenizer.load(args.tokenizer)
    if tokenizer.n_tokens > cfg.frontend.vocab_size:
        print(f"\ntokenizer has {tokenizer.n_tokens} ids, the model {cfg.frontend.vocab_size}",
              file=sys.stderr)
        raise SystemExit(2)

    decontaminator = None
    if args.benchmarks:
        decontaminator = Decontaminator()
        for path in sorted(Path(args.benchmarks).glob("*.jsonl")):
            items = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
            n = decontaminator.add_benchmark(path.stem, items)
            print(f"decontaminate  {path.stem}: {n} items indexed")

    extra = []
    if args.quarantine:
        if not args.tools:
            print("\n--quarantine needs --tools: the episodes' schemas must be in context for "
                  "the selection anchors to exist.", file=sys.stderr)
            raise SystemExit(2)
        from prophet.agent.actions import ToolRegistry, ToolSchema
        from prophet.agent.quarantine import Quarantine
        from prophet.agent.render import QuarantineSource
        from prophet.data.corpus import TokenisedSource
        registry = ToolRegistry()
        for spec in json.loads(Path(args.tools).read_text()):
            registry.add(ToolSchema(spec["name"], spec.get("description", ""), spec.get("parameters", {})))
        episodes = QuarantineSource(Quarantine(args.quarantine), registry, weight=args.quarantine_weight)
        print(f"quarantine     {episodes.n_documents()} promoted episodes at weight {args.quarantine_weight}")
        extra.append(TokenisedSource(episodes, tokenizer, max_epochs=None, parse_special=True))

    batch_tokens = args.batch_size * args.seq_len * args.grad_accum
    total_tokens = args.tokens if args.tokens else args.steps * batch_tokens
    mixture = prophet_v1_mixture(total_tokens=total_tokens)
    # The loader serves micro-batches; the trainer draws grad_accum of them per step.
    loader = build_loader(
        mixture, tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed, decontaminator=decontaminator, local_root=args.data_root,
        allow_hub=args.hub, extra_sources=extra,
    )
    return loader, max(1, loader.total_steps() // args.grad_accum)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a ProphetConfig JSON file")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adamw-lr", type=float, default=3e-3)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="short run on a synthetic corpus; verifies the whole path")
    ap.add_argument("--tokenizer", default=None,
                    help="Prophet-Tok artifact from scripts/train_tokenizer.py (real runs)")
    ap.add_argument("--data-root", default=None,
                    help="directory of <source>.jsonl or <source>/*.jsonl corpora, named "
                         "after the mixture's sources")
    ap.add_argument("--hub", action="store_true",
                    help="stream sources missing from --data-root from the HuggingFace Hub")
    ap.add_argument("--benchmarks", default=None,
                    help="directory of <benchmark>.jsonl test sets ({\"text\": ...}) to "
                         "decontaminate against; strongly recommended for a real run")
    ap.add_argument("--quarantine", default=None,
                    help="agent quarantine JSON; its promoted episodes join the anneal "
                         "phase as a source (docs/08_AGENT.md §4 bis)")
    ap.add_argument("--tools", default=None,
                    help="JSON list of tool schemas the quarantined episodes ran with "
                         "({name, description, parameters}); required with --quarantine")
    ap.add_argument("--quarantine-weight", type=float, default=0.05,
                    help="sampler weight of the episode source next to a phase summing to 1")
    ap.add_argument("--tokens", type=float, default=None,
                    help="total training tokens; sets --steps from the batch shape and "
                         "scales the mixture's phases (default: --steps x batch tokens)")
    ap.add_argument("--max-steps-this-session", type=int, default=None,
                    help="stop early without ending the run, e.g. to fit a Colab window")
    ap.add_argument("--session-minutes", type=float, default=None,
                    help="wall-clock budget for this session; the run checkpoints and "
                         "exits cleanly before Colab kills it, and resumes next time")
    ap.add_argument("--allow-slow-scan", action="store_true",
                    help="run without flash-linear-attention (reference scan). For tiny "
                         "CPU checks of the real data path only; never for a budgeted run")
    args = ap.parse_args()

    cfg = ProphetConfig.from_json(args.config)
    cfg.validate()

    # Budget check before anything expensive. A configuration that cannot fit should fail
    # in a second, not after an hour of downloading.
    params = count_parameters(cfg)
    # The gate estimates with the trainer's real policy: fp32 params (the master copy
    # under bf16 autocast), fp32 grads, Muon/AdamW state from the actual split. It once
    # assumed an 8-bit optimiser that nothing implements and said "fits" for a run that
    # did not.
    mem = training_memory(cfg, batch_tokens=args.batch_size * args.seq_len)
    print(f"config           {cfg.name}")
    print(f"parameters       {params.total / 1e6:.1f}M total / "
          f"{params.active_per_token / 1e6:.1f}M active per token")
    print(f"effective depth  {cfg.effective_depth()} "
          f"(from {cfg.parameterised_depth()} parameterised blocks)")
    print(f"training memory  {mem.total_gb:.1f} GB estimated, "
          f"{'fits' if mem.fits else 'DOES NOT FIT'} on {mem.device}")
    for warning in allocation_warnings(cfg):
        print(f"  warning: {warning}")
    if not mem.fits and not args.smoke:
        print("\nRefusing to start: the configuration does not fit. Reduce d_model, "
              "batch size, or sequence length.", file=sys.stderr)
        return 1

    from prophet.modeling.layers import HAS_FLA
    if not args.smoke and not HAS_FLA and not args.allow_slow_scan:
        print(
            "\nRefusing a real run without flash-linear-attention: the reference delta-rule "
            "scan is a Python loop over every token of every core iteration, and autograd "
            "retains each state -- roughly 144 GB of activations for 8k tokens on the main "
            "config. Install `fla` (pinned in pyproject) and run the GPU equivalence test first.",
            file=sys.stderr,
        )
        return 3

    if args.smoke:
        args.steps = min(args.steps, 60)
        args.seq_len = min(args.seq_len, 128)
        args.batch_size = min(args.batch_size, 4)
        args.checkpoint_every = 20
        print("\n[smoke] synthetic corpus, short run, no downloads")

    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)

    tokenizer = None
    if args.smoke:
        sources = synthetic_sources(cfg.frontend.vocab_size)
        loader = StreamingLoader(
            sources, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed
        )
        if cfg.heads.action_head:
            tokenizer = ProphetTokenizer(merges=[])  # byte-level: enough to parse targets
    else:
        loader, args.steps = build_real_loader(args, cfg)
        tokenizer = ProphetTokenizer.load(args.tokenizer)
        print(f"\ndata\n{loader.describe()}")
    train_cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum,
        peak_lr_muon=args.muon_lr,
        peak_lr_adamw=args.adamw_lr,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        max_wall_seconds=None if args.session_minutes is None else args.session_minutes * 60.0,
    )
    trainer = Trainer(model, loader, train_cfg, model_config=cfg, tokenizer=tokenizer)
    global _TRAINER
    _TRAINER = trainer

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if trainer.maybe_resume():
        print(f"\nresumed from step {trainer.step} "
              f"({trainer.tokens_seen / 1e6:.3f}M tokens already seen)")
    else:
        print("\nstarting from scratch (no checkpoint found)")
    routing = {k: f"{v / 1e6:.1f}M" for k, v in trainer.param_counts.items()}
    print(f"schedule         {trainer.schedule.describe()}")
    print(f"optimisers       {[type(o).__name__ for o in trainer.optimizers]}")
    print(f"routing          {routing}")
    print()

    try:
        trainer.train(max_steps=args.max_steps_this_session)
    finally:
        if trainer.step > 0:
            meta = trainer.ckpt.save(trainer.state_dict(), trainer.step)
            print(f"\ncheckpointed at step {meta.step} (slot {meta.slot}, "
                  f"{meta.bytes / 1e6:.1f} MB)")

    if trainer.history:
        first, last = trainer.history[0], trainer.history[-1]
        print(f"loss {first.loss:.4f} -> {last.loss:.4f} over {len(trainer.history)} steps, "
              f"{last.tokens / 1e6:.2f}M tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
