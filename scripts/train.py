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
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.budget import allocation_warnings, count_parameters, training_memory  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.streaming import StreamingLoader, sources_from_iterables  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402

_STOP = False


def _handle_signal(signum, frame) -> None:
    """Checkpoint and exit cleanly if the platform gives us any warning at all.

    Colab usually does not, which is why the checkpoint rotation has to survive being
    killed mid-write regardless. This just makes the polite case tidy.
    """
    global _STOP
    print(f"\n[signal {signum}] finishing the current step, then checkpointing.", flush=True)
    _STOP = True


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
    ap.add_argument("--max-steps-this-session", type=int, default=None,
                    help="stop early without ending the run, e.g. to fit a Colab window")
    args = ap.parse_args()

    cfg = ProphetConfig.from_json(args.config)
    cfg.validate()

    # Budget check before anything expensive. A configuration that cannot fit should fail
    # in a second, not after an hour of downloading.
    params = count_parameters(cfg)
    mem = training_memory(cfg, batch_tokens=args.batch_size * args.seq_len,
                          optimizer_bytes_per_param=2.0)
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

    if args.smoke:
        args.steps = min(args.steps, 60)
        args.seq_len = min(args.seq_len, 128)
        args.batch_size = min(args.batch_size, 4)
        args.checkpoint_every = 20
        print("\n[smoke] synthetic corpus, short run, no downloads")

    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)

    if args.smoke:
        sources = synthetic_sources(cfg.frontend.vocab_size)
    else:
        print("\nReal-corpus streaming is not wired up yet: the data mixture in "
              "configs/data_mixture_v1.yaml lists the sources, but every dataset id in it "
              "still needs verifying against the Hub before download (see docs/02_DATA.md, "
              "'Statut de vérification'). Run with --smoke meanwhile.", file=sys.stderr)
        return 2

    loader = StreamingLoader(
        sources, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed
    )
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
    )
    trainer = Trainer(model, loader, train_cfg, model_config=cfg)

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
