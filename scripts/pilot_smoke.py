#!/usr/bin/env python3
"""Train, interrupt, resume and score a tiny model on the prepared real-text pilot.

This verifies the real data path. The deliberately short run is not an architecture
quality comparison. Checkpoints and reports stay local under outputs/ by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from prophet.train.loss import compute_loss  # noqa: E402


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def score(model, batches, device):
    model.eval()
    total, tokens = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        out = model(batch)
        loss = compute_loss(out, batch, mtp_weight=0, z_loss_weight=0, loss_chunk_tokens=64)
        count = batch.shape[0] * (batch.shape[1] - 1)
        total += loss.lm.item() * count
        tokens += count
    return {"nats_per_token": total / tokens, "scored_tokens": tokens}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/fineweb-pilot-v1"))
    ap.add_argument("--out", type=Path, default=Path("outputs/real-pilot-smoke"))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    if args.steps < 2 or args.seq_len < 2 or args.batch_size < 1:
        ap.error("steps and sequence length must be >=2; batch-size must be positive")
    if args.out.exists():
        ap.error("choose a fresh output directory; existing checkpoints are preserved")
    if args.device == "cuda" and (not torch.cuda.is_available() or not HAS_FLA):
        ap.error("CUDA and the validated FLA extra are required")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    assert manifest["complete"]
    for relative, metadata in manifest["artifacts"].items():
        assert sha256(args.corpus / relative) == metadata["sha256"], relative
    tokenizer_path = args.corpus / "tokenizer.json"
    tokenizer_meta = json.loads(tokenizer_path.with_suffix(".json.metadata.json").read_text())
    assert sha256(tokenizer_path) == tokenizer_meta["tokenizer_sha256"]
    train_hash = manifest["artifacts"]["train/fineweb-edu/part-00000.jsonl"]["sha256"]
    assert [s["fingerprint"]["files"] for s in tokenizer_meta["sources"]] == [[train_hash]]
    tokenizer = ProphetTokenizer.load(tokenizer_path)
    cfg = ProphetConfig.from_json(Path(__file__).resolve().parent.parent / "configs/prophet_tiny_smoke.json")
    cfg.name = "prophet-real-text-pipeline-smoke"
    cfg.frontend.vocab_size = tokenizer.vocab_size
    cfg.heads.n_multi_token_predict = 0
    cfg.heads.confidence_head = False
    cfg.recurrent.train_loop_min = cfg.recurrent.train_loop_max = 2
    cfg.validate()
    args.out.mkdir(parents=True)
    cfg.to_json(args.out / "config.json")
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)
    # Eight complete document prefixes; no cross-document packing or padding in eval.
    heldout = [torch.tensor([tokenizer.encode(text, add_eos=True)[:args.seq_len]])
               for text in islice(validation.open(), 8)]

    def fresh():
        torch.manual_seed(0)
        model = ProphetModel(cfg)
        source = LocalTextSource.from_root(args.corpus / "train", "fineweb-edu", 1.0)
        loader = StreamingLoader([TokenisedSource(source, tokenizer)], seq_len=args.seq_len,
                                 batch_size=args.batch_size, seed=0)
        return Trainer(model, loader, TrainConfig(
            total_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
            peak_lr_muon=0.005, peak_lr_adamw=0.001, warmup_frac=0.1,
            loss_chunk_tokens=64, checkpoint_dir=str(args.out / "checkpoints"),
            checkpoint_every=0, log_every=8, device=args.device,
        ), model_config=cfg)

    trainer = fresh()
    before = score(trainer.model, heldout, args.device)
    print("INITIAL_HELDOUT", before, flush=True)
    trainer.train(max_steps=args.steps // 2)
    interrupted_step = trainer.step
    del trainer
    trainer = fresh()
    assert trainer.maybe_resume() and trainer.step == interrupted_step
    trainer.train()
    after = score(trainer.model, heldout, args.device)
    assert trainer.skipped_nonfinite == 0
    report = {
        "purpose": "real-text pipeline smoke; not a language-quality or architecture claim",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=Path(__file__).resolve().parent.parent, text=True).strip(),
        "source_revision": manifest["source_revision"], "corpus_artifacts": manifest["artifacts"],
        "tokenizer_sha256": tokenizer_meta["tokenizer_sha256"],
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "device": args.device, "torch": torch.__version__, "steps": trainer.step,
        "resumed_from_step": interrupted_step, "tokens_seen": trainer.tokens_seen,
        "skipped_nonfinite": trainer.skipped_nonfinite, "before": before, "after": after,
        "evaluation": f"first {args.seq_len} tokens of each of the first eight held-out documents",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
