#!/usr/bin/env python3
"""One bounded session of a loop-core arm (docs/29_LOOP_CORE_PREREG.md).

Three arms (``lc_gdn``, ``lc_attn``, ``lc_plain``), three seeds, one pass over the
loop-core corpus with the composition tasks mixed in at a fixed token share. Repeat the
same command to resume; the protocol is frozen on first launch and every later session
must match it. Milestone evaluations happen at exact step boundaries, so a session stops
early at a milestone, evaluates, and continues.

With ``--persistent DIR`` (a mounted Drive), a session that finds no local checkpoint
restores the latest verified snapshot of its run, and every session ends by publishing
a fresh audited snapshot there (one checkpoint slot, reports and logs), then removing
the older snapshot of the same run. Training itself always runs on local disk.

    python scripts/run_loop_core.py --corpus data/loop-core-v1 --tokenizer TOK.json \
        --arm lc_gdn --seed 0 --out /content/work/lc_gdn-seed0 \
        --persistent /content/drive/MyDrive/Prophet_AGI/loop-core \
        --max-session-steps 1500 --session-minutes 55

Prints ``RUN_COMPLETE`` when the final step is evaluated, else
``SESSION_COMPLETE_RESUME_REQUIRED``; the queue repeats until the former.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.composition import read_jsonl  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.composition import evaluate_composition  # noqa: E402
from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from scripts.gpu_check import kernel_agreement  # noqa: E402
from scripts.run_r04_pilot import (  # noqa: E402
    PILOT_TOKENIZER_SHA256,
    freeze_protocol,
    sha256,
    tokenizer_semantic_hash,
    write_json,
)
from scripts.snapshot_r04 import snapshot  # noqa: E402

ARMS = ("lc_gdn", "lc_attn", "lc_plain")
CORPUS_NAME = "loop-core-v1"
DEFAULT_TOTAL_STEPS = 18_310  # 300M tokens at 8 x 2048
DEFAULT_MILESTONE_TOKENS = (100_000_000, 200_000_000)
CONFIG_DIR = ROOT / "configs" / "loop_core"


def verify_corpus(root: Path, *, hash_shards: bool = True) -> dict:
    """Bind the run to the exact corpus bytes and to its recorded token estimate."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("name") != CORPUS_NAME:
        raise ValueError("corpus is incomplete or is not the loop-core corpus")
    tokens = manifest.get("tokens") or {}
    if (
        "train" not in tokens.get("splits", {})
        or "composition_train_tokens_first_2000" not in tokens
    ):
        raise ValueError("corpus manifest lacks token counts; prepare it with --tokenizer")
    artifacts = {}
    for relative, meta in manifest["artifacts"].items():
        path = root / relative
        if path.stat().st_size != meta["bytes"]:
            raise ValueError(f"corpus shard size differs: {relative}")
        if hash_shards and sha256(path) != meta["sha256"]:
            raise ValueError(f"corpus shard differs: {relative}")
        artifacts[relative] = meta["sha256"]
    composition = manifest["composition"]
    for split in ("train", "test"):
        if sha256(root / composition[split]["path"]) != composition[split]["sha256"]:
            raise ValueError(f"composition {split} file differs from the manifest")
    return {
        "name": manifest["name"],
        "manifest_sha256": sha256(root / "manifest.json"),
        "source_revision": manifest["source_revision"],
        "artifacts": artifacts,
        "composition": {s: composition[s]["sha256"] for s in ("train", "test")},
        "composition_seed": composition["seed"],
        "tokens": tokens,
        "train_docs": manifest["stats"]["train_docs"],
        "validation_docs": manifest["stats"]["validation_docs"],
        "composition_train_docs": composition["train"]["documents"],
    }


def mixture_weights(provenance: dict, share: float) -> dict:
    """Per-document sampling weights that give the composition source ``share`` of tokens."""
    if not 0 < share < 1:
        raise ValueError("composition share must be in (0, 1)")
    tokens = provenance["tokens"]
    web_per_doc = tokens["splits"]["train"]["estimated_tokens"] / provenance["train_docs"]
    sampled = min(2000, provenance["composition_train_docs"])
    comp_per_doc = tokens["composition_train_tokens_first_2000"] / sampled
    weight = share * web_per_doc / ((1 - share) * comp_per_doc + share * web_per_doc)
    return {
        "composition_share": share,
        "web_weight": 1 - weight,
        "composition_weight": weight,
        "mean_web_tokens_per_doc": web_per_doc,
        "mean_composition_tokens_per_doc": comp_per_doc,
    }


class DepthTrainer(Trainer):
    """Record the depth actually used at every step and bind it to the checkpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.grad_accum_steps != 1:
            raise ValueError("one microbatch per step: the depth history is per step")
        self.depth_history: list[int] = []

        def observe(model, inputs, output):
            if model.training:
                self.depth_history.append(int(output.loop_k))

        self.depth_hook = self.model.register_forward_hook(observe)

    def state_dict(self):
        if len(self.depth_history) != self.step:
            raise ValueError("depth history does not match completed steps")
        return {**super().state_dict(), "depth_history": list(self.depth_history)}

    def load_state_dict(self, state):
        history = state.get("depth_history")
        r = self.model_config.recurrent
        if not (
            isinstance(history, list)
            and len(history) == state["step"]
            and all(type(k) is int and r.train_loop_min <= k <= r.train_loop_max for k in history)
        ):
            raise ValueError("invalid recorded depth history")
        super().load_state_dict(state)
        self.depth_history = list(history)


def latest_snapshot(persistent_run: Path) -> Path | None:
    candidates = []
    for path in sorted(persistent_run.glob("step-*")):
        marker = path / "SNAPSHOT_COMPLETE.json"
        if marker.exists() and json.loads(marker.read_text()).get("complete"):
            candidates.append(path)
    return candidates[-1] if candidates else None


def restore_from_persistent(out: Path, persistent_run: Path) -> dict | None:
    """Copy the latest verified snapshot into an output directory that has no checkpoint."""
    if (out / "checkpoints" / "manifest.json").exists():
        return None
    source = latest_snapshot(persistent_run)
    if source is None:
        return None
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: not empty and without a checkpoint; inspect before restoring")
    staging = out.with_name(out.name + ".restoring")
    if staging.exists():
        raise ValueError(f"{staging}: inspect the earlier partial restore")
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("SNAPSHOT_COMPLETE.json"))
    staging.rename(out)
    return {"restored_from": str(source)}


def publish_snapshot(out: Path, step: int, persistent_run: Path) -> dict:
    """One audited snapshot per run on persistent storage; older ones are removed."""
    destination = persistent_run / f"step-{step:06d}"
    marker_path = destination / "SNAPSHOT_COMPLETE.json"
    if destination.exists():
        if marker_path.exists() and json.loads(marker_path.read_text()).get("complete"):
            marker = json.loads(marker_path.read_text())
        else:
            shutil.rmtree(destination)
            marker = snapshot(out, step, destination)
    else:
        marker = snapshot(out, step, destination)
    for older in sorted(persistent_run.glob("step-*")):
        if older != destination:
            shutil.rmtree(older)
    return {**marker, "destination": str(destination)}


def eval_subset(examples: list[dict], per_level: int) -> list[dict]:
    """The first ``per_level`` items of every (kind, level), in source order."""
    return [e for e in examples if int(e.get("index", 0)) < per_level]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument(
        "--tokenizer-sha256",
        default=PILOT_TOKENIZER_SHA256,
        help="semantic hash the tokenizer must have (the pilot's by default)",
    )
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--config", type=Path, default=None, help="defaults to configs/loop_core/<arm>.json"
    )
    ap.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--milestone-tokens", type=int, nargs="*", default=list(DEFAULT_MILESTONE_TOKENS)
    )
    ap.add_argument("--composition-share", type=float, default=0.03)
    ap.add_argument("--composition-eval-per-level", type=int, default=500)
    ap.add_argument("--max-session-steps", type=int, default=512)
    ap.add_argument("--session-minutes", type=float, default=50)
    ap.add_argument("--persistent", type=Path, default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--allow-cpu", action="store_true", help="miniature runs only; recorded")
    ap.add_argument("--skip-shard-hashes", action="store_true", help="size check only; recorded")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if (
        args.max_session_steps < 1
        or not math.isfinite(args.session_minutes)
        or args.session_minutes <= 0
    ):
        ap.error("session limits must be positive")
    if args.total_steps < 1 or args.batch_size < 1 or args.seq_len < 2:
        ap.error("positive schedule required")
    if args.device == "cpu" and not args.allow_cpu:
        ap.error("CPU runs are miniature diagnostics; pass --allow-cpu")

    provenance = verify_corpus(args.corpus, hash_shards=not args.skip_shard_hashes)
    tokenizer = ProphetTokenizer.load(args.tokenizer)
    semantic = tokenizer_semantic_hash(args.tokenizer)
    if semantic != args.tokenizer_sha256:
        raise ValueError("tokenizer vocabulary differs from the expected one")
    config_path = args.config or CONFIG_DIR / f"{args.arm}.json"
    cfg = ProphetConfig.from_json(config_path)
    cfg.validate()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if args.config is None:
        summary = json.loads((CONFIG_DIR / "summary.json").read_text())
        if summary["arms"][args.arm]["config_sha256"] != config_sha:
            raise ValueError("arm configuration differs from the generated summary; regenerate")
    if tokenizer.vocab_size != cfg.frontend.vocab_size:
        raise ValueError("tokenizer and model vocabulary widths differ")
    tokens_per_step = args.batch_size * args.seq_len
    planned = args.total_steps * tokens_per_step
    web_tokens = provenance["tokens"]["splits"]["train"]["estimated_tokens"]
    if planned * (1 - args.composition_share) > web_tokens:
        raise ValueError("planned web tokens exceed one pass over the corpus")
    milestones = sorted(
        m
        for m in {round(t / tokens_per_step) for t in args.milestone_tokens}
        if 0 < m < args.total_steps
    )
    mixture = mixture_weights(provenance, args.composition_share)
    protocol = {
        "format_version": 1,
        "programme": "loop-core",
        "variant": args.arm,
        "arm": args.arm,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "config_sha256": config_sha,
        "data": provenance,
        "tokenizer_semantic_sha256": semantic,
        "steps": args.total_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": tokens_per_step,
        "planned_tokens": planned,
        "milestone_steps": milestones,
        "loss_chunk_tokens": 512,
        "muon_lr": 0.01,
        "adamw_lr": 0.0003,
        "mixture": mixture,
        "web_max_epochs": 1.0,
        "composition_max_epochs": 4.0,
        "composition_eval_per_level": args.composition_eval_per_level,
        "evaluation": "all held-out documents at the default depth; composition test subset",
        "allow_cpu": bool(args.allow_cpu),
        "shard_hashes_checked": not args.skip_shard_hashes,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.device == "cuda":
        if not torch.cuda.is_available() or not HAS_FLA:
            raise RuntimeError(
                "validated CUDA/FLA environment required; run tests/test_gpu.py first"
            )
        if os.environ.get("TRITON_F32_DEFAULT") != "tf32x3":
            raise RuntimeError("set TRITON_F32_DEFAULT=tf32x3 before Python starts")
        protocol["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fla": version("fla-core"),
            "triton": version("triton"),
            "device": torch.cuda.get_device_name(0),
            "triton_f32_default": "tf32x3",
        }
    else:
        protocol["runtime"] = {"torch": torch.__version__, "device": "cpu"}

    run_name = f"{args.arm}-seed{args.seed}"
    persistent_run = args.persistent / run_name if args.persistent else None
    args.out.mkdir(parents=True, exist_ok=True)
    restored = restore_from_persistent(args.out, persistent_run) if persistent_run else None
    if restored:
        print("RESTORED", json.dumps(restored), flush=True)
    if not (args.out / "protocol.json").exists() and any(args.out.iterdir()):
        raise ValueError("output contains artifacts without a protocol; use a fresh directory")
    freeze_protocol(args.out / "protocol.json", protocol)
    if args.device == "cuda":
        error = kernel_agreement(cfg, seed=args.seed)
        write_json(args.out / "kernel-check.json", {"max_abs_error": error, "tolerance": 0.002})
        if not math.isfinite(error) or error > 0.002:
            raise RuntimeError("fused/reference gate failed")
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)
    web = LocalTextSource.from_root(args.corpus / "train", "fineweb-edu", mixture["web_weight"])
    comp = LocalTextSource.from_root(
        args.corpus / "train", "composition", mixture["composition_weight"]
    )
    loader = StreamingLoader(
        [
            TokenisedSource(web, tokenizer, max_epochs=1.0),
            TokenisedSource(comp, tokenizer, max_epochs=4.0),
        ],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    def log(metrics):
        print(metrics.format(), flush=True)
        with (args.out / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(metrics)) + "\n")

    trainer = DepthTrainer(
        model,
        loader,
        TrainConfig(
            total_steps=args.total_steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            peak_lr_muon=0.01,
            peak_lr_adamw=0.0003,
            loss_chunk_tokens=512,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=str(args.out / "checkpoints"),
            checkpoint_every=512,
            log_every=16,
            max_wall_seconds=args.session_minutes * 60,
        ),
        model_config=cfg,
        tokenizer=tokenizer,
        on_log=log,
    )
    resumed = trainer.maybe_resume()
    if not resumed and (args.out / "train.jsonl").exists():
        raise RuntimeError(
            "training history exists but no checkpoint survived; use a new output directory"
        )
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)
    composition_test = eval_subset(
        read_jsonl(args.corpus / "composition-test.jsonl"), args.composition_eval_per_level
    )

    def evaluate(label: str, checkpoint=None) -> None:
        result = evaluate_documents(
            model,
            validation.open(),
            tokenizer,
            device=args.device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        composition_report = evaluate_composition(
            model,
            composition_test,
            tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=args.device,
            loop_k=cfg.recurrent.default_loop_k,
        )
        result.update(
            {
                "step": trainer.step,
                "train_tokens": trainer.tokens_seen,
                "checkpoint": checkpoint,
                "run_protocol": protocol,
                "loop_k": cfg.recurrent.default_loop_k,
                "composition": {k: v for k, v in composition_report.items() if k != "items"},
                "composition_items": composition_report["items"],
                "depth_counts": {
                    str(k): trainer.depth_history.count(k)
                    for k in sorted(set(trainer.depth_history))
                },
            }
        )
        write_json(args.out / f"{label}.json", result)
        summary = {k: result[k] for k in ("step", "nats_per_token", "bits_per_byte")}
        summary["composition"] = {
            k: round(v["accuracy"], 4) for k, v in composition_report["levels"].items()
        }
        print(label, summary, flush=True)

    if trainer.step == 0:
        evaluate("initial")
    old_handlers = {
        sig: signal.signal(sig, lambda *_: setattr(trainer, "stop_requested", True))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    began = time.time()
    done_this_session = 0
    try:
        while trainer.step < args.total_steps and done_this_session < args.max_session_steps:
            pending = [
                m
                for m in milestones
                if m > trainer.step and not (args.out / f"milestone-step-{m:06d}.json").exists()
            ]
            target = min(
                trainer.step + args.max_session_steps - done_this_session,
                pending[0] if pending else args.total_steps,
                args.total_steps,
            )
            remaining = args.session_minutes * 60 - (time.time() - began)
            if remaining <= 0:
                trainer.stop_requested = True
                break
            trainer.cfg.max_wall_seconds = remaining
            before = trainer.step
            trainer.train(max_steps=target - trainer.step)
            done_this_session += trainer.step - before
            if (
                trainer.step in milestones
                and not (args.out / f"milestone-step-{trainer.step:06d}.json").exists()
            ):
                evaluate(f"milestone-step-{trainer.step:06d}")
            if trainer.stop_requested:
                break
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if trainer.step:
        meta = trainer.ckpt.save(trainer.state_dict(), trainer.step)
        print("CHECKPOINT", meta.to_dict(), flush=True)
        evaluate(f"evaluation-step-{trainer.step:06d}", meta.to_dict())
        if persistent_run:
            published = publish_snapshot(args.out, trainer.step, persistent_run)
            print("SNAPSHOT", json.dumps(published), flush=True)
    print(
        "RUN_COMPLETE" if trainer.step == args.total_steps else "SESSION_COMPLETE_RESUME_REQUIRED",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
