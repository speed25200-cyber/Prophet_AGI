#!/usr/bin/env python3
"""Final measurements of one loop-core run (docs/29_LOOP_CORE_PREREG.md §6).

On the exact published checkpoint of a finished run, in FP32 with TF32 off:

1. held-out bits per byte at every requested depth (hypotheses H2, H3);
2. composition accuracy per family and level at the requested depths (H4);
3. the real bytes held by the inference cache after prefilling a context of each
   requested length at each requested depth, by section and by cache kind (H1);
4. held-out bits per byte after weight-only int8 and int4 round-to-nearest
   quantization of the block linears, at the requested depths (H5).

An unshared arm (``train_loop_max == 1``) is measured at depth 1 only. The report is
written incrementally with ``complete: false`` until every block has run, so an
interrupted evaluation is visibly partial rather than silently short.

    python scripts/eval_loop_core.py --run RUN --step 18310 --corpus CORPUS \
        --tokenizer TOK.json --out RUN/final-measurements.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.composition import read_jsonl  # noqa: E402
from prophet.data.corpus import LocalTextSource  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.composition import evaluate_composition  # noqa: E402
from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from prophet.quant.rtn import quantize_model_rtn  # noqa: E402
from prophet.train.checkpoint import CheckpointManager, CheckpointMeta  # noqa: E402
from scripts.run_loop_core import eval_subset, verify_corpus  # noqa: E402
from scripts.run_r04_pilot import (  # noqa: E402
    PILOT_TOKENIZER_SHA256,
    sha256,
    tokenizer_semantic_hash,
)


def load_published(run: Path, step: int) -> tuple[dict, dict, CheckpointMeta, dict]:
    """The protocol, its published evaluation and the exact checkpoint state it names."""
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    saved = json.loads((run / f"evaluation-step-{step:06d}.json").read_text(encoding="utf-8"))
    if saved["run_protocol"] != protocol or saved["step"] != step:
        raise ValueError("published evaluation does not belong to this run or step")
    checkpoint = CheckpointMeta.from_dict(saved["checkpoint"])
    manager = CheckpointManager(run / "checkpoints")
    if checkpoint not in manager.read_manifest() or checkpoint.step != step:
        raise ValueError("published checkpoint has rotated; evaluate a snapshot")
    path = manager.slot_path(checkpoint.slot)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != checkpoint.sha256 or os.fstat(stream.fileno()).st_size != checkpoint.bytes:
            raise ValueError("checkpoint checksum or size differs")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if state["step"] != step or json.loads(json.dumps(state["config"])) != protocol["config"]:
        raise ValueError("checkpoint step or configuration differs from the protocol")
    return protocol, saved, checkpoint, state


@torch.no_grad()
def cache_bytes(
    model: ProphetModel, *, context: int, loop_k: int, vocab: int, seed: int, device: str
) -> dict:
    """Prefill ``context`` random tokens at depth ``loop_k`` and weigh every cache slot."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, context), generator=generator).to(device)
    cache = ProphetCache()
    model(ids, cache=cache, loop_k=loop_k, return_mtp=False)
    by_section: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    total = 0
    for (section, _block, _iteration), slot in cache.slots.items():
        size = slot.n_bytes()
        total += size
        by_section[section] = by_section.get(section, 0) + size
        kind = type(slot).__name__
        by_kind[kind] = by_kind.get(kind, 0) + size
    return {
        "context": context,
        "loop_k": loop_k,
        "bytes": total,
        "slots": len(cache.slots),
        "by_section": by_section,
        "by_kind": by_kind,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--tokenizer-sha256", default=PILOT_TOKENIZER_SHA256)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--composition-depths", type=int, nargs="*", default=[1, 2, 4, 6, 8])
    ap.add_argument("--composition-per-level", type=int, default=2000)
    ap.add_argument("--cache-contexts", type=int, nargs="*", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--cache-depths", type=int, nargs="*", default=[2, 4, 6])
    ap.add_argument("--quant-bits", type=int, nargs="*", default=[8, 4])
    ap.add_argument("--quant-depths", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-shard-hashes", action="store_true")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out}: preserve the previous measurement")
    if args.device == "cpu" and not args.allow_cpu:
        ap.error("CPU evaluation is for miniature runs; pass --allow-cpu")
    if any(
        k < 1 for k in args.depths + args.composition_depths + args.cache_depths + args.quant_depths
    ):
        ap.error("depths must be positive")
    if args.device == "cuda":
        if not torch.cuda.is_available() or not HAS_FLA:
            raise RuntimeError("validated CUDA/FLA environment required")
        if os.environ.get("TRITON_F32_DEFAULT") != "tf32x3":
            raise RuntimeError("set TRITON_F32_DEFAULT=tf32x3 before Python starts")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    began = time.time()
    protocol, saved, checkpoint, state = load_published(args.run, args.step)
    provenance = verify_corpus(args.corpus, hash_shards=not args.skip_shard_hashes)
    if provenance["manifest_sha256"] != protocol["data"]["manifest_sha256"]:
        raise ValueError("evaluation corpus differs from the training corpus")
    tokenizer = ProphetTokenizer.load(args.tokenizer)
    semantic = tokenizer_semantic_hash(args.tokenizer)
    if semantic != args.tokenizer_sha256 or semantic != protocol["tokenizer_semantic_sha256"]:
        raise ValueError("tokenizer differs from the training tokenizer")
    cfg = ProphetConfig.from_dict(state["config"])
    model = ProphetModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.to(args.device).float().eval()
    looped = cfg.recurrent.train_loop_max > 1
    depths = sorted(set(args.depths)) if looped else [1]
    composition_depths = (
        [k for k in sorted(set(args.composition_depths)) if k in depths] if looped else [1]
    )
    cache_depths = sorted(set(args.cache_depths)) if looped else [1]
    quant_depths = sorted(set(args.quant_depths)) if looped else [1]
    seq_len, batch = protocol["seq_len"], args.batch_size

    result = {
        "complete": False,
        "programme": "loop-core",
        "arm": protocol["arm"],
        "seed": protocol["seed"],
        "step": args.step,
        "checkpoint": checkpoint.to_dict(),
        "run_protocol_sha256": sha256(args.run / "protocol.json"),
        "corpus_manifest_sha256": provenance["manifest_sha256"],
        "tokenizer_semantic_sha256": semantic,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "precision": "float32, TF32 off, no autocast",
        "looped": looped,
        "default_loop_k": cfg.recurrent.default_loop_k,
        "saved_evaluation": {
            k: saved[k] for k in ("nats_per_token", "bits_per_byte", "loop_k") if k in saved
        },
        "text": {},
        "composition": {},
        "cache": [],
        "quantization": {},
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)

    save()
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)

    def text_at(k: int) -> dict:
        started = time.time()
        scores = evaluate_documents(
            model,
            validation.open(),
            tokenizer,
            device=args.device,
            seq_len=seq_len,
            batch_size=batch,
            precision="float32",
            loop_k=k,
        )
        keep = ("nats_per_token", "bits_per_byte", "scored_tokens", "payload_bytes", "documents")
        return {
            **{key: scores[key] for key in keep if key in scores},
            "seconds": time.time() - started,
        }

    for k in depths:
        result["text"][str(k)] = text_at(k)
        print("TEXT", k, result["text"][str(k)].get("bits_per_byte"), flush=True)
        save()

    examples = eval_subset(
        read_jsonl(args.corpus / "composition-test.jsonl"), args.composition_per_level
    )
    for k in composition_depths:
        started = time.time()
        report = evaluate_composition(
            model,
            examples,
            tokenizer,
            batch_size=batch,
            seq_len=seq_len,
            device=args.device,
            loop_k=k,
        )
        result["composition"][str(k)] = {
            "levels": report["levels"],
            "examples": report["examples"],
            "seconds": time.time() - started,
            "items": [
                {
                    key: item[key]
                    for key in (
                        "kind",
                        "level",
                        "index",
                        "gold",
                        "prediction",
                        "prediction_character_normalized",
                    )
                }
                for item in report["items"]
            ],
        }
        print(
            "COMPOSITION",
            k,
            {n: round(v["accuracy"], 4) for n, v in report["levels"].items()},
            flush=True,
        )
        save()

    for k in cache_depths:
        for context in sorted(set(args.cache_contexts)):
            result["cache"].append(
                cache_bytes(
                    model,
                    context=context,
                    loop_k=k,
                    vocab=cfg.frontend.vocab_size,
                    seed=protocol["seed"],
                    device=args.device,
                )
            )
            print("CACHE", k, context, result["cache"][-1]["bytes"], flush=True)
            save()

    for bits in sorted(set(args.quant_bits), reverse=True):
        quantized, report = quantize_model_rtn(model, bits)
        quantized = quantized.to(args.device).eval()
        entry = {"report": report, "text": {}}
        original = model
        model = quantized
        try:
            for k in quant_depths:
                entry["text"][str(k)] = text_at(k)
                print("QUANT", bits, k, entry["text"][str(k)].get("bits_per_byte"), flush=True)
        finally:
            model = original
            del quantized
            if args.device == "cuda":
                torch.cuda.empty_cache()
        result["quantization"][str(bits)] = entry
        save()

    result["seconds"] = time.time() - began
    result["complete"] = True
    save()
    print("MEASUREMENTS_COMPLETE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
