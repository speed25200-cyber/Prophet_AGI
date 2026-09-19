#!/usr/bin/env python3
"""Compare the initialization and donor on four held-out 128-target prefixes.

CPU float32 forward smoke only. This is neither full validation nor a recovery result.
Both arms use the same pinned donor vocabulary and exact next-token targets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the existing smoke report")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    audit = json.loads(args.audit.read_text())
    assert audit["complete"] and digest(args.checkpoint) == audit["checkpoint_sha256"]
    assert digest(args.source / "model.safetensors") == audit["donor_weights_sha256"]
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    samples = []
    for path in sorted(args.validation.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)[:129]
            if len(ids) != 129:
                continue
            samples.append((hashlib.sha256(text.encode()).hexdigest(), torch.tensor([ids])))
            if len(samples) == 4:
                break
        if len(samples) == 4:
            break
    assert len(samples) == 4
    result = {
        "scope": "four held-out 128-target prefixes; forward smoke only; not whole-validation quality",
        "precision": "CPU float32",
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "donor_weights_sha256": audit["donor_weights_sha256"],
        "arms": {},
    }
    for arm in ("hybrid_initialization", "donor"):
        began = time.perf_counter()
        if arm == "hybrid_initialization":
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            model = ProphetModel(ProphetConfig.from_dict(payload["config"]))
            model.load_state_dict(payload["model"], strict=True)
            del payload
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.source, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
            )
        model.eval()
        rows = []
        with torch.inference_mode():
            for sha, ids in samples:
                if arm == "hybrid_initialization":
                    logits = model(ids[:, :-1], return_mtp=False).logits
                else:
                    logits = model(ids[:, :-1], use_cache=False).logits
                if not torch.isfinite(logits).all():
                    raise ValueError(f"non-finite {arm} logits")
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
                ).item()
                rows.append(
                    {
                        "document_sha256": sha,
                        "tokens": 128,
                        "total_nats": loss,
                        "target_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    }
                )
                del logits
        result["arms"][arm] = {
            "documents": rows,
            "nats_per_token": sum(r["total_nats"] for r in rows) / 512,
            "scored_tokens": 512,
            "all_logits_finite": True,
            "seconds_including_load": time.perf_counter() - began,
        }
        print("FORWARD_SMOKE", arm, result["arms"][arm]["nats_per_token"], flush=True)
        del model
        gc.collect()
    result["complete"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
