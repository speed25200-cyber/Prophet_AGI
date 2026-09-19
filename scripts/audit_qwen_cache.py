#!/usr/bin/env python3
"""Check cached execution of the real hybrid initialization on one held-out prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "audit", "reference", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-scan", action="store_true")
    parser.add_argument("--donor-control", action="store_true")
    parser.add_argument("--trace-blocks", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve existing evidence")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    audit = json.loads(args.audit.read_text())
    assert digest(args.checkpoint) == audit["checkpoint_sha256"]
    reference = json.loads(args.reference.read_text())["arms"]["donor"]["documents"][0]
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    ids = None
    for path in sorted(args.validation.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            if hashlib.sha256(text.encode()).hexdigest() == reference["document_sha256"]:
                ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)[:129]])
                break
        if ids is not None:
            break
    assert ids is not None
    assert hashlib.sha256(ids.numpy().tobytes()).hexdigest() == reference["target_ids_sha256"]
    ids = ids[:, :-1]
    torch.set_num_threads(2)
    if args.donor_control:
        assert not args.reference_scan
        assert digest(args.source / "model.safetensors") == audit["donor_weights_sha256"]
        model = AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
        )
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model = ProphetModel(ProphetConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["model"], strict=True)
        del payload
    model.eval()
    if args.reference_scan:
        for block in model.sections["core"]:
            block.mixer.chunk_size = None
    result = {
        "scope": "one fixed 128-token prefix, CPU float32; hybrid uses fixed k=5",
        "model": "unchanged_donor" if args.donor_control else "hybrid_initialization",
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "donor_weights_sha256": audit["donor_weights_sha256"],
        "gdn_scan": None
        if args.donor_control
        else ("sequential_reference" if args.reference_scan else "chunk64"),
        "document_sha256": reference["document_sha256"],
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "atol": 1e-4,
        "rtol": 1e-4,
        "paths": {},
    }
    full_hidden, stage_counts, trace = {}, {}, {}
    tracing_reference, position = True, 0
    handles = []
    if args.trace_blocks:
        assert not args.donor_control, "block tracing currently targets Prophet sections"

        def make_hook(name):
            def hook(module, inputs, output):
                iteration = stage_counts.get(name, 0)
                stage_counts[name] = iteration + 1
                key = f"{name}/iteration-{iteration}"
                if tracing_reference:
                    full_hidden[key] = output.detach().clone()
                else:
                    expected = full_hidden[key][:, position : position + output.shape[1]]
                    difference = (output - expected).float()
                    row = trace.setdefault(
                        key, {"max_absolute_error": 0.0, "max_relative_l2_error": 0.0}
                    )
                    row["max_absolute_error"] = max(
                        row["max_absolute_error"], difference.abs().max().item()
                    )
                    relative = difference.norm() / expected.float().norm().clamp_min(1e-12)
                    row["max_relative_l2_error"] = max(
                        row["max_relative_l2_error"], relative.item()
                    )

            return hook

        for section, blocks in model.sections.items():
            for index, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_hook(f"{section}/{index}")))
    with torch.inference_mode():
        full = (
            model(ids, use_cache=False)
            if args.donor_control
            else model(ids, loop_k=5, return_mtp=False)
        ).logits
        assert torch.isfinite(full).all()
        tracing_reference = False
        for name, lengths in (
            ("chunked_prefill_then_decode", [64, 63, 1]),
            ("tokenwise", [1] * 128),
        ):
            cache = None if args.donor_control else ProphetCache()
            position, max_error, max_scaled_error, argmax_matches = 0, 0.0, 0.0, 0
            finite = True
            trace = {}
            for length in lengths:
                stage_counts.clear()
                if args.donor_control:
                    call = model(
                        ids[:, position : position + length], past_key_values=cache, use_cache=True
                    )
                    output, cache = call.logits, call.past_key_values
                    del call
                else:
                    output = model(
                        ids[:, position : position + length],
                        cache=cache,
                        loop_k=5,
                        return_mtp=False,
                    ).logits
                expected = full[:, position : position + length]
                finite = finite and bool(torch.isfinite(output).all())
                error = (output - expected).abs()
                max_error = max(max_error, error.max().item())
                max_scaled_error = max(
                    max_scaled_error, (error / (1e-4 + 1e-4 * expected.abs())).max().item()
                )
                argmax_matches += (output.argmax(-1) == expected.argmax(-1)).sum().item()
                position += length
                del output, error
            assert (
                position
                == (cache.get_seq_length() if args.donor_control else cache.position)
                == 128
            )
            result["paths"][name] = {
                "all_logits_finite": finite,
                "max_absolute_error": max_error,
                "max_tolerance_ratio": max_scaled_error,
                "passed": finite and max_scaled_error <= 1,
                "argmax_matches": argmax_matches,
                "positions": position,
                "cache": {"position": cache.get_seq_length()}
                if args.donor_control
                else cache.summary(),
            }
            if args.trace_blocks:
                result["paths"][name]["block_errors"] = trace
            print("CACHE_AUDIT", name, result["paths"][name], flush=True)
        del full
    for handle in handles:
        handle.remove()
    result["passed"] = all(path["passed"] for path in result["paths"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit("cache numerical tolerance failed; failure report preserved")


if __name__ == "__main__":
    main()
