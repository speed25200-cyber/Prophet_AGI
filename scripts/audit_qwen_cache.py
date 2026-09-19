#!/usr/bin/env python3
"""Check cached execution of a donor, initialization or recovery checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.modeling.layers import (  # noqa: E402
    CausalSelfAttention,
    GatedDeltaNet,
    RMSNorm,
    RotaryEmbedding,
)
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402


def gdn_fp64_forward(layer, x, *, state=None):
    """Diagnostic rank-one delta recurrence; no production FP32 scan or fused kernel."""
    if x.dtype != torch.float64:
        raise TypeError("FP64 diagnostic received a lower-precision GDN input")
    b, length, _ = x.shape
    h, dk, dv = layer.n_heads, layer.head_k, layer.head_v
    qkv = torch.cat([layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)], dim=-1)
    qkv = F.silu(layer._causal_conv(qkv, state))
    q, k, v = qkv.split([h * dk, h * dk, h * dv], dim=-1)
    q, k, v = q.reshape(b, length, h, dk), k.reshape(b, length, h, dk), v.reshape(b, length, h, dv)
    k = F.normalize(k, dim=-1, eps=1e-6)
    alpha, beta = torch.sigmoid(layer.a_proj(x)), layer.beta_max * torch.sigmoid(layer.b_proj(x))
    memory = x.new_zeros(b, h, dv, dk) if state is None or state.state is None else state.state.to(x)
    outputs = []
    for t in range(length):
        key = k[:, t].unsqueeze(-1)
        memory = alpha[:, t, :, None, None] * memory
        correction = beta[:, t, :, None, None] * (v[:, t].unsqueeze(-1) - memory @ key)
        memory = memory + correction @ key.transpose(-1, -2)
        outputs.append((memory @ q[:, t].unsqueeze(-1)).squeeze(-1))
    if state is not None:
        state.state = memory.detach()
        state.seen += length
    return layer.o_proj(layer.o_norm(torch.stack(outputs, dim=1)).reshape(b, length, h * dv))


def configure_fp64_oracle(model: ProphetModel, *, include_gdn: bool = False) -> None:
    """Diagnostic instance only: double arithmetic, including RMS and optional GDN.

    Preserve the production FP32 rotary tables so position construction remains
    identical. Do not change production implementations or silently use FP32 scans.
    """
    allowed = (CausalSelfAttention, GatedDeltaNet) if include_gdn else (CausalSelfAttention,)
    if any(not isinstance(block.mixer, allowed)
           for blocks in model.sections.values() for block in blocks):
        raise ValueError("FP64 oracle requires attention-only sections or explicitly enabled GDN")
    if include_gdn and not any(isinstance(m, GatedDeltaNet) for m in model.modules()):
        raise ValueError("hybrid FP64 oracle requires GDN")

    def rms_forward(layer, x):
        if x.dtype != torch.float64:
            raise TypeError("FP64 diagnostic received a lower-precision RMS input")
        out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + layer.eps)
        return out * layer.weight if layer.weight is not None else out

    model.double()
    for layer in model.modules():
        if isinstance(layer, RMSNorm):
            layer.forward = MethodType(rms_forward, layer)
        elif isinstance(layer, RotaryEmbedding):
            layer.float()
        elif isinstance(layer, GatedDeltaNet):
            layer.forward = MethodType(gdn_fp64_forward, layer)


def attention_fp64_oracle(model: ProphetModel) -> None:
    configure_fp64_oracle(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--checkpoint", type=Path)
    artifact.add_argument("--recovery-run", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--reference-scan", action="store_true")
    parser.add_argument("--donor-control", action="store_true")
    parser.add_argument("--trace-blocks", action="store_true")
    parser.add_argument("--loop-k", type=int, default=5)
    oracle = parser.add_mutually_exclusive_group()
    oracle.add_argument("--attention-fp64-oracle", action="store_true",
                        help="diagnostic only: double attention/residual/RMS arithmetic; not the FP32 gate")
    oracle.add_argument("--hybrid-fp64-oracle", action="store_true",
                        help="diagnostic only: double arithmetic with a sequential GDN oracle")
    args = parser.parse_args()
    if args.checkpoint is not None and (args.audit is None or args.step is not None):
        parser.error("initialization requires --audit and does not accept --step")
    if args.recovery_run is not None and (args.step is None or args.audit is not None or args.donor_control):
        parser.error("recovery requires --step and does not accept --audit or --donor-control")
    if args.loop_k < 1:
        parser.error("--loop-k must be positive")
    fp64 = args.attention_fp64_oracle or args.hybrid_fp64_oracle
    if fp64 and (args.donor_control or args.reference_scan):
        parser.error("FP64 oracles require a Prophet initialization and select their own scan")
    if args.out.exists():
        raise FileExistsError("preserve existing evidence")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recovery_audit, payload = None, None
    if args.recovery_run is not None:
        from scripts.audit_recovery_checkpoint import load_evaluated_checkpoint
        payload, recovery_audit = load_evaluated_checkpoint(args.recovery_run, args.step)
        identity = recovery_audit["training_contract"]["run_identity"]
        audit = {"checkpoint_sha256": recovery_audit["checkpoint"]["sha256"],
                 "donor_weights_sha256": identity["donor_weights_sha256"]}
    else:
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
        if payload is None:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model = ProphetModel(ProphetConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["model"], strict=True)
        del payload
    model.eval()
    gdn_layers = [layer for layer in model.modules() if isinstance(layer, GatedDeltaNet)]
    if args.reference_scan:
        if not gdn_layers:
            raise ValueError("--reference-scan requires at least one GDN layer")
        for layer in gdn_layers:
            layer.chunk_size = None
    if fp64:
        configure_fp64_oracle(model, include_gdn=args.hybrid_fp64_oracle)
    tolerance = 1e-8 if fp64 else 1e-4
    result = {
        "scope": "one fixed 128-token prefix; diagnostic FP64 oracle, not the production FP32 gate"
        if fp64
        else "one fixed 128-token prefix, CPU float32; unchanged elementwise tolerance",
        "precision": "float64 with FP32 rotary tables" if fp64 else "float32",
        "attention_fp64_oracle": args.attention_fp64_oracle,
        "hybrid_fp64_oracle": args.hybrid_fp64_oracle,
        "model": ("unchanged_donor" if args.donor_control else
                  "recovered_checkpoint" if recovery_audit else "converted_initialization"),
        "recovery_checkpoint_audit": recovery_audit,
        "core_pattern": None if args.donor_control else model.cfg.recurrent.core_pattern,
        "loop_k": None if args.donor_control else args.loop_k,
        "torch": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "donor_weights_sha256": audit["donor_weights_sha256"],
        "gdn_scan": None
        if not gdn_layers
        else ("fp64_sequential_oracle" if args.hybrid_fp64_oracle else
              "sequential_reference" if args.reference_scan else "chunk64"),
        "document_sha256": reference["document_sha256"],
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "atol": tolerance,
        "rtol": tolerance,
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
                    difference = output - expected
                    row = trace.setdefault(
                        key, {"max_absolute_error": 0.0, "max_relative_l2_error": 0.0}
                    )
                    row["max_absolute_error"] = max(
                        row["max_absolute_error"], difference.abs().max().item()
                    )
                    relative = difference.norm() / expected.norm().clamp_min(1e-12)
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
            else model(ids, loop_k=args.loop_k, return_mtp=False)
        ).logits
        assert torch.isfinite(full).all()
        assert full.dtype == (torch.float64 if fp64 else torch.float32)
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
                        loop_k=args.loop_k,
                        return_mtp=False,
                    ).logits
                expected = full[:, position : position + length]
                finite = finite and bool(torch.isfinite(output).all())
                error = (output - expected).abs()
                max_error = max(max_error, error.max().item())
                max_scaled_error = max(
                    max_scaled_error, (error / (tolerance + tolerance * expected.abs())).max().item()
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
