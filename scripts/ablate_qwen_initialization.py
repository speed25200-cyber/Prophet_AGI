#!/usr/bin/env python3
"""Diagnose conversion loss on fixed held-out prefixes, without recovery training.

The predeclared arms separate core initialization, input reinjection, GDN and
NoPE. All forwards use float32 after BF16 storage. No architecture is adopted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.convert.donors import get_donor  # noqa: E402
from prophet.convert.plan import plan_conversion  # noqa: E402
from prophet.convert.weights import convert_state_dict  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts.audit_qwen_blocks import WEIGHTS_SHA256  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402

ARMS = (
    ("attention_average_serial", "full_attn", "average", False, False),
    ("attention_average_injected", "full_attn", "average", True, False),
    ("attention_stride_serial", "full_attn", "stride", False, False),
    ("gdn_rope_serial", "gdn", "average", False, False),
    ("gdn_rope_injected", "gdn", "average", True, False),
    ("gdn_nope_serial", "gdn", "average", False, True),
    ("gdn_nope_injected", "gdn", "average", True, True),
)


def attention_weights(model, source_path, mode):
    """Stream layer groups into the resident BF16 model to bound CPU memory."""
    donor = replace(get_donor("qwen3-0.6b"), verified=True)
    plan = plan_conversion(donor, model.cfg, core_init=mode)
    with safe_open(source_path, framework="pt", device="cpu") as source:
        assert {source.get_slice(k).get_dtype() for k in source.keys()} == {"BF16"}  # noqa: SIM118
        with torch.no_grad():
            model.embed.weight.copy_(source.get_tensor("model.embed_tokens.weight"))
            model.norm_out.weight.copy_(source.get_tensor("model.norm.weight"))
        for origin in plan.blocks:
            prefixes = tuple(f"model.layers.{i}." for i in origin.donor_layers)
            values = {
                k: source.get_tensor(k)
                for k in source.keys()  # noqa: SIM118
                if k.startswith(prefixes)
            }
            block = model.sections[origin.section][origin.index]
            prefix = f"sections.{origin.section}.{origin.index}."
            state, report = convert_state_dict(
                values,
                replace(plan, blocks=[origin]),
                {prefix + k: v for k, v in block.state_dict().items()},
            )
            if report.fresh or report.mismatched:
                raise ValueError((report.fresh, report.mismatched))
            block.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()})
            del values, state
    assert model.embed.weight is model.lm_head.weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "audit", "reference", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve existing evidence")
    from transformers import AutoTokenizer

    audit = json.loads(args.audit.read_text())
    reference = json.loads(args.reference.read_text())
    assert digest(args.checkpoint) == audit["checkpoint_sha256"]
    assert digest(args.source / "model.safetensors") == WEIGHTS_SHA256
    assert audit["donor_weights_sha256"] == reference["donor_weights_sha256"] == WEIGHTS_SHA256
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    samples, targets = [], []
    for path in sorted(args.validation.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)[:129]
            if len(ids) != 129:
                continue
            samples.append(torch.tensor([ids]))
            targets.append(
                {
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "tokens": 128,
                    "target_ids_sha256": hashlib.sha256(samples[-1].numpy().tobytes()).hexdigest(),
                }
            )
            if len(samples) == 4:
                break
        if len(samples) == 4:
            break
    assert len(samples) == 4
    for row, expected in zip(targets, reference["arms"]["donor"]["documents"], strict=True):
        assert all(row[key] == expected[key] for key in row)
    torch.set_num_threads(2)
    result = {
        "scope": "fixed four-prefix diagnostic; 512 targets; no recovery or adoption claim",
        "donor_weights_sha256": WEIGHTS_SHA256,
        "hybrid_checkpoint_sha256": audit["checkpoint_sha256"],
        "precision": "BF16 stored weights, CPU float32 forwards; donor source is entirely BF16",
        "donor_nats_per_token": reference["arms"]["donor"]["nats_per_token"],
        "predeclared_arms": [list(arm) for arm in ARMS],
        "arms": {},
    }
    for name, mixer, mode, injected, nope in ARMS:
        torch.manual_seed(0)
        cfg = ProphetConfig.from_dict(audit["config"])
        cfg.recurrent.core_pattern = [mixer]
        cfg.recurrent.inject_input_each_step = injected
        cfg.recurrent.eval_state_init = "zeros" if injected else "prelude"
        cfg.mixer.nope_layers = (1,) if nope else ()
        # SWA remains unchanged: the 128-token prefixes fit entirely in its window.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = ProphetModel(cfg)
        finally:
            torch.set_default_dtype(previous_dtype)
        if mixer == "gdn":
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(payload["model"], strict=True)
            del payload
        else:
            attention_weights(model, args.source / "model.safetensors", mode)
        model.float().eval()
        rows = []
        with torch.inference_mode():
            for ids, target in zip(samples, targets, strict=True):
                logits = model(ids[:, :-1], loop_k=5, return_mtp=False).logits
                assert torch.isfinite(logits).all()
                nats = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
                ).item()
                rows.append({**target, "total_nats": nats})
                del logits
        ce = sum(row["total_nats"] for row in rows) / 512
        result["arms"][name] = {
            "nats_per_token": ce,
            "all_logits_finite": True,
            "documents": rows,
            "config": cfg.to_dict(),
            "core_init": mode,
        }
        print("INITIALIZATION_ABLATION", name, ce, flush=True)
        if name == "gdn_nope_injected":
            assert abs(ce - reference["arms"]["hybrid_initialization"]["nats_per_token"]) <= 1e-5
        del model
        gc.collect()
    result["complete"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
