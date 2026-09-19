#!/usr/bin/env python3
"""Stream a complete copied Qwen stack through Prophet blocks on four prefixes.

One block is resident at a time to bound workstation memory. This checks the
unmodified full-attention baseline; no GDN, sharing, NoPE or recovery is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from prophet.convert.donors import get_donor  # noqa: E402
from prophet.convert.plan import plan_conversion, prophet_config_for_donor  # noqa: E402
from prophet.convert.weights import convert_state_dict  # noqa: E402
from prophet.modeling.layers import RMSNorm, RotaryEmbedding  # noqa: E402
from prophet.modeling.model import ProphetBlock  # noqa: E402
from scripts.audit_qwen_blocks import REVISION, WEIGHTS_SHA256  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402
from scripts.verify_donors import compare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the existing audit")
    from transformers import AutoTokenizer

    weights = args.source / "model.safetensors"
    assert digest(weights) == WEIGHTS_SHA256
    donor = get_donor("qwen3-0.6b")
    assert not compare(donor, json.loads((args.source / "config.json").read_text()))
    donor = replace(donor, verified=True)
    cfg = prophet_config_for_donor(donor)
    cfg.recurrent.enabled = False
    cfg.n_layers = donor.n_layers
    cfg.mixer.pattern = ["full_attn"]
    cfg.mixer.nope_layers = ()
    cfg.heads.n_multi_token_predict = 0
    cfg.heads.confidence_head = False
    plan = plan_conversion(donor, cfg)
    assert [block.donor_layers for block in plan.blocks] == [(i,) for i in range(28)]
    torch.set_num_threads(2)
    reference = json.loads(args.reference.read_text())
    assert reference["donor_weights_sha256"] == WEIGHTS_SHA256
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    samples, targets = [], []
    for path in sorted(args.validation.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)[:129]
            if len(ids) != 129:
                continue
            samples.append(torch.tensor(ids))
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
    for measured, expected in zip(targets, reference["arms"]["donor"]["documents"], strict=True):
        assert all(measured[key] == expected[key] for key in measured)
    ids = torch.stack(samples)
    with safe_open(weights, framework="pt", device="cpu") as source, torch.inference_mode():
        embedding = source.get_tensor("model.embed_tokens.weight").float()
        h = F.embedding(ids[:, :-1], embedding)
        cos, sin = RotaryEmbedding(donor.head_dim, theta=donor.rope_theta)(
            torch.arange(128).expand(4, -1)
        )
        for origin in plan.blocks:
            prefix = f"model.layers.{origin.index}."
            source_state = {
                name: source.get_tensor(name)
                for name in source.keys()  # noqa: SIM118
                if name.startswith(prefix)
            }
            block = ProphetBlock(
                cfg, kind="full_attn", is_moe=False, layer_index=origin.index, section="trunk"
            ).eval()
            target_prefix = f"sections.trunk.{origin.index}."
            state, report = convert_state_dict(
                source_state,
                replace(plan, blocks=[origin]),
                {target_prefix + k: v for k, v in block.state_dict().items()},
            )
            assert not report.fresh and not report.mismatched
            block.load_state_dict({k.removeprefix(target_prefix): v for k, v in state.items()})
            h = block(h, cos=cos, sin=sin)
            assert torch.isfinite(h).all()
            del block, state, source_state
        norm = RMSNorm(donor.d_model, donor.norm_eps)
        norm.weight.copy_(source.get_tensor("model.norm.weight"))
        logits = F.linear(norm(h), embedding)
        assert torch.isfinite(logits).all()
        for i, row in enumerate(targets):
            row["total_nats"] = F.cross_entropy(logits[i], ids[i, 1:], reduction="sum").item()
    ce = sum(row["total_nats"] for row in targets) / 512
    expected_ce = reference["arms"]["donor"]["nats_per_token"]
    if abs(ce - expected_ce) > 1e-5:
        raise ValueError(f"copied stack loss differs: {ce} vs {expected_ce}")
    result = {
        "passed": True,
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "scope": "28 full-attention Prophet blocks streamed on four 128-target prefixes; no architecture change",
        "precision": "CPU float32",
        "nats_per_token": ce,
        "reference_nats_per_token": expected_ce,
        "absolute_ce_difference": abs(ce - expected_ce),
        "tolerance": 1e-5,
        "config": cfg.to_dict(),
        "documents": targets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("FULL_COPIED_STACK_PASSED", ce, expected_ce, flush=True)


if __name__ == "__main__":
    main()
