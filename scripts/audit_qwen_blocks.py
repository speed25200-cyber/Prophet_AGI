#!/usr/bin/env python3
"""Check real Qwen3-0.6B block transfers against Transformers on CPU.

This is a bounded copy-equivalence check, not an evaluation of a hybrid model.
Download the public snapshot separately; no remote code or network access is used.
Requires transformers==5.17.0 and safetensors==0.8.0 for the recorded experiment.
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
from safetensors import safe_open  # noqa: E402

from prophet.convert.donors import get_donor  # noqa: E402
from prophet.convert.plan import (  # noqa: E402
    BlockSource,
    plan_conversion,
    prophet_config_for_donor,
)
from prophet.convert.weights import convert_state_dict  # noqa: E402
from prophet.modeling.layers import RMSNorm, RotaryEmbedding  # noqa: E402
from prophet.modeling.model import ProphetBlock  # noqa: E402
from scripts.verify_donors import compare  # noqa: E402

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the existing audit")
    import transformers
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

    torch.set_num_threads(2)
    weights = args.source / "model.safetensors"
    with weights.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != WEIGHTS_SHA256:
        raise ValueError("weights differ from the pinned public Hub LFS hash")
    raw_config = json.loads((args.source / "config.json").read_text())
    donor = get_donor("qwen3-0.6b")
    differences = compare(donor, raw_config)
    if differences:
        raise ValueError(differences)
    donor = replace(donor, verified=True)
    cfg = prophet_config_for_donor(donor)
    # Isolate unchanged full-attention blocks; SWA/NoPE/GDN are separate adaptations.
    cfg.mixer.pattern = ["full_attn"]
    cfg.mixer.nope_layers = ()
    reference_config = Qwen3Config(**raw_config)
    reference_config._attn_implementation = "sdpa"
    rows = []
    with safe_open(weights, framework="pt", device="cpu") as source, torch.inference_mode():
        for layer in (0, 13, 27):
            prefix = f"model.layers.{layer}."
            source_state = {
                name: source.get_tensor(name)
                for name in source.keys()  # noqa: SIM118
                if name.startswith(prefix)
            }
            reference = Qwen3DecoderLayer(reference_config, layer).float().eval()
            reference.load_state_dict(
                {name.removeprefix(prefix): value for name, value in source_state.items()},
                strict=True,
            )
            target = (
                ProphetBlock(cfg, kind="full_attn", is_moe=False, layer_index=0, section="prelude")
                .float()
                .eval()
            )
            plan = plan_conversion(donor, cfg)
            plan.blocks = [BlockSource("prelude", 0, "full_attn", (layer,), True)]
            target_prefix = "sections.prelude.0."
            state, transfer = convert_state_dict(
                source_state,
                plan,
                {target_prefix + name: tensor for name, tensor in target.state_dict().items()},
            )
            if transfer.mismatched or transfer.fresh:
                raise ValueError("block transfer left uncopied or mismatched tensors")
            target.load_state_dict(
                {name.removeprefix(target_prefix): tensor for name, tensor in state.items()},
                strict=True,
            )
            rotary = RotaryEmbedding(donor.head_dim, theta=donor.rope_theta)
            reference_rotary = Qwen3RotaryEmbedding(reference_config)
            rng = torch.Generator().manual_seed(1000 + layer)
            x = torch.randn(2, 17, donor.d_model, generator=rng)
            for offset in (0, 257):
                positions = torch.arange(offset, offset + x.shape[1]).expand(x.shape[0], -1)
                cos, sin = rotary(positions)
                ref_cos, ref_sin = reference_rotary(x, positions)
                torch.testing.assert_close(cos, ref_cos, rtol=0, atol=1e-6)
                torch.testing.assert_close(sin, ref_sin, rtol=0, atol=1e-6)
                mask = torch.ones(17, 17, dtype=torch.bool).tril()[None, None]
                expected = reference(x, attention_mask=mask, position_embeddings=(ref_cos, ref_sin))
                measured = target(x, cos=cos, sin=sin)
                torch.testing.assert_close(measured, expected, rtol=1e-5, atol=1e-4)
                error = (measured - expected).abs().max().item()
                # Quantify the previous conversion default on these actual weights.
                for module in target.modules():
                    if isinstance(module, RMSNorm):
                        module.eps = 1e-5
                previous = target(x, cos=cos, sin=sin)
                for module in target.modules():
                    if isinstance(module, RMSNorm):
                        module.eps = donor.norm_eps
                rows.append(
                    {
                        "donor_layer": layer,
                        "position_offset": offset,
                        "max_abs_error": error,
                        "old_epsilon_max_abs_error": (previous - expected).abs().max().item(),
                        "copied_tensors": len(transfer.copied),
                    }
                )
                print("BLOCK_PARITY", rows[-1], flush=True)
    result = {
        "passed": True,
        "hf_id": donor.hf_id,
        "revision": REVISION,
        "weights_sha256": actual_sha,
        "weights_bytes": weights.stat().st_size,
        "config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "precision": "CPU float32; donor tensors converted from saved precision",
        "input_shape": [2, 17, donor.d_model],
        "atol": 1e-4,
        "rtol": 1e-5,
        "scope": "three real full-attention blocks; synthetic inputs; not whole-model or hybrid quality",
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
