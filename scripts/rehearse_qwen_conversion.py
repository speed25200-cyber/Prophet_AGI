#!/usr/bin/env python3
"""Create and audit a pinned Qwen3-0.6B shared initialization on CPU.

This does not train or adopt the candidate. The donor tokenizer must be retained.
The BF16 initialization limits workstation memory; recovery quality is unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from prophet.budget import count_parameters  # noqa: E402
from prophet.convert.donors import get_donor  # noqa: E402
from prophet.convert.plan import plan_conversion, prophet_config_for_donor  # noqa: E402
from prophet.convert.weights import convert_state_dict  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts.audit_qwen_blocks import REVISION, WEIGHTS_SHA256  # noqa: E402
from scripts.verify_donors import compare  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--core-mixer", choices=("gdn", "full_attn"), default="gdn",
                        help="change only the core mixer; preserve outer positions and reinjection")
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise FileExistsError("preserve previous initialization and reports")
    source_weights = args.source / "model.safetensors"
    if digest(source_weights) != WEIGHTS_SHA256:
        raise ValueError("source weights differ from the pinned Hub snapshot")
    donor = get_donor("qwen3-0.6b")
    differences = compare(donor, json.loads((args.source / "config.json").read_text()))
    if differences:
        raise ValueError(differences)
    donor = replace(donor, verified=True)
    cfg = prophet_config_for_donor(donor, loop_k=5)
    cfg.recurrent.core_pattern = [args.core_mixer]
    cfg.validate()
    plan = plan_conversion(donor, cfg)
    if plan.coverage()["coverage"] < 0.5:
        raise ValueError("planned direct/averaged coverage below the project floor")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = ProphetModel(cfg)
    finally:
        torch.set_default_dtype(previous_dtype)
    print("MODEL_CREATED", sum(p.numel() for p in model.parameters()), flush=True)
    source = load_file(source_weights, device="cpu")
    converted, transfer = convert_state_dict(source, plan, model.state_dict())
    if transfer.mismatched:
        raise ValueError(transfer.mismatched)
    # Copy into the live module so its tied Parameter identity survives loading.
    model.load_state_dict(converted, strict=True)
    del source, converted
    origins = {
        kind: set(getattr(transfer, kind)) for kind in ("copied", "averaged", "seeded", "fresh")
    }
    parameters = list(model.named_parameters())  # tied embedding/head counted once
    counts = {kind: 0 for kind in origins}
    for name, parameter in parameters:
        if not torch.isfinite(parameter).all():
            raise ValueError(f"non-finite initialized parameter: {name}")
        matches = [kind for kind, names in origins.items() if name in names]
        if len(matches) != 1:
            raise ValueError(f"ambiguous or missing transfer origin: {name}")
        counts[matches[0]] += parameter.numel()
    total = sum(counts.values())
    estimated_total = count_parameters(cfg).total
    assert model.embed.weight is model.lm_head.weight
    actual_coverage = (counts["copied"] + counts["averaged"]) / total
    if actual_coverage < 0.5:
        raise ValueError("actual direct/averaged coverage below the project floor")
    payload = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "donor": donor.hf_id,
        "donor_revision": REVISION,
        "donor_weights_sha256": WEIGHTS_SHA256,
        "core_init": "average",
        "seed": 0,
        "scope": "BF16 initialization only; no recovery training",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("preserve the interrupted save")
    torch.save(payload, temporary)
    temporary.replace(args.out)
    restored = torch.load(args.out, map_location="cpu", weights_only=True, mmap=True)
    for name, parameter in model.state_dict().items():
        if not torch.equal(restored["model"][name], parameter):
            raise ValueError(f"serialized initialization differs: {name}")
    report = {
        "complete": True,
        "hf_id": donor.hf_id,
        "donor_revision": REVISION,
        "donor_weights_sha256": WEIGHTS_SHA256,
        "donor_weights_bytes": source_weights.stat().st_size,
        "checkpoint_sha256": digest(args.out),
        "checkpoint_bytes": args.out.stat().st_size,
        "config": cfg.to_dict(),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "seed": 0,
        "all_parameters_finite": True,
        "serialization_exact": True,
        "unique_parameter_counts_by_origin": counts,
        "total_parameters": total,
        "budget_estimated_parameters": estimated_total,
        "budget_parameter_count_difference": total - estimated_total,
        "actual_direct_or_averaged_coverage": actual_coverage,
        "planned_coverage": plan.coverage(),
        "transfer": asdict(transfer),
        "warnings": cfg.design_warnings(),
        "scope": "shared initialization only; no recovery training, whole-model parity, or quality claim",
        "core_mixer": args.core_mixer,
        "tokenizer": "retain the pinned donor tokenizer; Prophet pilot vocabulary is incompatible",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CONVERSION_AUDITED", counts, actual_coverage, report["checkpoint_sha256"], flush=True)


if __name__ == "__main__":
    main()
