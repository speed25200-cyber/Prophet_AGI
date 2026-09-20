"""Resumable, train-only activation calibration for adjacent shared Qwen weights.

Experimental initialization scout, not an adopted architecture or recovery run.
The original donor supplies inputs before any sharing. Development text is scored
only after fitting; no development activations enter the fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from prophet.convert.shared_fit import fit_shared_projection  # noqa: E402
from scripts.audit_recovery_cache_suite import select_prefixes  # noqa: E402
from scripts.probe_qwen_relaxed import PROJECTIONS  # noqa: E402
from scripts.recover_qwen import (  # noqa: E402
    REVISION,
    WEIGHTS_SHA256,
    digest,
    load_source,
    read_documents,
    write_report,
)

INPUTS = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")
INPUT_FOR = {
    path: (
        "self_attn.q_proj"
        if path in PROJECTIONS[:3]
        else "mlp.gate_proj"
        if path == "mlp.up_proj"
        else path
    )
    for path in PROJECTIONS
}


def token_hash(ids):
    return hashlib.sha256(torch.tensor([ids], dtype=torch.long).numpy().tobytes()).hexdigest()


def evaluate(model, prefixes, device):
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for sha, ids in prefixes:
            tokens = torch.tensor([ids], device=device)
            logits = model(tokens, use_cache=False).logits[:, :-1].float()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite scout logits")
            nats = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum"
                )
            )
            rows.append(
                {
                    "document_sha256": sha,
                    "input_ids_sha256": token_hash(ids),
                    "targets": len(ids) - 1,
                    "total_nats": nats,
                }
            )
    total = sum(r["total_nats"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    return {
        "nats_per_token": total / targets,
        "total_nats": total,
        "targets": targets,
        "documents": rows,
        "seconds": time.perf_counter() - started,
    }


def save_moments(path, contract, next_document, moments):
    temporary = path.with_suffix(".tmp")
    payload = {
        "contract": contract,
        "next_document": next_document,
        "moments": {k: value.detach().cpu() for k, value in moments.items()},
    }
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "train", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-documents-per-session", type=int, default=64)
    args = parser.parse_args()
    if args.max_documents_per_session < 1:
        parser.error("session document limit must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_config, tokenizer = load_source(args.source)
    if digest(args.train) != "542508d6418b13e7c06def4b3de7cbcbd3ff0d21dd1f045487e1d0f4f3a2f3d6":
        raise ValueError("training source changed")
    if (
        digest(args.validation)
        != "06903e9925fb4e2ceed27eedc09f5d4b33b32e3e2ce1bfe6e1deed62a491e8af"
    ):
        raise ValueError("development source changed")
    train = [
        (sha, ids[:512])
        for sha, ids in select_prefixes(read_documents(args.train), tokenizer, 64, [512])
    ]
    validation = [
        (sha, ids[:512])
        for sha, ids in select_prefixes(read_documents(args.validation), tokenizer, 16, [512])
    ]
    assert not {sha for sha, _ in train} & {sha for sha, _ in validation}
    import transformers
    from transformers import AutoModelForCausalLM

    contract = {
        "protocol": "qwen-adjacent-second-moment-fit-v1",
        "script_sha256": digest(Path(__file__)),
        "fit_source_sha256": digest(
            Path(__file__).resolve().parent.parent / "prophet/convert/shared_fit.py"
        ),
        "donor_revision": REVISION,
        "donor_weights_sha256": WEIGHTS_SHA256,
        "source_config": source_config,
        "tokenizer": tokenizer.fingerprint(),
        "train_sha256": digest(args.train),
        "validation_sha256": digest(args.validation),
        "train_prefixes": [
            {"document_sha256": sha, "input_ids_sha256": token_hash(ids)} for sha, ids in train
        ],
        "validation_prefixes": [
            {"document_sha256": sha, "input_ids_sha256": token_hash(ids)} for sha, ids in validation
        ],
        "prefix_selection": "First distinct eligible document hashes ascending within each split, first 512 tokens, no EOS.",
        "ridge_fraction": 0.01,
        "groups": [[i, i + 1] for i in range(4, 24, 2)],
        "train_input_tokens": 32768,
        "expected_registered_parameters": 438763520,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "forward_dtype": "float32",
            "moment_and_solve_dtype": "float64",
            "allow_tf32": False,
            "deterministic_algorithms": True,
            "threads": 2,
        },
        "scope": "Train-only teacher-activation local ridge fit; no gradient training, GDN, reinjection, auxiliary heads, adaptive depth, benchmark selection or architecture adoption.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    if report_path.exists():
        prior = json.loads(report_path.read_bytes())
        if prior["contract"] != contract or prior["complete"]:
            raise ValueError("changed contract or completed experiment; preserve prior results")
    report = {"complete": False, "contract": contract, "status": "loading", "results": {}}
    write_report(report_path, report)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    layers = model.model.layers
    assert len(layers) == 28 and sum(p.numel() for p in model.parameters()) == 596049920
    moment_file = args.out / "moments.pt"
    moments, next_document = {}, 0
    if moment_file.exists():
        saved = torch.load(moment_file, weights_only=True, map_location="cpu")
        assert saved["contract"] == contract
        next_document = saved["next_document"]
        assert type(next_document) is int and 0 <= next_document <= len(train)
        moments = {k: value.to(device) for k, value in saved["moments"].items()}
        del saved
    expected_keys = {f"{i}:{path}" for i in range(4, 24) for path in INPUTS}
    if moments:
        assert set(moments) == expected_keys
    for i in range(4, 24):
        for path in INPUTS:
            key = f"{i}:{path}"
            width = layers[i].get_submodule(path).weight.shape[1]
            if key not in moments:
                moments[key] = torch.zeros(width, width, dtype=torch.float64, device=device)
            assert moments[key].shape == (width, width) and moments[key].dtype == torch.float64
            assert torch.isfinite(moments[key]).all()
    report["moment_storage_bytes"] = sum(x.numel() * x.element_size() for x in moments.values())
    handles, seen = [], set()

    def collector(key):
        def hook(module, inputs):
            if key in seen:
                raise ValueError("calibration projection executed twice in one donor forward")
            seen.add(key)
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            assert x.shape[0] == 512 and torch.isfinite(x).all()
            moments[key].addmm_(x.T, x)

        return hook

    resumed_from = next_document
    try:
        for i in range(4, 24):
            for path in INPUTS:
                handles.append(
                    layers[i]
                    .get_submodule(path)
                    .register_forward_pre_hook(collector(f"{i}:{path}"))
                )
        with torch.no_grad():
            for index in range(
                next_document, min(len(train), next_document + args.max_documents_per_session)
            ):
                seen.clear()
                ids = train[index][1]
                model.model(torch.tensor([ids], device=device), use_cache=False)
                assert seen == expected_keys
                next_document = index + 1
                if next_document % 16 == 0 or next_document == min(
                    len(train), resumed_from + args.max_documents_per_session
                ):
                    save_moments(moment_file, contract, next_document, moments)
                report.update(
                    status="collecting", next_document=next_document, resumed_from=resumed_from
                )
                write_report(report_path, report)
                print("MOMENTS", next_document, len(train), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if next_document < len(train):
        print("CALIBRATION_PAUSED", next_document, flush=True)
        return
    report["results"]["donor"] = evaluate(model, validation, device)
    write_report(report_path, report)
    fitted, means, fit_reports = {}, {}, {}
    for first in range(4, 24, 2):
        for path in PROJECTIONS:
            key = f"{first}:{path}"
            weights = [layers[i].get_submodule(path).weight.detach() for i in (first, first + 1)]
            hs = [moments[f"{i}:{INPUT_FOR[path]}"] for i in (first, first + 1)]
            fitted[key], fit_reports[key] = fit_shared_projection(weights, hs, ridge=0.01)
            means[key] = torch.stack(weights).mean(0)
        print("FITTED_PAIR", first, first + 1, flush=True)
    report["fits"] = fit_reports
    report["fit_count"] = len(fitted)
    assert len(fitted) == 70
    moments.clear()
    originals = []
    for first in range(4, 24, 2):
        for path in PROJECTIONS:
            parent, name = path.rsplit(".", 1)
            for index in (first, first + 1):
                owner = layers[index].get_submodule(parent)
                originals.append((owner, name, owner.get_submodule(name)))
    for label, values in [("mean-control", means), ("activation-fit", fitted)]:
        try:
            for first in range(4, 24, 2):
                for path in PROJECTIONS:
                    weight = values[f"{first}:{path}"]
                    replacement = nn.Linear(
                        weight.shape[1], weight.shape[0], bias=False, device=device
                    )
                    replacement.weight = nn.Parameter(weight)
                    parent, name = path.rsplit(".", 1)
                    for index in (first, first + 1):
                        setattr(layers[index].get_submodule(parent), name, replacement)
            count = sum(p.numel() for p in model.parameters())
            assert count == 438763520
            report["results"][label] = evaluate(model, validation, device)
            report["results"][label]["registered_unique_parameters"] = count
            write_report(report_path, report)
            print(
                "COVARIANCE_RESULT", label, report["results"][label]["nats_per_token"], flush=True
            )
        finally:
            for owner, name, module in originals:
                setattr(owner, name, module)
    report.update(
        complete=True,
        status="complete",
        allocation_scope="Donor modules retained for restoration; registered counts do not measure resident memory.",
    )
    write_report(report_path, report)
    print("COVARIANCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
