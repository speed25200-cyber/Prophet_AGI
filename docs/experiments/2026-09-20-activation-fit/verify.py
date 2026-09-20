"""Recheck exported identities and arithmetic, without repeating GPU inference.

Run from the repository root. Local source weights/corpora remain outside Git.
"""

import hashlib
import json
import math
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prophet.data.donor_tokenizer import DonorByteTokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((HERE / name).read_bytes())


def main():
    manifest = read("export-manifest.json")
    for name, expected in manifest["files"].items():
        data = (HERE / name).read_bytes()
        assert len(data) == expected["bytes"] and sha(data) == expected["sha256"], name
    report = read("fit/report.json")
    contract = report["contract"]
    for name, key in [
        ("scripts/probe_qwen_covariance.py", "script_sha256"),
        ("prophet/convert/shared_fit.py", "fit_source_sha256"),
    ]:
        source = subprocess.check_output(
            ["git", "show", f"{manifest['revision']}:{name}"], cwd=ROOT
        )
        assert sha(source) == contract[key]
    tokenizer_path = ROOT / "data/donor-qwen3-0.6b/source/tokenizer.json"
    assert sha(tokenizer_path.read_bytes()) == contract["tokenizer"]["source_sha256"]
    tokenizer = DonorByteTokenizer(tokenizer_path, eos_id=151645, pad_id=151643, vocab_size=151936)
    assert tokenizer.fingerprint() == contract["tokenizer"]
    identities = {}
    for split, count in [("train", 64), ("validation", 16)]:
        path = ROOT / f"data/qwen-recovery-v1/{split}.jsonl"
        data = path.read_bytes()
        assert sha(data) == contract[f"{split}_sha256"]
        texts = [json.loads(line)["text"] for line in data.decode().splitlines() if line.strip()]
        documents = {sha(text.encode()): text for text in texts}
        selected = []
        for document_sha, text in sorted(documents.items()):
            ids = tokenizer.encode(text, add_eos=False)
            if len(ids) < 512:
                continue
            selected.append(
                {
                    "document_sha256": document_sha,
                    "input_ids_sha256": sha(struct.pack("<512q", *ids[:512])),
                }
            )
            if len(selected) == count:
                break
        assert len(selected) == count and selected == contract[f"{split}_prefixes"]
        identities[split] = selected
    assert not (
        {r["document_sha256"] for r in identities["train"]}
        & {r["document_sha256"] for r in identities["validation"]}
    )
    assert contract["groups"] == [[i, i + 1] for i in range(4, 24, 2)]
    assert contract["ridge_fraction"] == 0.01 and contract["train_input_tokens"] == 32768
    assert contract["expected_registered_parameters"] == 438763520
    assert contract["runtime"] == {
        "torch": "2.11.0+cu128",
        "transformers": "5.17.0",
        "device": "cuda",
        "device_name": "NVIDIA A100-SXM4-40GB",
        "forward_dtype": "float32",
        "moment_and_solve_dtype": "float64",
        "allow_tf32": False,
        "deterministic_algorithms": True,
        "threads": 2,
    }
    assert report["complete"] and report["next_document"] == 64 and report["resumed_from"] == 1
    assert report["moment_storage_bytes"] == 2516582400
    first = read("first-document-report.json")
    assert not first["complete"] and first["next_document"] == 1
    assert first["contract"] == contract
    fits = report["fits"]
    projections = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    assert set(fits) == {f"{i}:{p}" for i in range(4, 24, 2) for p in projections}
    assert report["fit_count"] == len(fits) == 70
    for fit in fits.values():
        assert all(math.isfinite(value) for value in fit.values())
        assert fit["ridge_fraction"] == 0.01 and fit["ridge_penalty"] > 0
        assert 0 <= fit["fit_data_error_fp64"] <= fit["fit_regularized_objective_fp64"]
        assert fit["fit_regularized_objective_fp64"] <= fit["mean_data_error"] * (1 + 1e-8)
        assert 0 <= fit["fit_data_error_fp32"] <= fit["mean_data_error"] * (1 + 1e-8)
    scores = {}
    assert set(report["results"]) == {"donor", "mean-control", "activation-fit"}
    for name, result in report["results"].items():
        rows = result["documents"]
        assert [
            {k: row[k] for k in ("document_sha256", "input_ids_sha256")} for row in rows
        ] == identities["validation"]
        assert all(row["targets"] == 511 and math.isfinite(row["total_nats"]) for row in rows)
        targets, nats = sum(row["targets"] for row in rows), sum(row["total_nats"] for row in rows)
        assert targets == result["targets"] == 8176 and nats == result["total_nats"]
        assert nats / targets == result["nats_per_token"]
        if name != "donor":
            assert result["registered_unique_parameters"] == 438763520
        scores[name] = nats / targets
    suites = list(ET.parse(HERE / "solver-tests.xml").getroot().iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 5
    assert all(int(s.attrib[key]) == 0 for s in suites for key in ["failures", "errors", "skipped"])
    assert any("[cuda]" in case.attrib["name"] for s in suites for case in s.iter("testcase"))
    queue = read("queue.json")
    assert queue["status"] == "complete" and queue["results"] == scores
    assert queue["actual_cpu_cuda_solver_tests_passed"] == 5
    assert queue["elapsed_seconds"] < queue["maximum_seconds"] == 1800
    verification = {
        "complete": True,
        "source_revision": manifest["revision"],
        "source_zip_sha256": "7f0f68ba39494df8d4e8f61e26e79ce1f9b97a83fcbeb06a5de5cf98d86fc0eb",
        "scope": "Byte identities, independently selected tokenized inputs, arithmetic and recorded objective bounds; no second GPU forward or refit.",
        "scores_nats_per_token": scores,
        "fit_count": 70,
        "cpu_cuda_solver_tests": 5,
        "calibration_documents": 64,
        "resumed_from": 1,
        "validation_targets": 8176,
        "fit_vs_mean_ce_change_percent": 100
        * (scores["activation-fit"] / scores["mean-control"] - 1),
        "maximum_fit_to_mean_local_objective_ratio": max(
            f["fit_regularized_objective_fp64"] / f["mean_data_error"] for f in fits.values()
        ),
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
