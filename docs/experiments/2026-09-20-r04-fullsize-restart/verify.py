"""Verify exported evidence; checkpoint tensor equality was audited on the A100."""

import gzip
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def raw(name):
    path = HERE / name
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(name):
    return json.loads(raw(name))


def main():
    manifest = read("export-manifest.json")
    assert len(manifest["files"]) == 46
    for name, meta in manifest["files"].items():
        value = raw(name)
        assert len(value) == meta["bytes"] and hashlib.sha256(value).hexdigest() == meta["sha256"]
    queue = read("queue.json")
    assert queue["status"] == "both_real_shape_restart_gates_passed"
    assert queue["elapsed_seconds"] < queue["maximum_seconds"] == 1800
    assert queue["production_prefix"] == "continuous" and not queue["long_training_queued"]
    processes = manifest["processes"]
    assert len(processes) == 13 and all(p["returncode"] == 0 for p in processes.values())
    assert len({p["pid"] for p in processes.values()}) == 13
    suites = list(ET.fromstring(raw("cpu-cuda-tests.xml")).iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 9
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    assert sum(c.attrib["name"].endswith("cuda]") for s in suites for c in s.iter("testcase")) == 2
    original = json.loads(
        (
            ROOT / "docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json"
        ).read_bytes()
    )
    revision = manifest["revision"]
    assert revision == queue["revision"] == "f5a7d71be1723324e36a7be2546c940a69de9fdf"
    driver = subprocess.check_output(
        ["git", "show", revision + ":scripts/adapt_r04_depth.py"], cwd=ROOT
    )
    plan_raw = (
        ROOT / "docs/experiments/2026-09-20-r04-depth-adaptation-plan/protocol.json"
    ).read_bytes()
    results = {}
    for arm in ("fixed4", "uniform2to6"):
        audit = read(arm + "-restart-audit.json")
        identity = audit["identity"]
        assert identity["code"] == {
            "revision": revision,
            "driver_sha256": hashlib.sha256(driver).hexdigest(),
        }
        assert identity["plan_sha256"] == hashlib.sha256(plan_raw).hexdigest()
        assert identity["parent_checkpoint"] == original["checkpoint"]
        assert identity["numerical_policy"]["name"] == "r04-strict-determinism-v1"
        assert identity["numerical_policy"]["deterministic_algorithms"]
        assert audit["passed"] and audit["per_document_evaluation_equal"]
        assert audit["step"] == 8 and audit["tokens_seen"] == 131072
        assert audit["state_comparison"] == {"tensors": 315, "tensor_elements": 867172946}
        history = [4] * 8 if arm == "fixed4" else [6, 6, 5, 2, 5, 6, 4, 5]
        assert audit["depth_history"] == history
        reports, logs = [], []
        for kind, side in (("continuous", "left"), ("interrupted", "right")):
            folder = arm + "-" + kind
            baseline = read(folder + "/baseline.json")
            assert baseline["passed"] and baseline["identity"] == identity
            assert baseline["evaluation"]["documents"] == original["documents"]
            adaptation = read(folder + "/adaptation.json")
            assert (
                adaptation["identity"]
                == adaptation["training_contract"]["run_identity"]
                == identity
            )
            report = read(folder + "/evaluation-step-000008.json")
            assert report["identity"] == identity and not report["complete"]
            assert report["step"] == 8 and report["tokens_seen"] == 131072
            assert report["loader_step"] == 32832 and report["skipped_nonfinite"] == 0
            assert report["depth_history"] == history
            assert report["checkpoint"] == audit[side + "_checkpoint"]
            assert (
                report["checkpoint"] in read(folder + "/checkpoints/manifest.json")["checkpoints"]
            )
            evaluation = report["results"]["4"]
            assert set(report["results"]) == {"4"} and evaluation["loop_k"] == 4
            for row, ref in zip(evaluation["documents"], original["documents"], strict=True):
                assert all(
                    row[k] == ref[k] for k in ("index", "sha256", "scored_tokens", "scored_bytes")
                )
                assert math.isfinite(row["total_nats"]) and row["total_nats"] > 0
            total = sum(row["total_nats"] for row in evaluation["documents"])
            assert math.isclose(total, evaluation["total_nats"], rel_tol=1e-12)
            assert math.isclose(total / 393040, evaluation["nats_per_token"], rel_tol=1e-12)
            assert math.isclose(
                total / 1750592 / math.log(2), evaluation["bits_per_byte"], rel_tol=1e-12
            )
            reports.append(report)
            records = [json.loads(line) for line in raw(folder + "/training.jsonl").splitlines()]
            assert [r["step"] for r in records] == list(range(1, 9))
            assert [r["loop_k"] for r in records] == history
            logs.append([{k: v for k, v in r.items() if k != "seconds"} for r in records])
        assert reports[0]["results"] == reports[1]["results"] and logs[0] == logs[1]
        preflight = read("preflight-" + arm + "/preflight.json")
        assert preflight["passed"] and preflight["full_backprop"]
        assert preflight["actual_depths"] == [6, 6, 6]
        assert (
            preflight["peak_allocated_bytes"]
            == 13444140032
            < 0.9 * preflight["device_capacity_bytes"]
        )
        results[arm] = {
            "depth_history": history,
            "exact_restart": True,
            **audit["state_comparison"],
        }
    a, b = (read("preflight-" + arm + "/preflight.json") for arm in ("fixed4", "uniform2to6"))
    assert a["losses"] == b["losses"]
    verification = {
        "verified": True,
        "source_files": 46,
        "results": results,
        "queue_seconds": queue["elapsed_seconds"],
        "cpu_cuda_tests_passed": 9,
        "source_zip_bytes": 349577,
        "source_zip_sha256": "e1935e51e12858e4d04099b34331501c824508f4f989fc667035fc3c54c02529",
        "scope": "Exported source identities, separate terminal process IDs, metadata, original baseline rows, paired evaluations and training logs recomputed locally. Exact checkpoint tensor/optimizer/RNG equality was checked by the recorded A100 auditor; weights are not included or reloaded here. No quality decision.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
