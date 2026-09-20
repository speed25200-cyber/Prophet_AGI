"""Verify gate exports locally; full checkpoint tensor comparison ran on A100."""

import gzip
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = "dce35abc439fdeeb8f1482b1219dee3cc242c267"
ARMS = ("fixed_sum", "learned_mix")


def raw(name):
    path = HERE / name
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(name):
    return json.loads(raw(name))


def source_hash(path):
    return hashlib.sha256(
        subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=ROOT)
    ).hexdigest()


def main():
    manifest = read("export-manifest.json")
    for name, meta in manifest["files"].items():
        content = raw(name)
        assert len(content) == meta["bytes"]
        assert hashlib.sha256(content).hexdigest() == meta["sha256"]
    queue = read("queue.json")
    assert manifest["revision"] == queue["revision"] == REVISION
    assert queue["status"] == "all_gates_passed"
    assert queue["elapsed_seconds"] < queue["maximum_seconds"] == 1800
    assert queue["continuation_preselected"] == [a + "-continuous" for a in ARMS]
    processes = manifest["processes"]
    expected_processes = {"fetch", "worktree", "tests", "initial-equality"} | {
        a + "-" + p for a in ARMS for p in ("preflight", "continuous8", "split1", "split7", "audit")
    }
    assert set(processes) == expected_processes and processes == queue["processes"]
    assert all(p["returncode"] == 0 for p in processes.values())
    assert len({p["pid"] for p in processes.values()}) == len(processes)
    suites = list(ET.fromstring(raw("tests.xml")).iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == queue["tests_passed"] == 52
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    names = [c.attrib["name"] for s in suites for c in s.iter("testcase")]
    assert "test_cuda_bfloat16_zero_adapter_preserves_predictions" in names
    assert sum("cuda" in n for n in names) >= 5

    plan_folder = ROOT / "docs/experiments/2026-09-20-r04-reinjection-plan"
    plan_bytes = (plan_folder / "protocol.json").read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    original_path = (
        ROOT / "docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json"
    )
    original = json.loads(original_path.read_bytes())
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == plan["parent_report_sha256"]
    assert plan["parent_checkpoint"] == original["checkpoint"]
    code = {"revision": REVISION, "driver_sha256": source_hash("scripts/adapt_r04_depth.py")}
    equal = read("initial-equality.json")
    assert equal["code"] == {
        **code,
        "gate_sha256": source_hash("scripts/gate_r04_input_adapter.py"),
    }
    assert equal["passed"] and equal["input_shape"] == [8, 2048]
    assert equal["loader_step_after"] == 32776
    assert equal["plan_sha256"] == plan_sha and equal["parent_checkpoint"] == original["checkpoint"]
    assert set(equal["depths"]) == {"4", "6"}
    assert all(all(r.values()) for r in equal["depths"].values())

    results = {}
    history = [6, 6, 5, 2, 5, 6, 4, 5]
    assert queue["paired_depth_history"] == history
    for arm in ARMS:
        audit = read(arm + "-restart.json")
        identity = audit["identity"]
        assert identity["protocol"] == "r04-input-adapter-v1" and identity["arm"] == arm
        assert identity["mode"] == "train" and identity["evaluation_depths"] == [4, 1, 2, 6, 8]
        assert identity["code"] == {
            **code,
            "entrypoint_sha256": source_hash("scripts/adapt_r04_reinjection.py"),
        }
        assert (
            identity["plan_sha256"] == plan_sha
            and identity["parent_checkpoint"] == original["checkpoint"]
        )
        assert identity["numerical_policy"]["name"] == "r04-strict-determinism-v1"
        assert identity["numerical_policy"]["deterministic_algorithms"]
        assert not identity["numerical_policy"]["warn_only"]
        assert identity["runtime"] == original["run_protocol"]["runtime"] == equal["runtime"]
        assert audit["passed"] and audit["per_document_evaluation_equal"]
        assert audit["step"] == 8 and audit["tokens_seen"] == 131072
        assert audit["depth_history"] == history
        assert audit["state_comparison"]["tensors"] >= 315
        assert audit["state_comparison"]["tensor_elements"] >= 867172946
        reports, logs = [], []
        for kind, side in (("continuous", "left"), ("resumed", "right")):
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
            assert adaptation["config"] == json.loads((plan_folder / (arm + ".json")).read_bytes())
            report = read(folder + "/evaluation-step-000008.json")
            assert report["identity"] == identity and not report["complete"]
            assert report["step"] == 8 and report["tokens_seen"] == 131072
            assert report["loader_step"] == 32832 and report["skipped_nonfinite"] == 0
            assert report["depth_history"] == history
            assert report["checkpoint"] == audit[side + "_checkpoint"]
            assert (
                report["checkpoint"] in read(folder + "/checkpoints/manifest.json")["checkpoints"]
            )
            assert set(report["results"]) == {"4"}
            evaluation = report["results"]["4"]
            assert evaluation["loop_k"] == 4
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
        preflight = read(arm + "-preflight/preflight.json")
        assert (
            preflight["passed"]
            and preflight["full_backprop"]
            and preflight["actual_depths"] == [6] * 3
        )
        assert preflight["steps"] == 3 and preflight["maximum_loop_k"] == 6
        assert preflight["peak_allocated_bytes"] < 0.9 * preflight["device_capacity_bytes"]
        assert all(math.isfinite(x) and x > 0 for row in preflight["losses"] for x in row.values())
        results[arm] = {
            "exact_restart": True,
            "peak_allocated_bytes": preflight["peak_allocated_bytes"],
            **audit["state_comparison"],
        }
    verification = {
        "verified": True,
        "source_files": len(manifest["files"]),
        "revision": REVISION,
        "queue_seconds": queue["elapsed_seconds"],
        "cpu_cuda_tests_passed": 52,
        "initial_predictions_equal": True,
        "results": results,
        "scope": "Exported source hashes, separate terminal PIDs, exact original baseline rows, paired evaluation and training log equality recomputed locally. Checkpoint model/optimizer/RNG tensor equality was checked on A100 by the recorded auditor; weights are not included or reloaded here. No quality or capability decision.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
