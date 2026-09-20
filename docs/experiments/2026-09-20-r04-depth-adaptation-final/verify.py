"""Verify exported final evidence; does not reload weights or repeat GPU inference."""

import gzip
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE = HERE.parent / "2026-09-20-r04-fullsize-restart"
TRAINING = "f5a7d71be1723324e36a7be2546c940a69de9fdf"
ANALYSIS = "7f74a23038eef534bd6e828503c3abaa5ac663db"
GATE_SHA = "e1935e51e12858e4d04099b34331501c824508f4f989fc667035fc3c54c02529"
sys.path.insert(0, str(ROOT))
from scripts.summarize_depth_adaptation import PLAN, REFERENCE, summarize  # noqa: E402


def raw(name, root=HERE):
    path = root / name
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(name, root=HERE):
    return json.loads(raw(name, root))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    manifest = read("export-manifest.json")
    assert manifest["training_revision"] == TRAINING
    assert manifest["analysis_revision"] == ANALYSIS
    assert manifest["restart_gate_archive_sha256"] == GATE_SHA
    for name, meta in manifest["files"].items():
        value = raw(name)
        assert len(value) == meta["bytes"] and digest(value) == meta["sha256"], name
    training = read("training-queue/queue.json")
    analysis = read("analysis/queue.json")
    assert training["status"] == "both_512_steps_and_depth_evaluations_complete"
    assert training["revision"] == TRAINING
    assert training["elapsed_seconds"] < training["maximum_seconds"] == 5400
    assert analysis["status"] == "complete" and analysis["analysis_revision"] == ANALYSIS
    assert analysis["analysis_seconds"] < analysis["post_training_maximum_seconds"] == 600
    train_processes = manifest["training_processes"]
    analysis_processes = manifest["analysis_processes"]
    assert set(train_processes) == {
        f"{arm}-to{step}" for arm in ("fixed4", "uniform2to6") for step in (64, 512)
    }
    assert set(analysis_processes) == {
        "fetch",
        "worktree",
        "screen-tests",
        "checkpoint-audit",
        "depth-screen",
    }
    processes = list(train_processes.values()) + list(analysis_processes.values())
    assert all(p["returncode"] == 0 for p in processes)
    assert len({p["pid"] for p in processes}) == len(processes)
    suites = list(ET.fromstring(raw("analysis/screen-tests.xml")).iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 14
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    source = subprocess.check_output(
        ["git", "show", TRAINING + ":scripts/adapt_r04_depth.py"], cwd=ROOT
    )
    summary_source = subprocess.check_output(
        ["git", "show", ANALYSIS + ":scripts/summarize_depth_adaptation.py"], cwd=ROOT
    )
    assert (ROOT / "scripts/summarize_depth_adaptation.py").read_text() == summary_source.decode()
    original = json.loads(REFERENCE.read_bytes())
    audit = read("analysis/final-state-audits.json")
    assert audit["passed"] and set(audit["arms"]) == {"fixed4", "uniform2to6"}
    reports, durations = {}, {}
    for arm in ("fixed4", "uniform2to6"):
        folder = arm + "/"
        final_name = folder + "evaluation-step-000512.json"
        report = reports[arm] = read(final_name)
        identity = report["identity"]
        assert identity["code"] == {"revision": TRAINING, "driver_sha256": digest(source)}
        assert identity == read(arm + "-continuous/adaptation.json", GATE)["identity"]
        adaptation = read(folder + "adaptation.json")
        assert adaptation["identity"] == adaptation["training_contract"]["run_identity"] == identity
        baseline = read(folder + "baseline.json")
        assert baseline["passed"] and baseline["identity"] == identity
        assert baseline["evaluation"]["documents"] == original["documents"]
        assert report["checkpoint"] in read(folder + "checkpoints/manifest.json")["checkpoints"]
        assert report["loader_step"] == 36864
        item = audit["arms"][arm]
        assert item["all_checkpoint_tensors_finite"]
        assert item["inspected"] == {"tensors": 315, "tensor_elements": 867172946}
        assert item["report_sha256"] == digest(raw(final_name))
        for key in (
            "checkpoint",
            "identity",
            "step",
            "tokens_seen",
            "loader_step",
            "depth_history",
        ):
            assert item[key] == report[key], (arm, key)
        records = [json.loads(line) for line in raw(folder + "training.jsonl").splitlines()]
        assert [r["step"] for r in records] == list(range(1, 513))
        assert [r["loop_k"] for r in records] == report["depth_history"]
        for record in records:
            assert record["tokens"] == record["step"] * 16384
            assert record["loader_step"] == 32768 + record["step"] * 8
            assert all(math.isfinite(record[k]) for k in ("loss", "lr", "seconds"))
            assert math.isfinite(record["extra"]["train/grad_norm"])
            assert record["seconds"] > 0
        prefix = [
            json.loads(line) for line in raw(arm + "-continuous/training.jsonl", GATE).splitlines()
        ]
        assert records[:8] == prefix
        assert read(folder + "evaluation-step-000008.json") == read(
            arm + "-continuous/evaluation-step-000008.json", GATE
        )
        durations[arm] = sum(r["seconds"] for r in records)
    summary = summarize(
        reports["fixed4"],
        reports["uniform2to6"],
        json.loads(PLAN.read_bytes()),
        digest(PLAN.read_bytes()),
        original,
    )
    summary["input_files"] = {
        arm: {
            "sha256": digest(raw(arm + "/evaluation-step-000512.json")),
            "bytes": len(raw(arm + "/evaluation-step-000512.json")),
        }
        for arm in reports
    }
    assert summary == read("analysis/summary.json")
    assert summary["checks"] == analysis["checks"]
    assert summary["seed0_screen_passed"] == analysis["seed0_screen_passed"]
    result = {
        "verified": True,
        "source_files": len(manifest["files"]),
        "training_queue_seconds": training["elapsed_seconds"],
        "analysis_seconds": analysis["analysis_seconds"],
        "sum_recorded_update_seconds": durations,
        "seed0_screen_passed": summary["seed0_screen_passed"],
        "checks": summary["checks"],
        "scope": "Exported file hashes, terminal process records, source identities, prior prefix continuity, checkpoint publication metadata and final document scores verified locally. Bootstrap and decision recomputed exactly. Recorded update seconds exclude evaluations and checkpoint writes. Checkpoint tensor finiteness was inspected by the recorded Colab CPU audit; weights are not reloaded here. No repeated inference or cross-seed claim.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
