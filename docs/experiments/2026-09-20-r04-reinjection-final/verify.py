"""Recompute the final input-adapter screen from immutable exported evidence."""

import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE = HERE.parent / "2026-09-20-r04-reinjection-gates"
REVISION = "dce35abc439fdeeb8f1482b1219dee3cc242c267"
ARMS = ("fixed_sum", "learned_mix")


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


def source(path):
    return subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=ROOT)


def main():
    manifest = read("export-manifest.json")
    assert manifest["revision"] == REVISION
    for name, meta in manifest["files"].items():
        content = raw(name)
        assert len(content) == meta["bytes"] and digest(content) == meta["sha256"], name
    queue = read("queue.json")
    assert queue["status"] == "paired_training_and_screen_complete"
    assert (
        queue["revision"] == REVISION
        and queue["elapsed_seconds"] < queue["maximum_seconds"] == 5400
    )
    assert queue["gate_export"]["sha256"] == read("archive-receipt.json", GATE)["sha256"]
    gate_verification = read("verification.json", GATE)
    assert gate_verification["verified"] and gate_verification["revision"] == REVISION
    # Check source bytes again so a stale verification receipt cannot mask edits.
    for name, meta in read("export-manifest.json", GATE)["files"].items():
        content = raw(name, GATE)
        assert len(content) == meta["bytes"] and digest(content) == meta["sha256"]
    processes = manifest["processes"]
    assert set(processes) == {"fixed_sum-to512", "learned_mix-to512", "final-state-audit", "screen"}
    assert processes == queue["processes"]
    assert all(p["returncode"] == 0 for p in processes.values())
    assert len({p["pid"] for p in processes.values()}) == 4

    namespace = {
        "__name__": "_frozen_input_adapter_screen",
        "__file__": str(ROOT / "scripts/summarize_depth_adaptation.py"),
    }
    exec(
        compile(source("scripts/summarize_depth_adaptation.py"), namespace["__file__"], "exec"),
        namespace,
    )
    plan_bytes = source("docs/experiments/2026-09-20-r04-reinjection-plan/protocol.json")
    plan = json.loads(plan_bytes)
    original = json.loads(
        source("docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json")
    )
    code = {
        "revision": REVISION,
        "driver_sha256": digest(source("scripts/adapt_r04_depth.py")),
        "entrypoint_sha256": digest(source("scripts/adapt_r04_reinjection.py")),
    }
    audit = read("final-state-audit.json")
    assert audit["passed"] and set(audit["arms"]) == set(ARMS)
    reports, durations = {}, {}
    for arm in ARMS:
        folder = arm + "/"
        final_name = folder + "evaluation-step-000512.json"
        report = reports[arm] = read(final_name)
        identity = report["identity"]
        assert identity["code"] == code
        assert identity == read(arm + "-continuous/adaptation.json", GATE)["identity"]
        assert digest(raw(final_name)) == queue["results"][arm]["report_sha256"]
        adaptation = read(folder + "adaptation.json")
        assert adaptation["identity"] == adaptation["training_contract"]["run_identity"] == identity
        assert adaptation["config"] == read(arm + "-continuous/adaptation.json", GATE)["config"]
        baseline = read(folder + "baseline.json")
        assert baseline["passed"] and baseline["identity"] == identity
        assert baseline["evaluation"]["documents"] == original["documents"]
        assert report["checkpoint"] in read(folder + "checkpoints/manifest.json")["checkpoints"]
        assert report["loader_step"] == 36864
        item = audit["arms"][arm]
        assert item["checkpoint"] == report["checkpoint"] and item["identity"] == identity
        assert item["state"] == read(arm + "-restart.json", GATE)["state_comparison"]
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
    summary = namespace["summarize"](
        reports["fixed_sum"],
        reports["learned_mix"],
        plan,
        digest(plan_bytes),
        original,
        experiment="reinjection",
    )
    summary["input_files"] = {
        arm: {
            "sha256": digest(raw(arm + "/evaluation-step-000512.json")),
            "bytes": len(raw(arm + "/evaluation-step-000512.json")),
        }
        for arm in ARMS
    }
    assert summary == read("summary.json")
    assert (
        summary["checks"] == queue["checks"]
        and summary["seed0_screen_passed"] == queue["seed0_screen_passed"]
    )
    verification = {
        "verified": True,
        "revision": REVISION,
        "source_files": len(manifest["files"]),
        "queue_seconds": queue["elapsed_seconds"],
        "sum_recorded_update_seconds": durations,
        "seed0_screen_passed": summary["seed0_screen_passed"],
        "checks": summary["checks"],
        "scope": "Exported source hashes, terminal processes, checkpoint metadata, continuity with audited eight-step prefixes, all training records and all final document scores verified locally. Paired bootstrap and decision reproduced with the exact frozen analysis source. Checkpoint tensor finiteness was inspected by the recorded Colab auditor; weights are not reloaded here. No repeated inference, cross-seed, reasoning or adoption claim.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
