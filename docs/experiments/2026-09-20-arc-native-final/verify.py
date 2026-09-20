"""Verify native ARC export provenance and reproduce its frozen paired analysis."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = "46321013c2b3a52eabd296ac173c255cd1369bec"
TRAINING = HERE.parent / "2026-09-20-r04-reinjection-final"


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
    export = read("export-manifest.json")
    for name, meta in export["files"].items():
        value = raw(name)
        assert len(value) == meta["bytes"] and digest(value) == meta["sha256"], name
    queue = read("queue.json")
    assert queue["status"] == "all_six_and_analysis_complete" and queue["revision"] == REVISION
    assert queue["elapsed_execution_seconds"] < queue["maximum_execution_seconds"] == 1800
    assert queue["tests_passed"] == 28
    assert read("preceding-training-queue.json") == read("queue.json", TRAINING)
    assert queue["preceding_export"]["sha256"] == read("archive-receipt.json", TRAINING)["sha256"]
    assert read("verification.json", TRAINING)["verified"]
    for name, meta in read("export-manifest.json", TRAINING)["files"].items():
        value = raw(name, TRAINING)
        assert len(value) == meta["bytes"] and digest(value) == meta["sha256"]
    namespace = {
        "__name__": "_frozen_native_summary",
        "__file__": str(ROOT / "scripts/summarize_arc_native.py"),
    }
    exec(
        compile(source("scripts/summarize_arc_native.py"), namespace["__file__"], "exec"), namespace
    )
    keys = namespace["KEYS"]
    assert set(queue["results"]) == set(keys)
    processes = queue["processes"]
    assert set(processes) == {*keys, "fetch", "worktree", "tests", "prepare", "analysis"}
    assert all(p["returncode"] == 0 for p in processes.values())
    assert len({p["pid"] for p in processes.values()}) == len(processes)
    manifest_raw = source("docs/experiments/2026-09-20-arc-native-protocol/manifest.json")
    manifest = json.loads(manifest_raw)
    assert manifest == read("items/manifest.json") == queue["input_manifest"]
    item_bytes = (ROOT / "data/arc-native-eval-v2/items.jsonl").read_bytes()
    assert digest(item_bytes) == manifest["items_sha256"]
    items = [json.loads(line) for line in item_bytes.decode("utf-8").splitlines()]
    original = json.loads(
        source("docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json")
    )
    expected_source = {
        "revision": REVISION,
        "driver_sha256": digest(source("scripts/adapt_r04_depth.py")),
        "evaluator_sha256": digest(source("scripts/eval_arc_native.py")),
        "scorer_sha256": digest(source("prophet/eval/choices.py")),
    }
    reports = {}
    for key in keys:
        report = reports[key] = read(key + ".json")
        assert report["source"] == expected_source
        assert queue["results"][key]["report_sha256"] == digest(raw(key + ".json"))
        assert queue["results"][key]["seconds"] == report["seconds"] > 0
        arm = report["arm"]
        published = (
            original
            if arm == "original4096"
            else read(arm + "/evaluation-step-000512.json", TRAINING)
        )
        assert (
            report["checkpoint"]["checkpoint"]
            == published["checkpoint"]
            == queue["results"][key]["checkpoint"]
        )
        assert (
            report["checkpoint"]["training_identity"]
            == published["run_protocol" if arm == "original4096" else "identity"]
        )
        numerical = report["numerical_policy"]
        assert numerical["name"] == "r04-native-choice-fp32-v1"
        assert numerical["deterministic_algorithms"] and not numerical["warn_only"]
        assert not numerical["matmul_allow_tf32"] and not numerical["cudnn_allow_tf32"]
        assert numerical["cublas_workspace_config"] == ":4096:8"
        assert (
            report["runtime"]["triton_f32_default"] == "tf32x3"
            and "A100" in report["runtime"]["device"]
        )
    result = namespace["summarize"](reports, items, manifest, digest(manifest_raw))
    result.update(
        report_sha256={key: digest(raw(key + ".json")) for key in keys},
        input_items_sha256=digest(item_bytes),
        analysis_sha256=digest(source("scripts/summarize_arc_native.py")),
    )
    assert result == read("summary.json")
    verification = {
        "verified": True,
        "revision": REVISION,
        "source_files": len(export["files"]),
        "rows": len(items),
        "reports": len(reports),
        "queue_seconds": queue["elapsed_execution_seconds"],
        "scope": "All original export hashes, process completion, source/inputs/numerical contracts and previously audited checkpoint identities verified locally. Per-choice scores, ties, aggregates and all frozen paired contrasts recomputed; GPU inference and checkpoint tensor loads are not repeated here. No cross-seed or architectural adoption claim.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
