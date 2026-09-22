"""Recompute exported gates and partial-run accounting, without loading weights."""

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


manifest = read("export-manifest.json")
for name, expected in manifest["files"].items():
    content = raw(name)
    assert len(content) == expected["bytes"]
    assert hashlib.sha256(content).hexdigest() == expected["sha256"]
assert len(manifest["files"]) == 22
assert all(p["returncode"] == 0 for p in manifest["processes"].values())
assert set(manifest["processes"]) == {
    "fetch",
    "worktree",
    "cuda-restart-tests",
    "preflight-fixed4",
    "preflight-uniform2to6",
    "fixed4-to-64",
}
suites = list(ET.fromstring(raw("cuda-restart-tests.xml")).iter("testsuite"))
assert sum(int(s.attrib["tests"]) for s in suites) == 6
assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
cases = [c for s in suites for c in s.iter("testcase")]
assert sum(c.attrib["name"].endswith("cuda]") for c in cases) == 2

plan_bytes = (HERE.parent / "2026-09-20-r04-depth-adaptation-plan/protocol.json").read_bytes()
plan = json.loads(plan_bytes)
frozen = json.loads(
    gzip.decompress((HERE.parent / "2026-09-20-r04-final-depth/report.json.gz").read_bytes())
)
baseline = frozen["results"][0]
assert baseline["loop_k"] == 4
driver = subprocess.check_output(
    ["git", "show", manifest["source_revision"] + ":scripts/adapt_r04_depth.py"],
    cwd=ROOT,
)
for arm in ("preflight-fixed4", "preflight-uniform2to6", "fixed4"):
    report = read(arm + "/baseline.json")
    identity = report["identity"]
    assert report["passed"] and report["evaluation"]["documents"] == baseline["documents"]
    assert identity["code"] == {
        "revision": manifest["source_revision"],
        "driver_sha256": hashlib.sha256(driver).hexdigest(),
    }
    assert identity["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert identity["parent_checkpoint"] == frozen["checkpoint"] == plan["parent_checkpoint"]
    assert identity["runtime"] == frozen["run_protocol"]["runtime"]
    assert identity["data"] == frozen["run_protocol"]["data"]

preflights = [read(a + "/preflight.json") for a in ("preflight-fixed4", "preflight-uniform2to6")]
for p in preflights:
    assert p["passed"] and p["full_backprop"] and p["actual_depths"] == [6, 6, 6]
    assert p["steps"] == 3 and len(p["losses"]) == 3
    assert p["peak_allocated_bytes"] == 13_444_140_032 < 0.9 * p["device_capacity_bytes"]
    assert all(math.isfinite(v) for row in p["losses"] for v in row.values())
assert preflights[0]["losses"][0]["loss"] == preflights[1]["losses"][0]["loss"]
assert preflights[0]["losses"][0]["gradient_norm"] != preflights[1]["losses"][0]["gradient_norm"]
partial = read("fixed4/evaluation-step-000054.json")
assert not partial["complete"] and partial["step"] == 54 and partial["skipped_nonfinite"] == 0
assert partial["tokens_seen"] == 54 * 8 * 2048 == 884736
assert partial["depth_history"] == [4] * 54 and partial["depth_counts"] == {"4": 54}
assert partial["checkpoint"] in read("fixed4/checkpoints/manifest.json")["checkpoints"]
assert set(partial["results"]) == {"4"}
rows = [json.loads(line) for line in raw("fixed4/training.jsonl").splitlines()]
assert len(rows) == 54
for step, row in enumerate(rows, 1):
    assert row["step"] == step and row["tokens"] == step * 8 * 2048
    assert row["loop_k"] == 4 and row["loader_step"] == (4096 + step) * 8
    assert math.isfinite(row["loss"]) and math.isfinite(row["extra"]["train/grad_norm"])
assert partial["loader_step"] == rows[-1]["loader_step"]
evaluation = partial["results"]["4"]
documents = evaluation["documents"]
assert len(documents) == len(baseline["documents"]) == 376
for row, original in zip(documents, baseline["documents"], strict=True):
    assert all(row[k] == original[k] for k in ("index", "sha256", "scored_tokens", "scored_bytes"))
for key in ("total_nats", "scored_tokens", "scored_bytes"):
    assert math.isclose(sum(r[key] for r in documents), evaluation[key], rel_tol=1e-12)
assert math.isclose(
    evaluation["nats_per_token"],
    evaluation["total_nats"] / evaluation["scored_tokens"],
    rel_tol=1e-12,
)
assert math.isclose(
    evaluation["bits_per_byte"],
    evaluation["total_nats"] / math.log(2) / evaluation["scored_bytes"],
    rel_tol=1e-12,
)
queue = read("queue.json")
assert queue["status"] == "stopped_requires_inspection"
assert read("operator-stop.json")["pid"] == manifest["processes"]["fixed4-to-64"]["pid"]
assert "evaluation-step-000064.json" in raw("error.txt").decode()
result = {
    "verified_source_files": 22,
    "cuda_suite_passed": 6,
    "cuda_restart_cases": 2,
    "preflight_peak_allocated_bytes": [p["peak_allocated_bytes"] for p in preflights],
    "first_gradient_norms": [p["losses"][0]["gradient_norm"] for p in preflights],
    "partial_step": 54,
    "partial_tokens": partial["tokens_seen"],
    "partial_ce": evaluation["nats_per_token"],
    "paired_adaptation_complete": False,
    "scope": "Export hashes, code/protocol identities, miniature CUDA tests, reported memory gates and partial-run aggregates. No checkpoint tensor reload, repeated GPU forward, real-shape restart equivalence or cause of gradient discrepancy is established here.",
}
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
