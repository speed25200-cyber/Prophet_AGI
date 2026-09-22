"""Verify the exported separate-process, real-shape deterministic preflights."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def raw(name):
    p = HERE / name
    return p.read_bytes() if p.exists() else gzip.decompress(Path(str(p) + ".gz").read_bytes())


def read(name):
    return json.loads(raw(name))


m = read("export-manifest.json")
assert len(m["files"]) == 24
for name, expected in m["files"].items():
    value = raw(name)
    assert len(value) == expected["bytes"]
    assert hashlib.sha256(value).hexdigest() == expected["sha256"]
assert set(m["processes"]) == {"strict-a", "strict-b"}
assert all(p["returncode"] == 0 for p in m["processes"].values())
assert m["processes"]["strict-a"]["pid"] != m["processes"]["strict-b"]["pid"]
frozen = json.loads(
    gzip.decompress((HERE.parent / "2026-09-20-r04-final-depth/report.json.gz").read_bytes())
)
driver = subprocess.check_output(
    ["git", "show", m["source_revision"] + ":scripts/adapt_r04_depth.py"], cwd=ROOT
)
a, b = [read("paired/strict-" + arm + ".json") for arm in ("a", "b")]
for label, trace in (("a", a), ("b", b)):
    assert trace["complete"] and trace["policy"] == "strict"
    assert trace["deterministic_algorithms"] and trace["cudnn_deterministic"]
    assert trace["script_sha256"] == hashlib.sha256(raw("paired/probe.py")).hexdigest()
    assert len(trace["steps"]) == 3 and len(trace["final_weights"]) == 107
    assert all(len(s["gradients"]) == 106 and all(s["gradients"].values()) for s in trace["steps"])
    report = read("paired/strict-" + label + "/preflight.json")
    identity = report.pop("identity")
    assert report == trace["result"]
    assert report["passed"] and report["actual_depths"] == [6, 6, 6]
    assert report["full_backprop"] and report["peak_allocated_bytes"] == 13_444_140_032
    assert identity["parent_checkpoint"] == frozen["checkpoint"]
    assert identity["runtime"] == frozen["run_protocol"]["runtime"]
    assert identity["data"] == frozen["run_protocol"]["data"]
    assert identity["code"] == {
        "revision": m["source_revision"],
        "driver_sha256": hashlib.sha256(driver).hexdigest(),
    }
    baseline = read("paired/strict-" + label + "/baseline.json")
    assert (
        baseline["passed"]
        and baseline["evaluation"]["documents"] == frozen["results"][0]["documents"]
    )
assert a["initial_weights"] == b["initial_weights"]
assert a["steps"] == b["steps"]
assert a["final_weights"] == b["final_weights"]
assert a["result"]["losses"] == b["result"]["losses"]
assert "SyntaxError" in raw("syntax-failure/strict-a.txt").decode()
assert read("paired/queue.json")["status"] == "complete"
comparison = read("paired/comparison.json")
assert comparison["initial_equal"] and comparison["final_different"] == []
assert len(comparison["steps"]) == 3
for s in comparison["steps"]:
    assert s["inputs_equal"] and s["rng_before_equal"] and s["rng_after_equal"]
    assert s["different_gradients"] == []
result = {
    "verified_source_files": 24,
    "independent_processes": 2,
    "updates_per_process": 3,
    "gradient_tensor_hashes_equal_each_step": 106,
    "final_state_tensor_hashes_equal": 107,
    "input_and_rng_hashes_equal_each_step": True,
    "losses_and_gradient_norms_equal": True,
    "peak_allocated_bytes_each": 13_444_140_032,
    "scope": "Repeated disposable full-size k6 preflights, not interrupted training. Hashes are reported by the instrumented GPU processes and checked here; no second local GPU execution or isolated-kernel cause is claimed. Instrumented times include CPU hashing and are not throughput measurements.",
}
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
