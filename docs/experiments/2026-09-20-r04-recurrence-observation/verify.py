"""Check exported recurrence observations and summarize scalar rows, without weights."""

import gzip
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FINAL = HERE.parent / "2026-09-20-r04-depth-adaptation-final"
REVISION = "2e4d25ecd521980ee71607b2d96d3ed13b224210"
METRICS = (
    "state_rms",
    "token_centered_energy_fraction",
    "mean_distinct_token_cosine",
    "previous_loop_cosine",
    "relative_previous_loop_change",
)


def raw(name, root=HERE):
    path = root / name
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(name, root=HERE):
    return json.loads(raw(name, root))


def main():
    manifest = read("export-manifest.json")
    assert manifest["revision"] == REVISION
    assert manifest["model_revision"] == "f5a7d71be1723324e36a7be2546c940a69de9fdf"
    assert len(manifest["files"]) == 11
    for name, metadata in manifest["files"].items():
        value = raw(name)
        assert len(value) == metadata["bytes"]
        assert hashlib.sha256(value).hexdigest() == metadata["sha256"], name
    source = subprocess.check_output(
        ["git", "show", REVISION + ":scripts/diagnose_r04_recurrence.py"], cwd=ROOT
    )
    assert source == raw("executed-driver.py")
    queue = read("queue.json")
    assert queue["status"] == "complete" and queue["revision"] == REVISION
    assert queue["pair_seconds"] < queue["maximum_pair_seconds"] == 600
    assert (
        queue["final_evidence_sha256"] == read("archive-receipt.json", FINAL)["source_zip_sha256"]
    )
    processes = manifest["processes"]
    assert set(processes) == {"fetch", "worktree", "tests", "fixed4", "uniform2to6"}
    assert all(p["returncode"] == 0 for p in processes.values())
    assert len({p["pid"] for p in processes.values()}) == 5
    suites = list(ET.fromstring(raw("tests.xml")).iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 8
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    arms = {}
    input_identities = []
    for arm in ("fixed4", "uniform2to6"):
        report = read(arm + ".json")
        final = read(arm + "/evaluation-step-000512.json", FINAL)
        assert report["complete"] and report["protocol"] == "r04-recurrence-observation-v1"
        assert (
            report["identity"] == final["identity"] and report["checkpoint"] == final["checkpoint"]
        )
        assert report["driver_sha256"] == hashlib.sha256(source).hexdigest()
        documents = report["documents"]
        assert len(documents) == 16
        input_identities.append(
            [(d["index"], d["sha256"], d["input_ids_sha256"]) for d in documents]
        )
        for index, document in enumerate(documents):
            reference = final["results"]["4"]["documents"][index]
            assert document["index"] == reference["index"] == index
            assert document["sha256"] == reference["sha256"]
            assert len(document["input_ids_sha256"]) == 64
            assert document["observer_hidden_equal"]
            assert [row["loop"] for row in document["loops"]] == list(range(1, 9))
            for row in document["loops"]:
                assert row["tokens"] == min(2048, reference["scored_tokens"] + 1)
                for metric in METRICS:
                    value = row[metric]
                    if row["loop"] == 1 and metric in METRICS[-2:]:
                        assert value is None
                    else:
                        assert isinstance(value, (int, float)) and math.isfinite(value)
                assert row["state_rms"] > 0
                assert -1e-12 <= row["token_centered_energy_fraction"] <= 1 + 1e-12
                assert (
                    -1 / (row["tokens"] - 1) - 1e-12
                    <= row["mean_distinct_token_cosine"]
                    <= 1 + 1e-12
                )
                if row["loop"] > 1:
                    assert -1 - 1e-12 <= row["previous_loop_cosine"] <= 1 + 1e-12
                    assert row["relative_previous_loop_change"] >= 0
        arms[arm] = []
        for loop in range(8):
            row = {"loop": loop + 1}
            for metric in METRICS:
                values = [document["loops"][loop][metric] for document in documents]
                row[metric] = (
                    None
                    if values[0] is None
                    else {
                        "mean": math.fsum(values) / len(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
            arms[arm].append(row)
    assert input_identities[0] == input_identities[1]
    result = {
        "verified": True,
        "source_files": 11,
        "pair_seconds": queue["pair_seconds"],
        "document_forwards_with_recorded_exact_observer_equality": 32,
        "aggregation": "Unweighted mean/minimum/maximum over the fixed sixteen documents, separately for every loop. Descriptive only; no inferential interval or collapse threshold.",
        "arms": arms,
        "scope": "File/source hashes, terminal process records, final checkpoint/report identities, selected document identities and scalar domains verified locally. Full activation arrays and weights are not exported, so raw-state metric calculations and observed-versus-ordinary hidden equality are recorded Colab measurements, not repeated locally.",
    }
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
    for arm, rows in arms.items():
        print(arm)
        for row in rows:
            print(
                row["loop"],
                {key: None if row[key] is None else row[key]["mean"] for key in METRICS},
            )


if __name__ == "__main__":
    main()
