"""Recompute native ARC scores and paired question uncertainty without model inference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("original4096", "fixed_sum", "learned_mix")
KEYS = tuple(f"{arm}-k{k}" for arm in ARMS for k in (4, 6))
CONTRASTS = [(f"{arm}-k6", f"{arm}-k4") for arm in ARMS] + [
    (f"{left}-k{k}", f"{right}-k{k}")
    for left, right in (
        ("learned_mix", "fixed_sum"),
        ("fixed_sum", "original4096"),
        ("learned_mix", "original4096"),
    )
    for k in (4, 6)
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bytes(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())


def recompute(evaluation, items):
    rows = evaluation["items"]
    require(len(rows) == evaluation["rows"] == len(items) > 0, "question count differs")
    require(len({item["id"] for item in items}) == len(items), "duplicate question IDs")
    vectors = {
        key: []
        for key in (
            "raw",
            "normalized",
            "nats",
            "bytes",
            "tokens",
            "prediction",
            "normalized_prediction",
        )
    }
    ties = normalized_ties = 0
    chance = 0.0
    for row, item in zip(rows, items, strict=True):
        require(
            all(row[key] == item[key] for key in ("id", "source_row_sha256", "gold"))
            and row["candidate_tokens"] == item["tokens"]
            and row["choice_characters"] == [len(choice) for choice in item["choices"]],
            "question identity or token metadata differs",
        )
        values = row["choice_nats"]
        require(
            len(values) == len(item["choices"]) >= 2
            and all(math.isfinite(value) and value >= 0 for value in values)
            and all(length > 0 for length in row["choice_characters"])
            and 0 <= item["gold"] < len(values),
            "invalid candidate scores",
        )
        norm = [
            value / length for value, length in zip(values, row["choice_characters"], strict=True)
        ]
        a, b = values.index(min(values)), norm.index(min(norm))
        t, u = values.count(values[a]), norm.count(norm[b])
        require(
            (
                row["prediction"],
                row["prediction_character_normalized"],
                row["ties"],
                row["ties_character_normalized"],
            )
            == (a, b, t, u),
            "choice ranking or ties differ",
        )
        gold = item["gold"]
        token = item["tokens"][gold]
        require(
            token["answer_tokens"] > 0 and token["answer_bytes"] > 0, "invalid answer denominator"
        )
        for key, value in {
            "raw": int(a == gold),
            "normalized": int(b == gold),
            "nats": values[gold],
            "bytes": token["answer_bytes"],
            "tokens": token["answer_tokens"],
            "prediction": a,
            "normalized_prediction": b,
        }.items():
            vectors[key].append(value)
        ties += t > 1
        normalized_ties += u > 1
        chance += 1 / len(values)
    nats, tokens, size = (sum(vectors[key]) for key in ("nats", "tokens", "bytes"))
    aggregate = {
        "accuracy": sum(vectors["raw"]) / len(items),
        "accuracy_character_normalized": sum(vectors["normalized"]) / len(items),
        "uniform_choice_chance": chance / len(items),
        "tied_items": ties,
        "tied_items_character_normalized": normalized_ties,
        "gold_answer_total_nats": nats,
        "gold_answer_scored_tokens": tokens,
        "gold_answer_scored_bytes": size,
        "gold_answer_nats_per_token": nats / tokens,
        "gold_answer_bits_per_byte": nats / size / math.log(2),
    }
    for key, value in aggregate.items():
        require(
            math.isclose(evaluation[key], value, rel_tol=1e-12, abs_tol=1e-12),
            f"aggregate differs: {key}",
        )
    aggregate.update(
        correct=sum(vectors["raw"]), correct_character_normalized=sum(vectors["normalized"])
    )
    return aggregate, {key: np.asarray(value) for key, value in vectors.items()}


def paired(left, right, draws):
    require(
        np.array_equal(left["bytes"], right["bytes"])
        and np.array_equal(left["tokens"], right["tokens"]),
        "paired answer denominators differ",
    )
    result = {}
    for metric, scale, denominator in (
        ("raw", 100.0, None),
        ("normalized", 100.0, None),
        ("nats_per_token", 1.0, "tokens"),
        ("bits_per_byte", 1 / math.log(2), "bytes"),
    ):
        delta = (
            left[metric if denominator is None else "nats"]
            - right[metric if denominator is None else "nats"]
        )
        if denominator is None:
            estimate = float(delta.mean()) * scale
            samples = delta[draws].mean(axis=1) * scale
            result[metric] = {
                "left_only_correct": int((delta == 1).sum()),
                "right_only_correct": int((delta == -1).sum()),
                "same_correctness": int((delta == 0).sum()),
                "different_predictions": int(
                    (
                        left["prediction" if metric == "raw" else "normalized_prediction"]
                        != right["prediction" if metric == "raw" else "normalized_prediction"]
                    ).sum()
                ),
                "units": "percentage points",
            }
        else:
            estimate = float(delta.sum() / left[denominator].sum()) * scale
            samples = delta[draws].sum(axis=1) / left[denominator][draws].sum(axis=1) * scale
            result[metric] = {"units": metric}
        result[metric].update(
            difference=estimate, ci95=np.quantile(samples, [0.025, 0.975], method="linear").tolist()
        )
    return result


def summarize(reports, items, manifest, manifest_sha256, *, replicates=10000):
    require(set(reports) == set(KEYS), "all six preselected reports required")
    require(replicates >= 100, "at least 100 question resamples required")
    reference = reports[KEYS[0]]
    scores, vectors = {}, {}
    for key in KEYS:
        report = reports[key]
        arm, depth = key.rsplit("-k", 1)
        require(
            report["complete"]
            and report["protocol"] == "arc-easy-native-r04-evaluation-v1"
            and report["arm"] == arm
            and report["loop_k"] == int(depth),
            "unplanned or incomplete evaluation",
        )
        require(
            report["manifest"] == manifest
            and report["manifest_sha256"] == manifest_sha256
            and len(items) == manifest["rows"],
            "evaluation manifest differs",
        )
        for field in ("source", "runtime", "numerical_policy"):
            require(
                report[field] == reference[field], f"paired evaluation contract differs: {field}"
            )
        require(
            report["checkpoint"] == reports[f"{arm}-k4"]["checkpoint"],
            "depths use different checkpoints",
        )
        require(report["evaluation"]["buckets"] == manifest["buckets"], "batch shapes differ")
        oracle = report["evaluation"]["batch_oracle"]
        require(
            oracle["candidates"]
            == sum(
                min(manifest["batch_size"], bucket["candidates"]) for bucket in manifest["buckets"]
            )
            and oracle["atol"] == 1e-4
            and oracle["rtol"] == 1e-5
            and math.isfinite(oracle["max_absolute_nats_error"])
            and oracle["max_absolute_nats_error"] >= 0,
            "batch oracle contract differs",
        )
        scores[key], vectors[key] = recompute(report["evaluation"], items)
    draws = np.random.Generator(np.random.PCG64(0)).integers(
        0, len(items), (replicates, len(items))
    )
    return {
        "complete": True,
        "rows": len(items),
        "scores": scores,
        "paired": {
            f"{left} minus {right}": paired(vectors[left], vectors[right], draws)
            for left, right in CONTRASTS
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": 0,
            "generator": "PCG64",
            "quantiles": "linear",
            "unit": "whole question, jointly across models and metrics",
            "scope": "Unadjusted descriptive 95% intervals for nine contrasts per metric, conditional on these trained models; no training-seed or contamination uncertainty.",
        },
        "direction": "left minus right; positive accuracy and negative CE/BPB favor left",
        "scope": "Recomputed score aggregates and paired questions only; no repeated GPU forward, primary language-loss verdict change, architectural adoption or assistant-capability claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "items", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "preserve prior analysis")
    manifest_raw = (
        ROOT / "docs/experiments/2026-09-20-arc-native-protocol/manifest.json"
    ).read_bytes()
    manifest = json.loads(manifest_raw)
    items_raw = (args.items / "items.jsonl").read_bytes()
    require(
        digest(items_raw) == manifest["items_sha256"]
        and json.loads((args.items / "manifest.json").read_bytes()) == manifest,
        "frozen native inputs differ",
    )
    items = [json.loads(line) for line in items_raw.decode("utf-8").splitlines()]
    raw = {key: read_bytes(args.reports / (key + ".json")) for key in KEYS}
    reports = {key: json.loads(value) for key, value in raw.items()}
    source = reports[KEYS[0]]["source"]
    for key, path in (
        ("driver_sha256", "scripts/adapt_r04_depth.py"),
        ("evaluator_sha256", "scripts/eval_arc_native.py"),
        ("scorer_sha256", "prophet/eval/choices.py"),
    ):
        blob = subprocess.check_output(["git", "show", source["revision"] + ":" + path], cwd=ROOT)
        require(digest(blob) == source[key], "recorded scoring source differs")
    result = summarize(reports, items, manifest, digest(manifest_raw))
    result.update(
        report_sha256={key: digest(value) for key, value in raw.items()},
        input_items_sha256=digest(items_raw),
        analysis_sha256=digest(Path(__file__).read_bytes()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__":
    main()
