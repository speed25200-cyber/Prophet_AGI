"""Full exploratory ARC-Easy scoring of frozen native R04 weights at k4 or k6."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.choices import (  # noqa: E402
    batched_continuation_nats,
    continuation_bucket,
    continuation_nats,
    rank_choices,
)
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts import adapt_r04_depth as engine  # noqa: E402
from scripts.adapt_r04_reinjection import PLAN_DIR  # noqa: E402
from scripts.audit_r04_restart import load_run  # noqa: E402
from scripts.eval_arc_recovery import encoded_items  # noqa: E402
from scripts.run_r04_pilot import (  # noqa: E402
    PILOT_TOKENIZER_SHA256,
    sha256,
    tokenizer_semantic_hash,
)

MANIFEST = ROOT / "docs/experiments/2026-09-20-arc-native-protocol/manifest.json"
TRAINING_REVISION = "dce35abc439fdeeb8f1482b1219dee3cc242c267"


def evaluate(model, items, tokenizer, *, batch_size, seq_len, device, loop_k, progress=None):
    model.eval()
    encoded = encoded_items(items, tokenizer, max_tokens=seq_len + 1)
    candidates = [candidate for _, choices in encoded for candidate in choices]
    groups = {}
    for index, (ids, _) in enumerate(candidates):
        groups.setdefault(continuation_bucket(len(ids), max_seq_len=seq_len), []).append(index)
    nats = [None] * len(candidates)
    done = 0
    oracle, bucket_reports = [], []
    for length, indices in sorted(groups.items()):
        values = batched_continuation_nats(
            model,
            [candidates[i] for i in indices],
            pad_id=tokenizer.pad_id,
            batch_size=batch_size,
            seq_len=length,
            device=device,
            loop_k=loop_k,
            progress=(lambda completed, _, base=done: progress(base + completed, len(candidates)))
            if progress
            else None,
        )
        for index, value in zip(indices, values, strict=True):
            nats[index] = value
        done += len(indices)
        # First batch of each occupied shape, selected solely from source order.
        for index in indices[:batch_size]:
            ids, start = candidates[index]
            oracle.append(
                (index, continuation_nats(model, ids, start, device=device, loop_k=loop_k))
            )
        bucket_reports.append(
            {
                "seq_len": length,
                "candidates": len(indices),
                "batches": (len(indices) + batch_size - 1) // batch_size,
            }
        )
    errors = [abs(nats[index] - value) for index, value in oracle]
    engine.require(
        all(
            math.isclose(nats[index], value, rel_tol=1e-5, abs_tol=1e-4) for index, value in oracle
        ),
        "batch versus single score differs",
    )
    rows, offset = [], 0
    for item, choices in encoded:
        values = nats[offset : offset + len(choices)]
        offset += len(choices)
        lengths = [len(choice) for choice in item["choices"]]
        rows.append(
            {
                "id": item["id"],
                "source_row_sha256": item["source_row_sha256"],
                "gold": item["gold"],
                "choice_nats": values,
                "choice_characters": lengths,
                "candidate_tokens": item["tokens"],
                **rank_choices(values, lengths),
            }
        )
    gold_nats = sum(row["choice_nats"][row["gold"]] for row in rows)
    gold_bytes = sum(row["candidate_tokens"][row["gold"]]["answer_bytes"] for row in rows)
    gold_tokens = sum(row["candidate_tokens"][row["gold"]]["answer_tokens"] for row in rows)
    return {
        "items": rows,
        "rows": len(rows),
        "accuracy": sum(r["prediction"] == r["gold"] for r in rows) / len(rows),
        "accuracy_character_normalized": sum(
            r["prediction_character_normalized"] == r["gold"] for r in rows
        )
        / len(rows),
        "uniform_choice_chance": sum(1 / len(r["choice_nats"]) for r in rows) / len(rows),
        "tied_items": sum(r["ties"] > 1 for r in rows),
        "tied_items_character_normalized": sum(r["ties_character_normalized"] > 1 for r in rows),
        "gold_answer_total_nats": gold_nats,
        "gold_answer_scored_tokens": gold_tokens,
        "gold_answer_scored_bytes": gold_bytes,
        "gold_answer_nats_per_token": gold_nats / gold_tokens,
        "gold_answer_bits_per_byte": gold_nats / gold_bytes / math.log(2),
        "buckets": bucket_reports,
        "batch_oracle": {
            "candidates": len(oracle),
            "max_absolute_nats_error": max(errors),
            "atol": 1e-4,
            "rtol": 1e-5,
        },
    }


def load_weights(run, step):
    plan_bytes = (PLAN_DIR / "protocol.json").read_bytes()
    plan = json.loads(plan_bytes)
    if step == plan["parent_checkpoint"]["step"]:
        state, identity, report = engine.load_parent(run, plan)
        engine.require(
            sha256(run / f"evaluation-step-{step:06d}.json") == plan["parent_report_sha256"],
            "original parent report changed",
        )
        expected_config = identity["config"]
        arm = "original4096"
    else:
        engine.require(step == plan["steps_each"], "only preselected final checkpoints supported")
        state, report = load_run(run, step)
        identity = report["identity"]
        arm = identity["arm"]
        engine.require(
            report["complete"]
            and arm in ("fixed_sum", "learned_mix")
            and identity["protocol"] == plan["experiment"]
            and identity["code"]["revision"] == TRAINING_REVISION
            and identity["plan_sha256"] == sha256(PLAN_DIR / "protocol.json")
            and identity["parent_checkpoint"] == plan["parent_checkpoint"],
            "unplanned native checkpoint",
        )
        expected_config = json.loads((PLAN_DIR / (arm + ".json")).read_bytes())
    cfg = ProphetConfig.from_dict(state["config"])
    engine.require(
        cfg.to_dict() == ProphetConfig.from_dict(expected_config).to_dict(),
        "checkpoint configuration differs",
    )
    engine.require(
        identity["data"]["tokenizer_semantic_sha256"] == PILOT_TOKENIZER_SHA256,
        "checkpoint tokenizer differs",
    )
    engine.require(
        all(torch.isfinite(t).all() for t in state["model"].values()), "nonfinite checkpoint model"
    )
    model = ProphetModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    return (
        model,
        arm,
        {
            "checkpoint": report["checkpoint"],
            "training_identity": identity,
            "config": cfg.to_dict(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "items", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--loop-k", type=int, choices=(4, 6), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    engine.require(not args.out.exists(), "preserve previous capability scores")
    manifest = json.loads(MANIFEST.read_bytes())
    engine.require(
        json.loads((args.items / "manifest.json").read_bytes()) == manifest
        and sha256(args.items / "items.jsonl") == manifest["items_sha256"],
        "native questions changed",
    )
    engine.require(
        tokenizer_semantic_hash(args.tokenizer) == manifest["tokenizer_semantic_sha256"],
        "native tokenizer changed",
    )
    tokenizer = ProphetTokenizer.load(args.tokenizer)
    items = [
        json.loads(line)
        for line in (args.items / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    engine.require(len(items) == manifest["rows"], "question count changed")
    torch.set_num_threads(2)
    numerics = engine.configure_numerics(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    numerics.update(
        name="r04-native-choice-fp32-v1", matmul_allow_tf32=False, cudnn_allow_tf32=False
    )
    runtime = engine.runtime_for(args.device)
    source = {
        **engine.code_identity(),
        "evaluator_sha256": sha256(Path(__file__)),
        "scorer_sha256": sha256(ROOT / "prophet/eval/choices.py"),
    }
    model, arm, checkpoint = load_weights(args.run, args.step)
    model.to(args.device).eval()
    torch.manual_seed(0)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    next_progress = 800

    def progress(done, total):
        nonlocal next_progress
        if done >= next_progress or done == total:
            print(
                "NATIVE_ARC_PROGRESS",
                done,
                total,
                round(time.perf_counter() - began, 2),
                flush=True,
            )
            next_progress = (done // 800 + 1) * 800

    result = evaluate(
        model,
        items,
        tokenizer,
        batch_size=manifest["batch_size"],
        seq_len=manifest["maximum_input_length"],
        device=args.device,
        loop_k=args.loop_k,
        progress=progress,
    )
    engine.require(result["buckets"] == manifest["buckets"], "input-only bucket plan differs")
    report = {
        "complete": True,
        "protocol": "arc-easy-native-r04-evaluation-v1",
        "arm": arm,
        "loop_k": args.loop_k,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "manifest_sha256": sha256(MANIFEST),
        "source": source,
        "runtime": runtime,
        "numerical_policy": numerics,
        "evaluation": result,
        "seconds": time.perf_counter() - began,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "scope": "Full frozen exploratory ARC-Easy, answer-only FP32 likelihood with fixed causal padding and a single-candidate oracle. No training, chat, instruction-following, held-out architectural adoption or cross-seed claim. Known lexical-overlap screen is incomplete.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        "NATIVE_ARC_COMPLETE",
        arm,
        args.loop_k,
        result["accuracy"],
        result["accuracy_character_normalized"],
        flush=True,
    )


if __name__ == "__main__":
    main()
