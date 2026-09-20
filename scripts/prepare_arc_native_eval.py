"""Retokenize the already frozen ARC-Easy questions for native R04 evaluation."""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.choices import continuation_bucket, encode_choice, token_ids_sha256  # noqa: E402
from scripts.eval_arc_recovery import encoded_items  # noqa: E402
from scripts.run_r04_pilot import (  # noqa: E402
    PILOT_TOKENIZER_SHA256,
    sha256,
    tokenizer_semantic_hash,
)

ORIGINAL = ROOT / "docs/experiments/2026-09-20-arc-recovery-final/items/manifest.json"


def retokenize(items, tokenizer):
    prepared = copy.deepcopy(items)
    for item in prepared:
        item["tokens"] = []
        for choice in item["choices"]:
            ids, start = encode_choice(tokenizer, item["prompt"], choice, max_tokens=2048)
            item["tokens"].append(
                {
                    "input_ids_sha256": token_ids_sha256(ids),
                    "tokens": len(ids),
                    "context_tokens": start,
                    "answer_tokens": len(ids) - start,
                    "answer_bytes": tokenizer.byte_length(ids[start:]),
                }
            )
    encoded_items(prepared, tokenizer, max_tokens=2048)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the frozen native evaluation")
    original = json.loads(ORIGINAL.read_bytes())
    if (
        json.loads((args.items / "manifest.json").read_bytes()) != original
        or sha256(args.items / "items.jsonl") != original["items_sha256"]
    ):
        raise ValueError("original frozen questions differ")
    if tokenizer_semantic_hash(args.tokenizer) != PILOT_TOKENIZER_SHA256:
        raise ValueError("native pilot tokenizer differs")
    tokenizer = ProphetTokenizer.load(args.tokenizer)
    items = [
        json.loads(line)
        for line in (args.items / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(items) != original["rows"]:
        raise ValueError("source row count differs")
    prepared = retokenize(items, tokenizer)
    maximum = max(t["tokens"] for item in prepared for t in item["tokens"])
    maximum_shape = 1 << (maximum - 2).bit_length()
    counts = Counter(
        continuation_bucket(t["tokens"], max_seq_len=maximum_shape)
        for item in prepared
        for t in item["tokens"]
    )
    buckets = [
        {"seq_len": length, "candidates": count, "batches": (count + 7) // 8}
        for length, count in sorted(counts.items())
    ]
    args.out.mkdir(parents=True)
    output = args.out / "items.jsonl"
    output.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in prepared
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "complete": True,
        "protocol": "arc-easy-native-r04-v1",
        "source": original["source"],
        "original_items_sha256": original["items_sha256"],
        "rows": len(prepared),
        "items_sha256": sha256(output),
        "tokenizer_semantic_sha256": PILOT_TOKENIZER_SHA256,
        "max_tokens": 2048,
        "max_observed_tokens": maximum,
        "maximum_input_length": maximum_shape,
        "buckets": buckets,
        "padded_input_positions": sum(b["batches"] * 8 * b["seq_len"] for b in buckets),
        "batch_size": 8,
        "total_candidate_tokens": sum(t["tokens"] for item in prepared for t in item["tokens"]),
        "candidates": sum(len(item["choices"]) for item in prepared),
        "prompt_template": original["prompt_template"],
        "continuation_prefix": " ",
        "fewshot": 0,
        "precision": "fp32; no autocast or TF32 matmul",
        "scoring": original["scoring"],
        "tie_policy": original["tie_policy"],
        "scope": "All original questions and choices in source order, retokenized only. Exploratory capability probe; no training, filtering, truncation or reserved benchmark. Original R04 overlap screen is incomplete and does not establish benchmark-clean training.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
