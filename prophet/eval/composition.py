"""Score composition tasks (``prophet.data.composition``) by continuation likelihood.

Each example is a prompt ending in ``answer:`` and a closed list of candidates that
contains the answer. The model scores every candidate's continuation in FP32 with the
same batched causal scorer as the ARC evaluation; the prediction is the candidate with
the lowest summed loss. Accuracy is reported per family and per level (hops for tables,
digit count for addition) at one inference depth, so a sweep over ``loop_k`` gives the
curve hypothesis H4 asks for.
"""

from __future__ import annotations

import math
from collections import defaultdict

from prophet.eval.choices import (
    batched_continuation_nats,
    continuation_bucket,
    encode_choice,
    rank_choices,
)


def encode_examples(examples: list[dict], tokenizer, *, max_tokens: int) -> list[list[tuple]]:
    """Per example, the encoded ``(ids, start)`` of every candidate, in source order."""
    encoded = []
    for example in examples:
        if example["answer"] not in example["choices"]:
            raise ValueError("the answer must be one of the candidates")
        if len(set(example["choices"])) != len(example["choices"]):
            raise ValueError("candidates must be distinct")
        encoded.append(
            [
                encode_choice(tokenizer, example["prompt"], choice, max_tokens=max_tokens)
                for choice in example["choices"]
            ]
        )
    return encoded


def evaluate_composition(
    model,
    examples: list[dict],
    tokenizer,
    *,
    batch_size: int = 8,
    seq_len: int = 512,
    device: str = "cpu",
    loop_k: int | None = None,
    progress=None,
) -> dict:
    """Accuracy per ``(kind, level)`` plus every item's prediction, at one depth."""
    if not examples:
        raise ValueError("no examples")
    encoded = encode_examples(examples, tokenizer, max_tokens=seq_len + 1)
    flat = [candidate for choices in encoded for candidate in choices]
    groups: dict[int, list[int]] = {}
    for index, (ids, _) in enumerate(flat):
        groups.setdefault(continuation_bucket(len(ids), max_seq_len=seq_len), []).append(index)
    nats: list[float | None] = [None] * len(flat)
    for length, indices in sorted(groups.items()):
        values = batched_continuation_nats(
            model,
            [flat[i] for i in indices],
            pad_id=tokenizer.pad_id,
            batch_size=batch_size,
            seq_len=length,
            device=device,
            loop_k=loop_k,
            progress=progress,
        )
        for index, value in zip(indices, values, strict=True):
            nats[index] = value

    items = []
    correct: dict[tuple[str, int], int] = defaultdict(int)
    correct_normalized: dict[tuple[str, int], int] = defaultdict(int)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    chance: dict[tuple[str, int], float] = defaultdict(float)
    cursor = 0
    for example, choices in zip(examples, encoded, strict=True):
        scores = nats[cursor : cursor + len(choices)]
        cursor += len(choices)
        if any(score is None for score in scores):
            raise RuntimeError("unscored candidate")
        ranking = rank_choices(scores, [len(choice) for choice in example["choices"]])
        gold = example["choices"].index(example["answer"])
        key = (example["kind"], int(example["hops"]))
        counts[key] += 1
        chance[key] += 1 / len(example["choices"])
        correct[key] += ranking["prediction"] == gold
        correct_normalized[key] += ranking["prediction_character_normalized"] == gold
        items.append(
            {
                "kind": example["kind"],
                "level": int(example["hops"]),
                "index": example.get("index"),
                "gold": gold,
                "prediction": ranking["prediction"],
                "prediction_character_normalized": ranking["prediction_character_normalized"],
                "ties": ranking["ties"],
                "nats": [float(s) for s in scores],
            }
        )
    levels = {}
    for key in sorted(counts):
        kind, level = key
        levels[f"{kind}:{level}"] = {
            "kind": kind,
            "level": level,
            "count": counts[key],
            "accuracy": correct[key] / counts[key],
            "accuracy_character_normalized": correct_normalized[key] / counts[key],
            "chance": chance[key] / counts[key],
        }
    return {
        "loop_k": loop_k,
        "examples": len(examples),
        "candidates": len(flat),
        "levels": levels,
        "items": items,
    }


def paired_accuracy_difference(left: dict, right: dict, *, kind: str, level: int) -> dict:
    """Left-minus-right accuracy on the same items, with discordant counts.

    Both reports must come from the same example list in the same order; this is the
    per-item pairing the H4 contrast (k=6 against k=2) uses before bootstrapping.
    """
    pairs = [
        (a, b)
        for a, b in zip(left["items"], right["items"], strict=True)
        if a["kind"] == kind and a["level"] == level
    ]
    if not pairs:
        raise ValueError("no items at that kind/level")
    for a, b in pairs:
        if a["gold"] != b["gold"] or a["index"] != b["index"]:
            raise ValueError("reports are not paired on the same items")
    left_right = [(a["prediction"] == a["gold"], b["prediction"] == b["gold"]) for a, b in pairs]
    n = len(left_right)
    diff = (sum(x for x, _ in left_right) - sum(y for _, y in left_right)) / n
    return {
        "count": n,
        "difference": diff,
        "left_only": sum(x and not y for x, y in left_right),
        "right_only": sum(y and not x for x, y in left_right),
        "both": sum(x and y for x, y in left_right),
        "finite": math.isfinite(diff),
    }
