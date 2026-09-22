"""Composition tasks for the loop-core programme (docs/29, hypothesis H4).

Two families, generated deterministically from a seed, written as plain text so they
enter the training mixture like any other document and are scored by continuation
likelihood like any other choice task:

* **Table lookup chains.** A small table of ``a -> b`` lines defines a function on
  pseudowords; the question asks for the value after ``hops`` applications from a start
  key. One hop is retrieval; three or four hops require composing retrievals, which is
  the serial computation a looped core is supposed to buy. The candidates are the table's
  own keys, so chance is ``1 / table_size`` and no token-frequency prior helps.
* **Addition.** Two integers with a fixed digit count and their sum; candidates are the
  sum and seven near misses (carry and digit errors).

Train and test tables draw from **disjoint pseudoword vocabularies**, so a test chain
cannot be answered by remembering a training pair. Every example is a short document
ending in ``answer: <target>``; the evaluation prompt is the document cut after
``answer:`` and the continuation is one space plus the candidate, exactly the layout
``prophet.eval.choices.encode_choice`` expects.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"
HOPS = (1, 2, 3, 4)
DIGITS = (2, 3, 4, 5, 6)
TABLE_SIZE = 12
ADDITION_CHOICES = 8


def pseudowords(count: int, *, seed: int, syllables: int = 3) -> list[str]:
    """``count`` distinct consonant-vowel pseudowords in a seeded order."""
    rng = random.Random(seed)
    words: list[str] = []
    seen: set[str] = set()
    while len(words) < count:
        word = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables))
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words


def split_vocabulary(count: int, *, seed: int) -> dict[str, list[str]]:
    """Disjoint halves: train tables never share a pseudoword with test tables."""
    words = pseudowords(count, seed=seed)
    half = count // 2
    return {"train": words[:half], "test": words[half:]}


def table_example(
    rng: random.Random, names: list[str], *, hops: int, size: int = TABLE_SIZE
) -> dict:
    if hops < 1 or size < 3 or len(names) < size:
        raise ValueError("hops >= 1, size >= 3 and enough names are required")
    keys = rng.sample(names, size)
    # A random function without fixed points: every key maps to a different key.
    mapping = {}
    for key in keys:
        target = rng.choice(keys)
        while target == key:
            target = rng.choice(keys)
        mapping[key] = target
    start = rng.choice(keys)
    answer = start
    for _ in range(hops):
        answer = mapping[answer]
    lines = [f"{key} -> {mapping[key]}" for key in keys]
    rng.shuffle(lines)
    prompt = "table:\n" + "\n".join(lines) + f"\nstart: {start}\nhops: {hops}\nanswer:"
    return {
        "kind": "table",
        "hops": hops,
        "prompt": prompt,
        "answer": answer,
        "choices": list(keys),
        "text": f"{prompt} {answer}\n",
    }


def _near_misses(rng: random.Random, total: int, count: int) -> list[str]:
    candidates: set[int] = set()
    digits = list(str(total))
    while len(candidates) < count:
        kind = rng.randrange(4)
        if kind == 0:
            value = total + rng.choice((-1, 1)) * 10 ** rng.randrange(len(digits))
        elif kind == 1 and len(digits) > 1:
            i, j = rng.sample(range(len(digits)), 2)
            swapped = digits[:]
            swapped[i], swapped[j] = swapped[j], swapped[i]
            value = int("".join(swapped))
        elif kind == 2:
            value = total + rng.randrange(-20, 21)
        else:
            value = total + rng.choice((-1, 1)) * rng.randrange(1, 10) * 10 ** rng.randrange(
                len(digits)
            )
        if value != total and value >= 0:
            candidates.add(value)
    return [str(v) for v in candidates]


def addition_example(rng: random.Random, *, digits: int) -> dict:
    if digits < 1:
        raise ValueError("digits >= 1")
    low, high = 10 ** (digits - 1), 10**digits - 1
    a, b = rng.randint(low, high), rng.randint(low, high)
    total = a + b
    choices = [str(total)] + _near_misses(rng, total, ADDITION_CHOICES - 1)
    rng.shuffle(choices)
    prompt = f"add: {a} + {b}\nanswer:"
    return {
        "kind": "addition",
        "hops": digits,
        "prompt": prompt,
        "answer": str(total),
        "choices": choices,
        "text": f"{prompt} {total}\n",
    }


def generate(
    split: str,
    *,
    per_level: int,
    seed: int,
    vocabulary_size: int = 2000,
    table_size: int = TABLE_SIZE,
    hops: tuple[int, ...] = HOPS,
    digits: tuple[int, ...] = DIGITS,
) -> Iterator[dict]:
    """``per_level`` table examples per hop count and addition examples per digit count.

    The split decides the vocabulary half and the RNG stream, so train and test never
    coincide even for the same seed and counts. Examples interleave families and levels
    so a truncated prefix is still balanced.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    names = split_vocabulary(vocabulary_size, seed=seed)[split]
    rng = random.Random(f"{seed}:{split}")
    for index in range(per_level):
        for h in hops:
            example = table_example(rng, names, hops=h, size=table_size)
            example["index"] = index
            yield example
        for d in digits:
            example = addition_example(rng, digits=d)
            example["index"] = index
            yield example


def write_jsonl(
    examples: Iterator[dict], path: Path, *, fields: tuple[str, ...] | None = None
) -> dict:
    """Write one JSON object per line; ``fields=("text",)`` keeps only the training text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, characters = 0, 0
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            row = example if fields is None else {k: example[k] for k in fields}
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
            characters += len(example["text"])
    return {"documents": count, "characters": characters}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
