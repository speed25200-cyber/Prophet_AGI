#!/usr/bin/env python3
"""Fetch the evaluation sets the training stream must be decontaminated against.

    python scripts/fetch_benchmarks.py --out benchmarks/ --dry-run   # the list, no network
    python scripts/fetch_benchmarks.py --out benchmarks/             # every set the Hub serves

Writes ``<out>/<name>.jsonl`` with one ``{"text": ...}`` per test item -- the question
and its answer options joined, since a training document that contains either is the
leak the decontaminator (13-gram containment, ``prophet.data.decontaminate``) exists to
catch. ``scripts/train.py --benchmarks <out>`` indexes every file in the directory.

The dataset ids below are **claims to verify**: they were written without Hub access
(the same provenance caveat as the mixture) and ``--dry-run`` marks every one of them
so. A wrong id fails loudly at fetch time rather than silently producing an empty set;
an empty set is refused (a benchmark with zero items would decontaminate nothing while
looking done). Gated sets (GPQA) need a Hub token and are skipped with a note when
absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    fields: tuple[str, ...]
    """Row fields whose text is joined; a field may be a list (every element is taken)
    or a dict with a ``text`` list (ARC-style choices)."""
    gated: bool = False


#: Test splits of the tier-1 and tier-2 suites (``prophet.eval.harness``) plus the
#: dashboard's targets. VERIFY every id against the Hub before trusting a fetch.
BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("lambada_openai", "EleutherAI/lambada_openai", "default", "test", ("text",)),
    BenchmarkSpec("sciq", "allenai/sciq", None, "test", ("question", "support", "correct_answer", "distractor1", "distractor2", "distractor3")),
    BenchmarkSpec("piqa", "ybisk/piqa", None, "validation", ("goal", "sol1", "sol2")),
    BenchmarkSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "test", ("question", "choices")),
    BenchmarkSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", ("question", "choices")),
    BenchmarkSpec("hellaswag", "Rowan/hellaswag", None, "validation", ("ctx", "endings")),
    BenchmarkSpec("social_iqa", "allenai/social_i_qa", None, "validation", ("context", "question", "answerA", "answerB", "answerC")),
    BenchmarkSpec("openbookqa", "allenai/openbookqa", "main", "test", ("question_stem", "choices")),
    BenchmarkSpec("commonsense_qa", "tau/commonsense_qa", None, "validation", ("question", "choices")),
    BenchmarkSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation", ("sentence", "option1", "option2")),
    BenchmarkSpec("mmlu", "cais/mmlu", "all", "test", ("question", "choices")),
    BenchmarkSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", ("question", "options")),
    BenchmarkSpec("gsm8k", "openai/gsm8k", "main", "test", ("question", "answer")),
    BenchmarkSpec("humaneval", "openai/openai_humaneval", None, "test", ("prompt", "canonical_solution", "test")),
    BenchmarkSpec("mbpp", "google-research-datasets/mbpp", "full", "test", ("text", "code")),
    BenchmarkSpec("ifeval", "google/IFEval", None, "train", ("prompt",)),
    BenchmarkSpec("gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond", "train",
                  ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"), gated=True),
)


def row_text(row: dict[str, Any], fields: Iterable[str]) -> str:
    parts: list[str] = []
    for name in fields:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, dict) and "text" in value:
            value = value["text"]
        if isinstance(value, (list, tuple)):
            parts += [str(v) for v in value]
        else:
            parts.append(str(value))
    return "\n".join(p for p in parts if p)


def write_benchmark(name: str, rows: Iterable[dict[str, Any]], fields: Iterable[str], out: Path) -> int:
    """Write ``<out>/<name>.jsonl``; returns the item count and refuses to write an
    empty set."""
    fields = tuple(fields)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            text = row_text(row, fields)
            if not text.strip():
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: no items with text in fields {fields}; nothing written")
    tmp.replace(path)
    return n


def hub_rows(spec: BenchmarkSpec) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fetching benchmarks needs the `datasets` package") from exc
    yield from load_dataset(spec.hf_id, spec.config, split=spec.split, streaming=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated benchmark names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    specs = list(BENCHMARKS)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.name in wanted]

    print("| benchmark | hf_id | config | split | fields | status |\n|---|---|---|---|---|---|")
    for spec in specs:
        status = "gated: token needed" if spec.gated else "VERIFY id"
        print(f"| {spec.name} | `{spec.hf_id}` | {spec.config or ''} | {spec.split} | {', '.join(spec.fields)} | {status} |")
    if args.dry_run:
        print(f"\n{len(specs)} sets; nothing fetched.")
        return 0

    out = Path(args.out)
    failures = 0
    for spec in specs:
        try:
            n = write_benchmark(spec.name, hub_rows(spec), spec.fields, out)
            print(f"{spec.name}: {n} items")
        except Exception as exc:  # noqa: BLE001 - one bad id must not hide the others
            failures += 1
            print(f"{spec.name}: FAILED ({exc})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
