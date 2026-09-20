"""Prepare a fresh bounded pilot, excluding prior corpus and frozen ARC text.

CPU-only data preparation. This does not launch training or fit a tokenizer.
All old train AND validation documents protect both new splits. Long lexical
overlap is conservative (including boilerplate), not semantic decontamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.decontaminate import normalise  # noqa: E402
from scripts.prepare_pilot import (  # noqa: E402
    DATASET,
    REVISION,
    SUBSET,
    content_hash,
    gram_hashes,
    partition_rows,
    write_pilot,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def url_key(url: str | None) -> str:
    parsed = urlsplit(url or "")
    return parsed.netloc.lower() + parsed.path.rstrip("/") if parsed.netloc else ""


class Exclusions:
    def __init__(self):
        self.content: set[str] = set()
        self.urls: set[str] = set()
        self.grams: set[bytes] = set()
        self.short_phrases: dict[str, set[str]] = defaultdict(set)
        self.stats: Counter = Counter()

    def add(self, text: str, *, url: str | None = None, benchmark: bool = False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("protected text must be a nonempty string")
        cleaned = normalise(text)
        words = cleaned.split()
        self.content.add(content_hash(cleaned))
        key = url_key(url)
        if key:
            self.urls.add(key)
        if len(words) >= 13:
            self.grams.update(gram_hashes(cleaned))
        elif len(words) >= 5:
            self.short_phrases[words[0]].add(f" {cleaned} ")
        elif benchmark:
            self.stats["benchmark_fragments_under_five_words"] += 1
        self.stats["benchmark_fragments" if benchmark else "prior_documents"] += 1

    def reason(self, row: dict) -> str | None:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("source text must be a string")
        cleaned = normalise(text)
        if content_hash(cleaned) in self.content:
            return "prior_exact_content"
        if url_key(row.get("url")) in self.urls:
            return "prior_url"
        if self.grams.intersection(gram_hashes(cleaned)):
            return "protected_13_word_span"
        padded = f" {cleaned} "
        for word in set(cleaned.split()).intersection(self.short_phrases):
            if any(phrase in padded for phrase in self.short_phrases[word]):
                return "protected_short_phrase"
        return None


def load_exclusions(prior: Path, arc: Path, expected_arc_sha256: str):
    manifest_path = prior / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != 1
        or not manifest.get("complete")
        or (manifest.get("source"), manifest.get("source_revision"), manifest.get("subset"))
        != (DATASET, REVISION, SUBSET)
    ):
        raise ValueError("prior source identity or completion differs")
    if sha256(arc) != expected_arc_sha256:
        raise ValueError("ARC file SHA256 differs")
    artifacts = manifest["artifacts"]
    if {Path(name).parts[0] for name in artifacts} != {"train", "validation"}:
        raise ValueError("prior artifacts must include train and validation only")
    index = Exclusions()
    root = prior.resolve()
    for name, metadata in sorted(artifacts.items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".jsonl":
            raise ValueError("invalid prior artifact path")
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError("prior artifact bytes or SHA256 differs")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if content_hash(normalise(row["text"])) != row["content_sha256"]:
                    raise ValueError("prior normalized content hash differs")
                index.add(row["text"], url=row.get("url"))
                count += 1
        if count != metadata["documents"]:
            raise ValueError("prior document count differs")
    questions = 0
    with arc.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompt = row["prompt"]
            if not prompt.startswith("Question: ") or not prompt.endswith("\nAnswer:"):
                raise ValueError("ARC prompt wrapper differs")
            if not isinstance(row["choices"], list) or len(row["choices"]) < 2:
                raise ValueError("ARC choices missing")
            # Exclude the question and EVERY choice, without joining unrelated spans.
            for text in [prompt[len("Question: ") : -len("\nAnswer:")], *row["choices"]]:
                index.add(text, benchmark=True)
            questions += 1
    if not questions:
        raise ValueError("empty ARC file")
    return (
        index,
        manifest,
        {
            "prior_manifest_sha256": sha256(manifest_path),
            "prior_artifacts": artifacts,
            "arc_items_sha256": expected_arc_sha256,
            "arc_questions": questions,
            "protected": dict(index.stats),
            "unique_13_word_hashes": len(index.grams),
        },
    )


def fresh_rows(rows, index: Exclusions, *, skip_rows: int, max_scan_rows: int, stats: Counter):
    if skip_rows < 0 or max_scan_rows < 1:
        raise ValueError("nonnegative skip and positive scan cap required")
    # Bound the RAW source, not the number of accepted rows after exclusions.
    for ordinal, row in enumerate(islice(rows, skip_rows, skip_rows + max_scan_rows), skip_rows):
        stats["raw_rows_scanned"] += 1
        stats["last_source_ordinal"] = ordinal
        reason = index.reason(row)
        if reason:
            stats[reason] += 1
        else:
            stats["rows_passed_exclusions"] += 1
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--arc-items", type=Path, required=True)
    parser.add_argument("--arc-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--max-bytes", type=int, default=200000000)
    parser.add_argument("--max-scan-rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_name(args.out.name + ".building").exists():
        parser.error("output or staging already exists")
    if min(args.max_docs, args.max_bytes, args.max_scan_rows) < 1:
        parser.error("caps must be positive")
    index, prior, receipt = load_exclusions(args.prior, args.arc_items, args.arc_sha256)
    skip = prior["stats"]["rows_seen"]
    if not isinstance(skip, int) or skip < 1:
        raise ValueError("invalid prior source cursor")
    print(json.dumps({"exclusions_verified": receipt, "skip_rows": skip}), flush=True)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET, revision=REVISION)
    if info.gated or info.card_data.get("license") != "odc-by" or info.sha != REVISION:
        raise RuntimeError("source licence, access or revision differs")
    rows = load_dataset(DATASET, name=SUBSET, revision=REVISION, split="train", streaming=True)
    exclusions = Counter()
    groups, stats = partition_rows(
        fresh_rows(rows, index, skip_rows=skip, max_scan_rows=args.max_scan_rows, stats=exclusions),
        max_docs=args.max_docs,
        max_bytes=args.max_bytes,
        seed=args.seed,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True
    ).strip()
    result = write_pilot(
        groups,
        args.out,
        {
            "format_version": 2,
            "source": DATASET,
            "source_revision": REVISION,
            "subset": SUBSET,
            "source_split": "train",
            "source_license": "odc-by",
            "attribution_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
            "code_revision": revision,
            "seed": args.seed,
            "validation_per_mille": 20,
            "skip_rows": skip,
            "max_scan_rows": args.max_scan_rows,
            "max_docs": args.max_docs,
            "max_bytes": args.max_bytes,
            "exclusion_inputs": receipt,
            "exclusion_stats": dict(exclusions),
            "stats": stats,
            "decontamination": "both new splits exclude prior train/validation content, URL and any "
            "13-word span; ARC question and all choices exclude any 13-word span or entire 5-12-word "
            "phrase; new train excludes new validation 13-word spans",
            "limitations": "under-five-word ARC fragments excluded only on whole-document equality; "
            "no semantic or other-benchmark screen; bounded English prefix, not final mixture",
            "external_benchmarks_decontaminated": False,
            "tokenizer_policy": "reuse parent tokenizer; no tokenizer trained or copied by this script",
            "validation_policy": "freeze before experiment; do not use for iterative recipe selection",
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
