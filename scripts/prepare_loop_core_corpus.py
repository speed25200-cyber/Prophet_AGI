#!/usr/bin/env python3
"""Prepare the loop-core corpus: one pass of real text per run, plus composition tasks.

Docs/29 §4. The source is the pinned FineWeb-Edu sample used by every previous pilot,
read as a stream; the filters are the pilot's. What is new is the size (a single pass
for a 300M-token run) and the exclusions, applied before anything is written:

* prior pilots' train and validation documents (exact content and URL);
* any normalized 13-word span shared with this corpus' own validation split;
* any normalized 13-word span shared with the cached benchmark rows
  (``scripts/audit_pilot_overlap.py`` cache; long spans only, short rows untested);
* the composition test set, whose pseudowords never occur in web text anyway.

Two passes keep memory bounded at any size: pass one streams the source, filters,
deduplicates, assigns the URL-keyed split and stages train candidates on disk; pass two
re-reads the staged candidates against the validation split's spans and writes the
final shards. The composition tasks are generated last, train text and test items in
separate files, from a seed recorded in the manifest.

    python scripts/prepare_loop_core_corpus.py --out data/loop-core-v1 \
        --target-train-bytes 1700000000 --validation-docs 2000 \
        --prior data/fineweb-pilot-v1 --benchmark-cache data/benchmark-audit-v1 \
        --tokenizer data/fineweb-pilot-v1/tokenizer.json

Text stays outside Git; the manifest (revision, filters, exclusions, counts, hashes,
token estimate) is what gets committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data import composition  # noqa: E402
from prophet.data.decontaminate import normalise  # noqa: E402
from scripts.prepare_fresh_pilot import Exclusions  # noqa: E402
from scripts.prepare_pilot import (  # noqa: E402
    DATASET,
    REVISION,
    SUBSET,
    content_hash,
    gram_hashes,
    split_for,
)

ROOT = Path(__file__).resolve().parent.parent
FILTERS = {"min_int_score": 3, "min_bytes": 400, "max_bytes": 50000, "min_words": 30}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def passes_filters(row: dict) -> str | None:
    if not isinstance(row.get("text"), str) or "int_score" not in row:
        raise ValueError("FineWeb-Edu schema requires text and int_score")
    size = len(row["text"].encode("utf-8"))
    if row["int_score"] < FILTERS["min_int_score"]:
        return "quality"
    if not FILTERS["min_bytes"] <= size <= FILTERS["max_bytes"]:
        return "length"
    return None


def load_prior(index: Exclusions, prior: Path) -> dict:
    """Every document of a prior pilot, train and validation, by content and URL."""
    manifest_path = prior / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"{prior}: incomplete prior corpus")
    documents = 0
    for name in sorted(manifest["artifacts"]):
        path = prior / name
        if path.suffix != ".jsonl":
            continue
        if sha256(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"{path}: bytes differ from the prior manifest")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                index.add(row["text"], url=row.get("url"))
                documents += 1
    return {"path": str(prior), "manifest_sha256": sha256(manifest_path), "documents": documents}


def load_benchmarks(index: Exclusions, cache: Path) -> dict:
    files = {}
    for path in sorted(cache.glob("*.jsonl")):
        rows = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                index.add(json.loads(line)["text"], benchmark=True)
                rows += 1
        files[path.name] = {"sha256": sha256(path), "rows": rows}
    if not files:
        raise ValueError(f"{cache}: no benchmark rows cached")
    return files


class ShardWriter:
    """JSONL shards bounded in bytes, each hashed as it is closed."""

    def __init__(self, directory: Path, *, shard_bytes: int):
        if shard_bytes < 1:
            raise ValueError("shard_bytes must be positive")
        self.directory, self.shard_bytes = directory, shard_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, dict] = {}
        self._stream = None
        self._digest = None
        self._bytes = self._documents = 0

    def _open(self) -> None:
        index = len(self.artifacts)
        self._path = self.directory / f"part-{index:05d}.jsonl"
        self._stream = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._bytes = self._documents = 0

    def write(self, row: dict) -> None:
        if self._stream is None:
            self._open()
        data = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if self._bytes and self._bytes + len(data) > self.shard_bytes:
            self.close()
            self._open()
        self._stream.write(data)
        self._digest.update(data)
        self._bytes += len(data)
        self._documents += 1

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.close()
        self.artifacts[self._path.name] = {
            "sha256": self._digest.hexdigest(),
            "bytes": self._bytes,
            "documents": self._documents,
        }
        self._stream = None


def stage_candidates(
    rows: Iterable[dict],
    *,
    index: Exclusions,
    staging: Path,
    target_train_bytes: int,
    validation_docs: int,
    validation_per_mille: int,
    seed: int,
    max_scan_rows: int,
    shard_bytes: int,
) -> tuple[list[dict], Counter]:
    """Pass one: filter, exclude, deduplicate, split, stage train text on disk."""
    if target_train_bytes < 1 or validation_docs < 1 or max_scan_rows < 1:
        raise ValueError("positive targets required")
    stats: Counter = Counter()
    seen: set[str] = set()
    validation: list[dict] = []
    writer = ShardWriter(staging / "train-candidates", shard_bytes=shard_bytes)
    for row in islice(rows, max_scan_rows):
        stats["rows_scanned"] += 1
        reason = passes_filters(row)
        if reason:
            stats[f"rejected_{reason}"] += 1
            continue
        cleaned = normalise(row["text"])
        if len(cleaned.split()) < FILTERS["min_words"]:
            stats["rejected_short"] += 1
            continue
        digest = content_hash(cleaned)
        if digest in seen:
            stats["rejected_duplicate"] += 1
            continue
        excluded = index.reason(row)
        if excluded:
            stats[f"excluded_{excluded}"] += 1
            continue
        seen.add(digest)
        record = {
            "text": row["text"],
            "content_sha256": digest,
            "source_id": row.get("id"),
            "url": row.get("url"),
            "int_score": row["int_score"],
        }
        split = split_for(row.get("url", ""), digest, seed, validation_per_mille)
        if split == "validation":
            if len(validation) < validation_docs:
                validation.append(record)
                stats["validation_selected"] += 1
            else:
                stats["validation_overflow_dropped"] += 1
        else:
            writer.write(record)
            stats["train_candidates"] += 1
            stats["train_candidate_bytes"] += len(row["text"].encode("utf-8"))
        if (
            stats["train_candidate_bytes"] >= target_train_bytes
            and len(validation) >= validation_docs
        ):
            stats["targets_reached"] = 1
            break
    writer.close()
    stats["staged_shards"] = len(writer.artifacts)
    return validation, stats


def write_final(
    staging: Path,
    out: Path,
    *,
    validation: list[dict],
    stats: Counter,
    shard_bytes: int,
) -> dict:
    """Pass two: drop train candidates sharing a 13-word span with validation; shard."""
    protected: set[bytes] = set()
    for record in validation:
        protected.update(gram_hashes(normalise(record["text"])))
    artifacts: dict[str, dict] = {}
    train = ShardWriter(out / "train" / "fineweb-edu", shard_bytes=shard_bytes)
    for path in sorted((staging / "train-candidates").glob("part-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if protected.intersection(gram_hashes(normalise(record["text"]))):
                    stats["train_rejected_validation_span"] += 1
                    continue
                train.write(record)
                stats["train_docs"] += 1
                stats["train_bytes"] += len(record["text"].encode("utf-8"))
    train.close()
    for name, meta in train.artifacts.items():
        artifacts[f"train/fineweb-edu/{name}"] = meta
    held = ShardWriter(out / "validation" / "fineweb-edu", shard_bytes=shard_bytes)
    for record in validation:
        held.write(record)
    held.close()
    for name, meta in held.artifacts.items():
        artifacts[f"validation/fineweb-edu/{name}"] = meta
    stats["validation_docs"] = len(validation)
    stats["validation_bytes"] = sum(len(r["text"].encode("utf-8")) for r in validation)
    if not train.artifacts or not held.artifacts:
        raise ValueError("empty split; increase the scan or lower the targets")
    return artifacts


def write_composition(
    out: Path,
    *,
    seed: int,
    per_level: int,
    test_per_level: int,
    table_size: int,
    vocabulary_size: int,
) -> dict:
    train_path = out / "train" / "composition" / "part-00000.jsonl"
    train_stats = composition.write_jsonl(
        composition.generate(
            "train",
            per_level=per_level,
            seed=seed,
            table_size=table_size,
            vocabulary_size=vocabulary_size,
        ),
        train_path,
        fields=("text",),
    )
    test_path = out / "composition-test.jsonl"
    test_stats = composition.write_jsonl(
        composition.generate(
            "test",
            per_level=test_per_level,
            seed=seed,
            table_size=table_size,
            vocabulary_size=vocabulary_size,
        ),
        test_path,
    )
    return {
        "seed": seed,
        "per_level": per_level,
        "test_per_level": test_per_level,
        "table_size": table_size,
        "vocabulary_size": vocabulary_size,
        "hops": list(composition.HOPS),
        "digits": list(composition.DIGITS),
        "train": {
            "path": "train/composition/part-00000.jsonl",
            "sha256": sha256(train_path),
            "bytes": train_path.stat().st_size,
            **train_stats,
        },
        "test": {
            "path": "composition-test.jsonl",
            "sha256": sha256(test_path),
            "bytes": test_path.stat().st_size,
            **test_stats,
        },
    }


def estimate_tokens(out: Path, artifacts: dict, tokenizer, *, docs_per_shard: int) -> dict:
    """Bytes per token measured on a prefix of every shard, then scaled to the split."""
    report = {"method": f"first {docs_per_shard} documents of every shard", "splits": {}}
    for split in ("train", "validation"):
        sampled_bytes = sampled_tokens = 0
        total_bytes = 0
        for name, meta in artifacts.items():
            if not name.startswith(f"{split}/"):
                continue
            total_bytes += meta["bytes"]
            with (out / name).open(encoding="utf-8") as stream:
                for line in islice(stream, docs_per_shard):
                    text = json.loads(line)["text"]
                    sampled_bytes += len(text.encode("utf-8"))
                    sampled_tokens += len(tokenizer.encode(text, add_eos=True))
        if sampled_tokens:
            per_token = sampled_bytes / sampled_tokens
            report["splits"][split] = {
                "sampled_bytes": sampled_bytes,
                "sampled_tokens": sampled_tokens,
                "bytes_per_token": per_token,
                "estimated_tokens": int(total_bytes / per_token),
            }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--target-train-bytes", type=int, default=1_700_000_000)
    ap.add_argument("--validation-docs", type=int, default=2000)
    ap.add_argument("--validation-per-mille", type=int, default=25)
    ap.add_argument("--max-scan-rows", type=int, default=3_000_000)
    ap.add_argument("--shard-bytes", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--prior", type=Path, action="append", default=[])
    ap.add_argument("--benchmark-cache", type=Path, default=None)
    ap.add_argument("--composition-seed", type=int, default=20260921)
    ap.add_argument("--composition-per-level", type=int, default=30000)
    ap.add_argument("--composition-test-per-level", type=int, default=2000)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--token-sample-docs", type=int, default=200)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: never replace a corpus")
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    info = dataset_info(DATASET, revision=args.revision)
    rows = load_dataset(DATASET, name=SUBSET, revision=info.sha, split="train", streaming=True)
    manifest = build(rows, args, source_sha=info.sha)
    print(json.dumps({k: v for k, v in manifest.items() if k in ("stats", "tokens")}, indent=2))
    return 0


def build(rows: Iterable[dict], args, *, source_sha: str) -> dict:
    began = time.time()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: never replace a corpus")
    index = Exclusions()
    exclusions = {
        "prior": [load_prior(index, p) for p in args.prior],
        "benchmarks": load_benchmarks(index, args.benchmark_cache) if args.benchmark_cache else {},
    }
    staging = args.out.with_name(args.out.name + ".building")
    if staging.exists():
        raise FileExistsError(f"{staging}: inspect the earlier partial build")
    staging.mkdir(parents=True)
    scratch = staging / "scratch"
    validation, stats = stage_candidates(
        rows,
        index=index,
        staging=scratch,
        target_train_bytes=args.target_train_bytes,
        validation_docs=args.validation_docs,
        validation_per_mille=args.validation_per_mille,
        seed=args.seed,
        max_scan_rows=args.max_scan_rows,
        shard_bytes=args.shard_bytes,
    )
    artifacts = write_final(
        scratch, staging, validation=validation, stats=stats, shard_bytes=args.shard_bytes
    )
    shutil.rmtree(scratch)
    comp = write_composition(
        staging,
        seed=args.composition_seed,
        per_level=args.composition_per_level,
        test_per_level=args.composition_test_per_level,
        table_size=composition.TABLE_SIZE,
        vocabulary_size=2000,
    )
    tokens = None
    if args.tokenizer is not None:
        from prophet.data.tokenizer import ProphetTokenizer

        tokenizer = ProphetTokenizer.load(args.tokenizer)
        tokens = estimate_tokens(
            staging, artifacts, tokenizer, docs_per_shard=args.token_sample_docs
        )
        tokens["tokenizer_sha256"] = sha256(args.tokenizer)
        comp_tokens = sum(
            len(tokenizer.encode(r["text"], add_eos=True))
            for r in islice(composition.read_jsonl(staging / comp["train"]["path"]), 2000)
        )
        tokens["composition_train_tokens_first_2000"] = comp_tokens
    manifest = {
        "format_version": 1,
        "name": "loop-core-v1",
        "source": DATASET,
        "subset": SUBSET,
        "source_revision": args.revision,
        "source_sha": source_sha,
        "filters": FILTERS,
        "split": {
            "seed": args.seed,
            "validation_per_mille": args.validation_per_mille,
            "validation_docs": args.validation_docs,
            "policy": "URL host/path keyed",
        },
        "exclusions": {
            **exclusions,
            "unique_13_word_hashes": len(index.grams),
            "protected": dict(index.stats),
        },
        "targets": {"train_bytes": args.target_train_bytes, "max_scan_rows": args.max_scan_rows},
        "stats": dict(stats),
        "artifacts": artifacts,
        "composition": comp,
        "tokens": tokens,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "seconds": time.time() - began,
        "complete": True,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(args.out)
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
