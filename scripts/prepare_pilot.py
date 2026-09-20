#!/usr/bin/env python3
"""Prepare a bounded, pinned FineWeb-Edu pilot with a separate held-out corpus.

Artifacts stay under data/ by default. Train the tokenizer on <out>/train only.
This is a finite English web pilot, not the final multilingual/code data mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.decontaminate import ngrams, normalise  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SUBSET = "sample-10BT"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(url: str, digest: str, seed: int, validation_per_mille: int) -> str:
    """Keep versions of one URL together; exact content is deduplicated globally."""
    parsed = urlsplit(url)
    key = (parsed.netloc.lower() + parsed.path.rstrip("/")) if parsed.netloc else digest
    value = int(content_hash(f"{seed}:{key}"), 16) % 1000
    return "validation" if value < validation_per_mille else "train"


def gram_hashes(text: str) -> set[bytes]:
    return {hashlib.blake2b(g.encode("utf-8"), digest_size=16).digest() for g in ngrams(text, 13)}


def partition_rows(rows: Iterable[dict], *, max_docs: int, max_bytes: int,
                   seed: int = 42, validation_per_mille: int = 20):
    """Global normalized dedup, URL split, then conservative held-out decontamination.

    Reject any training document sharing a normalized 13-word span with validation.
    This intentionally also rejects boilerplate overlap. It does not establish freedom
    from semantic paraphrases or contamination of external benchmark suites.
    """
    if max_docs < 1 or max_bytes < 1 or not 1 <= validation_per_mille <= 999:
        raise ValueError("positive caps and validation_per_mille in [1,999] required")
    groups = {"train": [], "validation": []}
    seen: set[str] = set()
    stats: Counter = Counter()
    for row in islice(rows, max_docs * 10):
        stats["rows_seen"] += 1
        if not isinstance(row.get("text"), str) or "int_score" not in row:
            raise ValueError("FineWeb-Edu schema requires text and int_score")
        text = row["text"]
        encoded_bytes = len(text.encode("utf-8"))
        if row["int_score"] < 3 or not 400 <= encoded_bytes <= 50000:
            stats["quality_or_length_rejected"] += 1
            continue
        cleaned = normalise(text)
        if len(cleaned.split()) < 30:
            stats["short_rejected"] += 1
            continue
        digest = content_hash(cleaned)
        if digest in seen:
            stats["duplicate_rejected"] += 1
            continue
        if stats["selected_bytes"] + encoded_bytes > max_bytes:
            stats["byte_cap_reached"] = 1
            break
        seen.add(digest)
        split = split_for(row.get("url", ""), digest, seed, validation_per_mille)
        groups[split].append({"text": text, "content_sha256": digest,
                              "source_id": row.get("id"), "url": row.get("url"),
                              "int_score": row["int_score"]})
        stats["selected_docs"] += 1
        stats["selected_bytes"] += encoded_bytes
        if stats["selected_docs"] >= max_docs:
            stats["document_cap_reached"] = 1
            break
    protected = set()
    for row in groups["validation"]:
        protected.update(gram_hashes(normalise(row["text"])))
    train = []
    for row in groups["train"]:
        if protected.intersection(gram_hashes(normalise(row["text"]))):
            stats["heldout_overlap_rejected"] += 1
        else:
            train.append(row)
    groups["train"] = train
    for name, items in groups.items():
        stats[f"{name}_docs"] = len(items)
        stats[f"{name}_bytes"] = sum(len(row["text"].encode("utf-8")) for row in items)
    return groups, dict(stats)


def write_pilot(groups: dict, out: Path, manifest: dict) -> dict:
    """Publish only a complete corpus; never replace an existing corpus."""
    if out.exists():
        raise FileExistsError(out)
    staging = out.with_name(out.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for split, rows in groups.items():
        if not rows:
            raise ValueError(f"empty {split}; increase the sample before training")
        relative = Path(split) / "fineweb-edu" / "part-00000.jsonl"
        path = staging / relative
        path.parent.mkdir(parents=True)
        digest = hashlib.sha256()
        with path.open("wb") as stream:
            for row in rows:
                data = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                stream.write(data)
                digest.update(data)
        artifacts[relative.as_posix()] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                                         "documents": len(rows)}
    manifest = {**manifest, "artifacts": artifacts, "complete": True}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(out)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/fineweb-pilot-v1"))
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--max-bytes", type=int, default=200000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-per-mille", type=int, default=20)
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        ap.error("revision must be an immutable 40-character Hub commit")
    if args.out.exists() or args.out.with_name(args.out.name + ".building").exists():
        ap.error("output or staging directory already exists; choose a new corpus path")
    if args.max_docs < 1 or args.max_bytes < 1 or not 1 <= args.validation_per_mille <= 999:
        ap.error("positive caps and validation-per-mille in [1,999] required")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET, revision=args.revision)
    license_name = info.card_data.get("license")
    if info.gated or license_name != "odc-by" or info.sha != args.revision:
        raise RuntimeError("Dataset access, licence or revision differs from audited pilot source")
    print(f"Streaming {DATASET}@{info.sha}, {SUBSET}; capped at {args.max_docs} documents", flush=True)
    rows = load_dataset(DATASET, name=SUBSET, revision=info.sha, split="train", streaming=True)
    groups, stats = partition_rows(rows, max_docs=args.max_docs, max_bytes=args.max_bytes,
                                  seed=args.seed, validation_per_mille=args.validation_per_mille)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parent.parent, text=True).strip()
    manifest = write_pilot(groups, args.out, {
        "format_version": 1, "source": DATASET, "source_revision": info.sha, "subset": SUBSET,
        "source_split": "train", "source_license": license_name,
        "attribution_url": f"https://huggingface.co/datasets/{DATASET}/tree/{info.sha}",
        "code_revision": revision, "max_docs": args.max_docs, "max_bytes": args.max_bytes,
        "max_rows_scanned": args.max_docs * 10,
        "seed": args.seed, "validation_per_mille": args.validation_per_mille,
        "split_policy": "seeded SHA256 URL host/path; global normalized-content deduplication",
        "decontamination": "reject any training document sharing a normalized 13-word span with validation",
        "external_benchmarks_decontaminated": False,
        "filters": {"int_score_min": 3, "utf8_bytes_min": 400, "utf8_bytes_max": 50000,
                    "normalized_words_min": 30},
        "sampling": "bounded prefix of sample-10BT; not a representative final training mixture",
        "stats": stats,
    })
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
