#!/usr/bin/env python3
"""Retrospective exact 13-word overlap audit, without changing training data.

Reads a pinned Hub source manifest. Corpus and fetched benchmark text stay outside
git; the report contains counts and document hashes only. This is a lexical screen,
not semantic decontamination or a benchmark performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.decontaminate import ngrams, normalise  # noqa: E402
from scripts.fetch_benchmarks import write_benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the previous audit")
    from datasets import load_dataset

    allowed = {"mit", "apache-2.0", "cc-by-2.0", "cc-by-4.0", "cc-by-sa-4.0"}
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    result = {
        "complete": False,
        "sources": [],
        "splits": {},
        "n": 13,
        "policy": "any exact normalized 13-word overlap; rows shorter than 13 words untested",
        "scope": "retrospective lexical screen; corpus unchanged; no semantic or benchmark-quality claim",
    }
    index: dict[str, int] = {}
    names = []

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)

    for source in sources:
        record = dict(source)
        licenses = source["license"]
        licenses = licenses if isinstance(licenses, list) else [licenses]
        if source["gated"] or not licenses or any(item not in allowed for item in licenses):
            record["status"] = "not fetched: gated or outside this audit's metadata allowlist"
        else:
            try:
                rows = load_dataset(
                    source["hf_id"],
                    source["config"],
                    split=source["split"],
                    revision=source["revision"],
                    streaming=True,
                    token=False,
                )
                record["rows"] = write_benchmark(source["name"], rows, source["fields"], args.cache)
                path = args.cache / f"{source['name']}.jsonl"
                record["sha256"] = digest(path)
                bit = 1 << len(names)
                names.append(source["name"])
                record["short_rows_untested"] = 0
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        cleaned = normalise(json.loads(line)["text"])
                        if len(cleaned.split()) < 13:
                            record["short_rows_untested"] += 1
                        for gram in ngrams(cleaned, 13):
                            index[gram] = index.get(gram, 0) | bit
                record["status"] = "indexed"
            except Exception as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        result["sources"].append(record)
        save()
        print(source["name"], record["status"], record.get("rows"), flush=True)

    result["unique_benchmark_ngrams"] = len(index)
    if not index:
        raise ValueError("no benchmark n-grams were indexed")
    for split in ("train", "validation"):
        paths = sorted((args.corpus / split).rglob("*.jsonl"))
        if not paths:
            raise ValueError(f"missing corpus split: {split}")
        counts = dict.fromkeys(names, 0)
        matched = []
        seen = 0
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    text = json.loads(line)["text"]
                    mask = 0
                    for gram in ngrams(normalise(text), 13):
                        mask |= index.get(gram, 0)
                    hits = [name for i, name in enumerate(names) if mask & (1 << i)]
                    if hits:
                        matched.append(
                            {
                                "document_index": seen,
                                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "benchmarks": hits,
                            }
                        )
                        for name in hits:
                            counts[name] += 1
                    seen += 1
        result["splits"][split] = {
            "documents": seen,
            "matching_documents": len(matched),
            "per_benchmark": counts,
            "matches": matched,
            "files": [
                {
                    "path": str(path.relative_to(args.corpus)).replace("\\", "/"),
                    "sha256": digest(path),
                }
                for path in paths
            ],
        }
        save()
        print(split, seen, "documents", len(matched), "potential overlaps", flush=True)
    result["complete"] = True
    result["all_requested_sources_indexed"] = all(
        s["status"] == "indexed" for s in result["sources"]
    )
    save()


if __name__ == "__main__":
    main()
