#!/usr/bin/env python3
"""Create a separate recovery corpus using the frozen overlap and smoke audits.

Remove known lexical overlaps from training and previously inspected donor-smoke
documents from validation. This does not repair unavailable benchmark coverage or
constitute a final untouched test set. Original R04 corpus files stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rehearse_qwen_conversion import digest  # noqa: E402


def prepare(corpus: Path, overlap_path: Path, smoke_path: Path, out: Path):
    if out.exists():
        raise FileExistsError("use a new output directory; preserve existing corpus")
    overlap = json.loads(overlap_path.read_bytes())
    smoke = json.loads(smoke_path.read_bytes())
    if not overlap.get("complete") or not smoke.get("complete"):
        raise ValueError("input audits must be complete")
    excluded_diagnostics = {row["document_sha256"]
                            for row in smoke["arms"]["hybrid_initialization"]["documents"]}
    source_paths = {}
    for split in ("train", "validation"):
        source_paths[split] = []
        for item in overlap["splits"][split]["files"]:
            path = (corpus / item["path"]).resolve()
            if not path.is_relative_to(corpus.resolve()) or digest(path) != item["sha256"]:
                raise ValueError("corpus source differs from the overlap audit")
            source_paths[split].append(path)
    out.mkdir(parents=True)
    report = {"complete": False, "overlap_audit_sha256": digest(overlap_path),
              "donor_smoke_sha256": digest(smoke_path), "splits": {},
              "scope": "recovery development data; known lexical overlaps removed; not benchmark-clean or a final untouched test set",
              "all_requested_benchmarks_screened": overlap["all_requested_sources_indexed"]}
    retained = {}
    for split in ("train", "validation"):
        excluded_overlap = {row["sha256"] for row in overlap["splits"][split]["matches"]}
        excluded = excluded_overlap | (excluded_diagnostics if split == "validation" else set())
        seen, removed, kept_hashes = 0, set(), set()
        destination = out / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as writer:
            for path in source_paths[split]:
                with path.open(encoding="utf-8") as reader:
                    for line in reader:
                        if not line.strip():
                            continue
                        text = json.loads(line)["text"]
                        sha = hashlib.sha256(text.encode()).hexdigest()
                        seen += 1
                        if sha in excluded:
                            removed.add(sha)
                            continue
                        if sha in kept_hashes:
                            raise ValueError("unexpected duplicate document in frozen corpus")
                        kept_hashes.add(sha)
                        writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        if removed != excluded or seen != overlap["splits"][split]["documents"]:
            raise ValueError("exclusion identities or source counts differ from the audits")
        retained[split] = kept_hashes
        report["splits"][split] = {"input_documents": seen, "retained_documents": len(kept_hashes),
                                    "removed_sha256": sorted(removed), "sha256": digest(destination),
                                    "bytes": destination.stat().st_size, "path": destination.name}
    if retained["train"] & retained["validation"]:
        raise ValueError("train and validation share an exact document")
    report["complete"] = True
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("corpus", "overlap", "smoke", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.overlap, args.smoke, args.out), indent=2))


if __name__ == "__main__":
    main()
