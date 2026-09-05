#!/usr/bin/env python3
"""Materialise the mixture's sources as local shards the loader reads.

    python scripts/prepare_corpus.py --out corpus/ --dry-run            # the plan, no network
    python scripts/prepare_corpus.py --out corpus/ --max-bytes 2e9      # every source, capped
    python scripts/prepare_corpus.py --out corpus/ --sources fineweb-edu,stack-edu

For each source of ``configs/data_mixture_v1.yaml`` (deduplicated across phases), the
Hub stream is read with the source's declarative filters and written as
``<out>/<name>/part-NNNNN.jsonl`` shards of ``--shard-docs`` documents, with a
``manifest.json`` per source: dataset id, config, filters, licence as declared in the
mixture, documents and bytes written, shards, and the commit of this repository.

Resumable by shard: a rerun skips the documents already written and continues the
stream from there (a Hub stream cannot seek, so that skip costs one pass over the
skipped rows -- documented in ``prophet.data.corpus.HubSource``). Caps are per source:
``--max-docs`` and ``--max-bytes`` stop a source early, which is how a 300-hour budget
takes 20B tokens from a 1.3T-token source without downloading the rest.

What this script does not do: decide what is in the mixture (``prophet.data.recipes``),
verify that a dataset id exists and carries the licence the mixture claims
(``scripts/verify_datasets.py`` -- run it first), or decontaminate (the loader does that
in the stream, against ``benchmarks/``). It moves bytes, records where they came from,
and stops when told.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.corpus import HubSource  # noqa: E402
from prophet.data.mixture import Mixture, Source  # noqa: E402

MIXTURE = Path(__file__).resolve().parent.parent / "configs" / "data_mixture_v1.yaml"


def unique_sources(mixture: Mixture) -> list[Source]:
    seen: dict[str, Source] = {}
    for phase in mixture.phases:
        for src in phase.sources:
            seen.setdefault(src.name, src)
    return list(seen.values())


def load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"docs": 0, "bytes": 0, "shards": [], "complete": False}


def write_shards(
    name: str,
    docs: Iterable[str],
    out: Path,
    *,
    shard_docs: int = 10_000,
    max_docs: int | None = None,
    max_bytes: float | None = None,
    provenance: dict[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Write ``docs`` as shards under ``out/name``; returns the manifest.

    ``docs`` must already start after the documents a previous run wrote (the caller
    opens the stream at ``manifest["docs"]``). Shards are written to a temporary file
    and renamed, so an interrupted shard never counts.
    """
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(directory) if resume else {"docs": 0, "bytes": 0, "shards": [], "complete": False}
    manifest.update(provenance or {})
    manifest.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if manifest.get("complete"):
        return manifest

    shard_index = len(manifest["shards"])
    buffer: list[str] = []
    stopped_by_cap = False

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        path = directory / f"part-{shard_index:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for text in buffer:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        tmp.replace(path)
        manifest["shards"].append(path.name)
        manifest["docs"] += len(buffer)
        manifest["bytes"] += sum(len(t.encode("utf-8")) for t in buffer)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shard_index += 1
        buffer = []

    pending_bytes = 0
    for text in docs:
        if max_docs is not None and manifest["docs"] + len(buffer) >= max_docs:
            stopped_by_cap = True
            break
        if max_bytes is not None and manifest["bytes"] + pending_bytes >= max_bytes:
            stopped_by_cap = True
            break
        buffer.append(text)
        pending_bytes += len(text.encode("utf-8"))
        if len(buffer) >= shard_docs:
            flush()
            pending_bytes = 0
    flush()
    manifest["complete"] = True
    manifest["stopped_by_cap"] = stopped_by_cap
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def plan(sources: list[Source]) -> str:
    lines = ["| source | hf_id | config | filters | licence | available tokens |", "|---|---|---|---|---|---:|"]
    for s in sources:
        avail = "unverified" if s.available_tokens is None else f"{s.available_tokens / 1e9:.0f}B"
        lines.append(f"| {s.name} | `{s.hf_id}` | {s.config or ''} | {json.dumps(s.filters) if s.filters else ''} | {s.license} | {avail} |")
    return "\n".join(lines)


def stream(source: Source, start: int) -> Iterator[str]:
    return HubSource(source.name, source.weight, source.hf_id, config=source.config,
                     filters=dict(source.filters)).open(start)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixture", default=str(MIXTURE))
    ap.add_argument("--sources", default=None, help="comma-separated subset of source names")
    ap.add_argument("--shard-docs", type=int, default=10_000)
    ap.add_argument("--max-docs", type=int, default=None, help="per source")
    ap.add_argument("--max-bytes", type=float, default=None, help="per source")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; touch nothing, need no network")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    mixture = Mixture.from_yaml(args.mixture)
    mixture.validate()
    sources = unique_sources(mixture)
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        unknown = wanted - {s.name for s in sources}
        if unknown:
            raise SystemExit(f"unknown sources: {sorted(unknown)}")
        sources = [s for s in sources if s.name in wanted]

    print(plan(sources))
    if args.dry_run:
        print(f"\n{len(sources)} sources; nothing written. Every id above is a claim until "
              "scripts/verify_datasets.py has checked it against the Hub.")
        return 0

    out = Path(args.out)
    commit = _commit()
    for source in sources:
        manifest = load_manifest(out / source.name)
        if manifest.get("complete") and not args.no_resume:
            print(f"{source.name}: complete ({manifest['docs']} docs), skipped")
            continue
        start = 0 if args.no_resume else int(manifest.get("docs", 0))
        if start:
            print(f"{source.name}: resuming after {start} documents (a Hub stream skips, it cannot seek)")
        provenance = {"hf_id": source.hf_id, "config": source.config, "filters": source.filters,
                      "license": source.license, "domain": source.domain, "commit": commit}
        try:
            manifest = write_shards(source.name, stream(source, start), out, shard_docs=args.shard_docs,
                                    max_docs=args.max_docs, max_bytes=args.max_bytes, provenance=provenance,
                                    resume=not args.no_resume)
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{source.name}: {manifest['docs']} docs, {manifest['bytes'] / 1e9:.2f} GB, "
              f"{len(manifest['shards'])} shards{' (cap reached)' if manifest.get('stopped_by_cap') else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
