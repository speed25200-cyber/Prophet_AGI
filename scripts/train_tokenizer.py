#!/usr/bin/env python3
"""Train a Prophet-Tok v1 vocabulary and save the artifact a real run needs.

    python scripts/train_tokenizer.py --data-root corpus/ --out tokenizer.json \\
        --vocab-size 32768 --max-docs 200000

Reads every ``*.jsonl`` / ``*.txt`` under ``--data-root`` (the same layout the training
loader uses), samples up to ``--max-docs`` documents round-robin across sources so no
single corpus dominates the merges, and trains with :class:`prophet.data.tokenizer.
BPETrainer` -- the *reference* trainer: correct, inspectable, and slow (it re-counts
pairs after every merge). Budget roughly a minute per thousand merges on a few hundred
thousand short documents; for a full 32k vocabulary on a large sample, run it once on a
CPU box and commit nothing but the resulting JSON somewhere outside git.

The artifact records the pre-tokenisation pattern; :meth:`ProphetTokenizer.load` refuses
a vocabulary built with a different one, and ``check_invariants`` is run here so a
vocabulary that violates the digit or newline rules never leaves this script.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.corpus import LocalTextSource  # noqa: E402
from prophet.data.tokenizer import BPETrainer, ProphetTokenizer  # noqa: E402


def _sources(root: Path) -> list[LocalTextSource]:
    names: set[str] = set()
    for p in root.iterdir():
        if p.suffix in (".jsonl", ".txt"):
            names.add(p.stem)
        elif p.is_dir() and any(q.suffix in (".jsonl", ".txt") for q in p.iterdir()):
            names.add(p.name)
    if not names:
        raise SystemExit(f"no *.jsonl / *.txt corpora under {root}")
    return [LocalTextSource.from_root(root, name, 1.0) for name in sorted(names)]


def sample_documents(sources: list[LocalTextSource], max_docs: int) -> list[str]:
    """Round-robin across sources so the merges reflect the mixture, not the largest file."""
    iterators = [src.open(0) for src in sources]
    out: list[str] = []
    for it in itertools.cycle(list(iterators)):
        if len(out) >= max_docs or not iterators:
            break
        try:
            out.append(next(it))
        except StopIteration:
            iterators.remove(it)
            if not iterators:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, help="where to write the tokenizer JSON")
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--max-docs", type=int, default=100_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    sources = _sources(Path(args.data_root))
    docs = sample_documents(sources, args.max_docs)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"sources    {[s.name for s in sources]}")
    print(f"sample     {len(docs)} documents, {n_bytes / 1e6:.1f} MB")

    trainer = BPETrainer(args.vocab_size, min_frequency=args.min_frequency)
    merges = trainer.train(docs)
    tok = ProphetTokenizer(merges=merges, vocab_size=args.vocab_size)
    problems = tok.check_invariants()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    tok.save(args.out)
    sample = "\n".join(docs[:50])
    print(f"merges     {len(merges)} (ids in use: {tok.n_tokens})")
    print(f"fertility  {tok.fertility(sample):.2f} bytes/token on the first 50 documents")
    print(f"saved      {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
