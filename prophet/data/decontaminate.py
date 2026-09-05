"""Small, deterministic n-gram benchmark decontaminator."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

__all__ = ["Decontaminator", "ngrams", "normalise"]


def normalise(text: str) -> str:
    """Fold case and accents, replace punctuation, and collapse whitespace."""

    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.category(character).startswith("M")
    )
    words: list[str] = []
    current: list[str] = []
    for character in without_marks:
        if character.isalnum():
            current.append(character)
        elif current:
            words.append("".join(current))
            current.clear()
    if current:
        words.append("".join(current))
    return " ".join(words)


def ngrams(text: str, n: int) -> Iterator[str]:
    """Yield contiguous word n-grams from already-normalised or raw text."""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    words = str(text).split()
    for index in range(len(words) - n + 1):
        yield " ".join(words[index : index + n])


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    normalised: str
    grams: frozenset[str]


class Decontaminator:
    """Reject documents containing too much of a registered benchmark item.

    Overlap is measured as the fraction of an item's unique n-grams found in the
    candidate document.  Items shorter than ``n`` use token-boundary-aware substring
    matching so that short answers are not silently ignored.
    """

    def __init__(self, n: int = 13, threshold: float = 0.5) -> None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must lie above 0 and at most 1")
        self.n = n
        self.threshold = float(threshold)
        self._benchmarks: dict[str, list[_Fingerprint]] = {}
        self._rejected_by_benchmark: Counter[str] = Counter()
        self.documents_seen = 0
        self.documents_rejected = 0

    def add_benchmark(self, name: str, examples: Iterable[str]) -> None:
        if not name or not name.strip():
            raise ValueError("benchmark name must not be empty")
        fingerprints = self._benchmarks.setdefault(name, [])
        for example in examples:
            cleaned = normalise(example)
            if not cleaned:
                continue
            fingerprints.append(_Fingerprint(cleaned, frozenset(ngrams(cleaned, self.n))))

    @staticmethod
    def _contains_phrase(document: str, phrase: str) -> bool:
        return f" {phrase} " in f" {document} "

    def is_contaminated(self, text: str) -> bool:
        cleaned = normalise(text)
        document_grams = frozenset(ngrams(cleaned, self.n))
        matched: list[str] = []

        for benchmark, fingerprints in self._benchmarks.items():
            for fingerprint in fingerprints:
                if not fingerprint.grams:
                    hit = self._contains_phrase(cleaned, fingerprint.normalised)
                else:
                    overlap = len(document_grams.intersection(fingerprint.grams))
                    hit = overlap / len(fingerprint.grams) >= self.threshold
                if hit:
                    matched.append(benchmark)
                    break

        self.documents_seen += 1
        if matched:
            self.documents_rejected += 1
            self._rejected_by_benchmark.update(matched)
            return True
        return False

    @property
    def rejected_by_benchmark(self) -> dict[str, int]:
        return {name: self._rejected_by_benchmark[name] for name in self._benchmarks}

    def report(self) -> str:
        lines = [
            "| Benchmark | Rejected documents |",
            "|---|---:|",
        ]
        lines.extend(
            f"| {name} | {self._rejected_by_benchmark[name]} |" for name in self._benchmarks
        )
        lines.extend(
            (
                "",
                f"Documents seen: {self.documents_seen}",
                f"Documents rejected: {self.documents_rejected}",
            )
        )
        return "\n".join(lines)
