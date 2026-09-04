"""Deterministic mixture sampling, sequence packing, and resumable loading.

The loader's random choice is a pure function of ``(seed, sequence_number)``. Its
checkpoint contains cursors, small per-source packing buffers, and a versioned manifest
fingerprint; it never serialises a random-number generator or corpus contents.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "IterableSource",
    "LOADER_STATE_FORMAT_VERSION",
    "LoaderState",
    "MixtureSampler",
    "SequencePacker",
    "StreamingLoader",
    "sources_from_iterables",
]

_MASK64 = (1 << 64) - 1
LOADER_STATE_FORMAT_VERSION = 1
"""Version of the serialized :class:`LoaderState` checkpoint contract."""


class MixtureSampler:
    """Stateless deterministic categorical sampler.

    SplitMix64 supplies a stable, well-distributed 64-bit value without depending on
    Python's process-randomised hash or mutable RNG state.
    """

    def __init__(self, weights: Sequence[float], *, seed: int = 0) -> None:
        if not weights:
            raise ValueError("at least one source weight is required")
        parsed = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0 for weight in parsed):
            raise ValueError("source weights must be finite and non-negative")
        total = math.fsum(parsed)
        if total <= 0:
            raise ValueError("at least one source weight must be positive")

        running = 0.0
        self.weights = tuple(weight / total for weight in parsed)
        cumulative: list[float] = []
        for weight in self.weights:
            running += weight
            cumulative.append(running)
        cumulative[-1] = 1.0
        self._cumulative = tuple(cumulative)
        self.seed = int(seed) & _MASK64

    @staticmethod
    def _splitmix64(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & _MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return value ^ (value >> 31)

    def source_for_step(self, step: int) -> int:
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        mixed = self._splitmix64((step ^ self.seed) & _MASK64)
        # The high 53 bits map exactly to the precision of a Python float in [0, 1).
        uniform = (mixed >> 11) * (1.0 / (1 << 53))
        return min(bisect.bisect_right(self._cumulative, uniform), len(self.weights) - 1)

    def empirical_shares(self, steps: int) -> list[float]:
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        counts = [0] * len(self.weights)
        for step in range(steps):
            counts[self.source_for_step(step)] += 1
        return [count / steps for count in counts]


class SequencePacker:
    """Concatenate documents and emit fixed-length token sequences."""

    def __init__(self, seq_len: int, *, separator: int | None = None) -> None:
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        self.seq_len = seq_len
        self.separator = None if separator is None else int(separator)
        self._tokens: deque[int] = deque()

    def add(self, document: Iterable[int]) -> None:
        self._tokens.extend(int(token) for token in document)
        if self.separator is not None:
            self._tokens.append(self.separator)

    def ready(self) -> bool:
        return len(self._tokens) >= self.seq_len

    def pop(self) -> list[int]:
        if not self.ready():
            raise IndexError(f"packer has {len(self._tokens)} tokens, needs {self.seq_len}")
        return [self._tokens.popleft() for _ in range(self.seq_len)]

    @property
    def carry(self) -> list[int]:
        return list(self._tokens)

    def load_carry(self, tokens: Iterable[int]) -> None:
        self._tokens = deque(int(token) for token in tokens)

    def __len__(self) -> int:
        return len(self._tokens)


@dataclass(frozen=True, slots=True)
class IterableSource:
    """A restartable in-memory source used by tests and smoke training."""

    name: str
    weight: float
    documents: tuple[tuple[int, ...], ...]


def _manifest_fingerprint(
    sources: Sequence[IterableSource],
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
    separator: int | None,
) -> str:
    """Hash every value that determines the loader's future output.

    Fields are length-prefixed rather than represented with ``repr`` or Python's salted
    ``hash`` so the digest is stable across processes and cannot be made ambiguous by a
    source name or token value containing a delimiter.
    """

    digest = hashlib.sha256()

    def add(value: str | bytes | int) -> None:
        encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add("prophet-stream-manifest")
    add(LOADER_STATE_FORMAT_VERSION)
    add(seq_len)
    add(batch_size)
    add(seed)
    add("none" if separator is None else "integer")
    if separator is not None:
        add(separator)
    add(len(sources))
    for source in sources:
        add(source.name)
        add(float(source.weight).hex())
        add(len(source.documents))
        for document in source.documents:
            add(len(document))
            for token in document:
                add(int(token))
    return digest.hexdigest()


def sources_from_iterables(
    sources: Mapping[str, tuple[float, Iterable[Iterable[int]]]],
) -> list[IterableSource]:
    """Materialise small restartable sources from ``name -> (weight, documents)``."""

    result: list[IterableSource] = []
    for name, specification in sources.items():
        if len(specification) != 2:
            raise ValueError(f"source {name!r} must be a (weight, documents) pair")
        weight, documents = specification
        materialised = tuple(tuple(int(token) for token in doc) for doc in documents)
        result.append(IterableSource(str(name), float(weight), materialised))
    return result


@dataclass(slots=True)
class LoaderState:
    """Everything needed to resume the exact next packed sequence."""

    step: int = 0
    cursors: dict[str, int] = field(default_factory=dict)
    carry: dict[str, list[int]] = field(default_factory=dict)
    format_version: int = LOADER_STATE_FORMAT_VERSION
    manifest_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": int(self.format_version),
            "manifest_fingerprint": self.manifest_fingerprint,
            "step": int(self.step),
            "cursors": {name: int(cursor) for name, cursor in self.cursors.items()},
            "carry": {
                name: [int(token) for token in tokens] for name, tokens in self.carry.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoaderState:
        if not isinstance(data, Mapping):
            raise TypeError("loader state must be a mapping")
        cursors = data.get("cursors") or {}
        carry = data.get("carry") or {}
        if not isinstance(cursors, Mapping) or not isinstance(carry, Mapping):
            raise TypeError("loader cursors and carry must be mappings")
        format_version = data.get("format_version")
        if not isinstance(format_version, int) or isinstance(format_version, bool):
            raise ValueError("loader state is missing a valid format_version")
        manifest_fingerprint = data.get("manifest_fingerprint")
        if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint:
            raise ValueError("loader state is missing a manifest_fingerprint")
        step = data.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("loader state is missing a valid integer step")
        parsed_cursors: dict[str, int] = {}
        for name, cursor in cursors.items():
            if not isinstance(cursor, int) or isinstance(cursor, bool):
                raise ValueError(f"loader cursor for source {name!r} must be an integer")
            parsed_cursors[str(name)] = cursor
        return cls(
            step=step,
            cursors=parsed_cursors,
            carry={str(name): [int(token) for token in tokens] for name, tokens in carry.items()},
            format_version=format_version,
            manifest_fingerprint=manifest_fingerprint,
        )


class StreamingLoader:
    """Yield fixed-size batches while preserving exact preemption/resume semantics."""

    def __init__(
        self,
        sources: Sequence[IterableSource],
        *,
        seq_len: int,
        batch_size: int = 1,
        seed: int = 0,
        separator: int | None = None,
    ) -> None:
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not sources:
            raise ValueError("at least one source is required")

        self.sources = list(sources)
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        for source in self.sources:
            if not source.documents:
                raise ValueError(f"source {source.name!r} has no documents")
            if separator is None and not any(source.documents):
                raise ValueError(f"source {source.name!r} contains no tokens")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = int(seed)
        self.separator = None if separator is None else int(separator)
        self.sampler = MixtureSampler([source.weight for source in self.sources], seed=seed)
        self._source_by_name = {source.name: source for source in self.sources}
        self._cursors = {source.name: 0 for source in self.sources}
        self._packers = {
            source.name: SequencePacker(seq_len, separator=self.separator)
            for source in self.sources
        }
        self.manifest_fingerprint = _manifest_fingerprint(
            self.sources,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            seed=self.seed,
            separator=self.separator,
        )
        self._step = 0

    def _next_document(self, source: IterableSource) -> tuple[int, ...]:
        cursor = self._cursors[source.name]
        document = source.documents[cursor]
        self._cursors[source.name] = (cursor + 1) % len(source.documents)
        return document

    def _next_sequence(self) -> list[int]:
        source = self.sources[self.sampler.source_for_step(self._step)]
        packer = self._packers[source.name]
        empty_documents = 0
        while not packer.ready():
            before = len(packer)
            packer.add(self._next_document(source))
            if len(packer) == before:
                empty_documents += 1
                if empty_documents >= len(source.documents):
                    raise RuntimeError(f"source {source.name!r} cannot fill a sequence")
            else:
                empty_documents = 0
        sequence = packer.pop()
        self._step += 1
        return sequence

    def batches(self, count: int | None = None) -> Iterator[list[list[int]]]:
        """Yield ``count`` batches, or continue forever when ``count`` is ``None``."""

        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError("batch count must be a non-negative integer or None")
        emitted = 0
        while count is None or emitted < count:
            yield [self._next_sequence() for _ in range(self.batch_size)]
            emitted += 1

    def __iter__(self) -> Iterator[list[list[int]]]:
        return self.batches()

    def state(self) -> LoaderState:
        carry = {name: packer.carry for name, packer in self._packers.items() if len(packer)}
        return LoaderState(
            step=self._step,
            cursors=dict(self._cursors),
            carry=carry,
            manifest_fingerprint=self.manifest_fingerprint,
        )

    def validate_state(self, state: LoaderState | Mapping[str, Any]) -> LoaderState:
        """Validate and normalize a checkpoint without mutating this loader."""

        if not isinstance(state, LoaderState):
            state = LoaderState.from_dict(state)
        if state.format_version != LOADER_STATE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported loader state format version {state.format_version}; "
                f"expected {LOADER_STATE_FORMAT_VERSION}"
            )
        if state.manifest_fingerprint != self.manifest_fingerprint:
            raise ValueError(
                "loader state manifest fingerprint does not match this loader's "
                "sources or stream parameters"
            )
        known = set(self._source_by_name)
        cursor_names = set(state.cursors)
        carry_names = set(state.carry)
        unknown = sorted((cursor_names | carry_names) - known)
        if unknown:
            raise KeyError(f"unknown source in checkpoint: {unknown[0]}")
        missing = sorted(known - cursor_names)
        if missing:
            raise KeyError(f"missing source cursor in checkpoint: {missing[0]}")
        if not isinstance(state.step, int) or isinstance(state.step, bool) or state.step < 0:
            raise ValueError("loader step must be non-negative")

        cursors: dict[str, int] = {}
        for name, source in self._source_by_name.items():
            cursor = state.cursors[name]
            if not isinstance(cursor, int) or isinstance(cursor, bool):
                raise ValueError(f"cursor for source {name!r} must be an integer")
            if cursor < 0:
                raise ValueError(f"negative cursor for source {name!r}")
            if cursor >= len(source.documents):
                raise ValueError(
                    f"cursor {cursor} for source {name!r} exceeds its "
                    f"{len(source.documents)} documents"
                )
            cursors[name] = cursor

        carry = {
            name: [int(token) for token in tokens] for name, tokens in state.carry.items()
        }
        return LoaderState(
            step=int(state.step),
            cursors=cursors,
            carry=carry,
            format_version=state.format_version,
            manifest_fingerprint=state.manifest_fingerprint,
        )

    def load_state(self, state: LoaderState | Mapping[str, Any]) -> None:
        state = self.validate_state(state)
        self._step = state.step
        self._cursors = state.cursors
        for name, packer in self._packers.items():
            packer.load_carry(state.carry.get(name, []))
