"""Restartable text sources and deterministic phase transitions for real training."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from prophet.data.streaming import StreamingLoader


class EpochCapExceeded(RuntimeError):
    """The configured corpus repetition budget has been exhausted."""


def row_passes(row, filters):
    for key, expected in filters.items():
        op = next((suffix for suffix in ("_min", "_max", "_in") if key.endswith(suffix)), "")
        field = key[:-len(op)] if op else key
        if field not in row:
            return False
        value = row[field]
        try:
            ok = (value >= expected if op == "_min" else value <= expected if op == "_max"
                  else value in expected if op == "_in" else value == expected)
        except TypeError:
            return False
        if not ok:
            return False
    return True


class LocalTextSource:
    """JSONL text shards with content-validated byte-offset indexes.

    Only offsets are held in memory. A new source hashes the files, so a same-size edit
    cannot silently reuse stale offsets or resume an incompatible training stream.
    """

    def __init__(self, name, weight, paths, *, text_field="text"):
        self.name, self.weight = name, weight
        self.paths = [Path(p) for p in paths]
        self.text_field = text_field
        self._indexes = None
        self._identity = None

    @classmethod
    def from_root(cls, root, name, weight):
        root = Path(root)
        direct = root / f"{name}.jsonl"
        plain = root / f"{name}.txt"
        paths = ([direct] if direct.is_file() else [plain] if plain.is_file() else
                 sorted(p for p in (root / name).glob("*") if p.suffix in (".jsonl", ".txt")))
        if not paths:
            raise FileNotFoundError(f"missing {direct} or {root / name}/*.jsonl")
        return cls(name, weight, paths)

    def _index(self):
        if self._indexes is not None:
            return
        indexes, identities = [], []
        for path in self.paths:
            digest = hashlib.sha256()
            offsets = []
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    digest.update(line)
                    if line.strip():
                        offsets.append(offset)
            identity = digest.hexdigest()
            payload = {"sha256": identity, "offsets": offsets}
            cache = path.with_suffix(path.suffix + ".lines")
            try:
                previous = json.loads(cache.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None
            if previous != payload:
                temporary = cache.with_suffix(cache.suffix + ".tmp")
                temporary.write_text(json.dumps(payload), encoding="utf-8")
                temporary.replace(cache)
            indexes.append(offsets)
            identities.append(identity)
        self._indexes = indexes
        self._identity = {"files": identities, "text_field": self.text_field}

    def fingerprint(self):
        self._index()
        return self._identity

    def n_documents(self):
        self._index()
        return sum(map(len, self._indexes))

    def open(self, start=0):
        self._index()
        if start < 0:
            raise ValueError("start must be non-negative")
        for path, offsets in zip(self.paths, self._indexes, strict=True):
            if start >= len(offsets):
                start -= len(offsets)
                continue
            with path.open("rb") as stream:
                for offset in offsets[start:]:
                    stream.seek(offset)
                    line = stream.readline().decode("utf-8")
                    text = (line.rstrip("\r\n") if path.suffix == ".txt"
                            else json.loads(line)[self.text_field])
                    if not isinstance(text, str):
                        raise TypeError(f"{path}: {self.text_field} must be text")
                    yield text
            start = 0


class HubSource:
    """Stream a pinned Hub revision. Cursor counts rows surviving source filters.

    Resuming seeks through the filtered prefix; it is exact but can be slow. Materialise
    local shards for long runs. Dataset scripts are never executed.
    """

    def __init__(self, name, weight, hf_id, *, config=None, split="train",
                 text_field="text", filters=None, revision=None):
        self.name, self.weight, self.hf_id = name, weight, hf_id
        self.config, self.split, self.text_field = config, split, text_field
        self.filters = dict(filters or {})
        self.revision = revision

    def n_documents(self):
        return None

    def fingerprint(self):
        if self.revision is None:
            from huggingface_hub import HfApi
            self.revision = HfApi().dataset_info(self.hf_id).sha
        return {k: getattr(self, k) for k in
                ("hf_id", "config", "split", "text_field", "filters", "revision")}

    def open(self, start=0):
        from datasets import load_dataset
        self.fingerprint()
        rows = load_dataset(self.hf_id, name=self.config, split=self.split,
                            revision=self.revision, streaming=True)
        for row in rows:
            if not row_passes(row, self.filters):
                continue
            if start:
                start -= 1
                continue
            text = row[self.text_field]
            if not isinstance(text, str):
                raise TypeError(f"{self.hf_id}: {self.text_field} must be text")
            yield text


@dataclass
class SourceStats:
    seen: int = 0
    rejected: int = 0


class TokenisedSource:
    def __init__(self, source, tokenizer, *, decontaminator=None, max_epochs=4,
                 parse_special=False):
        if max_epochs is not None and (not math.isfinite(max_epochs) or max_epochs <= 0):
            raise ValueError("max_epochs must be finite and positive")
        self.source, self.tokenizer = source, tokenizer
        self.name, self.weight = source.name, source.weight
        self.decontaminator, self.max_epochs = decontaminator, max_epochs
        self.parse_special = parse_special
        self.cursor = 0
        self.stats = SourceStats()
        self._iterator = None
        self._size = source.n_documents()
        if self._size == 0:
            raise ValueError(f"source {self.name!r} has no documents")
        # Unknown-size Hub streams do not wrap: a repetition cap cannot be enforced.
        self.empty_limit = self._size or 100_000

    def fingerprint(self):
        if hasattr(self.source, "fingerprint"):
            identity = self.source.fingerprint()
        else:
            # Small application sources (e.g. quarantine) become an immutable snapshot.
            docs = tuple(self.source.open(0))
            identity = docs
            self._snapshot = docs
        tok = self.tokenizer
        decon = self.decontaminator
        tokenizer_identity = ({"tokenizer": tok.fingerprint()} if hasattr(tok, "fingerprint") else {
            "vocab_size": tok.vocab_size,
            "merges": [[a.hex(), b.hex()] for a, b in tok.merges],
            "special_tokens": tok._special_to_id,
        })
        return {"source": identity, "max_epochs": self.max_epochs,
                "parse_special": self.parse_special, **tokenizer_identity,
                "decontamination": None if decon is None else {
                    "n": decon.n, "threshold": decon.threshold,
                    "items": {k: [v.normalised for v in values]
                              for k, values in decon._benchmarks.items()}}}

    def epochs(self):
        return self.cursor / self._size if self._size else 0.0

    def validate_cursor(self, cursor):
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if self._size and self.max_epochs is not None and cursor > math.ceil(self._size * self.max_epochs):
            raise ValueError("cursor exceeds epoch cap")

    def restore_cursor(self, cursor):
        self.validate_cursor(cursor)
        self.cursor, self._iterator = cursor, None

    def next_document(self):
        if self._size and self.max_epochs is not None and self.epochs() >= self.max_epochs:
            raise EpochCapExceeded(f"{self.name}: {self.epochs():.2f} epochs reached")
        offset = self.cursor % self._size if self._size else self.cursor
        if self._iterator is None or offset == 0:
            self._iterator = (iter(self._snapshot[offset:]) if hasattr(self, "_snapshot")
                              else iter(self.source.open(offset)))
        text = next(self._iterator)  # Unknown-size sources stop instead of silently wrapping.
        self.cursor += 1
        self.stats.seen += 1
        if self.decontaminator and self.decontaminator.is_contaminated(text):
            self.stats.rejected += 1
            return []
        return self.tokenizer.encode(text, add_eos=not self.parse_special,
                                     parse_special=self.parse_special)


@dataclass
class PhasedState:
    payload: dict

    def to_dict(self):
        return self.payload


class PhasedLoader:
    def __init__(self, phases):
        if not phases or any(steps < 1 for _, steps, _ in phases):
            raise ValueError("phases need positive step counts")
        self.names, self.steps, self.loaders = map(list, zip(*phases, strict=True))
        self.phase = self.step_in_phase = 0
        self.seq_len = self.loaders[0].seq_len
        self.batch_size = self.loaders[0].batch_size
        if any((v.seq_len, v.batch_size) != (self.seq_len, self.batch_size) for v in self.loaders):
            raise ValueError("phases must share batch and sequence shapes")

    def total_steps(self):
        return sum(self.steps)

    def describe(self):
        return "; ".join(f"phase {name}: {steps} steps" for name, steps in zip(self.names, self.steps, strict=True)) + "; last phase open-ended (epoch caps still apply)"

    def batches(self, count=None):
        emitted = 0
        while count is None or emitted < count:
            if self.step_in_phase >= self.steps[self.phase] and self.phase < len(self.loaders) - 1:
                self.phase += 1
                self.step_in_phase = 0
            batch = next(self.loaders[self.phase].batches(1))
            self.step_in_phase += 1
            emitted += 1
            yield batch

    def state(self):
        return PhasedState({"phase": self.phase, "step_in_phase": self.step_in_phase,
                            "phases": [[n, s] for n, s in zip(self.names, self.steps, strict=True)],
                            "loaders": [loader.state().to_dict() for loader in self.loaders]})

    def validate_state(self, state):
        state = state.to_dict() if isinstance(state, PhasedState) else state
        if state.get("phases") != self.state().payload["phases"] or len(state.get("loaders", [])) != len(self.loaders):
            raise ValueError("checkpoint phases differ from this loader")
        phase, step = state.get("phase"), state.get("step_in_phase")
        if not isinstance(phase, int) or isinstance(phase, bool) or not 0 <= phase < len(self.loaders):
            raise ValueError("invalid phase")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or (phase < len(self.loaders) - 1 and step > self.steps[phase]):
            raise ValueError("invalid step_in_phase")
        for loader, saved in zip(self.loaders, state["loaders"], strict=True):
            loader.validate_state(saved)
        return PhasedState(state)

    def load_state(self, state):
        state = self.validate_state(state).payload
        for loader, saved in zip(self.loaders, state["loaders"], strict=True):
            loader.load_state(saved)
        self.phase, self.step_in_phase = state["phase"], state["step_in_phase"]

    restore = load_state


def phase_steps(mixture, *, batch_tokens):
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    return [max(1, round(mixture.total_tokens * p.weight / batch_tokens)) for p in mixture.phases]


def build_sources(phase, *, tokenizer, local_root=None, allow_hub=False,
                  overrides=None, decontaminator=None, max_epochs=4):
    sources = []
    for spec in phase.sources:
        source = (overrides or {}).get(spec.name)
        if source is None and local_root is not None:
            with suppress(FileNotFoundError):
                source = LocalTextSource.from_root(local_root, spec.name, spec.weight)
        if source is None:
            if not allow_hub:
                raise FileNotFoundError(f"source {spec.name!r} missing locally; use --hub to stream")
            source = HubSource(spec.name, spec.weight, spec.hf_id, config=spec.config, filters=spec.filters)
        sources.append(TokenisedSource(source, tokenizer, decontaminator=decontaminator,
                                       max_epochs=max_epochs))
    return sources


def build_loader(mixture, *, tokenizer, seq_len, batch_size=1, seed=0,
                 extra_sources=(), extra_phases=None, **kwargs):
    mixture.validate()
    names = [p.name for p in mixture.phases]
    extra_phases = [names[-1]] if extra_phases is None else extra_phases
    if set(extra_phases) - set(names):
        raise ValueError("extra sources specify unknown phases")
    steps = phase_steps(mixture, batch_tokens=seq_len * batch_size)
    phases = []
    for phase, n_steps in zip(mixture.phases, steps, strict=True):
        sources = build_sources(phase, tokenizer=tokenizer, **kwargs)
        if phase.name in extra_phases:
            import copy
            sources.extend(copy.deepcopy(extra_sources))
        phases.append((phase.name, n_steps, StreamingLoader(sources, seq_len=seq_len,
                                                           batch_size=batch_size, seed=seed)))
    return PhasedLoader(phases)
