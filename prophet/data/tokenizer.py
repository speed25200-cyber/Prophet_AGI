"""Prophet-Tok v1: deterministic byte-fallback BPE.

The identifier layout is part of the checkpoint format and must not be changed:

* ids 0..255 are the corresponding byte values;
* the next 256 ids are reserved for control and future modality tokens;
* learned BPE tokens start after the whole reserved range.

Pre-tokenisation deliberately makes ASCII digits and line endings indivisible units.
Consequently BPE training cannot create a token spanning either boundary.  Leading
indentation is also kept as one unit so common indentation depths can be learned as a
single token.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

N_BYTES = 256
N_RESERVED = 256
DEFAULT_VOCAB_SIZE = 32_768

# The order is frozen: these ids are serialized in datasets and checkpoints.  Most of
# the reserved range intentionally remains unused so later modalities do not require an
# embedding-table resize.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|mask|>",
    "<|idk|>",
    "<|retrieve|>",
    "<|img|>",
    "<|audio|>",
    "<|video|>",
    "<|think|>",
)

if len(SPECIAL_TOKENS) >= N_RESERVED:  # pragma: no cover - import-time guard
    raise RuntimeError("SPECIAL_TOKENS exhausts the reserved tokenizer range")

# This string is persisted verbatim.  Changing it is a tokenizer-format change, not a
# harmless refactor: an existing merge table was trained against these boundaries.
# Alternatives, in order: indentation, CR/LF, one ASCII digit, an optional leading
# space plus a Unicode word (with common contractions), other whitespace, punctuation.
PRE_TOKENIZE_PATTERN = (
    r"(?m:^[ \t]+)|\r|\n|[0-9]| ?[^\W\d]+(?:['’](?:s|t|re|ve|ll|d|m))?"
    r"|[^\S\r\n]+|[^\w\s]+"
)
_PRE_TOKENIZE_RE = re.compile(PRE_TOKENIZE_PATTERN)

Merge = tuple[bytes, bytes]


def pre_tokenize(text: str) -> list[str]:
    """Split *text* at boundaries across which BPE is forbidden to merge.

    NFC is the sole normalization, as specified by Prophet-Tok v1.  The fallback for a
    regex gap keeps the function total for unusual Unicode categories; ordinary input
    is entirely covered by the pattern.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text:
        return []

    text = unicodedata.normalize("NFC", text)
    units: list[str] = []
    cursor = 0
    for match in _PRE_TOKENIZE_RE.finditer(text):
        if match.start() > cursor:
            # One code point per unit is the safest possible fallback: it cannot create
            # a cross-class merge that the pattern meant to prohibit.
            units.extend(text[cursor : match.start()])
        units.append(match.group(0))
        cursor = match.end()
    if cursor < len(text):
        units.extend(text[cursor:])
    return units


def _merge_pair(symbols: Sequence[bytes], pair: Merge) -> tuple[bytes, ...]:
    """Replace non-overlapping occurrences of *pair*, from left to right."""

    left, right = pair
    merged = left + right
    output: list[bytes] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
            output.append(merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return tuple(output)


def _as_merge(value: object, *, index: int) -> Merge:
    """Validate and normalize one public merge entry."""

    if not isinstance(value, tuple | list) or len(value) != 2:
        raise TypeError(f"merge {index} must be a pair of byte strings")
    raw_left, raw_right = value
    if not isinstance(raw_left, bytes | bytearray | memoryview) or not isinstance(
        raw_right, bytes | bytearray | memoryview
    ):
        raise TypeError(f"merge {index} must be a pair of byte strings")
    left, right = bytes(raw_left), bytes(raw_right)
    if not left or not right:
        raise ValueError(f"merge {index} contains an empty symbol")
    return left, right


class BPETrainer:
    """Train a deterministic byte-level BPE merge table.

    The implementation aggregates equal pre-tokens, which keeps the reference trainer
    small while avoiding work proportional to repeated corpus text.  It is intended for
    reproducible vocabulary construction, not for online tokenization.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        *,
        min_frequency: int = 2,
    ) -> None:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer")
        if vocab_size <= N_BYTES + N_RESERVED:
            raise ValueError(
                "no room for merges: vocab_size must exceed "
                f"N_BYTES + N_RESERVED ({N_BYTES + N_RESERVED})"
            )
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise TypeError("min_frequency must be an integer")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train(self, corpus: Iterable[str]) -> list[Merge]:
        """Return merge rules learned from *corpus*, in application order."""

        if isinstance(corpus, str):
            corpus = (corpus,)

        sequences: Counter[tuple[bytes, ...]] = Counter()
        for document in corpus:
            if not isinstance(document, str):
                raise TypeError(f"corpus entries must be str, got {type(document).__name__}")
            for unit in pre_tokenize(document):
                encoded = unit.encode("utf-8")
                if encoded:
                    sequences[tuple(bytes((byte,)) for byte in encoded)] += 1

        target_merges = self.vocab_size - N_BYTES - N_RESERVED
        vocabulary = {bytes((byte,)) for byte in range(N_BYTES)}
        merges: list[Merge] = []

        while len(merges) < target_merges:
            counts: Counter[Merge] = Counter()
            for symbols, frequency in sequences.items():
                for left, right in zip(symbols, symbols[1:], strict=False):
                    if left + right not in vocabulary:
                        counts[(left, right)] += frequency

            if not counts:
                break

            # Highest frequency wins.  Lexicographic byte ordering makes ties stable
            # across processes, platforms and Counter insertion order.
            pair, frequency = min(
                counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            if frequency < self.min_frequency:
                break

            updated: Counter[tuple[bytes, ...]] = Counter()
            for symbols, occurrences in sequences.items():
                updated[_merge_pair(symbols, pair)] += occurrences
            sequences = updated
            merges.append(pair)
            vocabulary.add(pair[0] + pair[1])

        return merges


class ProphetTokenizer:
    """Encode and decode text with a Prophet-Tok v1 merge table."""

    _FORMAT = "prophet-tok-v1"
    _FORMAT_VERSION = 1
    _CACHE_LIMIT = 65_536

    def __init__(
        self,
        merges: Iterable[Sequence[bytes]],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> None:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer")
        if vocab_size < N_BYTES + N_RESERVED:
            raise ValueError(
                f"vocab_size must be at least {N_BYTES + N_RESERVED}, got {vocab_size}"
            )

        normalized_merges = [_as_merge(merge, index=index) for index, merge in enumerate(merges)]
        if len(normalized_merges) > vocab_size - N_BYTES - N_RESERVED:
            raise ValueError(
                f"{len(normalized_merges)} merges do not fit in vocab_size={vocab_size}"
            )

        self.vocab_size = vocab_size
        self.merges: tuple[Merge, ...] = tuple(normalized_merges)
        self._token_to_id: dict[bytes | str, int] = {
            bytes((byte,)): byte for byte in range(N_BYTES)
        }
        self._token_to_id.update(
            {token: N_BYTES + index for index, token in enumerate(SPECIAL_TOKENS)}
        )
        self._id_to_token: dict[int, bytes | str] = {
            token_id: token for token, token_id in self._token_to_id.items()
        }

        available = {bytes((byte,)) for byte in range(N_BYTES)}
        self._merge_ranks: dict[Merge, int] = {}
        for rank, pair in enumerate(self.merges):
            left, right = pair
            if left not in available or right not in available:
                raise ValueError(
                    f"merge {rank} refers to a symbol not produced by an earlier merge"
                )
            token = left + right
            if token in available:
                raise ValueError(f"merge {rank} creates duplicate token {token!r}")
            token_id = N_BYTES + N_RESERVED + rank
            self._merge_ranks[pair] = rank
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            available.add(token)

        self._special_to_id = {token: N_BYTES + index for index, token in enumerate(SPECIAL_TOKENS)}
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_id(self) -> int:
        return self.special_id("<|pad|>")

    @property
    def bos_id(self) -> int:
        return self.special_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.special_id("<|eos|>")

    @property
    def unk_id(self) -> int:
        return self.special_id("<|unk|>")

    def special_id(self, token: str) -> int:
        """Return the stable reserved id for *token*."""

        try:
            return self._special_to_id[token]
        except KeyError:
            raise KeyError(f"unknown special token: {token!r}") from None

    @property
    def valid_token_ids(self) -> frozenset[int]:
        """Return ids backed by a byte, named special token, or learned merge.

        The remaining ids are reserved capacity and must be masked by generation code.
        """

        return frozenset(self._id_to_token)

    def _encode_unit(self, unit: str) -> tuple[int, ...]:
        raw = unit.encode("utf-8")
        cached = self._encode_cache.get(raw)
        if cached is not None:
            return cached

        symbols: tuple[bytes, ...] = tuple(bytes((byte,)) for byte in raw)
        while len(symbols) > 1:
            best_pair: Merge | None = None
            best_rank = len(self.merges) + 1
            for pair in zip(symbols, symbols[1:], strict=False):
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            symbols = _merge_pair(symbols, best_pair)

        result = tuple(self._token_to_id[symbol] for symbol in symbols)
        if len(self._encode_cache) >= self._CACHE_LIMIT:
            self._encode_cache.clear()
        self._encode_cache[raw] = result
        return result

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode Unicode text; byte fallback makes ``<|unk|>`` unnecessary."""

        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        for unit in pre_tokenize(text):
            token_ids.extend(self._encode_unit(unit))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode ids, replacing malformed generated UTF-8 and skipping unused capacity."""

        output: list[str] = []
        buffer = bytearray()

        def flush() -> None:
            if buffer:
                # Encoded text is valid UTF-8 and still round-trips exactly. Generated byte
                # tokens need not form valid UTF-8, so replacement keeps decoding total.
                output.append(bytes(buffer).decode("utf-8", errors="replace"))
                buffer.clear()

        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token ids must be integers, got {token_id!r}")
            token = self._id_to_token.get(token_id)
            if token is None:
                if 0 <= token_id < self.vocab_size:
                    if not skip_special:
                        flush()
                        output.append(f"<|unused_{token_id}|>")
                    continue
                raise ValueError(f"unknown or unused token id: {token_id}")
            if isinstance(token, str):
                if not skip_special:
                    flush()
                    output.append(token)
            else:
                buffer.extend(token)
        flush()
        return "".join(output)

    def check_invariants(self) -> list[str]:
        """Return human-readable checkpoint-format or boundary violations."""

        problems: list[str] = []
        for byte in range(N_BYTES):
            if self._token_to_id.get(bytes((byte,))) != byte:
                problems.append(f"byte {byte} does not have id {byte}")

        for index, token in enumerate(SPECIAL_TOKENS):
            expected = N_BYTES + index
            if self._special_to_id.get(token) != expected:
                problems.append(f"special token {token!r} does not have id {expected}")

        for rank, (left, right) in enumerate(self.merges):
            token = left + right
            if any(ord("0") <= byte <= ord("9") for byte in token):
                problems.append(f"merge {rank} contains an ASCII digit")
            if b"\n" in token or b"\r" in token:
                problems.append(f"merge {rank} contains a newline")
            token_id = self._token_to_id.get(token)
            expected = N_BYTES + N_RESERVED + rank
            if token_id != expected:
                problems.append(f"merge {rank} has id {token_id}, expected {expected}")

        return problems

    def save(self, path: str | Path) -> None:
        """Atomically save the merge table and its segmentation contract as JSON."""

        destination = Path(path)
        payload = {
            "format": self._FORMAT,
            "format_version": self._FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "n_bytes": N_BYTES,
            "n_reserved": N_RESERVED,
            "special_tokens": list(SPECIAL_TOKENS),
            "normalization": "NFC",
            "pre_tokenize_pattern": PRE_TOKENIZE_PATTERN,
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a tokenizer, rejecting any incompatible segmentation contract."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tokenizer JSON in {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")
        if payload.get("format") != cls._FORMAT or payload.get("format_version") != 1:
            raise ValueError("unsupported tokenizer format or version")
        if payload.get("pre_tokenize_pattern") != PRE_TOKENIZE_PATTERN:
            raise ValueError("tokenizer was built with a different pre-tokenisation pattern")
        if payload.get("normalization") != "NFC":
            raise ValueError("tokenizer was built with a different normalization")
        if payload.get("n_bytes") != N_BYTES or payload.get("n_reserved") != N_RESERVED:
            raise ValueError("tokenizer uses a different byte/reserved id layout")
        if payload.get("special_tokens") != list(SPECIAL_TOKENS):
            raise ValueError("tokenizer uses a different special-token id layout")

        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("tokenizer merges must be a JSON list")
        merges: list[Merge] = []
        try:
            for entry in raw_merges:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("each serialized merge must contain two hex strings")
                left_hex, right_hex = entry
                if not isinstance(left_hex, str) or not isinstance(right_hex, str):
                    raise ValueError("serialized merge symbols must be hex strings")
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid serialized merge table: {exc}") from exc

        vocab_size = payload.get("vocab_size")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise ValueError("tokenizer vocab_size must be an integer")
        return cls(merges=merges, vocab_size=vocab_size)


__all__ = [
    "BPETrainer",
    "DEFAULT_VOCAB_SIZE",
    "N_BYTES",
    "N_RESERVED",
    "PRE_TOKENIZE_PATTERN",
    "ProphetTokenizer",
    "SPECIAL_TOKENS",
    "pre_tokenize",
]
