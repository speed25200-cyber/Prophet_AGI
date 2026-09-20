"""Pinned ByteLevel donor vocabulary for plain-text recovery experiments.

Control-looking strings in a corpus remain ordinary text. Added-token matching
and post-processing are disabled; an explicit donor EOS is appended separately.
This adapter does not implement Prophet's action/control-token protocol.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


class DonorByteTokenizer:
    def __init__(self, path: str | Path, *, eos_id: int, pad_id: int, vocab_size: int):
        from tokenizers import Tokenizer

        source = Path(path).read_bytes()
        specification = json.loads(source)
        if specification.get("model", {}).get("type") != "BPE":
            raise ValueError("recovery tokenizer requires a BPE model")
        if specification.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("exact byte accounting requires a ByteLevel decoder")
        if specification.get("normalizer") not in (None, {"type": "NFC"}):
            raise ValueError("only identity or NFC normalization has been audited")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        for value in (eos_id, pad_id):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < vocab_size:
                raise ValueError("EOS/padding ids must be within the model vocabulary")
        # The GPT ByteLevel alphabet maps each raw byte to exactly one Unicode
        # codepoint. Counting these codepoints therefore counts bytes, even when
        # a Unicode character is split across several BPE tokens.
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = {chr(i) for i in byte_values}
        alphabet.update(chr(256 + i) for i in range(256 - len(byte_values)))
        self._lengths = {}
        for token, index in specification["model"]["vocab"].items():
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < vocab_size:
                raise ValueError("BPE id lies outside the model vocabulary")
            if not token or not all(character in alphabet for character in token):
                raise ValueError("BPE contains a token outside the byte alphabet")
            if index in self._lengths:
                raise ValueError("duplicate BPE id")
            self._lengths[index] = len(token)
        if eos_id in self._lengths or pad_id in self._lengths:
            raise ValueError("EOS/padding ids must be separate from payload BPE ids")
        self._lengths[eos_id] = self._lengths[pad_id] = 0
        self.eos_id, self.pad_id, self.vocab_size = eos_id, pad_id, vocab_size
        self.n_tokens = max(self._lengths) + 1
        self._identity = {
            "format": "donor-bytelevel-raw-text-v1",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "eos_id": eos_id, "pad_id": pad_id, "vocab_size": vocab_size,
            "normalizer": specification.get("normalizer"),
            "policy": "no added-token recognition; no postprocessor; explicit EOS",
        }
        specification["added_tokens"] = []
        specification["post_processor"] = None
        specification["padding"] = None
        specification["truncation"] = None
        self._backend = Tokenizer.from_str(json.dumps(specification))

    def fingerprint(self):
        return deepcopy(self._identity)

    def encode(self, text: str, *, add_eos: bool = False, parse_special: bool = False):
        if parse_special:
            raise ValueError("donor recovery only supports raw text, not Prophet control tokens")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if any(index not in self._lengths for index in ids):
            raise ValueError("tokenizer emitted an unaudited token")
        return ids + [self.eos_id] if add_eos else ids

    def byte_length(self, ids):
        try:
            return sum(self._lengths[index] for index in ids)
        except KeyError as error:
            raise ValueError("unknown token in byte accounting") from error

    def decode(self, ids):
        return self._backend.decode([i for i in ids if i not in (self.eos_id, self.pad_id)],
                                    skip_special_tokens=False)
