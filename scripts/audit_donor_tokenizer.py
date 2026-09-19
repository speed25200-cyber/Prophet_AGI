#!/usr/bin/env python3
"""Audit the plain-text adapter against the pinned local donor tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.donor_tokenizer import DonorByteTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the existing tokenizer audit")
    from transformers import AutoTokenizer
    reference = AutoTokenizer.from_pretrained(args.source, local_files_only=True, trust_remote_code=False)
    config = json.loads((args.source / "config.json").read_bytes())
    tokenizer = DonorByteTokenizer(args.source / "tokenizer.json", eos_id=reference.eos_token_id,
                                   pad_id=reference.pad_token_id, vocab_size=config["vocab_size"])
    rows = []
    paths = [args.validation] if args.validation.is_file() else sorted(args.validation.rglob("*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            normalized = unicodedata.normalize("NFC", text)
            ids = tokenizer.encode(text)
            same = ids == reference.encode(text, add_special_tokens=False)
            if not same:
                raise ValueError("validation includes a tokenization difference; inspect before training")
            if tokenizer.decode(ids) != normalized or tokenizer.byte_length(ids) != len(normalized.encode()):
                raise ValueError("donor payload byte accounting or roundtrip failed")
            rows.append({"sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": len(ids),
                         "normalized_bytes": len(normalized.encode()), "reference_ids_equal": same})
    probes = ["e\u0301 🍋 漢字\n\t", "<|im_end|> literal <|endoftext|>", ""]
    for text in probes:
        ids = tokenizer.encode(text, add_eos=True)
        normalized = unicodedata.normalize("NFC", text)
        if (ids[-1] != tokenizer.eos_id or tokenizer.eos_id in ids[:-1]
                or tokenizer.decode(ids) != normalized
                or tokenizer.byte_length(ids) != len(normalized.encode())):
            raise ValueError("Unicode/literal-control probe failed")
    if not rows:
        raise ValueError("no validation documents")
    report = {"complete": True, "tokenizer": tokenizer.fingerprint(), "documents": rows,
              "document_count": len(rows), "tokens_without_eos": sum(row["tokens"] for row in rows),
              "all_reference_ids_equal": True, "unicode_and_literal_control_probes_passed": True,
              "byte_protocol": "NFC-normalized UTF-8 payload; donor EOS/padding zero bytes",
              "scope": "tokenization/byte accounting only, not a model quality result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "documents"}, indent=2))


if __name__ == "__main__":
    main()
