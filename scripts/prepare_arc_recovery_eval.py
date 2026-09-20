#!/usr/bin/env python3
"""Freeze the full pinned ARC-Easy test split and its raw-text scoring contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.eval.choices import encode_choice, token_ids_sha256  # noqa: E402
from scripts.recover_qwen import digest, load_source, write_report  # noqa: E402

SOURCE = {"hf_id": "allenai/ai2_arc", "revision": "210d026faf9955653af8916fad021475a3f00453",
          "config": "ARC-Easy", "split": "test", "license": "cc-by-sa-4.0"}
PROMPT = "Question: {question}\nAnswer:"


def prepare_rows(rows, tokenizer, *, max_tokens=512):
    prepared, seen = [], set()
    for row in rows:
        labels, choices = row["choices"]["label"], row["choices"]["text"]
        if (not isinstance(row["id"], str) or not row["id"] or row["id"] in seen
                or not isinstance(row["question"], str) or not row["question"].strip()
                or not 2 <= len(choices) <= 5 or len(labels) != len(choices)
                or len(set(labels)) != len(labels) or row["answerKey"] not in labels
                or any(not isinstance(choice, str) for choice in choices)):
            raise ValueError("invalid or duplicate ARC row")
        seen.add(row["id"])
        prompt = PROMPT.format(question=row["question"])
        tokens = []
        for choice in choices:
            ids, start = encode_choice(tokenizer, prompt, choice, max_tokens=max_tokens)
            tokens.append({"input_ids_sha256": token_ids_sha256(ids), "tokens": len(ids),
                           "context_tokens": start, "answer_tokens": len(ids)-start,
                           "answer_bytes": tokenizer.byte_length(ids[start:])})
        raw = {key: row[key] for key in ("id", "question", "choices", "answerKey")}
        prepared.append({"id": row["id"], "prompt": prompt, "choices": choices,
                         "gold": labels.index(row["answerKey"]), "tokens": tokens,
                         "source_row_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True,
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()})
    if not prepared:
        raise ValueError("empty ARC split")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the frozen evaluation set")
    _, tokenizer = load_source(args.source)
    from datasets import load_dataset
    rows = load_dataset(SOURCE['hf_id'], SOURCE['config'], split=SOURCE['split'],
                        revision=SOURCE['revision'], token=False)
    prepared = prepare_rows(rows, tokenizer)
    if len(prepared) != 2376:
        raise ValueError("pinned ARC-Easy test row count changed")
    args.out.mkdir(parents=True)
    path = args.out/'items.jsonl'
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    manifest = {'complete': True, 'protocol': 'arc-easy-raw-choice-v1', 'source': SOURCE,
                'rows': len(prepared), 'items_sha256': digest(path), 'tokenizer': tokenizer.fingerprint(),
                'max_tokens': 512, 'max_observed_tokens': max(t['tokens'] for r in prepared for t in r['tokens']),
                'total_candidate_tokens': sum(t['tokens'] for r in prepared for t in r['tokens']),
                'prompt_template': PROMPT, 'continuation_prefix': ' ', 'fewshot': 0,
                'scoring': 'continuation-only summed nats; character-normalized nats divide by original choice length',
                'tie_policy': 'first source choice; report ties',
                'scope': 'Full exploratory test split; no chat, truncation, cached generation or instruction-following claim. Prior lexical overlap screen is incomplete; donor pretraining contamination is unknown.'}
    write_report(args.out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
