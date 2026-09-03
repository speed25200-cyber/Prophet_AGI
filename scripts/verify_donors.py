#!/usr/bin/env python3
"""Check every donor spec against the Hub's config.json.

The architecture figures in ``prophet/convert/donors.py`` were written while
huggingface.co was unreachable from the build environment, so every spec carries
``verified=False`` and the conversion script refuses to run without an explicit override.
This is what clears that flag.

A wrong ``head_dim`` or ``ffn_hidden`` does not fail loudly during conversion -- it
produces shape mismatches that are silently left at fresh initialisation, so the
conversion appears to succeed and the model is simply much worse than it should be.

    python scripts/verify_donors.py
    python scripts/verify_donors.py --donor qwen3-1.7b
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.convert.donors import DONORS, DonorSpec  # noqa: E402

CONFIG_URL = "https://huggingface.co/{hf_id}/resolve/main/config.json"
TIMEOUT = 30

#: Spec field -> the config.json key it should equal.
FIELD_MAP: dict[str, str] = {
    "n_layers": "num_hidden_layers",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "ffn_hidden": "intermediate_size",
    "vocab_size": "vocab_size",
    "tie_word_embeddings": "tie_word_embeddings",
}


def fetch_config(hf_id: str) -> dict | None:
    try:
        req = urllib.request.Request(
            CONFIG_URL.format(hf_id=hf_id), headers={"User-Agent": "prophet-donor-verifier"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            return json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None


def compare(spec: DonorSpec, config: dict) -> list[str]:
    problems: list[str] = []
    for field, key in FIELD_MAP.items():
        declared = getattr(spec, field)
        actual = config.get(key)
        if actual is None:
            if field == "head_dim":
                # Some configs omit it and imply hidden_size / num_attention_heads.
                actual = config.get("hidden_size", 0) // max(
                    config.get("num_attention_heads", 1), 1
                )
            else:
                problems.append(f"{field}: config.json has no {key!r}")
                continue
        if declared != actual:
            problems.append(f"{field}: declared {declared}, Hub says {actual}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--donor", choices=sorted(DONORS), help="check one donor")
    args = ap.parse_args()

    targets = {args.donor: DONORS[args.donor]} if args.donor else DONORS
    unreachable: list[str] = []
    failures: dict[str, list[str]] = {}
    verified: list[str] = []

    for key, spec in targets.items():
        config = fetch_config(spec.hf_id)
        if config is None:
            unreachable.append(key)
            continue
        problems = compare(spec, config)
        if problems:
            failures[key] = problems
        else:
            verified.append(key)

    if unreachable and not verified and not failures:
        print(
            "huggingface.co is not reachable from here, so nothing could be checked.\n"
            "Run this where the Hub is reachable before converting any donor weights.",
            file=sys.stderr,
        )
        return 2

    for key in verified:
        print(f"OK        {key}: every field matches the Hub")
    for key, problems in failures.items():
        print(f"MISMATCH  {key}:")
        for problem in problems:
            print(f"            {problem}")
    for key in unreachable:
        print(f"SKIPPED   {key}: could not fetch config.json")

    if verified and not failures:
        print(
            "\nAll checked donors match. Set verified=True on them in "
            "prophet/convert/donors.py."
        )
    return 1 if failures or unreachable else 0


if __name__ == "__main__":
    raise SystemExit(main())
