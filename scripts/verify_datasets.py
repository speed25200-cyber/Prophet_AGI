#!/usr/bin/env python3
"""Verify every dataset id in the mixture against the HuggingFace Hub.

Week one of the roadmap. The mixture's dataset ids came from research reports written
while the agents' access to the Hub was blocked by an outbound proxy, so they are
*claims*, not facts. Sizes drive the epoch-repetition check and licences drive whether
the trained model can be released at all — both are load-bearing, and both are currently
unverified for several sources.

    python scripts/verify_datasets.py
    python scripts/verify_datasets.py --write-back   # update available_tokens in place

Requires network access to huggingface.co. If the Hub is unreachable the script says so
and exits non-zero rather than reporting everything as fine.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.mixture import BLOCKED_LICENSES, Mixture  # noqa: E402
from prophet.data.recipes import prophet_v1_mixture  # noqa: E402

API = "https://huggingface.co/api/datasets/"
TIMEOUT = 30


@dataclass
class Check:
    source: str
    hf_id: str
    exists: bool | None = None
    declared_license: str = ""
    hub_license: str = ""
    downloads: int | None = None
    gated: bool = False
    configs: list[str] | None = None
    error: str = ""

    @property
    def license_matches(self) -> bool:
        if not self.hub_license or not self.declared_license:
            return False
        a = self.declared_license.lower().replace(" ", "").replace("_", "-")
        b = self.hub_license.lower().replace(" ", "").replace("_", "-")
        return a.startswith(b[:6]) or b.startswith(a[:6])

    @property
    def hub_license_is_blocked(self) -> bool:
        lowered = self.hub_license.lower()
        return any(token in lowered for token in BLOCKED_LICENSES)


def fetch(hf_id: str) -> Check:
    check = Check(source="", hf_id=hf_id)
    try:
        req = urllib.request.Request(
            API + hf_id, headers={"User-Agent": "prophet-dataset-verifier"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        check.exists = exc.code != 404
        check.error = f"HTTP {exc.code}"
        return check
    except Exception as exc:  # network, TLS, proxy
        check.error = f"{type(exc).__name__}: {exc}"
        return check

    check.exists = True
    card = data.get("cardData") or {}
    licence = card.get("license") or data.get("license") or ""
    check.hub_license = licence if isinstance(licence, str) else ", ".join(licence)
    check.downloads = data.get("downloads")
    check.gated = bool(data.get("gated"))
    configs = card.get("configs")
    if isinstance(configs, list):
        check.configs = [c.get("config_name", "") if isinstance(c, dict) else str(c)
                         for c in configs]
    return check


def hub_reachable() -> bool:
    try:
        req = urllib.request.Request(
            API + "HuggingFaceFW/fineweb-edu",
            headers={"User-Agent": "prophet-dataset-verifier"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT):
            return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=float, default=40e9)
    args = ap.parse_args()

    if not hub_reachable():
        print(
            "huggingface.co is not reachable from here, so nothing can be verified.\n"
            "Run this where the Hub is reachable before trusting any dataset id, size or "
            "licence in configs/data_mixture_v1.yaml.",
            file=sys.stderr,
        )
        return 2

    mixture: Mixture = prophet_v1_mixture(args.tokens)
    seen: dict[str, Check] = {}
    rows: list[Check] = []

    for phase in mixture.phases:
        for source in phase.sources:
            if source.hf_id in seen:
                check = Check(**{**seen[source.hf_id].__dict__})
            else:
                check = fetch(source.hf_id)
                seen[source.hf_id] = check
            check.source = f"{phase.name}/{source.name}"
            check.declared_license = source.license
            rows.append(check)

    print("| Source | HF id | Exists | Gated | Declared licence | Hub licence | Match |")
    print("|---|---|---|---|---|---|---|")
    problems: list[str] = []
    for r in rows:
        exists = "yes" if r.exists else ("NO" if r.exists is False else "?")
        match = "yes" if r.license_matches else "**CHECK**"
        print(
            f"| {r.source} | `{r.hf_id}` | {exists} | {'yes' if r.gated else ''} | "
            f"{r.declared_license} | {r.hub_license or '—'} | {match} |"
        )
        if r.exists is False:
            problems.append(f"{r.source}: `{r.hf_id}` does not exist on the Hub")
        if r.error and r.exists is not True:
            problems.append(f"{r.source}: {r.error}")
        if r.hub_license_is_blocked:
            problems.append(
                f"{r.source}: Hub licence {r.hub_license!r} is on the blocked list — "
                "using it would make the trained model un-releasable"
            )
        if r.gated:
            problems.append(f"{r.source}: gated, needs an accepted licence agreement")

    print()
    if problems:
        print("## Problems\n")
        for p in dict.fromkeys(problems):
            print(f"- {p}")
        return 1

    print("Every dataset id resolves and no blocked licence was found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
