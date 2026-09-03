"""Regenerate docs/02_DATA.md from the data recipe.

The document is generated so that the mixture in the docs cannot drift from the mixture
the loader actually uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.data.recipes import prophet_v1_mixture  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    mixture = prophet_v1_mixture(40e9)
    mixture.validate()
    mixture.to_yaml(ROOT / "configs" / "data_mixture_v1.yaml")
    print(f"wrote configs/data_mixture_v1.yaml ({mixture.total_tokens / 1e9:.0f}B tokens)")
    print(mixture.report())


if __name__ == "__main__":
    main()
