#!/usr/bin/env python3
"""Plan and execute a donor-to-Prophet conversion.

    # inspect the plan without downloading anything
    python scripts/convert_donor.py --donor qwen3-1.7b --plan-only

    # convert real weights
    python scripts/convert_donor.py --donor qwen3-1.7b \
        --weights /path/to/donor/model.safetensors --out checkpoints/prophet-main-init.pt

``--plan-only`` is the intended first step every time: it prints which donor layer
initialises which Prophet block, the parameter coverage, and every warning, at a cost of
zero bytes downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.budget import count_parameters  # noqa: E402
from prophet.convert.donors import (  # noqa: E402
    DONORS,
    LicenceProblem,
    assert_donor_is_usable,
    get_donor,
)
from prophet.convert.plan import plan_conversion, prophet_config_for_donor  # noqa: E402
from prophet.convert.weights import convert_state_dict  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402

MINIMUM_COVERAGE = 0.50
"""Below this a 'conversion' is really pretraining with a warm start, and the recovery
budget is wrong by an order of magnitude. Better to know before committing."""


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--donor", required=True, choices=sorted(DONORS))
    ap.add_argument("--weights", type=Path, help="donor .safetensors or .pt")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/prophet-init.pt"))
    ap.add_argument("--prelude", type=int, default=4)
    ap.add_argument("--core", type=int, default=4)
    ap.add_argument("--coda", type=int, default=4)
    ap.add_argument("--loop-k", type=int, default=5)
    ap.add_argument("--core-init", default="average", choices=["average", "stride", "first"])
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--allow-restricted-licence", action="store_true",
                    help="proceed with a donor whose licence follows the derivative")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="proceed although the donor's architecture figures are unchecked")
    args = ap.parse_args()

    donor = get_donor(args.donor)
    try:
        assert_donor_is_usable(donor, allow_restricted=args.allow_restricted_licence)
    except LicenceProblem as problem:
        print(f"\n{problem}", file=sys.stderr)
        return 1

    cfg = prophet_config_for_donor(
        donor,
        prelude_layers=args.prelude,
        core_layers=args.core,
        coda_layers=args.coda,
        loop_k=args.loop_k,
    )
    plan = plan_conversion(donor, cfg, core_init=args.core_init)
    print(plan.report())

    params = count_parameters(cfg)
    print()
    print(f"Donor    {donor.params_estimate / 1e9:.2f}B parameters, "
          f"{donor.n_layers} layers")
    print(f"Prophet  {params.total / 1e9:.2f}B total, "
          f"{params.active_per_token / 1e6:.0f}M active per token, "
          f"effective depth {cfg.effective_depth()} from "
          f"{cfg.parameterised_depth()} parameterised blocks")

    coverage = plan.coverage()["coverage"]
    if coverage < MINIMUM_COVERAGE:
        print(
            f"\nRefusing to convert: only {coverage:.0%} of parameters come from the "
            f"donor, below the {MINIMUM_COVERAGE:.0%} floor. At this coverage the run is "
            "pretraining with a warm start, and should be budgeted as such.",
            file=sys.stderr,
        )
        return 1

    if not donor.verified and not args.allow_unverified and not args.plan_only:
        print(
            f"\nRefusing to convert: {donor.name}'s architecture figures have not been "
            "checked against the Hub. Run scripts/verify_donors.py first, or pass "
            "--allow-unverified.",
            file=sys.stderr,
        )
        return 1

    if args.plan_only:
        return 0
    if args.weights is None:
        print("\n--weights is required unless --plan-only is given.", file=sys.stderr)
        return 2

    print(f"\nloading donor weights from {args.weights}")
    donor_state = load_weights(args.weights)

    model = ProphetModel(cfg)
    target = {k: v.clone() for k, v in model.state_dict().items()}
    converted, report = convert_state_dict(donor_state, plan, target)
    model.load_state_dict(converted)

    print()
    print(report.summary())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "donor": donor.hf_id, "core_init": args.core_init}, args.out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    print("This is an initialisation, not a working model: recovery training is what "
          "makes it one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
