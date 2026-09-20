"""Controlled learned-reinjection warm starts; experimental, disabled by default."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts import adapt_r04_depth as engine  # noqa: E402
from scripts.build_r04_reinjection_plan import configurations  # noqa: E402

PLAN_DIR = ROOT / "docs/experiments/2026-09-20-r04-reinjection-plan"
ADAPTER_KEY = "core_input_adapter.weight"


def configuration(parent, arm):
    engine.require(arm in ("fixed_sum", "learned_mix"), "unknown reinjection arm")
    engine.require(
        parent["recurrent"].get("input_adapter", "none") == "none",
        "parent already has an input adapter",
    )
    return configurations(parent)[arm]


def load_weights(model, parent):
    """Allow exactly the declared, zero-initialized new tensor; load all else strictly."""
    if model.core_input_adapter is None:
        model.load_state_dict(parent, strict=True)
        return
    target = model.state_dict()
    engine.require(
        set(target) - set(parent) == {ADAPTER_KEY} and not set(parent) - set(target),
        "parent keys differ beyond the declared adapter",
    )
    engine.require(
        torch.count_nonzero(target[ADAPTER_KEY]).item() == 0,
        "adapter must start exactly at zero",
    )
    model.load_state_dict({**parent, ADAPTER_KEY: target[ADAPTER_KEY]}, strict=True)


def specification():
    return engine.Experiment(
        plan_dir=PLAN_DIR,
        arms=("fixed_sum", "learned_mix"),
        protocol="r04-input-adapter-v1",
        config_factory=configuration,
        load_weights=load_weights,
        entrypoint=Path(__file__),
    )


def main():
    engine.main(experiment=specification())


if __name__ == "__main__":
    main()
