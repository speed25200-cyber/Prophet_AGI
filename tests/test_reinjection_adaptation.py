"""Real training/restart gates for the new component experiment."""

import json

import pytest
import torch

from prophet.modeling.model import ProphetModel
from scripts import adapt_r04_reinjection as driver
from scripts.audit_r04_restart import audit, load_run
from tests.test_depth_adaptation import assert_equal, fixture, restore_numerical_flags  # noqa: F401
from tests.test_input_adapter import pair


@pytest.mark.parametrize("arm", ["fixed_sum", "learned_mix"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="actual CUDA adapter restart"
            ),
        ),
    ],
)
def test_component_cli_exact_warm_start_and_restart(tmp_path, monkeypatch, arm, device):
    run, parent, _, _ = fixture(tmp_path, monkeypatch, device, driver.specification())
    resumed, continuous = tmp_path / "resumed", tmp_path / "continuous"
    run(resumed, arm, 1)
    initial = torch.load(
        resumed / "checkpoints/ckpt_slot0.pt", weights_only=True, map_location="cpu"
    )
    for key, value in parent["model"].items():
        assert_equal(initial["model"][key], value.cpu())
    if arm == "learned_mix":
        assert torch.count_nonzero(initial["model"][driver.ADAPTER_KEY]) == 0
        assert set(initial["model"]) - set(parent["model"]) == {driver.ADAPTER_KEY}
    else:
        assert set(initial["model"]) == set(parent["model"])
    assert initial["loader"] == parent["loader"]
    assert all(not opt["state"] for opt in initial["optimizers"])
    assert initial["training_contract"]["run_identity"]["protocol"] == "r04-input-adapter-v1"
    other = "fixed_sum" if arm == "learned_mix" else "learned_mix"
    with pytest.raises(ValueError, match="another adaptation experiment"):
        run(resumed, other)
    run(resumed, arm)
    run(continuous, arm)
    assert audit(resumed, continuous, 4)["passed"]
    final, _ = load_run(resumed, 4)
    assert len(set(final["adaptation_depth_history"])) > 1
    if arm == "learned_mix":
        assert torch.isfinite(final["model"][driver.ADAPTER_KEY]).all()
        assert torch.count_nonzero(final["model"][driver.ADAPTER_KEY]) > 0
    report = resumed / "evaluation-step-000004.json"
    before = report.read_bytes()
    run(resumed, arm)
    assert report.read_bytes() == before


@pytest.mark.parametrize(
    "corruption", ["missing_base", "unexpected", "existing_adapter", "nonzero_init"]
)
def test_warm_start_rejects_undeclared_changes(corruption):
    base, model = pair()
    weights = base.state_dict()
    if corruption == "missing_base":
        weights.pop(next(iter(weights)))
    elif corruption == "unexpected":
        weights["unexpected"] = torch.zeros(1)
    elif corruption == "existing_adapter":
        weights[driver.ADAPTER_KEY] = torch.zeros_like(model.core_input_adapter.weight)
    else:
        torch.nn.init.ones_(model.core_input_adapter.weight)
    with pytest.raises(ValueError, match="adapter"):
        driver.load_weights(model, weights)


def test_published_plan_matches_actual_topologies():
    protocol = json.loads((driver.PLAN_DIR / "protocol.json").read_bytes())
    parent = json.loads(
        (
            driver.ROOT
            / "docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json"
        ).read_bytes()
    )
    for arm in driver.specification().arms:
        cfg = driver.configuration(parent["run_protocol"]["config"], arm)
        assert json.loads(json.dumps(cfg.to_dict())) == json.loads(
            (driver.PLAN_DIR / f"{arm}.json").read_bytes()
        )
        with torch.device("meta"):
            model = ProphetModel(cfg)
        assert sum(p.numel() for p in model.parameters()) == protocol["parameters_each"][arm]
