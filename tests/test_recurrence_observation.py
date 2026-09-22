"""Closed-form state geometry and an observer that leaves real model behavior intact."""

import copy

import pytest
import torch

from prophet.modeling.model import ProphetModel
from scripts.diagnose_r04_recurrence import observe, state_metrics
from tests.test_training import tiny_model_config


def test_geometry_distinguishes_identical_and_orthogonal_tokens():
    identical = state_metrics([torch.ones(3, 3)])[0]
    orthogonal = state_metrics([torch.eye(3), 2 * torch.eye(3)])
    assert identical["mean_distinct_token_cosine"] == pytest.approx(1)
    assert identical["token_centered_energy_fraction"] == 0
    assert orthogonal[0]["mean_distinct_token_cosine"] == 0
    assert orthogonal[0]["token_centered_energy_fraction"] == pytest.approx(2 / 3)
    assert orthogonal[1]["state_rms"] == pytest.approx(2 / 3**0.5)
    assert orthogonal[1]["previous_loop_cosine"] == 1
    assert orthogonal[1]["relative_previous_loop_change"] == 1


def test_pair_mean_matches_explicit_distinct_pairs():
    values = torch.tensor([[1.0, 2, 3], [2.0, -1, 4], [-2.0, 3, 1]], dtype=torch.float64)
    unit = values / values.norm(dim=-1, keepdim=True)
    pairwise = unit @ unit.T
    expected = pairwise[~torch.eye(3, dtype=torch.bool)].mean().item()
    assert state_metrics([values])[0]["mean_distinct_token_cosine"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "value", [torch.zeros(2, 3), torch.ones(1, 3), torch.full((2, 3), float("nan"))]
)
def test_invalid_geometry_rejected(value):
    with pytest.raises(ValueError):
        state_metrics([value])


@pytest.mark.parametrize("pattern", [["swa", "full_attn"], ["gdn"]])
def test_observer_leaves_outputs_weights_and_rng_unchanged(pattern):
    cfg = tiny_model_config()
    cfg.mixer.pattern = pattern
    model = ProphetModel(cfg).eval()
    inputs = torch.tensor([[1, 2, 3, 4, 0, 0]])
    weights = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = model(inputs, loop_k=4, return_mtp=False)
    observed, metrics = observe(model, inputs, 4, 4)
    assert torch.equal(expected.hidden, observed.hidden)
    assert torch.equal(expected.logits, observed.logits)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert len(metrics) == 4 and all(row["tokens"] == 4 for row in metrics)
    assert not model.sections["core"][-1]._forward_hooks


def test_observer_removed_after_failure(monkeypatch):
    model = ProphetModel(tiny_model_config()).eval()

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        observe(model, torch.tensor([[1, 2, 3]]), 3, 4)
    assert not model.sections["core"][-1]._forward_hooks
