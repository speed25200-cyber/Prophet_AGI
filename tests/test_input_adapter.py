"""Behavioral gates for optional learned reinjection, before any architecture adoption."""

import copy

import pytest
import torch

from prophet.budget import count_parameters, inference_profile, tokens_affordable, training_memory
from prophet.config import ProphetConfig
from prophet.modeling.model import ProphetCache, ProphetModel
from scripts.gate_r04_input_adapter import compare_initial
from tests.test_modeling import tiny_config


def pair():
    base = tiny_config()
    base.recurrent.truncated_backprop_steps = 8
    adapted = copy.deepcopy(base)
    adapted.recurrent.input_adapter = "residual_linear"
    torch.manual_seed(123)
    control = ProphetModel(base)
    torch.manual_seed(123)
    model = ProphetModel(adapted)
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in control.state_dict().items()
    )
    return control, model


@pytest.mark.parametrize("depth", [1, 4, 8])
def test_zero_adapter_preserves_parent_predictions(depth):
    control, model = pair()
    control.eval()
    model.eval()
    inputs = torch.tensor([[1, 2, 8, 5, 7, 3, 2]])
    assert torch.count_nonzero(model.core_input_adapter.weight) == 0
    with torch.no_grad():
        a = control(inputs, loop_k=depth, return_mtp=False)
        b = model(inputs, loop_k=depth, return_mtp=False)
    assert torch.equal(a.hidden, b.hidden)
    assert torch.equal(a.logits, b.logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA BF16 zero adapter")
def test_cuda_bfloat16_zero_adapter_preserves_predictions():
    control, model = pair()
    control, model = control.cuda().eval(), model.cuda().eval()
    inputs = torch.tensor([[1, 2, 8, 5, 7, 3, 2]], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = control(inputs, loop_k=6, return_mtp=False)
        b = model(inputs, loop_k=6, return_mtp=False)
    assert torch.equal(a.hidden, b.hidden)
    assert torch.equal(a.logits, b.logits)


def test_initial_equality_gate_detects_changed_predictions():
    control, model = pair()
    ids = torch.tensor([[1, 2, 8, 5, 7, 3, 2]])
    assert compare_initial(control, model, ids)["passed"]
    torch.nn.init.normal_(model.core_input_adapter.weight, std=0.02)
    assert not compare_initial(control, model, ids)["passed"]


def test_adapter_receives_gradient_without_changing_initial_parent_gradients():
    control, model = pair()
    inputs = torch.tensor([[1, 2, 8, 5, 7, 3, 2]])
    for network in (control, model):
        torch.manual_seed(17)
        network(inputs, loop_k=4, return_mtp=False).logits.square().mean().backward()
    original_parameters = dict(control.named_parameters())
    for name, parameter in model.named_parameters():
        if name == "core_input_adapter.weight":
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad[:, : model.cfg.d_model].abs().sum() > 0
            assert parameter.grad[:, model.cfg.d_model :].abs().sum() > 0
        else:
            original = original_parameters[name]
            assert (parameter.grad is None) == (original.grad is None)
            if original.grad is not None:
                assert torch.equal(parameter.grad, original.grad), name
    model.eval()
    with torch.no_grad():
        before = model(inputs, loop_k=4, return_mtp=False).logits.clone()
    optimizer = torch.optim.SGD([model.core_input_adapter.weight], lr=0.01)
    optimizer.step()
    with torch.no_grad():
        after = model(inputs, loop_k=4, return_mtp=False).logits
    assert not torch.equal(before, after)


@pytest.mark.parametrize("depth", [1, 4, 8])
def test_learned_adapter_preserves_full_incremental_equivalence_and_cache_size(depth):
    control, model = pair()
    model.eval()
    control.eval()
    torch.nn.init.normal_(model.core_input_adapter.weight, std=0.02)
    inputs = torch.tensor([[1, 2, 8, 5, 7, 3, 2, 6, 9, 4, 2, 1]])
    with torch.no_grad():
        full = model(inputs, loop_k=depth, return_mtp=False).logits
        cache = ProphetCache()
        pieces = [
            model(inputs[:, i : i + 1], cache=cache, loop_k=depth, return_mtp=False).logits
            for i in range(inputs.shape[1])
        ]
        original_cache = ProphetCache()
        control(inputs, loop_k=depth, cache=original_cache, return_mtp=False)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=2e-5, rtol=1e-5)
    assert cache.summary() == original_cache.summary()


def test_learned_adapter_checkpointing_preserves_backward():
    _, model = pair()
    torch.nn.init.normal_(model.core_input_adapter.weight, std=0.02)
    checkpointed = copy.deepcopy(model)
    checkpointed.gradient_checkpointing = True
    inputs = torch.tensor([[1, 2, 8, 5, 7, 3, 2]])
    outputs = []
    for network in (model, checkpointed):
        torch.manual_seed(18)
        output = network(inputs, loop_k=4, return_mtp=False).logits
        output.square().mean().backward()
        outputs.append(output)
    assert torch.equal(*outputs)
    for a, b in zip(model.parameters(), checkpointed.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("change", ["unknown", "no_recurrence", "no_injection"])
def test_invalid_adapter_configuration_rejected(change):
    cfg = tiny_config()
    cfg.recurrent.input_adapter = "unknown" if change == "unknown" else "residual_linear"
    if change == "no_recurrence":
        cfg.recurrent.enabled = False
    if change == "no_injection":
        cfg.recurrent.inject_input_each_step = False
    with pytest.raises(ValueError, match="input_adapter"):
        cfg.validate()


def test_old_configuration_defaults_to_unchanged_topology():
    cfg = tiny_config().to_dict()
    del cfg["recurrent"]["input_adapter"]
    restored = ProphetConfig.from_dict(cfg)
    assert restored.recurrent.input_adapter == "none"
    assert ProphetModel(restored).core_input_adapter is None


def test_budget_counts_unique_adapter_and_each_execution():
    control, model = pair()
    added = 2 * model.cfg.d_model**2
    assert count_parameters(model.cfg).total == sum(p.numel() for p in model.parameters())
    assert count_parameters(model.cfg).total - count_parameters(control.cfg).total == added
    assert count_parameters(model.cfg).by_component["recurrent/input_adapter"] == added
    for depth in (1, 4, 8):
        a = inference_profile(control.cfg, loop_k=depth)
        b = inference_profile(model.cfg, loop_k=depth)
        assert b.flops_per_token - a.flops_per_token == pytest.approx(2 * added * depth)
        assert a.kv_state_gb == b.kv_state_gb
        a_train = tokens_affordable(control.cfg, loop_k_train=depth)
        b_train = tokens_affordable(model.cfg, loop_k_train=depth)
        assert b_train["flops_per_token"] - a_train["flops_per_token"] == pytest.approx(
            6 * added * depth
        )
    assert training_memory(model.cfg).total_gb > training_memory(control.cfg).total_gb
