"""Analytic guards for the experimental depth-specific residual projection."""

import pytest
import torch
from torch import nn

from scripts.probe_qwen_relaxed import ResidualLinear, install_residuals


def test_rank_sweep_recovers_known_singular_directions():
    base = nn.Parameter(torch.eye(3))
    target = base.detach() + torch.diag(torch.tensor([9.0, 4.0, 1.0]))
    module = ResidualLinear(base, target, 3)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    expected = ([1.0, 2.0, 3.0], [10.0, 2.0, 3.0], [10.0, 10.0, 3.0], [10.0, 10.0, 6.0])
    for rank, answer in enumerate(expected):
        module.set_rank(rank)
        torch.testing.assert_close(module(x), torch.tensor([answer]))


def test_shared_base_accumulates_gradients_from_distinct_depths():
    base = nn.Parameter(torch.zeros(2, 2))
    first = ResidualLinear(base, torch.diag(torch.tensor([2.0, 1.0])), 2)
    second = ResidualLinear(base, torch.diag(torch.tensor([1.0, 3.0])), 2)
    x = torch.tensor([[2.0, 4.0]])
    y = torch.tensor([[3.0, 5.0]])
    assert first.base is second.base
    assert first.up is not second.up and first.down is not second.down
    loss = first(x).sum() + 2 * second(y).sum()
    loss.backward()
    torch.testing.assert_close(base.grad, torch.tensor([[8.0, 14.0], [8.0, 14.0]]))
    assert first.up.grad.abs().sum() > 0 and second.down.grad.abs().sum() > 0
    first.set_rank(0)
    torch.testing.assert_close(first(x), torch.zeros(1, 2))
    torch.testing.assert_close(second(y), torch.tensor([[3.0, 15.0]]))


def test_rectangular_full_rank_matches_dense_projection_and_input_gradient():
    torch.manual_seed(7)
    base = nn.Parameter(torch.randn(4, 3))
    target = torch.randn(4, 3)
    module = ResidualLinear(base, target, 3)
    x = torch.randn(2, 5, 3, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_()
    actual = module(x)
    expected = torch.nn.functional.linear(reference_input, target)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_input.grad, atol=2e-5, rtol=3e-6)
    with pytest.raises(ValueError):
        module.set_rank(4)
    with pytest.raises(ValueError):
        module.set_rank(-1)


def test_full_rank_native_qwen_reconstruction_preserves_complete_forward():
    # Same 28-layer native topology as the real probe, with smaller dimensions.
    # This guards the layer mapping and replacement wiring beyond one linear op.
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).float().eval()
    tokens = torch.tensor([[1, 4, 9, 3, 12, 2], [2, 5, 8, 1, 13, 6]])
    with torch.no_grad():
        reference = model(tokens, use_cache=False).logits
        modules = install_residuals(model, 16)
        reconstructed = model(tokens, use_cache=False).logits
        torch.testing.assert_close(reconstructed, reference, atol=2e-5, rtol=2e-5)
        for module in modules:
            module.set_rank(0)
        shared_only = model(tokens, use_cache=False).logits
        assert not torch.allclose(shared_only, reference, atol=1e-4, rtol=1e-4)
