"""Behavioral guards for reversible native-donor research interventions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.probe_qwen_selective import remove_middle_eight, selective_sharing


def toy_layers():
    layers = nn.ModuleList()
    for index in range(28):
        layer = nn.Module()
        for family, names in [
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ]:
            parent = nn.Module()
            for name in names:
                projection = nn.Linear(2, 2, bias=False)
                projection.weight.data.copy_(torch.eye(2) * (index + 1))
                parent.add_module(name, projection)
            parent.norm = nn.LayerNorm(2)
            layer.add_module(family, parent)
        layer.self_attn.layer_idx = index
        layers.append(layer)
    return layers


def test_only_requested_family_is_shared_and_originals_restore_after_exception():
    layers = toy_layers()
    original_modules = {name: module for name, module in layers.named_modules()}
    original_values = {name: value.clone() for name, value in layers.state_dict().items()}
    with pytest.raises(RuntimeError, match="intentional"), selective_sharing(layers, "mlp"):
        x = torch.tensor([[2.0, 3.0]])
        # Cyclic group 4,8,12,16,20 has mean scalar weight 13.
        torch.testing.assert_close(layers[4].mlp.up_proj(x), 13 * x)
        assert layers[4].mlp.up_proj is layers[20].mlp.up_proj
        assert layers[4].mlp.up_proj is not layers[5].mlp.up_proj
        assert layers[4].self_attn.q_proj is original_modules["4.self_attn.q_proj"]
        assert layers[4].mlp.norm is original_modules["4.mlp.norm"]
        assert layers[3].mlp.up_proj is original_modules["3.mlp.up_proj"]
        assert layers[24].mlp.up_proj is original_modules["24.mlp.up_proj"]
        raise RuntimeError("intentional")
    assert dict(layers.named_modules()) == original_modules
    for name, value in layers.state_dict().items():
        torch.testing.assert_close(value, original_values[name], atol=0, rtol=0)


def test_pruning_preserves_order_and_restores_attention_indices_after_exception():
    original = toy_layers()
    model = SimpleNamespace(
        model=SimpleNamespace(layers=original), config=SimpleNamespace(num_hidden_layers=28)
    )
    with pytest.raises(RuntimeError, match="intentional"), remove_middle_eight(model):
        assert model.config.num_hidden_layers == 20
        assert list(model.model.layers) == list(original[:12]) + list(original[20:])
        assert [layer.self_attn.layer_idx for layer in model.model.layers] == list(range(20))
        raise RuntimeError("intentional")
    assert model.model.layers is original and model.config.num_hidden_layers == 28
    assert [layer.self_attn.layer_idx for layer in original] == list(range(28))
