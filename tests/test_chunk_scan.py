"""The blockwise delta-rule scan is the reference scan, to float precision.

The reference is a Python loop over every token; the chunked form is one triangular
solve and a few matmuls per chunk. They must agree on outputs, on the state handed to
the next call, and on gradients -- otherwise the CPU/Mac path trains a different model
from the one the fused kernel serves.
"""

from __future__ import annotations

import pytest
import torch

from prophet.config import ProphetConfig
from prophet.modeling.layers import GatedDeltaNet, RecurrentState
from prophet.modeling.model import ProphetCache, ProphetModel


def _layer(chunk: int | None) -> GatedDeltaNet:
    torch.manual_seed(0)
    layer = GatedDeltaNet(48, n_heads=3, head_dim=16, expand=2.0, allow_fused=False, chunk_size=chunk)
    # Push the gates away from their gentle init so the test covers the whole range:
    # strong forgetting and write strengths up to beta_max = 2.
    with torch.no_grad():
        layer.a_proj.bias.fill_(0.0)
        layer.a_proj.weight.mul_(4.0)
        layer.b_proj.weight.mul_(4.0)
        layer.b_proj.bias.fill_(1.0)
    return layer


@pytest.mark.parametrize("seq_len", [1, 5, 8, 37, 64])
@pytest.mark.parametrize("chunk", [4, 8, 32])
def test_chunked_matches_reference_outputs_and_state(seq_len, chunk):
    layer = _layer(chunk)
    x = torch.randn(2, seq_len, 48)
    ref_state, chunk_state = RecurrentState(), RecurrentState()
    with torch.no_grad():
        layer.chunk_size = None
        ref = layer(x, state=ref_state)
        layer.chunk_size = chunk
        out = layer(x, state=chunk_state)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()
    assert torch.allclose(ref_state.state, chunk_state.state, atol=1e-5)


def test_chunked_continues_from_a_carried_state_exactly():
    layer = _layer(8)
    x = torch.randn(1, 50, 48)
    ref_state, chunk_state = RecurrentState(), RecurrentState()
    with torch.no_grad():
        layer.chunk_size = None
        layer(x[:, :23], state=ref_state)
        ref = layer(x[:, 23:], state=ref_state)
        layer.chunk_size = 8
        layer(x[:, :23], state=chunk_state)
        out = layer(x[:, 23:], state=chunk_state)
    assert torch.allclose(ref, out, atol=1e-5)
    # ...and a chunked prefill matches token-by-token decode.
    with torch.no_grad():
        decode_state = RecurrentState()
        layer.chunk_size = 8
        layer(x[:, :23], state=decode_state)
        steps = torch.cat([layer(x[:, t : t + 1], state=decode_state) for t in range(23, 50)], 1)
    assert torch.allclose(ref, steps, atol=1e-5)


def test_chunked_gradients_match_reference():
    layer = _layer(8)
    x = torch.randn(2, 21, 48)
    grads = {}
    for chunk in (None, 8):
        layer.chunk_size = chunk
        layer.zero_grad()
        layer(x).square().mean().backward()
        grads[chunk] = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    assert set(grads[None]) == set(grads[8]) and grads[None]
    for name in grads[None]:
        assert torch.allclose(grads[None][name], grads[8][name], atol=1e-5, rtol=1e-4), name


def test_model_is_identical_under_either_scan():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    torch.manual_seed(1)
    model = ProphetModel(cfg).eval()
    ids = torch.randint(0, 2048, (1, 45))
    layers = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    assert layers and all(layer.chunk_size == cfg.mixer.linear_chunk_size for layer in layers)
    with torch.no_grad():
        chunked = model(ids, cache=ProphetCache(), loop_k=3).logits
        for layer in layers:
            layer.chunk_size = None
        reference = model(ids, cache=ProphetCache(), loop_k=3).logits
    assert torch.allclose(chunked, reference, atol=1e-4)


@pytest.mark.parametrize("bias", [-20.0, -40.0, -90.0])
def test_gradients_stay_finite_when_the_forget_gate_closes(bias):
    """alpha.log() has gradient 1/alpha; a closed gate made the chunked path's
    gradients NaN while the reference's were finite. logsigmoid of the gate logit has
    gradient 1 - alpha and cannot overflow."""
    torch.manual_seed(0)
    layer = GatedDeltaNet(64, n_heads=4, head_dim=16, allow_fused=False, chunk_size=64)
    with torch.no_grad():
        layer.a_proj.bias.fill_(bias)
    x = torch.randn(2, 128, 64, requires_grad=True)
    (layer(x).square().mean() * 1e6).backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
