"""``recurrent.iteration_readout``: the per-iteration read-out without halting."""

from __future__ import annotations

import torch

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.modeling.model import ProphetModel


def _model(readout: bool) -> ProphetModel:
    cfg = ProphetConfig(
        d_model=32,
        frontend=FrontendConfig(vocab_size=64),
        mixer=MixerConfig(pattern=["swa", "full_attn"], n_heads=2, n_kv_heads=2, head_dim=16,
                          nope_layers=(1,), linear_heads=1, linear_head_dim=16),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1, core_pattern=["gdn"],
            coda_pattern=["full_attn"], default_loop_k=3, train_loop_min=3, train_loop_max=3,
            halting="none", iteration_readout=readout, truncated_backprop_steps=3,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    cfg.validate()
    torch.manual_seed(0)
    return ProphetModel(cfg).eval()


def test_readout_returns_one_state_per_iteration_and_the_last_is_the_output():
    ids = torch.randint(0, 64, (2, 7))
    off = _model(False)(ids, loop_k=3)
    assert off.hidden_per_step is None  # without halting, nothing is probed by default
    on = _model(True)(ids, loop_k=3)
    assert on.hidden_per_step is not None and len(on.hidden_per_step) == 3
    assert all(tuple(h.shape) == (2, 7, 32) for h in on.hidden_per_step)
    # The last probe is the read-out the model emits from: same coda, same norm, no cache.
    assert torch.allclose(on.hidden_per_step[-1], on.hidden, atol=1e-5)
    # The switch changes what is returned, not what is computed.
    assert torch.allclose(on.logits, off.logits, atol=1e-5)


def test_readout_carries_gradient_to_every_iteration():
    model = _model(True).train()
    ids = torch.randint(0, 64, (1, 5))
    torch.manual_seed(1)
    out = model(ids, loop_k=3)
    # A loss on the first iteration alone must reach the core's weights.
    loss = model.lm_head(out.hidden_per_step[0])[0, -1].float().logsumexp(-1)
    loss.backward()
    core = [p for n, p in model.named_parameters() if ".core." in n and p.grad is not None]
    assert core and any(p.grad.abs().sum() > 0 for p in core)
