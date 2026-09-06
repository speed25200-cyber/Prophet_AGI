"""``recurrent.iteration_readout``: the per-iteration read-out without halting."""

from __future__ import annotations

import pytest
import torch

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.modeling.model import ProphetCache, ProphetModel


def _model(readout: bool, embedding: bool = False, requantize: str = "none") -> ProphetModel:
    cfg = ProphetConfig(
        d_model=32,
        frontend=FrontendConfig(vocab_size=64),
        mixer=MixerConfig(pattern=["swa", "full_attn"], n_heads=2, n_kv_heads=2, head_dim=16,
                          nope_layers=(1,), linear_heads=1, linear_head_dim=16),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1, core_pattern=["gdn"],
            coda_pattern=["full_attn"], default_loop_k=3, train_loop_min=3, train_loop_max=3,
            halting="none", iteration_readout=readout, iteration_embedding=embedding,
            iteration_requantize=requantize,
            truncated_backprop_steps=3,
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


def test_iteration_embedding_is_read_per_iteration_and_keeps_decode_exact():
    plain, embedded = _model(False), _model(False, embedding=True)
    assert embedded.iteration_embed is not None and embedded.iteration_embed.weight.shape == (3, 32)
    n_plain = sum(p.numel() for p in plain.parameters())
    assert sum(p.numel() for p in embedded.parameters()) == n_plain + 3 * 32
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        embedded.load_state_dict(plain.state_dict(), strict=False)
        embedded.iteration_embed.weight.zero_()
        assert torch.allclose(embedded(ids, loop_k=3).logits, plain(ids, loop_k=3).logits, atol=1e-5)
        embedded.iteration_embed.weight[1].fill_(0.5)  # only the second pass changes
        moved = embedded(ids, loop_k=3).logits
        assert not torch.allclose(moved, plain(ids, loop_k=3).logits, atol=1e-3)
        assert torch.allclose(embedded(ids, loop_k=1).logits, plain(ids, loop_k=1).logits, atol=1e-5)
        # Deeper than the table: the last row is reused, and decode stays exact.
        cache = ProphetCache()
        steps = [embedded(ids[:, t : t + 1], cache=cache, loop_k=4).logits for t in range(9)]
        full = embedded(ids, loop_k=4).logits
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_requantized_readout_feeds_the_next_iteration_and_decodes_exactly(mode):
    plain, fed = _model(True), _model(True, requantize=mode)
    fed.load_state_dict(plain.state_dict())
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        a, b = plain(ids, loop_k=3), fed(ids, loop_k=3)
        # The first iteration reads the same input; from the second on, the symbol is in.
        assert torch.allclose(a.hidden_per_step[0], b.hidden_per_step[0], atol=1e-5)
        assert not torch.allclose(a.hidden_per_step[1], b.hidden_per_step[1], atol=1e-3)
        assert not torch.allclose(a.logits, b.logits, atol=1e-3)
        # One iteration: nothing to feed back, so nothing changes.
        assert torch.allclose(plain(ids, loop_k=1).logits, fed(ids, loop_k=1).logits, atol=1e-5)
        cache = ProphetCache()
        steps = [fed(ids[:, t : t + 1], cache=cache, loop_k=3).logits for t in range(9)]
        assert torch.allclose(torch.cat(steps, dim=1), fed(ids, loop_k=3).logits, atol=1e-4)
    # Gradient reaches the embedding through the fed-back symbol.
    fed.train()
    out = fed(ids, loop_k=3)
    out.hidden_per_step[-1].float().pow(2).mean().backward()
    assert fed.embed.weight.grad is not None and fed.embed.weight.grad.abs().sum() > 0


def test_requantize_needs_a_readout():
    with pytest.raises(ValueError, match="iteration_requantize"):
        _model(False, requantize="soft")
