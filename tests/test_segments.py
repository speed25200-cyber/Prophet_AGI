"""Segment-masked attention: a training row of several episodes seen exactly as the loop
sees a carried session (attention empty at each episode's start, recurrent state kept)."""

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
from prophet.memory.session import extract_session, restore_session
from prophet.modeling.layers import CausalSelfAttention
from prophet.modeling.model import ProphetCache, ProphetModel
from prophet.train.loop import segment_ids_from_bos

VOCAB = 256


def _model(nope: bool) -> ProphetModel:
    cfg = ProphetConfig(
        d_model=64,
        frontend=FrontendConfig(vocab_size=VOCAB),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=4, n_kv_heads=2, head_dim=16,
            sliding_window=32, linear_heads=2, linear_head_dim=16, nope_layers=(1,) if nope else (),
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=2, core_pattern=["gdn"],
            coda_pattern=["swa", "full_attn"], default_loop_k=2, train_loop_min=2, train_loop_max=2,
            halting="none",
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    cfg.validate()
    torch.manual_seed(0)
    return ProphetModel(cfg).eval()


def test_bare_layer_sees_only_its_own_segment():
    torch.manual_seed(1)
    attn = CausalSelfAttention(32, n_heads=2, n_kv_heads=2, head_dim=16, window=None, sink_tokens=0).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        full = attn(x)
        attn.segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        masked = attn(x)
        attn.segment_ids = None
        alone = attn(x[:, 3:])
    assert torch.allclose(full[:, :3], masked[:, :3], atol=1e-6)  # the first segment is untouched
    assert torch.allclose(masked[:, 3:], alone, atol=1e-5)  # the second is as if it started the row
    assert not torch.allclose(full[:, 3:], masked[:, 3:], atol=1e-3)  # and the mask did something


@pytest.mark.parametrize("nope", [False, True])
def test_segment_mask_is_the_carried_session_to_1e4(nope):
    """Row = episode A then episode B under the mask == B run on a fresh cache that holds
    A's session state (positions continuing), which is what ``AgentLoop`` does."""
    model = _model(nope)
    g = torch.Generator().manual_seed(3)
    a = torch.randint(0, VOCAB, (1, 20), generator=g)
    b = torch.randint(0, VOCAB, (1, 14), generator=g)
    row = torch.cat([a, b], dim=1)
    segments = torch.tensor([[0] * 20 + [1] * 14])
    with torch.no_grad():
        masked = model(row, segment_ids=segments, loop_k=2).logits[:, 20:]
        cache = ProphetCache()
        model(a, cache=cache, loop_k=2)
        carried = ProphetCache()
        restore_session(extract_session(cache), carried)
        assert carried.position == 20
        session = model(b, cache=carried, loop_k=2).logits
        plain = model(row, loop_k=2).logits[:, 20:]
    assert torch.allclose(masked, session, atol=1e-4), f"max diff {(masked - session).abs().max():.2e}"
    assert not torch.allclose(plain, session, atol=1e-2)  # without the mask, B saw A through attention


def test_segments_are_derived_at_bos_and_refused_with_a_cache():
    batch = torch.tensor([[7, 1, 2, 7, 3, 7], [1, 2, 3, 7, 4, 5]])
    assert segment_ids_from_bos(batch, 7).tolist() == [[1, 1, 1, 2, 2, 3], [0, 0, 0, 1, 1, 1]]
    model = _model(True)
    with pytest.raises(ValueError, match="cache-free"):
        model(batch[:1], segment_ids=torch.zeros(1, 6, dtype=torch.long), cache=ProphetCache())
    with pytest.raises(ValueError, match="shaped"):
        model(batch[:1], segment_ids=torch.zeros(1, 5, dtype=torch.long))
    # A refused call leaves no segment behind on the layers, and a plain call clears them.
    assert all(layer.segment_ids is None for layer in model._attention_layers())
    model(batch[:1], segment_ids=torch.zeros(1, 6, dtype=torch.long))
    assert all(layer.segment_ids is not None for layer in model._attention_layers())
    model(batch[:1])
    assert all(layer.segment_ids is None for layer in model._attention_layers())


def test_segment_mask_survives_activation_checkpointing():
    """The forward is recomputed during backward under checkpointing; the mask must be
    the same then, or PyTorch refuses the recompute (the trap that crashed the first
    segment-masked run at its first step)."""
    model = _model(True).train()
    g = torch.Generator().manual_seed(4)
    row = torch.randint(0, VOCAB, (2, 24), generator=g)
    segments = torch.tensor([[0] * 12 + [1] * 12, [0] * 8 + [1] * 16])
    torch.manual_seed(0)  # the training-mode state init is random: same draw both times
    plain = model(row, segment_ids=segments, loop_k=2).logits
    model.gradient_checkpointing = True
    torch.manual_seed(0)
    out = model(row, segment_ids=segments, loop_k=2)
    out.logits.float().sum().backward()  # recompute happens here
    assert torch.allclose(out.logits, plain, atol=1e-5)
    assert all(p.grad is not None for n, p in model.named_parameters() if "embed" in n)
