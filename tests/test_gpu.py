"""GPU-only checks: the first tests to run on the A100, skipped everywhere else.

The fused delta-rule kernel (flash-linear-attention) has never executed in this
repository: no GPU, no ``fla``. Its layout contract is written down in
``GatedDeltaNet.forward`` and nothing else vouches for it. These tests are the vouching:
the kernel must match the reference scan on outputs *and* on the state it hands back,
and a chunked prefill must match token-by-token decode under the kernel, before any
budgeted run starts. ``scripts/train.py`` refuses a real run without ``fla`` for exactly
this reason; ``scripts/gpu_check.py`` runs the same checks outside pytest.
"""

from __future__ import annotations

import pytest
import torch

from prophet.config import ProphetConfig
from prophet.modeling.layers import HAS_FLA, GatedDeltaNet
from prophet.modeling.model import ProphetCache, ProphetModel

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _model(cfg_path: str = "configs/prophet_tiny_smoke.json") -> ProphetModel:
    torch.manual_seed(0)
    cfg = ProphetConfig.from_json(cfg_path)
    return ProphetModel(cfg).cuda().eval()


def _set_fused(model: ProphetModel, fused: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m.allow_fused = fused
            n += 1
    return n


@pytest.mark.skipif(not HAS_FLA, reason="flash-linear-attention is not installed")
def test_fused_kernel_matches_the_reference_scan_on_outputs_and_state():
    model = _model()
    assert _set_fused(model, False) > 0
    ids = torch.randint(0, 2048, (2, 96), device="cuda")
    with torch.no_grad():
        ref_cache = ProphetCache()
        ref = model(ids, cache=ref_cache, loop_k=3).logits.float()
        _set_fused(model, True)
        fused_cache = ProphetCache()
        out = model(ids, cache=fused_cache, loop_k=3).logits.float()
    assert torch.allclose(ref, out, atol=2e-3, rtol=1e-3), (ref - out).abs().max()
    for key, slot in ref_cache.slots.items():
        state = getattr(slot, "state", None)
        if state is None:
            continue
        other = fused_cache.slots[key].state
        assert other.shape == state.shape, f"state layout differs at {key}: {other.shape} vs {state.shape}"
        assert torch.allclose(state.float(), other.float(), atol=2e-3, rtol=1e-3), key


@pytest.mark.skipif(not HAS_FLA, reason="flash-linear-attention is not installed")
def test_fused_prefill_matches_incremental_decode():
    model = _model()
    _set_fused(model, True)
    ids = torch.randint(0, 2048, (1, 40), device="cuda")
    with torch.no_grad():
        full = model(ids, loop_k=2).logits.float()
        cache = ProphetCache()
        model(ids[:, :25], cache=cache, loop_k=2)
        steps = [model(ids[:, t : t + 1], cache=cache, loop_k=2).logits.float() for t in range(25, 40)]
    assert torch.allclose(full[:, 25:], torch.cat(steps, 1), atol=2e-3, rtol=1e-3)


def test_a_training_step_runs_under_autocast_with_checkpointing():
    """bf16 autocast, activation checkpointing, Muon + AdamW, on the real device."""
    from prophet.data.streaming import StreamingLoader, sources_from_iterables
    from prophet.train.loop import TrainConfig, Trainer

    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg)
    rows = [[int(x) for x in torch.randint(0, 2048, (64,))] for _ in range(8)]
    loader = StreamingLoader(sources_from_iterables({"a": (1.0, rows)}), seq_len=64, batch_size=2)
    trainer = Trainer(
        model, loader,
        TrainConfig(total_steps=2, seq_len=64, batch_size=2, device="cuda",
                    checkpoint_dir="/tmp/prophet-gpu-check", activation_checkpointing=True),
        model_config=cfg,
    )
    history = trainer.train(max_steps=2)
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(h.loss)) for h in history)
