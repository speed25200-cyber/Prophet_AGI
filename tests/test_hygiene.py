"""Every configuration field is read by the code that honours it.

The rule in CLAUDE.md: a field nothing reads is a bug, not a reservation. These tests pin
the fields the second review pass either wired in (``dropout``, ``norm_kind``,
``max_seq_len``, ``memory.max_writes``, ``memory.update_rule``) or removed, and the
frontend modes the model cannot build.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from prophet.config import FrontendConfig, MemoryConfig, ModalityConfig, ProphetConfig
from prophet.data.streaming import StreamingLoader, sources_from_iterables
from prophet.memory.consolidate import Episode, consolidate
from prophet.memory.ledger import LedgerConfig, ProductKeyMemory
from prophet.modeling.model import ProphetModel
from prophet.train.loop import TrainConfig, Trainer


def _cfg(**top) -> ProphetConfig:
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    return dataclasses.replace(cfg, **top)


def test_dropout_is_off_by_default_and_active_when_set():
    torch.manual_seed(0)
    off = ProphetModel(_cfg()).train()
    assert isinstance(off.sections["prelude"][0].dropout, nn.Identity)
    on = ProphetModel(_cfg(dropout=0.5)).train()
    ids = torch.randint(0, 2048, (1, 8))
    a, b = on(ids, loop_k=1).logits, on(ids, loop_k=1).logits
    assert not torch.equal(a, b)  # training: stochastic
    on.eval()
    with torch.no_grad():
        assert torch.equal(on(ids, loop_k=1).logits, on(ids, loop_k=1).logits)


def test_norm_kind_selects_the_normalisation_everywhere():
    model = ProphetModel(_cfg(norm_kind="layernorm")).eval()
    assert isinstance(model.norm_out, nn.LayerNorm)
    assert isinstance(model.sections["core"][0].norm1, nn.LayerNorm)
    with torch.no_grad():
        model(torch.randint(0, 2048, (1, 6)), loop_k=2)
    with pytest.raises(ValueError, match="norm_kind"):
        from prophet.modeling.layers import make_norm
        make_norm("batchnorm", 8, 1e-5)


def test_trainer_refuses_a_sequence_longer_than_max_seq_len(tmp_path):
    cfg = _cfg(max_seq_len=32)
    model = ProphetModel(cfg)
    loader = StreamingLoader(
        sources_from_iterables({"a": (1.0, [[1, 2, 3, 4] * 20])}), seq_len=64
    )
    with pytest.raises(ValueError, match="max_seq_len"):
        Trainer(
            model, loader,
            TrainConfig(total_steps=1, seq_len=64, checkpoint_dir=str(tmp_path), device="cpu"),
            model_config=cfg,
        )


def test_removed_fields_are_gone():
    for cls, name in (
        (FrontendConfig, "patch_max_bytes"),
        (FrontendConfig, "patch_target_bytes"),
        (FrontendConfig, "patch_entropy_threshold"),
        (FrontendConfig, "local_window"),
        (MemoryConfig, "persist_across_sessions"),
        (MemoryConfig, "max_persisted_writes"),
        (ModalityConfig, "bidirectional_spans"),
        (ModalityConfig, "adapter_mount_points"),
    ):
        assert name not in {f.name for f in dataclasses.fields(cls)}, f"{cls.__name__}.{name}"


def test_stale_keys_in_a_saved_config_are_ignored_not_fatal(tmp_path):
    """Old JSON files carry the removed keys; loading them must not break."""
    import json
    data = _cfg().to_dict()
    data["frontend"]["patch_max_bytes"] = 16
    data["memory"]["persist_across_sessions"] = True
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    ProphetConfig.from_json(path).validate()


# --------------------------------------------------------------------------------------
# Memory: write cap and surprise gating
# --------------------------------------------------------------------------------------


def test_ledger_refuses_writes_past_its_lifetime_cap():
    torch.manual_seed(1)
    ledger = ProductKeyMemory(LedgerConfig(dim=16, memory_dim=8, n_slots=16, top_k=4, n_heads=1, max_writes=3))
    x, t = torch.randn(1, 2, 16), torch.randn(1, 2, 16)
    assert ledger.write(x, t).accepted
    assert ledger.write(x, t).accepted  # 2 < 3 when checked, so still accepted
    before = ledger.values.clone()
    refused = ledger.write(x, t)
    assert not refused.accepted and refused.slots_touched == 0
    assert torch.equal(ledger.values, before)
    assert int(ledger.tokens_written) == 4
    ledger.reset()
    assert int(ledger.tokens_written) == 0 and ledger.write(x, t).accepted


def _memory_model(**memory) -> ProphetModel:
    torch.manual_seed(2)
    cfg = _cfg(memory=MemoryConfig(enabled=True, kind="product_key", mount="output",
                                   memory_dim=16, n_slots=64, **memory))
    cfg.validate()
    return ProphetModel(cfg).eval()


def _episodes():
    torch.manual_seed(3)
    return [Episode(torch.randint(0, 2048, (1, 6)), torch.randint(0, 2048, (1, 4))) for _ in range(2)]


def test_surprise_gating_follows_the_model_config_by_default():
    # A huge threshold gates every token that has a predictor. The first query token
    # of each episode has none and counts as infinitely surprising, so two of the eight
    # tokens are still written.
    gated = _memory_model(update_rule="surprise_gated", surprise_threshold=1e6)
    report = consolidate(gated, gated.ledgers["output"], _episodes(), passes=1)
    assert report.tokens_written == 2 and report.tokens_gated == 6

    plain = _memory_model(update_rule="delta")
    report = consolidate(plain, plain.ledgers["output"], _episodes(), passes=1)
    assert report.tokens_written == 8 and report.tokens_gated == 0


def test_surprise_gating_can_be_overridden_per_call():
    model = _memory_model(update_rule="delta")
    report = consolidate(model, model.ledgers["output"], _episodes(), passes=1, surprise_threshold=1e6)
    assert report.tokens_written == 2 and report.tokens_gated == 6
    report = consolidate(model, model.ledgers["output"], _episodes(), passes=1, surprise_threshold=None)
    assert report.tokens_written == 8


def test_consolidation_reports_refused_writes():
    model = _memory_model(update_rule="delta", max_writes=4)
    report = consolidate(model, model.ledgers["output"], _episodes(), passes=1)
    # Two episodes of four query tokens: the first is accepted, the second refused.
    assert report.tokens_written == 4 and report.writes_refused == 1
