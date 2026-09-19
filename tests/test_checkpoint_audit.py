import json

import pytest
import torch

from prophet.train.checkpoint import CheckpointManager
from scripts.audit_r04_checkpoint import audit_checkpoint


def publish(run, *, optimizer_value=2.0):
    protocol = {
        "variant": "loop",
        "seed": 0,
        "config": {"width": 2},
        "batch_size": 8,
        "seq_len": 2048,
    }
    state = {
        "step": 512,
        "config": protocol["config"],
        "tokens_seen": 512 * 8 * 2048,
        "trainer_state_version": 3,
        "skipped_nonfinite": 0,
        "consecutive_nonfinite": 0,
        "training_contract": {"seed": 0},
        "model": {"weight": torch.ones(2)},
        "optimizers": [{"state": {0: {"momentum": torch.tensor(optimizer_value)}}}],
    }
    manager = CheckpointManager(run / "checkpoints")
    manager.save({**state, "model": {"weight": torch.full((2,), float("nan"))}}, step=512)
    meta = manager.save(state, step=512)
    (run / "protocol.json").write_text(json.dumps(protocol))
    evaluation = {
        "run_protocol": protocol,
        "step": 512,
        "checkpoint": meta.to_dict(),
        "train_tokens": state["tokens_seen"],
    }
    (run / "evaluation-step-000512.json").write_text(json.dumps(evaluation))
    return manager, meta


def test_audit_uses_published_save_when_steps_tie(tmp_path):
    _, meta = publish(tmp_path)
    result = audit_checkpoint(tmp_path, 512)
    assert result["checkpoint"] == meta.to_dict()
    assert result["all_model_optimizer_tensors_finite"]
    assert result["inspected_tensors"] == 2


def test_audit_reports_nonfinite_optimizer_state(tmp_path):
    publish(tmp_path, optimizer_value=float("nan"))
    result = audit_checkpoint(tmp_path, 512)
    assert not result["all_model_optimizer_tensors_finite"]
    assert result["nonfinite_tensor_paths"] == ["optimizers/0/state/0/momentum"]


def test_audit_rejects_corrupt_published_save_without_fallback(tmp_path):
    manager, meta = publish(tmp_path)
    with manager.slot_path(meta.slot).open("ab") as stream:
        stream.write(b"bad-write")
    with pytest.raises(ValueError, match="checksum"):
        audit_checkpoint(tmp_path, 512)


def test_audit_rejects_rotated_published_weights(tmp_path):
    manager, _ = publish(tmp_path)
    manager.save({"step": 513}, step=513)
    manager.save({"step": 514}, step=514)
    with pytest.raises(ValueError, match="rotated"):
        audit_checkpoint(tmp_path, 512)
