"""Test the gate's failure detection; CPU passes do not certify CUDA kernels."""
import pytest
import torch

from prophet.modeling.layers import GatedDeltaNet
from prophet.train.distillation import state_sha256
from scripts.gate_qwen_recovery import (
    diagnostic_gdn_fp32,
    finite_state,
    measure_updates,
    training_kernel_agreement,
)
from tests.test_distillation import fresh
from tests.test_training import ProphetModel, tiny_model_config


def test_fp32_policy_disables_inherited_tf32_and_selects_teacher_storage():
    from scripts.recover_qwen import recovery_precision

    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        assert recovery_precision("float32", "cuda") == torch.float32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert recovery_precision("bfloat16", "cuda") == torch.bfloat16
        assert recovery_precision("bfloat16", "cpu") == torch.float32
        with pytest.raises(ValueError, match="precision"):
            recovery_precision("float16", "cuda")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_numerical_gate_preserves_weights_rng_and_execution_policy():
    torch.manual_seed(71)
    cfg = tiny_model_config()
    cfg.heads.n_multi_token_predict = 1
    cfg.heads.confidence_head = True
    model = ProphetModel(cfg).eval()
    ids = torch.randint(0, 64, (1, 17))
    before, rng = state_sha256(model), torch.get_rng_state().clone()
    layers = [(m, m.allow_fused, m.chunk_size) for m in model.modules()
              if isinstance(m, GatedDeltaNet)]
    report = training_kernel_agreement(model, ids, loop_k=2, autocast=False, chunk_tokens=7)
    assert report["passed"] and report["applicable"]
    assert not report["fused_cuda_exercised"]
    assert report["unused_parameters"]
    assert state_sha256(model) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.training and not model.gradient_checkpointing
    assert all(p.grad is None for p in model.parameters())
    assert all(m.allow_fused == fused and m.chunk_size == chunk for m, fused, chunk in layers)


def test_numerical_gate_detects_a_wrong_fused_backward():
    model = ProphetModel(tiny_model_config())
    layer = next(m for m in model.modules() if isinstance(m, GatedDeltaNet))
    # Identical forward logits, deliberately wrong gradient in the candidate pass.
    parameter = next(layer.parameters())
    handle = parameter.register_hook(lambda g: g * (2 if layer.allow_fused else 1))
    try:
        report = training_kernel_agreement(model, torch.randint(0, 64, (1, 17)),
                                           loop_k=2, autocast=False, chunk_tokens=7)
    finally:
        handle.remove()
    assert report["tensors"]["logits"]["passed"]
    assert not report["passed"]
    assert any(not v["passed"] for k, v in report["tensors"].items() if k != "logits")


def test_gdn_precision_probe_matches_explicit_fp32_without_changing_other_instances():
    import copy

    torch.manual_seed(83)
    reference = GatedDeltaNet(16, n_heads=2, head_dim=4, expand=2, allow_fused=False)
    probe = copy.deepcopy(reference)
    before = state_sha256(probe)
    assert diagnostic_gdn_fp32(probe) == 1
    data = torch.randn(1, 9, 16).to(torch.bfloat16)
    expected_input = data.float().requires_grad_()
    actual_input = data.clone().requires_grad_()
    expected = reference(expected_input)
    expected.square().mean().backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = probe(actual_input)
        untouched = reference(data)
    actual.square().mean().backward()
    assert actual.dtype == torch.float32 and untouched.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_input.grad, expected_input.grad.to(torch.bfloat16), atol=0, rtol=0)
    for left, right in zip(reference.parameters(), probe.parameters(), strict=True):
        torch.testing.assert_close(right.grad, left.grad, atol=0, rtol=0)
    assert state_sha256(probe) == before
    with pytest.raises(ValueError, match="requires GDN"):
        diagnostic_gdn_fp32(torch.nn.Linear(16, 16))


@pytest.mark.parametrize("enabled", [False, True], ids=["ce", "kl"])
def test_measurement_executes_updates_without_checkpoints(tmp_path, enabled):
    trainer = fresh(tmp_path, enabled=enabled)
    trainer.cfg.checkpoint_every = 0
    before = state_sha256(trainer.model)
    report = measure_updates(trainer, warmup_steps=1, measured_steps=2)
    assert report["passed"] and state_sha256(trainer.model) != before
    assert report["tokens_seen"] == 3 * 2 * 2 * 32
    assert report["teacher_tokens_seen"] == (report["tokens_seen"] if enabled else 0)
    assert report["teacher_unchanged_and_no_gradients"] is (True if enabled else None)
    assert len(report["synchronized_seconds_per_step"]) == 2
    assert all(t > 0 for t in report["synchronized_seconds_per_step"])
    assert len(report["metrics"]) == 3
    assert not list(tmp_path.rglob("*.pt"))


def test_measurement_rejects_skipped_nonfinite_steps(tmp_path):
    trainer = fresh(tmp_path)
    trainer.cfg.checkpoint_every = 0
    trainer.distillation.loss = lambda logits, batch, **kw: logits.sum() * 0 + float("inf")
    report = measure_updates(trainer, warmup_steps=1, measured_steps=1)
    assert not report["passed"] and report["skipped_nonfinite"] == 2
    assert report["model_optimizer_tensors_finite"]


def test_measurement_requires_fresh_trainer_and_disabled_checkpoints(tmp_path):
    trainer = fresh(tmp_path)
    with pytest.raises(ValueError, match="no checkpoints"):
        measure_updates(trainer)
    trainer.cfg.checkpoint_every = 0
    trainer.train(max_steps=1)
    with pytest.raises(ValueError, match="fresh trainer"):
        measure_updates(trainer)


def test_finite_state_checks_nested_optimizer_tensors_and_scalars():
    assert finite_state({"state": [torch.ones(3), {"step": 1.0}]})
    assert not finite_state({"state": [torch.ones(3), {"step": float("nan")}]})
    assert not finite_state({"state": [torch.tensor([float("inf")])]})
