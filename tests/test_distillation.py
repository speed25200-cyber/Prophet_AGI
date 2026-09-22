from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from prophet.train.distillation import (
    DistillationObjective,
    DistillationSettings,
    forward_kl,
    state_sha256,
)
from prophet.train.loop import TrainConfig, Trainer
from tests.test_training import ProphetModel, make_loader, tiny_model_config


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_kl_loss_and_gradient_match_reference(temperature, chunk):
    torch.manual_seed(7)
    student = torch.randn(2, 11, 5).transpose(1, 2).requires_grad_()
    teacher = torch.randn_like(student).requires_grad_()
    weights = torch.rand(student.shape[:-1])
    weights[:, -1] = 0  # excluded next-token position
    reference = student.detach().clone().requires_grad_()
    log_q = F.log_softmax(teacher.detach() / temperature, dim=-1)
    expected = F.kl_div(F.log_softmax(reference / temperature, dim=-1), log_q,
                        reduction="none", log_target=True).sum(-1) * temperature**2
    actual = forward_kl(student, teacher, temperature=temperature, chunk_tokens=chunk)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(student.grad, reference.grad, atol=2e-6, rtol=2e-5)
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:, -1]) == 0


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": float("nan")},
                                  {"chunk_tokens": 0}, {"chunk_tokens": True}])
def test_invalid_kl_settings(kwargs):
    with pytest.raises(ValueError):
        forward_kl(torch.zeros(2, 3), torch.zeros(2, 3), **kwargs)


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 64)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input_ids, use_cache=False):
        assert not use_cache
        return SimpleNamespace(logits=self.dropout(self.embed(input_ids)))


def fresh(path, *, alpha=0.5, change_teacher=False, enabled=True):
    torch.manual_seed(123)
    model_cfg = tiny_model_config()
    model = ProphetModel(model_cfg)
    teacher = TinyTeacher()
    if change_teacher:
        with torch.no_grad():
            teacher.embed.weight[0, 0] += 1
    objective = DistillationObjective(teacher, identity={"source": "test-fixed-donor"},
                                      settings=DistillationSettings(alpha=alpha, chunk_tokens=11))
    cfg = TrainConfig(total_steps=6, checkpoint_every=3, log_every=100,
                      grad_accum_steps=2, checkpoint_dir=str(path), loss_chunk_tokens=7)
    return Trainer(model, make_loader(), cfg, model_config=model_cfg,
                   distillation=objective if enabled else None, on_log=lambda m: None)


def test_distillation_resumes_exactly_and_donor_is_frozen(tmp_path):
    reference = fresh(tmp_path / "reference")
    donor = reference.distillation.teacher
    before = state_sha256(donor)
    reference.train()
    assert state_sha256(donor) == before
    assert not donor.training
    assert all(p.grad is None and not p.requires_grad for p in donor.parameters())
    assert "loss/donor_kl" in reference.history[-1].extra
    interrupted = fresh(tmp_path / "run")
    interrupted.train(max_steps=3)
    resumed = fresh(tmp_path / "run")
    assert resumed.maybe_resume()
    resumed.train()
    assert resumed.tokens_seen == reference.tokens_seen
    for key, value in reference.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    for left, right in zip(reference.history[3:], resumed.history, strict=True):
        assert left.loss == right.loss
        assert left.extra == right.extra


@pytest.mark.parametrize("change", [{"alpha": 0.7}, {"change_teacher": True}, {"enabled": False}])
def test_resume_rejects_changed_donor_or_objective_before_mutation(tmp_path, change):
    original = fresh(tmp_path / "original")
    state = deepcopy(original.state_dict())
    resumed = fresh(tmp_path / "resumed", **change)
    before = state_sha256(resumed.model)
    with pytest.raises(ValueError, match="training contract"):
        resumed.load_state_dict(state)
    assert resumed.step == 0
    assert state_sha256(resumed.model) == before


def test_ce_only_contract_has_no_new_field(tmp_path):
    assert "distillation" not in fresh(tmp_path, enabled=False).training_contract()


def test_nonfinite_kl_skips_even_with_finite_gradients(tmp_path):
    trainer = fresh(tmp_path)
    # A nonfinite constant has zero derivative: grad-norm checks alone miss it.
    trainer.distillation.loss = lambda logits, batch, **kw: logits.sum() * 0 + float("inf")
    before = state_sha256(trainer.model)
    trainer.train(max_steps=1)
    assert trainer.skipped_nonfinite == 1
    assert state_sha256(trainer.model) == before


def test_objective_excludes_last_position_and_matches_full_kl():
    teacher = TinyTeacher()
    objective = DistillationObjective(teacher, identity={"source": "tiny"})
    batch = torch.tensor([[1, 3, 4, 7], [4, 5, 8, 9]])
    logits = torch.randn(2, 4, 64, requires_grad=True)
    actual = objective.loss(logits, batch)
    expected = F.kl_div(F.log_softmax(logits[:, :-1], -1),
                        F.softmax(teacher(batch).logits[:, :-1], -1), reduction="sum") / 6
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0


def test_bf16_tensor_hash_and_identity_are_stable():
    teacher = TinyTeacher().bfloat16()
    identity = {"tokenizer": {"sha256": "abc"}}
    objective = DistillationObjective(teacher, identity=identity)
    identity["tokenizer"]["sha256"] = "changed"
    first = objective.fingerprint()
    first["identity"]["tokenizer"]["sha256"] = "also changed"
    assert objective.fingerprint()["identity"]["tokenizer"]["sha256"] == "abc"
    assert state_sha256(deepcopy(teacher)) == state_sha256(teacher)


def test_hf_qwen_teacher_api_without_downloads():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=16)
    donor = transformers.Qwen3ForCausalLM(config).eval()
    objective = DistillationObjective(donor, identity={"config": config.to_dict()})
    batch = torch.tensor([[1, 4, 5, 6]])
    student = torch.randn(1, 4, 64, requires_grad=True)
    loss = objective.loss(student, batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert all(p.grad is None for p in donor.parameters())


def test_recovery_identity_is_immutable_and_resume_checked(tmp_path):
    cfg = tiny_model_config()
    identity = {"initialization": {"sha256": "a"}}
    trainer = Trainer(ProphetModel(cfg), make_loader(), TrainConfig(checkpoint_dir=str(tmp_path)),
                      model_config=cfg, run_identity=identity)
    saved = deepcopy(trainer.state_dict())
    identity["initialization"]["sha256"] = "changed"
    assert trainer.training_contract()["run_identity"]["initialization"]["sha256"] == "a"
    saved["training_contract"]["run_identity"]["initialization"]["sha256"] = "b"
    with pytest.raises(ValueError, match="training contract"):
        trainer.load_state_dict(saved)
