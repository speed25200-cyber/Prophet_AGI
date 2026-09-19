"""Exercise the complete recovery command with local miniature donor artifacts.

Only Qwen snapshot pins and tokenizer IDs are substituted for fixture artifacts.
Loading, CE/KL training, checkpoint serialization/resume and evaluation are real.
"""
import json
import sys

import pytest
import torch

from prophet.data.donor_tokenizer import DonorByteTokenizer
from prophet.modeling.model import ProphetModel
from prophet.train.checkpoint import CheckpointManager
from scripts import recover_qwen
from tests.test_training import tiny_model_config


@pytest.fixture
def recovery_fixture(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "source"
    source.mkdir()
    teacher_cfg = transformers.Qwen3Config(vocab_size=260, hidden_size=32, intermediate_size=64,
                                           num_hidden_layers=1, num_attention_heads=2,
                                           num_key_value_heads=1, head_dim=16)
    torch.manual_seed(12)
    transformers.Qwen3ForCausalLM(teacher_cfg).save_pretrained(source)
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.save(str(source / "tokenizer.json"))
    weights_sha = recover_qwen.digest(source / "model.safetensors")
    monkeypatch.setattr(recover_qwen, "WEIGHTS_SHA256", weights_sha)
    monkeypatch.setattr(recover_qwen, "TOKENIZER_SHA256", recover_qwen.digest(source / "tokenizer.json"))
    monkeypatch.setattr(recover_qwen, "compare", lambda donor, config: {})
    monkeypatch.setattr(recover_qwen, "DonorByteTokenizer", lambda path, **kw:
                        DonorByteTokenizer(path, eos_id=257, pad_id=256, vocab_size=260))
    cfg = tiny_model_config()
    cfg.frontend.vocab_size = 260
    initialization = tmp_path / "initial.pt"
    payload = {"model": ProphetModel(cfg).state_dict(), "config": cfg.to_dict(),
               "donor_revision": recover_qwen.REVISION, "donor_weights_sha256": weights_sha}
    torch.save(payload, initialization)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"complete": True, "all_parameters_finite": True,
                                 "serialization_exact": True, "config": cfg.to_dict(),
                                 "donor_revision": recover_qwen.REVISION,
                                 "donor_weights_sha256": weights_sha,
                                 "checkpoint_sha256": recover_qwen.digest(initialization)}))
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text(json.dumps({"text": "abc def " * 100}) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({"text": "unique test"}) + "\n", encoding="utf-8")

    def run(out, objective, session_steps=2):
        args = ["recover_qwen.py", "--source", str(source), "--initialization", str(initialization),
                "--audit", str(audit), "--train", str(train), "--validation", str(validation),
                "--out", str(out), "--objective", objective, "--steps", "4", "--loop-k", "2",
                "--seq-len", "8", "--batch-size", "1", "--checkpoint-every", "2",
                "--muon-lr", "0.01", "--adamw-lr", "0.001", "--device", "cpu",
                "--chunk-tokens", "3", "--max-session-steps", str(session_steps)]
        monkeypatch.setattr(sys, "argv", args)
        recover_qwen.main()
    return run


@pytest.mark.parametrize("objective", ["ce", "kl"])
def test_recovery_command_resume_and_evaluate(recovery_fixture, tmp_path, objective):
    run = tmp_path / "run"
    reference = tmp_path / "reference"
    recovery_fixture(run, objective)
    first = json.loads((run / "evaluation-step-000002.json").read_text())
    assert first["step"] == 2 and first["skipped_nonfinite"] == 0
    recovery_fixture(run, objective)
    result = json.loads((run / "evaluation-step-000004.json").read_text())
    assert result["step"] == 4
    recovery_fixture(reference, objective, 4)
    expected = json.loads((reference / "evaluation-step-000004.json").read_text())
    assert result["evaluation"] == expected["evaluation"]
    resumed, _ = CheckpointManager(run).load_latest()
    uninterrupted, _ = CheckpointManager(reference).load_latest()
    for key, value in resumed["model"].items():
        assert torch.equal(value, uninterrupted["model"][key]), key
    # A repeated completed invocation does not overwrite reports or checkpoints.
    report_before = (run / "evaluation-step-000004.json").read_bytes()
    recovery_fixture(run, objective)
    assert report_before == (run / "evaluation-step-000004.json").read_bytes()
    # A session killed after the final checkpoint can still publish evaluation.
    (run / "evaluation-step-000004.json").unlink()
    recovery_fixture(run, objective)
    recovered = json.loads((run / "evaluation-step-000004.json").read_text())
    assert recovered["evaluation"] == result["evaluation"]


def test_recovery_config_freezes_depth_without_changing_initial_config():
    original = tiny_model_config().to_dict()
    config = recover_qwen.recovery_config(original, 3, 32)
    assert config.recurrent.train_loop_min == config.recurrent.train_loop_max == 3
    assert config.recurrent.default_loop_k == config.recurrent.truncated_backprop_steps == 3
    assert original == tiny_model_config().to_dict()
