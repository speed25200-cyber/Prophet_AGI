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


@pytest.mark.parametrize("arm", ["donor", "initialization"])
def test_full_development_evaluation_matches_unpadded_manual_windows(
    recovery_fixture, tmp_path, monkeypatch, arm
):
    import math

    from scripts import eval_qwen_recovery

    source = tmp_path / "source"
    validation = tmp_path / "validation.jsonl"
    texts = ["é🦊 abc def", "hello", "", "many windows " * 3]
    validation.write_text("\n".join(json.dumps({"text": text}) for text in texts), encoding="utf-8")
    data_audit = tmp_path / "data-audit.json"
    data_audit.write_text(json.dumps({"complete": True, "splits": {"validation": {
        "sha256": recover_qwen.digest(validation), "retained_documents": len(texts)}}}))
    out = tmp_path / "evaluation.json"
    args = ["eval_qwen_recovery.py", "--source", str(source), "--validation", str(validation),
            "--data-audit", str(data_audit), "--out", str(out), "--arm", arm,
            "--device", "cpu", "--seq-len", "8", "--batch-size", "3", "--loop-k", "2"]
    if arm == "initialization":
        args += ["--initialization", str(tmp_path / "initial.pt"), "--audit", str(tmp_path / "audit.json")]
    monkeypatch.setattr(sys, "argv", args)
    eval_qwen_recovery.main()
    report = json.loads(out.read_bytes())
    tokenizer = recover_qwen.DonorByteTokenizer(source / "tokenizer.json")
    if arm == "donor":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, attn_implementation="sdpa")
        extra = {"use_cache": False}
    else:
        payload = torch.load(tmp_path / "initial.pt", weights_only=True)
        model = ProphetModel(recover_qwen.recovery_config(payload["config"], 2, 8))
        model.load_state_dict(payload["model"])
        extra = {"loop_k": 2, "return_mtp": False}
    model.eval()
    expected_nats, expected_tokens, expected_bytes = 0.0, 0, 0
    with torch.no_grad():
        for text, measured in zip(texts, report["evaluation"]["documents"], strict=True):
            ids = tokenizer.encode(text, add_eos=True)
            doc_nats = 0.0
            for start in range(0, len(ids) - 1, 7):
                window = torch.tensor([ids[start:start + 8]])
                logits = model(window, **extra).logits
                doc_nats += torch.nn.functional.cross_entropy(
                    logits[0, :-1], window[0, 1:], reduction="none").double().sum().item()
            assert measured["total_nats"] == pytest.approx(doc_nats, abs=1e-5)
            expected_nats += doc_nats
            expected_tokens += len(ids) - 1
            expected_bytes += tokenizer.byte_length(ids[1:])
    assert report["complete"] and report["trained_steps"] == 0
    assert report["evaluation"]["scored_tokens"] == expected_tokens
    assert report["evaluation"]["scored_bytes"] == expected_bytes
    assert report["evaluation"]["bits_per_byte"] == pytest.approx(expected_nats / expected_bytes / math.log(2))
    assert (report["initialization_sha256"] is None) == (arm == "donor")
    with pytest.raises(FileExistsError, match="preserve"):
        eval_qwen_recovery.main()
    out.unlink()
    validation.write_text('{"text":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="validation differs"):
        eval_qwen_recovery.main()


def test_recovery_artifact_preflight_rejects_tampering(recovery_fixture, tmp_path):
    source = tmp_path / "source"
    with (source / "tokenizer.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="source differs"):
        recover_qwen.load_source(source)
    audit = tmp_path / "audit.json"
    record = json.loads(audit.read_bytes())
    record["config"]["d_model"] = 123
    audit.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="metadata mismatch"):
        recover_qwen.load_initialization(tmp_path / "initial.pt", audit)


def test_initialization_identity_ignores_archive_container(recovery_fixture, tmp_path, monkeypatch):
    from prophet.train.distillation import state_sha256
    from scripts import audit_initialization_identity

    original = tmp_path / "initial.pt"
    payload = torch.load(original, weights_only=True)
    other = tmp_path / "different_archive_name.pt"
    torch.save(payload, other)
    assert recover_qwen.digest(other) != recover_qwen.digest(original)
    reports = []
    for index, artifact in enumerate((original, other)):
        audit = json.loads((tmp_path / "audit.json").read_bytes())
        audit["checkpoint_sha256"] = recover_qwen.digest(artifact)
        audit_path = tmp_path / f"audit-{index}.json"
        audit_path.write_text(json.dumps(audit))
        out = tmp_path / f"identity-{index}.json"
        monkeypatch.setattr(sys, "argv", ["identity", "--initialization", str(artifact),
                                         "--audit", str(audit_path), "--out", str(out)])
        audit_initialization_identity.main()
        reports.append(json.loads(out.read_bytes()))
    assert reports[0]["checkpoint_sha256"] != reports[1]["checkpoint_sha256"]
    assert reports[0]["state_sha256"] == reports[1]["state_sha256"]
    assert reports[0]["config_sha256"] == reports[1]["config_sha256"]
    model = ProphetModel(recover_qwen.recovery_config(payload["config"], 2, 8))
    model.load_state_dict(payload["model"])
    assert state_sha256(model) == reports[0]["state_sha256"]
    reordered = dict(reversed(list(payload["model"].items())))
    assert state_sha256(reordered) == reports[0]["state_sha256"]
    payload["model"]["embed.weight"][0, 0] += 1
    assert state_sha256(payload["model"]) != reports[0]["state_sha256"]
