"""Real CLI warm-start/restart tests using miniature local R04 artifacts.

Only the published experiment's pins/scale and repository identity are replaced.
Data packing, checkpoint hashing, stochastic depth, training and evaluation are real.
"""

import copy
import json
import sys

import pytest
import torch

from prophet.config import HeadsConfig
from prophet.data.corpus import LocalTextSource, TokenisedSource
from prophet.data.streaming import StreamingLoader
from prophet.data.tokenizer import ProphetTokenizer
from prophet.eval.text import evaluate_documents
from prophet.modeling.model import ProphetModel
from prophet.train.checkpoint import CheckpointManager
from prophet.train.loop import TrainConfig, Trainer
from scripts import adapt_r04_depth as driver
from scripts.audit_r04_restart import audit, compare_states
from tests.test_training import tiny_model_config


@pytest.fixture(autouse=True)
def restore_numerical_flags():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def fixture(tmp_path, monkeypatch, device):
    corpus = tmp_path / "corpus"
    for split, text in [("train", "abc def " * 32), ("validation", "a new text")]:
        folder = corpus / split / "fineweb-edu"
        folder.mkdir(parents=True)
        (folder / "data.jsonl").write_text(json.dumps({"text": text}) + "\n")
    tokenizer = ProphetTokenizer([], vocab_size=512)
    tokenizer.save(corpus / "tokenizer.json")
    cfg = tiny_model_config()
    cfg.frontend.vocab_size = 512
    cfg.heads = HeadsConfig(n_multi_token_predict=0, confidence_head=False)
    cfg.recurrent.train_loop_min = cfg.recurrent.train_loop_max = cfg.recurrent.default_loop_k = 4
    cfg.recurrent.truncated_backprop_steps = 4
    cfg.validate()
    source = TokenisedSource(
        LocalTextSource.from_root(corpus / "train", "fineweb-edu", 1.0), tokenizer
    )
    loader = StreamingLoader([source], seq_len=8, batch_size=1, seed=0)
    torch.manual_seed(12)
    parent_run = tmp_path / "parent"
    trainer = Trainer(
        ProphetModel(cfg),
        loader,
        TrainConfig(
            total_steps=1,
            batch_size=1,
            seq_len=8,
            device=device,
            loss_chunk_tokens=3,
            checkpoint_dir=str(parent_run / "checkpoints"),
            checkpoint_every=0,
        ),
        model_config=cfg,
        tokenizer=tokenizer,
        on_log=lambda _: None,
    )
    trainer.train()
    meta = trainer.ckpt.save(trainer.state_dict(), 1)
    provenance = {"fixture": "miniature real text"}
    parent_protocol = {
        "config": cfg.to_dict(),
        "variant": "loop",
        "seed": 0,
        "batch_size": 1,
        "seq_len": 8,
        "data": provenance,
        "runtime": driver.runtime_for(device),
        "loss_chunk_tokens": 3,
    }
    evaluation = evaluate_documents(
        trainer.model,
        ["a new text"],
        tokenizer,
        seq_len=8,
        batch_size=1,
        device=device,
        loss_chunk_tokens=3,
        loop_k=4,
    )
    evaluation.update(
        step=1, train_tokens=8, checkpoint=meta.to_dict(), run_protocol=parent_protocol
    )
    driver.write_json(parent_run / "protocol.json", parent_protocol)
    driver.write_json(parent_run / "evaluation-step-000001.json", evaluation)
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    plan = {
        "parent_checkpoint": meta.to_dict(),
        "steps_each": 4,
        "batch_size": 1,
        "seq_len": 8,
        "original_training_corpus_tokens": 1000,
        "parameters_each": sum(p.numel() for p in trainer.model.parameters()),
        "recipe": {"muon_lr": 0.001, "adamw_lr": 0.00003, "weight_decay": 0.1, "grad_clip": 1.0},
    }
    driver.write_json(plan_dir / "protocol.json", plan)
    for arm in ["fixed4", "uniform2to6"]:
        driver.adaptation_config(cfg.to_dict(), arm).to_json(plan_dir / f"{arm}.json")
    monkeypatch.setattr(driver, "PLAN_DIR", plan_dir)
    monkeypatch.setattr(driver, "code_identity", lambda: {"revision": "fixture"})
    monkeypatch.setattr(driver, "verify_pilot", lambda path: provenance)
    parent_state = trainer.state_dict()
    parent_state = copy.deepcopy(parent_state)
    del trainer
    if device == "cuda":
        torch.cuda.empty_cache()

    def run(out, arm, session=4):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_r04_depth.py",
                "--parent-run",
                str(parent_run),
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--arm",
                arm,
                "--device",
                device,
                "--max-session-steps",
                str(session),
            ],
        )
        driver.main()

    return run, parent_state, parent_run, plan


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="actual CUDA depth restart"
            ),
        ),
    ],
)
@pytest.mark.parametrize("arm", ["fixed4", "uniform2to6"])
def test_cli_warm_start_resume_and_interrupted_final_evaluation(tmp_path, monkeypatch, device, arm):
    run, parent, _, _ = fixture(tmp_path, monkeypatch, device)
    resumed, continuous = tmp_path / "resumed", tmp_path / "continuous"
    run(resumed, arm, 1)
    initial = torch.load(
        resumed / "checkpoints/ckpt_slot0.pt", weights_only=True, map_location="cpu"
    )
    assert initial["step"] == initial["tokens_seen"] == 0
    assert_equal(initial["model"], {k: v.cpu() for k, v in parent["model"].items()})
    assert initial["loader"] == parent["loader"]
    assert all(not opt["state"] for opt in initial["optimizers"])
    identity = initial["training_contract"]["run_identity"]
    assert identity["protocol"] == "r04-depth-adaptation-v2"
    assert identity["numerical_policy"]["deterministic_algorithms"] is True
    assert identity["numerical_policy"]["warn_only"] is False
    other_arm = "fixed4" if arm == "uniform2to6" else "uniform2to6"
    with pytest.raises(ValueError, match="another adaptation experiment"):
        run(resumed, other_arm)
    run(resumed, arm)
    run(continuous, arm)
    resumed_state, _ = CheckpointManager(resumed / "checkpoints").load_latest()
    continuous_state, _ = CheckpointManager(continuous / "checkpoints").load_latest()
    assert_equal(resumed_state, continuous_state)
    audit_report = audit(resumed, continuous, 4)
    assert audit_report["passed"] and audit_report["state_comparison"]["tensors"] > 0
    assert len(resumed_state["adaptation_depth_history"]) == 4
    if arm == "uniform2to6":
        assert len(set(resumed_state["adaptation_depth_history"])) > 1
    else:
        assert resumed_state["adaptation_depth_history"] == [4] * 4
    path = resumed / "evaluation-step-000004.json"
    report = json.loads(path.read_bytes())
    expected = json.loads((continuous / path.name).read_bytes())
    assert report["complete"] and report["results"] == expected["results"]
    before = path.read_bytes()
    run(resumed, arm)
    assert path.read_bytes() == before
    partial = {**report, "complete": False, "results": {"4": report["results"]["4"]}}
    driver.write_json(path, partial)
    run(resumed, arm)
    recovered = json.loads(path.read_bytes())
    assert recovered == report
    after, _ = CheckpointManager(resumed / "checkpoints").load_latest()
    assert_equal(after, resumed_state)


def test_changed_numerical_policy_cannot_resume(tmp_path, monkeypatch):
    run, _, _, _ = fixture(tmp_path, monkeypatch, "cpu")
    output = tmp_path / "run"
    run(output, "fixed4", 1)
    manifest = output / "adaptation.json"
    changed = json.loads(manifest.read_bytes())
    changed["identity"]["numerical_policy"]["deterministic_algorithms"] = False
    driver.write_json(manifest, changed)
    with pytest.raises(ValueError, match="another adaptation experiment"):
        run(output, "fixed4", 1)


def test_cuda_policy_requires_workspace_before_start(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="before starting Python"):
        driver.configure_numerics("cuda")


def test_restart_audit_detects_optimizer_and_rng_corruption():
    saved = {
        "optimizers": [{"state": {0: {"momentum": torch.ones(2)}}}],
        "rng": torch.tensor([1, 2], dtype=torch.uint8),
    }
    for component in ("optimizers", "rng"):
        changed = copy.deepcopy(saved)
        if component == "optimizers":
            changed[component][0]["state"][0]["momentum"][1] += 1e-5
        else:
            changed[component][0] = 0
        with pytest.raises(ValueError, match=component):
            compare_states(saved, changed)


def test_depth_policy_does_not_change_topology_or_parent_config():
    cfg = tiny_model_config()
    cfg.heads = HeadsConfig(n_multi_token_predict=0, confidence_head=False)
    cfg.recurrent.train_loop_min = cfg.recurrent.train_loop_max = cfg.recurrent.default_loop_k = 4
    original = cfg.to_dict()
    changed = driver.adaptation_config(original, "uniform2to6").to_dict()
    expected = copy.deepcopy(original)
    expected["recurrent"].update(
        train_loop_min=2, train_loop_max=6, train_loop_dist="uniform", truncated_backprop_steps=6
    )
    assert changed == expected and cfg.to_dict() == original
    cfg.recurrent.token_depth = True
    with pytest.raises(ValueError, match="plain R04 policy"):
        driver.adaptation_config(cfg.to_dict(), "uniform2to6")


def test_parent_hash_and_publication_mismatch_are_rejected(tmp_path, monkeypatch):
    _, _, parent, plan = fixture(tmp_path, monkeypatch, "cpu")
    bad = copy.deepcopy(plan)
    bad["parent_checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="publication differs"):
        driver.load_parent(parent, bad)
    path = parent / f"checkpoints/ckpt_slot{plan['parent_checkpoint']['slot']}.pt"
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        driver.load_parent(parent, plan)
