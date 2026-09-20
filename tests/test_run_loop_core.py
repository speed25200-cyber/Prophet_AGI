"""The loop-core runner on a miniature corpus and model: frozen protocol, exact milestones,
depth history bound to the checkpoint, persistent snapshot and restore, all three arms."""

import json
import random
from pathlib import Path

import pytest
import torch

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.data.tokenizer import ProphetTokenizer
from scripts.prepare_loop_core_corpus import build
from scripts.run_loop_core import DepthTrainer, main, mixture_weights, verify_corpus
from scripts.run_r04_pilot import tokenizer_semantic_hash
from tests.test_loop_core_corpus import make_args, paragraph, row

SEQ_LEN, BATCH = 512, 2


def tiny_config(core: str, *, loop: bool = True) -> ProphetConfig:
    return ProphetConfig(
        name=f"tiny-{core}",
        d_model=32,
        n_layers=3,
        max_seq_len=SEQ_LEN,
        frontend=FrontendConfig(vocab_size=512, tie_word_embeddings=True),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"],
            n_heads=2,
            n_kv_heads=1,
            head_dim=16,
            sliding_window=SEQ_LEN,
            attention_sink_tokens=1,
            nope_layers=(1,),
            linear_heads=2,
            linear_head_dim=16,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=1,
            core_layers=1,
            coda_layers=1,
            train_loop_min=2 if loop else 1,
            train_loop_max=3 if loop else 1,
            train_loop_dist="uniform",
            default_loop_k=2 if loop else 1,
            halting="none",
            core_pattern=[core],
            coda_pattern=["full_attn"],
            truncated_backprop_steps=3 if loop else 1,
        ),
        heads=HeadsConfig(n_multi_token_predict=0, confidence_head=False),
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("loop-core")
    tokenizer_path = root / "tok.json"
    ProphetTokenizer([], vocab_size=512).save(tokenizer_path)
    rng = random.Random(3)
    stream = [row(paragraph(rng, 150), url=f"https://s{i}.org/p/{i}") for i in range(80)]
    args = make_args(
        root / "corpus",
        target_train_bytes=60_000,
        validation_docs=2,
        validation_per_mille=200,
        shard_bytes=10**6,
        composition_per_level=8,
        composition_test_per_level=2,
        tokenizer=tokenizer_path,
        token_sample_docs=5,
    )
    manifest = build(iter(stream), args, source_sha="x")
    assert manifest["tokens"]["splits"]["train"]["estimated_tokens"] > 0
    configs = {}
    for arm, core, loop in (
        ("lc_gdn", "gdn", True),
        ("lc_attn", "full_attn", True),
        ("lc_plain", "gdn", False),
    ):
        cfg = tiny_config(core, loop=loop)
        cfg.validate()
        path = root / f"{arm}.json"
        cfg.to_json(path)
        configs[arm] = path
    return {
        "root": root,
        "corpus": root / "corpus",
        "tokenizer": tokenizer_path,
        "tokenizer_sha256": tokenizer_semantic_hash(tokenizer_path),
        "configs": configs,
    }


def base_args(corpus: dict, arm: str, out: Path, **extra) -> list[str]:
    argv = [
        "--corpus",
        str(corpus["corpus"]),
        "--tokenizer",
        str(corpus["tokenizer"]),
        "--tokenizer-sha256",
        corpus["tokenizer_sha256"],
        "--arm",
        arm,
        "--seed",
        "0",
        "--out",
        str(out),
        "--config",
        str(corpus["configs"][arm]),
        "--batch-size",
        str(BATCH),
        "--seq-len",
        str(SEQ_LEN),
        "--composition-eval-per-level",
        "1",
        "--device",
        "cpu",
        "--allow-cpu",
        "--session-minutes",
        "10",
    ]
    for key, value in extra.items():
        argv += [
            f"--{key.replace('_', '-')}",
            *([str(v) for v in value] if isinstance(value, list) else [str(value)]),
        ]
    return argv


def test_corpus_provenance_and_mixture_weights(corpus):
    provenance = verify_corpus(corpus["corpus"])
    assert provenance["name"] == "loop-core-v1" and provenance["composition"]["test"]
    weights = mixture_weights(provenance, 0.03)
    assert 0 < weights["composition_weight"] < 1
    assert weights["web_weight"] + weights["composition_weight"] == pytest.approx(1.0)
    # The document weight yields a 3% token share given the two mean lengths.
    share = (
        weights["composition_weight"]
        * weights["mean_composition_tokens_per_doc"]
        / (
            weights["composition_weight"] * weights["mean_composition_tokens_per_doc"]
            + weights["web_weight"] * weights["mean_web_tokens_per_doc"]
        )
    )
    assert share == pytest.approx(0.03)
    with pytest.raises(ValueError, match="share"):
        mixture_weights(provenance, 1.5)


def test_dry_run_prints_the_protocol_without_touching_the_output(corpus, tmp_path, capsys):
    out = tmp_path / "dry"
    assert main(base_args(corpus, "lc_gdn", out, total_steps=5, dry_run=[])) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["arm"] == "lc_gdn" and protocol["steps"] == 5
    assert protocol["milestone_steps"] == []  # the default milestones are beyond 5 steps
    assert not out.exists()


def test_sessions_resume_hit_milestones_exactly_and_snapshot_to_persistent_storage(
    corpus, tmp_path, capsys
):
    out = tmp_path / "run"
    persistent = tmp_path / "drive"
    common = dict(
        total_steps=5,
        milestone_tokens=[2 * BATCH * SEQ_LEN],
        max_session_steps=3,
        persistent=persistent,
    )
    assert main(base_args(corpus, "lc_gdn", out, **common)) == 0
    first = capsys.readouterr().out
    assert "SESSION_COMPLETE_RESUME_REQUIRED" in first and "RUN_COMPLETE" not in first
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["milestone_steps"] == [2]
    milestone = json.loads((out / "milestone-step-000002.json").read_text())
    assert milestone["step"] == 2 and milestone["train_tokens"] == 2 * BATCH * SEQ_LEN
    assert "table:1" in milestone["composition"]["levels"]
    assert json.loads((out / "initial.json").read_text())["step"] == 0
    evaluation = json.loads((out / "evaluation-step-000003.json").read_text())
    assert evaluation["checkpoint"]["step"] == 3 and evaluation["run_protocol"] == protocol
    snapshot = persistent / "lc_gdn-seed0" / "step-000003"
    assert json.loads((snapshot / "SNAPSHOT_COMPLETE.json").read_text())["complete"]
    assert (snapshot / "checkpoints" / "manifest.json").exists()

    # A new machine: nothing local, the run restores from the snapshot and finishes.
    import shutil

    shutil.rmtree(out)
    assert main(base_args(corpus, "lc_gdn", out, **common)) == 0
    second = capsys.readouterr().out
    assert "RESTORED" in second and "RUN_COMPLETE" in second
    final = json.loads((out / "evaluation-step-000005.json").read_text())
    assert final["step"] == 5 and final["train_tokens"] == 5 * BATCH * SEQ_LEN
    state = torch.load(
        out / "checkpoints" / f"ckpt_slot{final['checkpoint']['slot']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert len(state["depth_history"]) == 5 and set(state["depth_history"]) <= {2, 3}
    assert final["depth_counts"] == {
        str(k): state["depth_history"].count(k) for k in sorted(set(state["depth_history"]))
    }
    assert sorted(p.name for p in (persistent / "lc_gdn-seed0").glob("step-*")) == ["step-000005"]

    # Repeating a finished run is idempotent; changing the recipe is refused.
    assert main(base_args(corpus, "lc_gdn", out, **common)) == 0
    assert "RUN_COMPLETE" in capsys.readouterr().out
    with pytest.raises(ValueError, match="protocol changed"):
        main(base_args(corpus, "lc_gdn", out, **{**common, "composition_share": 0.05}))


def test_attention_core_and_plain_arms_train_with_their_own_depth_policy(corpus, tmp_path, capsys):
    for arm, depths in (("lc_attn", {2, 3}), ("lc_plain", {1})):
        out = tmp_path / arm
        assert main(base_args(corpus, arm, out, total_steps=2, max_session_steps=2)) == 0
        assert "RUN_COMPLETE" in capsys.readouterr().out
        final = json.loads((out / "evaluation-step-000002.json").read_text())
        state = torch.load(
            out / "checkpoints" / f"ckpt_slot{final['checkpoint']['slot']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert len(state["depth_history"]) == 2 and set(state["depth_history"]) <= depths
        assert final["loop_k"] == (2 if arm == "lc_attn" else 1)


def test_depth_trainer_refuses_an_inconsistent_history(corpus):
    cfg = ProphetConfig.from_json(corpus["configs"]["lc_gdn"])
    trainer = DepthTrainer.__new__(DepthTrainer)
    trainer.model_config = cfg
    with pytest.raises(ValueError, match="depth history"):
        trainer.load_state_dict({"step": 2, "depth_history": [2, 9]})
