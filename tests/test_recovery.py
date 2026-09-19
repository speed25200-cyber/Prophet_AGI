"""Clone recovery: real data, interruption and the actual R04 experiment contract."""

import json

import pytest

from prophet.budget import block_passes_per_token, count_parameters, expected_train_loop_k
from prophet.data.corpus import LocalTextSource, TokenisedSource
from prophet.data.decontaminate import Decontaminator
from prophet.data.streaming import StreamingLoader
from prophet.data.tokenizer import SPECIAL_TOKENS, ProphetTokenizer
from scripts.build_configs import CONFIGS
from scripts.train_tokenizer import sample_documents
from tests.test_corpus import _mixture, _phase_loader


def test_phase_checkpoint_survives_json_serialisation():
    original = _phase_loader()
    list(original.batches(4))
    state = json.loads(json.dumps(original.state().to_dict()))
    expected = list(original.batches(6))
    resumed = _phase_loader()
    resumed.restore(state)
    assert list(resumed.batches(6)) == expected


@pytest.mark.parametrize("change", ["text", "tokenizer", "decontamination", "control", "cap"])
def test_real_resume_refuses_changed_data_contract_before_mutation(tmp_path, change):
    path = tmp_path / "web.jsonl"
    path.write_text(json.dumps({"text": "aa bb cc dd ee ff"}) + "\n", encoding="utf-8")

    def make(changed=False):
        tok = ProphetTokenizer([(b"a", b"a")] if changed and change == "tokenizer" else [])
        decon = Decontaminator()
        if changed and change == "decontamination":
            decon.add_benchmark("heldout", ["one two three four five six"])
        return StreamingLoader([TokenisedSource(
            LocalTextSource.from_root(tmp_path, "web", 1), tok,
            decontaminator=decon, max_epochs=2 if changed and change == "cap" else 4,
            parse_special=changed and change == "control",
        )], seq_len=4)

    original = make()
    list(original.batches(1))
    state = original.state().to_dict()
    if change == "text":
        # Same size, different bytes: mtime/length-only identity would miss this.
        path.write_text(path.read_text().replace("aa", "zz"), encoding="utf-8")
    resumed = make(True)
    before = resumed.state().to_dict()
    with pytest.raises(ValueError, match="fingerprint"):
        resumed.restore(state)
    assert resumed.state().to_dict() == before


def test_all_rejected_corpus_does_not_spin_forever(tmp_path):
    text = "one two three four five six"
    (tmp_path / "web.txt").write_text(text + "\n", encoding="utf-8")
    decon = Decontaminator()
    decon.add_benchmark("test", [text])
    source = TokenisedSource(LocalTextSource.from_root(tmp_path, "web", 1),
                             ProphetTokenizer([]), decontaminator=decon, max_epochs=None)
    loader = StreamingLoader([source], seq_len=8, separator=256)
    with pytest.raises(RuntimeError, match="cannot fill"):
        next(loader.batches(1))


def test_tokenizer_sampler_handles_unequal_source_lengths(tmp_path):
    (tmp_path / "small.txt").write_text("first\n", encoding="utf-8")
    (tmp_path / "large.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    sources = [LocalTextSource.from_root(tmp_path, name, 1) for name in ("small", "large")]
    assert sample_documents(sources, 99) == ["first", "a", "b", "c", "d"]


def test_control_tokens_are_opt_in_and_old_prefix_ids_stay_stable(tmp_path):
    tok = ProphetTokenizer([])
    text = "<|tool|>hi<|assistant|>"
    assert tok.special_id("<|tool|>") not in tok.encode(text)
    assert tok.decode(tok.encode(text, parse_special=True), skip_special=False) == text
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["special_tokens"] = list(SPECIAL_TOKENS[:11])
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = ProphetTokenizer.load(path)
    assert restored.special_id("<|think|>") == 266
    assert restored.encode("hello") == tok.encode("hello")
    payload["special_tokens"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="special-token"):
        ProphetTokenizer.load(path)


def test_mixture_rescaling_is_independent_and_honours_requested_tokens():
    mixture = _mixture()
    scaled = mixture.rescale(1234)
    assert scaled.total_tokens == 1234 and mixture.total_tokens == 4000
    scaled.phases[0].sources[0].weight = 0
    assert mixture.phases[0].sources[0].weight == 0.5


def test_r04_is_above_plan_floor_and_has_matched_executed_blocks():
    loop, plain = (CONFIGS[f"prophet_r04_{arm}.json"] for arm in ("loop", "plain"))
    assert count_parameters(loop).total >= 350_000_000
    for cfg, k in ((loop, 4), (plain, 1)):
        r = cfg.recurrent
        assert r.train_loop_min == r.train_loop_max == r.default_loop_k == k
        assert r.truncated_backprop_steps >= k
        assert r.halting == "none" and not r.iteration_readout
        assert cfg.heads.n_multi_token_predict == 0
        assert block_passes_per_token(cfg, expected_train_loop_k(cfg)) == 20
    assert loop.d_model == plain.d_model
    assert loop.mixer == plain.mixer
    assert loop.ffn == plain.ffn


def test_training_resume_is_bit_exact_across_a_real_corpus_phase_boundary(tmp_path):
    import copy

    import torch

    from prophet.data.corpus import build_loader
    from prophet.modeling.model import ProphetModel
    from prophet.train.loop import TrainConfig, Trainer
    from tests.test_training import tiny_model_config

    for name in ("web", "code"):
        (tmp_path / f"{name}.jsonl").write_text("".join(
            json.dumps({"text": f"{name}: sample {i} of distinct training text"}) + "\n"
            for i in range(30)
        ), encoding="utf-8")
    mixture = _mixture().rescale(64)
    mixture.phases[0].weight = mixture.phases[1].weight = 0.5

    def make():
        torch.manual_seed(13)
        cfg = tiny_model_config()
        cfg.frontend.vocab_size = 512
        loader = build_loader(mixture, tokenizer=ProphetTokenizer([]), seq_len=8,
                              batch_size=2, local_root=tmp_path, seed=13)
        return Trainer(ProphetModel(cfg), loader, TrainConfig(
            total_steps=4, batch_size=2, seq_len=8, device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"), checkpoint_every=0,
            log_every=100, activation_checkpointing=False,
        ), model_config=cfg)

    full = make()
    full.train()
    split = make()
    split.train(max_steps=2)  # Immediately before switching to phase B.
    saved = copy.deepcopy(split.state_dict())
    resumed = make()
    resumed.load_state_dict(saved)
    resumed.train()
    assert resumed.step == 4
    assert resumed.loader.phase == full.loader.phase == 1
    assert resumed.loader.state().to_dict() == full.loader.state().to_dict()
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
