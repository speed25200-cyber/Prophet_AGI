"""Check padded native scoring against independent and single-candidate oracles."""

import copy
import json
import sys

import pytest
import torch

from prophet.data.tokenizer import ProphetTokenizer
from prophet.eval.choices import batched_continuation_nats, continuation_bucket, continuation_nats
from scripts import adapt_r04_reinjection
from scripts import eval_arc_native as native
from scripts.eval_arc_recovery import encoded_items, evaluate_items
from scripts.prepare_arc_native_eval import retokenize
from scripts.prepare_arc_recovery_eval import prepare_rows
from scripts.run_r04_pilot import sha256, tokenizer_semantic_hash
from tests.test_choice_eval import CharacterTokenizer, FixedDistribution, source_row
from tests.test_depth_adaptation import fixture, restore_numerical_flags  # noqa: F401
from tests.test_input_adapter import pair


def test_fixed_shape_batch_matches_probability_oracle_and_preserves_mode():
    model = FixedDistribution().train()
    candidates = [([1, 2, 97], 2), ([3, 97, 98, 97], 1), ([4, 98], 1)]
    expected = [continuation_nats(model, ids, start) for ids, start in candidates]
    progress = []
    actual = batched_continuation_nats(
        model,
        candidates,
        pad_id=0,
        batch_size=2,
        seq_len=8,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert actual == pytest.approx(expected, abs=1e-6)
    assert progress == [(2, 3), (3, 3)] and model.training


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA fixed-shape native scoring"
            ),
        ),
    ],
)
def test_learned_native_model_padding_matches_single_candidates(device):
    if device == "cuda":
        native.engine.configure_numerics(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _, model = pair()
    torch.nn.init.normal_(model.core_input_adapter.weight, std=0.02)
    model.to(device).eval()
    candidates = [([1, 2, 8, 5, 7, 3, 2], 3), ([3, 4, 5], 1), ([8, 9], 1)]
    expected = [
        continuation_nats(model, ids, start, loop_k=6, device=device) for ids, start in candidates
    ]
    actual = batched_continuation_nats(
        model, candidates, pad_id=0, batch_size=2, seq_len=16, loop_k=6, device=device
    )
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-4)


def test_batch_rejects_truncation_and_nonfinite_answers():
    model = FixedDistribution().train()
    with pytest.raises(ValueError, match="truncation"):
        batched_continuation_nats(model, [([1, 2, 3], 1)], pad_id=0, batch_size=2, seq_len=1)
    model.logits[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        batched_continuation_nats(model, [([1, 2, 3], 1)], pad_id=0, batch_size=2, seq_len=8)
    assert model.training


def test_retokenization_preserves_all_questions_and_complete_scores():
    tokenizer = CharacterTokenizer()
    tokenizer.pad_id = 0
    rows = [source_row(), source_row("two")]
    rows[1]["choices"] = {"label": ["A", "B", "C"], "text": ["b", "a", "aa"]}
    rows[1]["answerKey"] = "B"
    original = prepare_rows(rows, tokenizer)
    before = copy.deepcopy(original)
    prepared = retokenize(original, tokenizer)
    assert prepared == original == before
    expected = evaluate_items(
        FixedDistribution(), encoded_items(prepared, tokenizer, max_tokens=64)
    )
    actual = native.evaluate(
        FixedDistribution(), prepared, tokenizer, batch_size=2, seq_len=64, device="cpu", loop_k=4
    )
    actual.pop("batch_oracle")
    actual.pop("buckets")
    assert actual == expected


def test_input_only_buckets_restore_source_order_and_match_single_scores():
    tokenizer = CharacterTokenizer()
    tokenizer.pad_id = 0
    rows = [source_row(str(i)) for i in range(3)]
    rows[0]["question"] = "A" * 110
    rows[1]["question"] = "B"
    rows[2]["question"] = "C" * 50
    prepared = prepare_rows(rows, tokenizer)
    expected = evaluate_items(
        FixedDistribution(), encoded_items(prepared, tokenizer, max_tokens=256)
    )
    actual = native.evaluate(
        FixedDistribution(), prepared, tokenizer, batch_size=2, seq_len=256, device="cpu", loop_k=6
    )
    assert [row["id"] for row in actual["items"]] == ["0", "1", "2"]
    assert [b["seq_len"] for b in actual.pop("buckets")] == [32, 128, 256]
    assert actual.pop("batch_oracle")["candidates"] == 6
    assert actual == expected
    with pytest.raises(ValueError, match="truncation"):
        continuation_bucket(258, max_seq_len=256)


def test_native_cli_loads_exact_final_weights_and_freezes_all_scores(tmp_path, monkeypatch):
    run, parent, parent_run, plan = fixture(
        tmp_path, monkeypatch, "cpu", adapt_r04_reinjection.specification()
    )
    plan["parent_report_sha256"] = sha256(parent_run / "evaluation-step-000001.json")
    native.engine.write_json(tmp_path / "plan/protocol.json", plan)
    folder = tmp_path / "trained"
    run(folder, "learned_mix")
    tokenizer_path = tmp_path / "corpus/tokenizer.json"
    tokenizer = ProphetTokenizer.load(tokenizer_path)
    tokenizer_hash = tokenizer_semantic_hash(tokenizer_path)
    items = prepare_rows([source_row(), source_row("two")], tokenizer)
    inputs = tmp_path / "choices"
    inputs.mkdir()
    (inputs / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    manifest = {
        "rows": len(items),
        "items_sha256": sha256(inputs / "items.jsonl"),
        "tokenizer_semantic_sha256": tokenizer_hash,
        "batch_size": 2,
        "maximum_input_length": 64,
        "buckets": [{"seq_len": 32, "candidates": 4, "batches": 2}],
    }
    (inputs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(native, "MANIFEST", inputs / "manifest.json")
    monkeypatch.setattr(native, "PLAN_DIR", tmp_path / "plan")
    monkeypatch.setattr(native, "PILOT_TOKENIZER_SHA256", tokenizer_hash)
    monkeypatch.setattr(native, "TRAINING_REVISION", "fixture")
    original, original_arm, _ = native.load_weights(parent_run, 1)
    assert original_arm == "original4096"
    assert all(
        torch.equal(original.state_dict()[key], value) for key, value in parent["model"].items()
    )
    output = tmp_path / "score.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_arc_native.py",
            "--run",
            str(folder),
            "--step",
            "4",
            "--items",
            str(inputs),
            "--tokenizer",
            str(tokenizer_path),
            "--loop-k",
            "6",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    native.main()
    result = json.loads(output.read_bytes())
    assert result["complete"] and result["arm"] == "learned_mix" and result["loop_k"] == 6
    assert result["evaluation"]["rows"] == 2
    assert result["evaluation"]["batch_oracle"]["max_absolute_nats_error"] < 1e-4
    with pytest.raises(ValueError, match="preserve"):
        native.main()
    monkeypatch.setattr(native, "TRAINING_REVISION", "changed")
    with pytest.raises(ValueError, match="unplanned"):
        native.load_weights(folder, 4)
