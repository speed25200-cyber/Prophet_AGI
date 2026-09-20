"""Final measurements on miniature runs: the depth sweep reproduces the saved score, the
cache law separates the two cores, quantization error orders by bit width."""

import json

import pytest
import torch

from prophet.modeling.model import ProphetModel
from prophet.quant.rtn import block_linear_modules, quantize_model_rtn, quantize_tensor_rtn
from scripts.eval_loop_core import main as evaluate
from scripts.run_loop_core import main as train
from tests.test_run_loop_core import base_args, build_loop_core_fixture, tiny_config

corpus = pytest.fixture(scope="module")(build_loop_core_fixture)


def _train(corpus, arm, out, capsys):
    assert train(base_args(corpus, arm, out, total_steps=2, max_session_steps=2)) == 0
    assert "RUN_COMPLETE" in capsys.readouterr().out


def _measure(corpus, out, report, **extra):
    argv = [
        "--run",
        str(out),
        "--step",
        "2",
        "--corpus",
        str(corpus["corpus"]),
        "--tokenizer",
        str(corpus["tokenizer"]),
        "--tokenizer-sha256",
        corpus["tokenizer_sha256"],
        "--out",
        str(report),
        "--depths",
        "1",
        "2",
        "3",
        "--composition-depths",
        "2",
        "--composition-per-level",
        "1",
        "--cache-contexts",
        "16",
        "32",
        "--cache-depths",
        "2",
        "3",
        "--quant-bits",
        "8",
        "4",
        "--quant-depths",
        "2",
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--allow-cpu",
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", *map(str, value)]
    assert evaluate(argv) == 0
    return json.loads(report.read_text())


def core_bytes(report, *, context, loop_k):
    entry = next(e for e in report["cache"] if e["context"] == context and e["loop_k"] == loop_k)
    return entry["by_section"]["core"]


def test_measurements_on_the_gdn_core(corpus, tmp_path, capsys):
    out = tmp_path / "gdn"
    _train(corpus, "lc_gdn", out, capsys)
    report = _measure(corpus, out, tmp_path / "gdn.json")
    assert report["complete"] and report["arm"] == "lc_gdn" and report["looped"]
    assert set(report["text"]) == {"1", "2", "3"}
    # Training evaluated on the CPU in FP32 too: the default depth must reproduce it.
    saved = json.loads((out / "evaluation-step-000002.json").read_text())
    assert report["text"]["2"]["nats_per_token"] == pytest.approx(saved["nats_per_token"], abs=1e-5)
    assert "table:1" in report["composition"]["2"]["levels"]
    assert all("nats" not in item for item in report["composition"]["2"]["items"])
    # A bounded-state core: the same bytes at any context, more slots with depth.
    assert core_bytes(report, context=16, loop_k=2) == core_bytes(report, context=32, loop_k=2)
    assert core_bytes(report, context=16, loop_k=3) > core_bytes(report, context=16, loop_k=2)
    assert set(report["quantization"]) == {"8", "4"}
    assert (
        report["quantization"]["4"]["report"]["relative_l2_error"]
        > report["quantization"]["8"]["report"]["relative_l2_error"]
    )
    for bits in ("8", "4"):
        assert torch.isfinite(
            torch.tensor(report["quantization"][bits]["text"]["2"]["bits_per_byte"])
        )
    with pytest.raises(FileExistsError):
        _measure(corpus, out, tmp_path / "gdn.json")


def test_measurements_on_the_attention_core_show_the_cache_growing_with_context(
    corpus, tmp_path, capsys
):
    out = tmp_path / "attn"
    _train(corpus, "lc_attn", out, capsys)
    report = _measure(corpus, out, tmp_path / "attn.json")
    assert report["complete"] and report["arm"] == "lc_attn"
    assert core_bytes(report, context=32, loop_k=2) > core_bytes(report, context=16, loop_k=2)
    assert core_bytes(report, context=16, loop_k=3) > core_bytes(report, context=16, loop_k=2)
    kinds = next(e for e in report["cache"] if e["loop_k"] == 2)["by_kind"]
    assert "AttentionCache" in kinds


def test_the_unshared_arm_is_measured_at_depth_one_only(corpus, tmp_path, capsys):
    out = tmp_path / "plain"
    _train(corpus, "lc_plain", out, capsys)
    report = _measure(corpus, out, tmp_path / "plain.json")
    assert not report["looped"]
    assert list(report["text"]) == ["1"] and list(report["composition"]) == ["1"]
    assert {e["loop_k"] for e in report["cache"]} == {1}
    assert list(report["quantization"]["8"]["text"]) == ["1"]


def test_wrong_step_is_refused(corpus, tmp_path, capsys):
    out = tmp_path / "wrong"
    _train(corpus, "lc_gdn", out, capsys)
    with pytest.raises(FileNotFoundError):
        _measure(corpus, out, tmp_path / "w.json", step=[3])


def test_rtn_rounds_to_the_grid_and_only_touches_block_linears():
    weight = torch.tensor([[0.75, -1.0, 0.25], [0.0, 0.0, 0.0]])
    dequantized, codes = quantize_tensor_rtn(weight, 4)
    assert codes.tolist() == [[5, -7, 2], [0, 0, 0]]
    assert dequantized[0].tolist() == pytest.approx([5 / 7, -1.0, 2 / 7])
    assert dequantized[1].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        quantize_tensor_rtn(weight, 1)

    model = ProphetModel(tiny_config("gdn"))
    names = block_linear_modules(model)
    assert names and all(name.startswith("sections.") for name in names)
    quantized, report = quantize_model_rtn(model, 8)
    assert report["quantized_linear_modules"] == len(names)
    assert torch.equal(quantized.embed.weight, model.embed.weight)
    changed = [
        n
        for n, m in quantized.named_modules()
        if isinstance(m, torch.nn.Linear)
        and not torch.equal(m.weight, dict(model.named_modules())[n].weight)
    ]
    assert changed and set(changed) <= set(names)
