"""The three arms of the loop-core programme differ by the core mixer and nothing else."""

import json
from pathlib import Path

import pytest

from prophet.budget import count_parameters
from prophet.config import ProphetConfig
from scripts.build_loop_core_configs import (
    ACCEPTED_WARNING_PREFIX,
    PARAM_MATCH_TOLERANCE,
    build_all,
    describe,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def arms():
    configs, match = build_all()
    return configs, match


def test_every_arm_validates_and_executes_twenty_blocks_per_token(arms):
    configs, _ = arms
    for name, cfg in configs.items():
        cfg.validate()
        info = describe(name, cfg)
        assert info["effective_depth"] == 20, name
        assert cfg.recurrent.halting == "none"
        assert cfg.heads.n_multi_token_predict == 0 and not cfg.heads.confidence_head
        assert cfg.recurrent.input_adapter == "none"


def test_looped_arms_share_the_elastic_depth_policy_with_full_backprop(arms):
    configs, _ = arms
    for name in ("lc_gdn", "lc_attn"):
        r = configs[name].recurrent
        assert (r.train_loop_min, r.train_loop_max, r.train_loop_dist) == (2, 6, "uniform")
        assert r.truncated_backprop_steps == 6, "no visited pass may be detached"
        assert r.default_loop_k == 4
    r = configs["lc_plain"].recurrent
    assert r.train_loop_min == r.train_loop_max == r.default_loop_k == 1


def test_only_the_core_mixer_differs_between_the_looped_arms(arms):
    configs, match = arms
    gdn, attn = configs["lc_gdn"], configs["lc_attn"]
    assert gdn.recurrent.core_pattern == ["gdn"]
    assert attn.recurrent.core_pattern == ["full_attn"]
    for section in ("prelude", "coda"):
        assert [k for s, _, k in gdn.section_layout() if s == section] == [
            k for s, _, k in attn.section_layout() if s == section
        ]
    assert gdn.d_model == attn.d_model and gdn.mixer == attn.mixer
    assert abs(match["relative_gap"]) <= PARAM_MATCH_TOLERANCE
    assert count_parameters(attn).total == match["params"]


def test_the_attention_arm_accepts_exactly_the_d1_warning_and_nothing_else(arms):
    configs, _ = arms
    warnings = configs["lc_attn"].design_warnings()
    assert len(warnings) == 1 and warnings[0].startswith(ACCEPTED_WARNING_PREFIX)
    assert configs["lc_gdn"].design_warnings() == []
    assert configs["lc_plain"].design_warnings() == []
    with pytest.raises(ValueError, match="only the lc_attn ablation"):
        describe("lc_gdn", configs["lc_attn"])
    with pytest.raises(ValueError, match="must trip"):
        describe("lc_attn", configs["lc_gdn"])


def test_shipped_configs_match_the_generator(arms):
    configs, _ = arms
    shipped = ROOT / "configs" / "loop_core"
    summary = json.loads((shipped / "summary.json").read_text())
    for name, cfg in configs.items():
        on_disk = ProphetConfig.from_json(shipped / f"{name}.json")
        assert on_disk.to_dict() == cfg.to_dict(), f"{name}: regenerate with the script"
        assert summary["arms"][name]["params_resident"] == count_parameters(cfg).total
