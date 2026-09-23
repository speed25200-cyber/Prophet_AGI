"""The miniature depth-as-a-dial probe (docs/36): the task is what it claims, the arms
read their schedule, and a run writes every hop count for every arm and seed."""

import json
import random

import pytest

from scripts import hops_cpu as h


def test_the_answer_is_the_table_applied_hops_times():
    rng = random.Random(3)
    for hops in range(1, h.MAX_HOPS + 1):
        ids, answer = h.make_example(rng, hops=hops)
        assert len(ids) == h.LENGTH and ids[-1] == h.EQ and ids[-2] == h.HOP0 + hops - 1
        table = {ids[i]: ids[i + 1] for i in range(0, 3 * h.TABLE, 3)}
        assert all(ids[i + 2] == h.SEP for i in range(0, 3 * h.TABLE, 3))
        assert sorted(table) == sorted(table.values())  # a permutation: every hop defined
        node = ids[-3]
        for _ in range(hops):
            node = table[node]
        assert node == answer
    with pytest.raises(ValueError):
        h.make_example(rng, hops=h.MAX_HOPS + 1)


def test_the_schedule_ties_or_fixes_the_loop_count():
    assert [h.loop_k("tied", n) for n in (1, 3, 6)] == [1, 3, 6]
    assert {h.loop_k("fixed", n) for n in (1, 3, 6)} == {h.FIXED_K}
    assert h.config("attn").recurrent.core_pattern == ["full_attn"]
    assert h.config("gdn").recurrent.core_pattern == ["gdn"]
    # Exact look-ups need a sharp attention: no qk_norm cap at head_dim 16 (docs/36 amend. 1).
    for core in ("gdn", "attn"):
        assert not any("qk_norm" in w for w in h.config(core).design_warnings())


def test_a_run_reports_every_hop_count_for_every_arm_and_seed(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "gdn-tied,attn-fixed",
        "--steps",
        "2",
        "--eval-n",
        "4",
        "--seeds",
        "0,1",
        "--warmup",
        "1",
    ]
    assert h.main(argv) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["arms"]) == {
        "gdn-tied-seed0",
        "attn-fixed-seed0",
        "gdn-tied-seed1",
        "attn-fixed-seed1",
    }
    for arm in report["arms"].values():
        assert set(arm["accuracy_by_hops"]) == {str(n) for n in range(1, h.MAX_HOPS + 1)}
        assert set(arm["k_sweep"]) == {"2", "3", "4"}
    with pytest.raises(SystemExit):
        h.main(["--out", str(tmp_path), "--arms", "rnn-tied"])


def test_the_one_hop_warm_start_feeds_one_hop_batches_first(monkeypatch):
    seen = []
    real = h.batch

    def spy(rng, *, n, hops):
        seen.append(hops)
        return real(rng, n=n, hops=hops)

    monkeypatch.setattr(h, "batch", spy)
    h.train(
        "gdn",
        "tied",
        steps=7,
        minutes=1,
        seed=0,
        lr=1e-3,
        warmup=1,
        warm_hops1=4,
        log=lambda _: None,
    )
    assert seen[:4] == [1, 1, 1, 1] and set(seen[4:]) <= set(h.TRAIN_HOPS) and len(seen) == 7


def test_the_small_width_recipe_leaves_the_attention_arms_no_warning():
    """docs/38: under the first recipe the attention arms sit in the regime where a
    look-up is learned unreliably, and design_warnings says so; the small-width recipe
    clears it (docs/36 amendment 3)."""
    for core in ("attn", "gdn"):
        assert any("small width" in w for w in h.config(core).design_warnings())
        cfg = h.config(core, **h.SMALL_WIDTH)
        assert not any("small width" in w for w in cfg.design_warnings())
        assert cfg.init_std == 0.06 and not cfg.frontend.tie_word_embeddings
        assert cfg.mixer.n_kv_heads == cfg.mixer.n_heads
