"""The committed programme queue is well formed and consistent with the runner."""

import json
import re
from pathlib import Path

import pytest

from scripts.colab_queue import load_queue
from scripts.run_loop_core import DEFAULT_TOTAL_STEPS
from scripts.stage_corpus import stage
from tests.test_run_loop_core import build_loop_core_fixture

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue" / "loop_core" / "programme.json"
corpus = pytest.fixture(scope="module")(build_loop_core_fixture)


def test_programme_queue_is_valid_and_binds_train_and_measure_commands():
    queue = load_queue(QUEUE)
    commands = {c["id"]: c for c in queue["commands"]}
    names = {m for c in queue["commands"] for a in c["argv"] for m in re.findall(r"\$\{(\w+)\}", a)}
    assert names == {"PROPHET_PILOT", "PROPHET_PERSISTENT"}
    trains = [c for c in queue["commands"] if c["id"].startswith("train-")]
    assert len(trains) == 7
    for train in trains:
        assert train["repeat_until"] == "RUN_COMPLETE" and train["max_attempts"] >= 12
        run_dir = train["argv"][train["argv"].index("--out") + 1]
        measure = commands["measure-" + train["id"][len("train-") :]]
        assert measure["argv"][measure["argv"].index("--run") + 1] == run_dir
        assert measure["argv"][measure["argv"].index("--step") + 1] == str(DEFAULT_TOTAL_STEPS)
        arm = train["argv"][train["argv"].index("--arm") + 1]
        seed = train["argv"][train["argv"].index("--seed") + 1]
        assert run_dir.endswith(f"{arm}-seed{seed}")
    order = [c["id"] for c in queue["commands"]]
    assert order.index("preflight-lc_attn") < order.index("train-lc_gdn-seed0")
    assert order.index("measure-lc_attn-seed0") < order.index("train-lc_gdn-seed1")
    assert order.index("train-lc_plain-seed0") > order.index("measure-lc_attn-seed2")
    assert queue["evidence"]["roots"] == ["outputs/loop-core"]
    assert "*.pt" in queue["evidence"]["exclude"]


def test_stage_corpus_copies_verifies_and_is_idempotent(corpus, tmp_path):
    dest = tmp_path / "local"
    first = stage(corpus["corpus"], dest)
    assert first["status"] == "staged" and (dest / "manifest.json").exists()
    assert stage(corpus["corpus"], dest)["status"] == "already-staged"
    manifest = json.loads((dest / "manifest.json").read_text())
    shard = next(iter(manifest["artifacts"]))
    (dest / shard).write_bytes(b"torn")
    again = stage(corpus["corpus"], dest)
    assert again["status"] == "staged"
    assert (dest / shard).read_bytes() == (corpus["corpus"] / shard).read_bytes()
