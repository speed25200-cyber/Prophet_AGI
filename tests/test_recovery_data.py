import hashlib
import json

import pytest

from scripts.prepare_recovery_data import prepare
from scripts.rehearse_qwen_conversion import digest


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    splits = {}
    for split, rows in (("train", ["retained train", "benchmark overlap"]),
                        ("validation", ["development holdout", "inspected diagnostic"])):
        path = corpus / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in rows))
        splits[split] = {"files": [{"path": path.name, "sha256": digest(path)}],
                         "documents": 2, "matches": [{"sha256": text_sha(rows[1])}] if split == "train" else []}
    overlap, smoke = tmp_path / "overlap.json", tmp_path / "smoke.json"
    overlap.write_text(json.dumps({"complete": True, "splits": splits,
                                   "all_requested_sources_indexed": False}))
    smoke.write_text(json.dumps({"complete": True, "arms": {"hybrid_initialization": {
        "documents": [{"document_sha256": text_sha("inspected diagnostic")}]}}}))
    return corpus, overlap, smoke


def test_preparation_removes_only_audited_hashes_and_preserves_sources(inputs, tmp_path):
    corpus, overlap, smoke = inputs
    originals = {p.name: p.read_bytes() for p in corpus.iterdir()}
    out = tmp_path / "prepared"
    report = prepare(corpus, overlap, smoke, out)
    assert report["complete"] and not report["all_requested_benchmarks_screened"]
    for split, expected in (("train", "retained train"), ("validation", "development holdout")):
        assert json.loads((out / f"{split}.jsonl").read_text())["text"] == expected
        assert report["splits"][split]["retained_documents"] == 1
    assert originals == {p.name: p.read_bytes() for p in corpus.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(corpus, overlap, smoke, out)


def test_preparation_refuses_changed_source_before_creating_output(inputs, tmp_path):
    corpus, overlap, smoke = inputs
    (corpus / "train.jsonl").write_text('{"text":"changed"}\n')
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="differs"):
        prepare(corpus, overlap, smoke, out)
    assert not out.exists()
