"""Hold-out leakage checks before the pilot consumes GPU time."""

import hashlib
import json

import pytest

from scripts.prepare_pilot import gram_hashes, partition_rows, split_for, write_pilot


def rows():
    return [{"text": " ".join(f"document{i}word{j}" for j in range(70)),
             "url": f"https://example.org/page{i}", "int_score": 3, "id": str(i)}
            for i in range(200)]


def test_deterministic_split_dedup_and_no_shared_heldout_spans():
    source = rows()
    train = next(r for r in source if split_for(r["url"], "", 42, 200) == "train")
    val = next(r for r in source if split_for(r["url"], "", 42, 200) == "validation")
    contaminated = {**train, "text": train["text"] + " " + val["text"]}
    duplicate = {**val, "text": val["text"].upper(), "url": train["url"]}
    source += [contaminated, duplicate]
    groups, stats = partition_rows(source, max_docs=1000, max_bytes=1000000, validation_per_mille=200)
    assert stats["duplicate_rejected"] == 1
    assert stats["heldout_overlap_rejected"] == 1
    from prophet.data.decontaminate import normalise
    train_spans = set().union(*(gram_hashes(normalise(r["text"])) for r in groups["train"]))
    val_spans = set().union(*(gram_hashes(normalise(r["text"])) for r in groups["validation"]))
    assert not train_spans & val_spans
    assert (groups, stats) == partition_rows(source, max_docs=1000, max_bytes=1000000,
                                            validation_per_mille=200)


def test_url_versions_stay_together():
    assert split_for("https://example.org/page/?x=2", "a", 42, 200) == split_for(
        "http://EXAMPLE.org/page?x=3#part", "b", 42, 200)


def test_recipe_filters_match_published_fineweb_schema():
    from prophet.data.corpus import row_passes
    from prophet.data.recipes import prophet_v1_mixture
    for phase in prophet_v1_mixture().phases:
        for source in phase.sources:
            if source.hf_id == "HuggingFaceFW/fineweb-edu":
                assert row_passes({"text": "example", "score": 4.1, "int_score": 4}, source.filters)
                assert not row_passes({"text": "example", "score": 2.1, "int_score": 2}, source.filters)


def test_caps_are_strict_and_schema_errors_fail_closed():
    groups, stats = partition_rows(rows(), max_docs=10, max_bytes=100000)
    assert stats["selected_docs"] == 10
    _, stats = partition_rows(rows(), max_docs=100, max_bytes=3000)
    assert stats["selected_bytes"] <= 3000 and stats["byte_cap_reached"]
    with pytest.raises(ValueError, match="schema"):
        partition_rows([{"text": "text", "edu_score": 3}], max_docs=1, max_bytes=100)


def test_artifact_hashes_and_existing_corpus_preserved(tmp_path):
    groups, stats = partition_rows(rows(), max_docs=200, max_bytes=1000000, validation_per_mille=200)
    out = tmp_path / "pilot"
    manifest = write_pilot(groups, out, {"stats": stats})
    assert json.loads((out / "manifest.json").read_text()) == manifest
    for relative, metadata in manifest["artifacts"].items():
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == metadata["sha256"]
    with pytest.raises(FileExistsError):
        write_pilot(groups, out, {})


def test_smoke_cli_writes_intermediate_and_final_checkpoints(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from prophet.data.tokenizer import ProphetTokenizer
    from prophet.train.checkpoint import CheckpointManager

    corpus, out = tmp_path / "corpus", tmp_path / "run"
    manifest = write_pilot({"train": [{"text": "abc def ghi " * 30}],
                            "validation": [{"text": "jkl mno pqr " * 30}]},
                           corpus, {"source_revision": "test-fixture"})
    tokenizer_path = corpus / "tokenizer.json"
    ProphetTokenizer(merges=[], vocab_size=512).save(tokenizer_path)
    metadata = {"tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                "sources": [{"fingerprint": {"files": [manifest["artifacts"][
                    "train/fineweb-edu/part-00000.jsonl"]["sha256"]]}}]}
    tokenizer_path.with_suffix(".json.metadata.json").write_text(json.dumps(metadata))
    subprocess.run([sys.executable, "scripts/pilot_smoke.py", "--corpus", str(corpus),
                    "--out", str(out), "--steps", "2", "--seq-len", "8", "--batch-size", "1"],
                   cwd=Path(__file__).resolve().parent.parent, check=True, capture_output=True,
                   env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}, timeout=60)
    report = json.loads((out / "report.json").read_text())
    assert report["resumed_from_step"] == 1 and report["steps"] == 2
    state, metadata = CheckpointManager(out / "checkpoints").load_latest()
    assert metadata.step == state["step"] == 2
    assert state["tokens_seen"] == 16
