"""Provenance gates for the paired pilot, without allocating a large model."""

import json

import pytest

from scripts.run_r04_pilot import freeze_protocol, tokenizer_semantic_hash


def test_protocol_resume_is_nonmutating_and_rejects_changed_experiment(tmp_path):
    path = tmp_path / "protocol.json"
    original = {"seed": 0, "settings": {"betas": (0.9, 0.95), "steps": 4096}}
    freeze_protocol(path, original)
    before = path.read_bytes()
    freeze_protocol(path, json.loads(json.dumps(original)))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="protocol changed"):
        freeze_protocol(path, {**original, "seed": 1})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_semantic_fingerprint_accepts_platform_serialization_but_rejects_merges(tmp_path):
    path = tmp_path / "tokenizer.json"
    vocabulary = {"merges": [["a", "b"], ["c", "d"]], "name": "français"}
    path.write_bytes(json.dumps(vocabulary, indent=2, ensure_ascii=False).encode())
    expected = tokenizer_semantic_hash(path)
    path.write_bytes(json.dumps(vocabulary, indent=4).replace("\n", "\r\n").encode())
    assert tokenizer_semantic_hash(path) == expected
    vocabulary["merges"].reverse()
    path.write_text(json.dumps(vocabulary))
    assert tokenizer_semantic_hash(path) != expected
