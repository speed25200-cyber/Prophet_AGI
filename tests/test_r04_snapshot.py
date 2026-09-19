import json

import pytest
from tests.test_checkpoint_audit import publish

import scripts.snapshot_r04 as snapshot_module
from scripts.snapshot_r04 import snapshot


def test_snapshot_copies_exact_published_checkpoint_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    manager, meta = publish(source)
    before = manager.read_manifest()
    destination = tmp_path / "snapshot"
    marker = snapshot(source, 512, destination)
    assert marker["complete"] and marker["checkpoint"] == meta.to_dict()
    assert manager.read_manifest() == before
    assert len(list((source / "checkpoints").glob("*.pt"))) == 2
    assert len(list((destination / "checkpoints").glob("*.pt"))) == 1
    saved = json.loads((destination / "checkpoints/manifest.json").read_text())
    assert saved["checkpoints"] == [meta.to_dict()]
    assert (destination / "SNAPSHOT_COMPLETE.json").is_file()


def test_snapshot_never_overwrites_existing_destination(tmp_path):
    source = tmp_path / "source"
    publish(source)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        snapshot(source, 512, destination)
    assert sentinel.read_text() == "previous evidence"
    assert not (destination / "SNAPSHOT_COMPLETE.json").exists()


def test_corrupt_source_never_produces_completed_snapshot(tmp_path):
    source = tmp_path / "source"
    manager, meta = publish(source)
    manager.slot_path(meta.slot).write_bytes(b"corrupt")
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="checksum"):
        snapshot(source, 512, destination)
    assert not destination.exists()


def test_interrupted_copy_cannot_inherit_a_stale_completion_marker(tmp_path, monkeypatch):
    source = tmp_path / "source"
    manager, meta = publish(source)
    (source / "SNAPSHOT_COMPLETE.json").write_text('{"complete": true}')
    original = manager.slot_path(meta.slot).read_bytes()
    copyfile = snapshot_module.shutil.copyfile

    def interrupted(src, dst):
        if src.suffix == ".pt":
            dst.write_bytes(b"partial")
            raise OSError("simulated storage interruption")
        return copyfile(src, dst)

    monkeypatch.setattr(snapshot_module.shutil, "copyfile", interrupted)
    destination = tmp_path / "snapshot"
    with pytest.raises(OSError, match="storage interruption"):
        snapshot(source, 512, destination)
    assert destination.exists()
    assert not (destination / "SNAPSHOT_COMPLETE.json").exists()
    assert manager.slot_path(meta.slot).read_bytes() == original
