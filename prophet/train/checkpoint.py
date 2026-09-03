"""Checkpointing that survives a Colab session dying mid-write.

The failure this module exists to prevent: the process is preempted while a checkpoint
file is being written, leaving a truncated file that looks valid until it is loaded —
days later, with no earlier copy to fall back on.

The defence is a **two-slot atomic rotation**. Writes go to a temporary file, are
fsynced, checksummed, and only then renamed into a slot (rename is atomic on POSIX).
Slots alternate, so a torn write can only ever damage the *older* of two copies and the
newer one is always intact. A manifest records which slot is current, and loading falls
back to the other slot if a checksum fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CheckpointManager", "CheckpointError", "CheckpointMeta"]


class CheckpointError(RuntimeError):
    """Raised when no intact checkpoint can be recovered."""


@dataclass
class CheckpointMeta:
    step: int
    slot: int
    sha256: str
    bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "slot": self.slot,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointMeta":
        return cls(
            step=int(d["step"]),
            slot=int(d["slot"]),
            sha256=str(d["sha256"]),
            bytes=int(d["bytes"]),
            extra=dict(d.get("extra", {})),
        )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class CheckpointManager:
    """Two-slot atomic checkpoint rotation.

    Parameters
    ----------
    directory:
        Where slots and the manifest live.
    keep_milestones:
        Steps whose checkpoints are copied aside and never rotated out. Use this for
        the WSD plateau branch point, from which several anneals are launched.
    """

    MANIFEST = "manifest.json"

    def __init__(self, directory: str | Path, *, keep_milestones: tuple[int, ...] = ()) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_milestones = set(keep_milestones)

    # -- paths -------------------------------------------------------------------------

    def slot_path(self, slot: int) -> Path:
        return self.dir / f"ckpt_slot{slot}.pt"

    def milestone_path(self, step: int) -> Path:
        return self.dir / f"ckpt_milestone_{step:09d}.pt"

    @property
    def manifest_path(self) -> Path:
        return self.dir / self.MANIFEST

    # -- manifest ----------------------------------------------------------------------

    def read_manifest(self) -> list[CheckpointMeta]:
        if not self.manifest_path.exists():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # The manifest itself can be torn. It is rewritten atomically, so this is
            # unlikely, but recovering from the slots directly is always possible.
            return []
        return [CheckpointMeta.from_dict(d) for d in data.get("checkpoints", [])]

    def _write_manifest(self, metas: list[CheckpointMeta]) -> None:
        payload = {"checkpoints": [m.to_dict() for m in metas]}
        self._atomic_write_text(self.manifest_path, json.dumps(payload, indent=2))

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- save --------------------------------------------------------------------------

    def save(self, state: dict[str, Any], step: int, *, extra: dict[str, Any] | None = None) -> CheckpointMeta:
        """Write a checkpoint into the next slot, atomically."""
        import torch

        metas = self.read_manifest()
        slot = 0 if not metas else 1 - metas[-1].slot
        target = self.slot_path(slot)

        fd, tmp_name = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with tmp.open("wb") as fh:
                torch.save(state, fh)
                fh.flush()
                os.fsync(fh.fileno())
            digest = _sha256(tmp)
            size = tmp.stat().st_size
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        meta = CheckpointMeta(step=step, slot=slot, sha256=digest, bytes=size,
                              extra=dict(extra or {}))
        metas = [m for m in metas if m.slot != slot][-1:] + [meta]
        self._write_manifest(metas)

        if step in self.keep_milestones:
            shutil.copy2(target, self.milestone_path(step))

        return meta

    # -- load --------------------------------------------------------------------------

    def load_latest(self, *, map_location: Any = "cpu") -> tuple[dict[str, Any], CheckpointMeta]:
        """Load the newest intact checkpoint, falling back to the older slot."""
        import torch

        metas = sorted(self.read_manifest(), key=lambda m: m.step, reverse=True)
        failures: list[str] = []

        for meta in metas:
            path = self.slot_path(meta.slot)
            if not path.exists():
                failures.append(f"slot {meta.slot} (step {meta.step}): missing")
                continue
            actual = _sha256(path)
            if actual != meta.sha256:
                failures.append(
                    f"slot {meta.slot} (step {meta.step}): checksum mismatch — "
                    "almost certainly a write interrupted by preemption"
                )
                continue
            return torch.load(path, map_location=map_location, weights_only=False), meta

        raise CheckpointError(
            "no intact checkpoint found in "
            f"{self.dir}:\n" + "\n".join(f"  - {f}" for f in failures)
        )

    def has_checkpoint(self) -> bool:
        return bool(self.read_manifest())

    def verify(self) -> dict[int, bool]:
        """Checksum every slot. Cheap insurance to run at startup."""
        return {
            m.slot: (self.slot_path(m.slot).exists() and _sha256(self.slot_path(m.slot)) == m.sha256)
            for m in self.read_manifest()
        }
