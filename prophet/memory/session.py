"""Memory that survives the end of a conversation.

Prophet's memory has two tiers, and this module is what makes either of them *persistent*
rather than merely present during one forward pass.

**Tier 1 — session state.** The gated-delta layers already carry a fixed-size matrix
state that accumulates within a sequence. Serialising it lets a conversation resume
tomorrow where it stopped today, at a cost independent of how long that conversation was.
For Prophet-mini the whole state is under a megabyte.

**Tier 2 — the ledger.** Slower, larger, and shared across sessions. Written during the
offline consolidation pass in :mod:`prophet.memory.consolidate`.

The split matters because the two tiers fail differently. Session state is cheap and
disposable: losing it costs one conversation. The ledger is durable and therefore
dangerous — anything written into it persists, including mistakes, so it is only written
by a deliberate consolidation step and never directly from a live conversation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prophet.modeling.layers import RecurrentState
from prophet.modeling.model import ProphetCache

__all__ = ["SessionMemory", "extract_session", "restore_session"]

FORMAT_VERSION = 1


@dataclass
class SessionMemory:
    """Serialisable per-conversation state.

    ``model_fingerprint`` guards the failure this class exists to prevent: restoring a
    state produced by different weights. The tensors would load without complaint and the
    model would behave subtly wrongly, with nothing in the output to say why.
    """

    states: dict[str, Tensor] = field(default_factory=dict)
    conv_states: dict[str, Tensor] = field(default_factory=dict)
    tokens_seen: int = 0
    model_fingerprint: str = ""
    version: int = FORMAT_VERSION

    def n_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (*self.states.values(), *self.conv_states.values())
        )

    def summary(self) -> dict[str, Any]:
        return {
            "slots": len(self.states),
            "tokens_seen": self.tokens_seen,
            "bytes": self.n_bytes(),
            "megabytes": round(self.n_bytes() / 1e6, 3),
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "version": self.version,
                "states": self.states,
                "conv_states": self.conv_states,
                "tokens_seen": self.tokens_seen,
                "model_fingerprint": self.model_fingerprint,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SessionMemory:
        # The format contains tensors and primitive metadata only.  ``weights_only``
        # avoids executing arbitrary pickle globals from a supplied session file.
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, Mapping):
            raise TypeError("session memory must contain a mapping")
        version = int(data.get("version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"session memory format v{version}, expected v{FORMAT_VERSION}; "
                "the state layout changed and loading it would be meaningless"
            )
        def tensor_map(name: str) -> dict[str, Tensor]:
            value = data.get(name, {})
            if not isinstance(value, Mapping):
                raise TypeError(f"session memory {name} must be a mapping")
            result: dict[str, Tensor] = {}
            for key, tensor in value.items():
                if not isinstance(key, str) or not isinstance(tensor, Tensor):
                    raise TypeError(f"session memory {name} must map strings to tensors")
                result[key] = tensor
            return result

        tokens_seen = data.get("tokens_seen", 0)
        if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
            raise ValueError("session memory tokens_seen must be a non-negative integer")
        fingerprint = data.get("model_fingerprint", "")
        if not isinstance(fingerprint, str):
            raise TypeError("session memory model_fingerprint must be a string")

        return cls(
            states=tensor_map("states"),
            conv_states=tensor_map("conv_states"),
            tokens_seen=tokens_seen,
            model_fingerprint=fingerprint,
            version=version,
        )


def model_fingerprint(model: torch.nn.Module, *, n_sampled: int = 8) -> str:
    """A cheap identity for a set of weights.

    Hashes shapes plus a few sampled values, which is enough to notice a different
    checkpoint without walking gigabytes of parameters. Mutable ledger values and write
    counts are excluded: a memory write must not make otherwise compatible recurrent
    state impossible to restore. Immutable addressing keys remain part of the identity.
    """
    if not isinstance(n_sampled, int) or isinstance(n_sampled, bool) or n_sampled < 1:
        raise ValueError("n_sampled must be a positive integer")

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(f"session-format:{FORMAT_VERSION}".encode())
    mutable_ledger_suffixes = (".values", ".write_counts")
    for name, param in sorted(model.state_dict().items()):
        if name.startswith("ledgers.") and name.endswith(mutable_ledger_suffixes):
            continue
        hasher.update(name.encode())
        hasher.update(str(tuple(param.shape)).encode())
        hasher.update(str(param.dtype).encode())
        flat = param.detach().reshape(-1)
        if flat.numel():
            step = max(flat.numel() // n_sampled, 1)
            # Sample first, then transfer/cast. Casting the complete parameter to FP32 on
            # device can transiently allocate gigabytes for a large checkpoint.
            sample = flat[::step][:n_sampled].to(device="cpu", dtype=torch.float32)
            hasher.update(sample.numpy().tobytes())
    return hasher.hexdigest()


def _key(section: str, block: int, iteration: int) -> str:
    return f"{section}.{block}.{iteration}"


def extract_session(cache: ProphetCache, *, fingerprint: str = "") -> SessionMemory:
    """Pull the persistable part out of a live cache.

    Only bounded-state slots are kept. Attention KV caches are deliberately dropped: they
    grow with context, so persisting them would reintroduce exactly the linear-memory
    problem the recurrent core exists to avoid.
    """
    memory = SessionMemory(model_fingerprint=fingerprint, tokens_seen=cache.position)
    for (section, block, iteration), slot in cache.slots.items():
        if not isinstance(slot, RecurrentState):
            continue
        key = _key(section, block, iteration)
        if slot.state is not None:
            memory.states[key] = slot.state.detach().clone()
        if slot.conv_state is not None:
            memory.conv_states[key] = slot.conv_state.detach().clone()
    return memory


def restore_session(
    memory: SessionMemory,
    cache: ProphetCache,
    *,
    fingerprint: str = "",
    strict: bool = True,
) -> int:
    """Write a saved session back into a cache. Returns the number of slots restored."""
    if (
        strict
        and fingerprint
        and memory.model_fingerprint
        and fingerprint != memory.model_fingerprint
    ):
        raise ValueError(
            "this session memory was produced by different weights; restoring it "
            "would leave the model subtly and silently wrong. Pass strict=False only "
            "if the mismatch is understood."
        )

    restored = 0
    for key, state in memory.states.items():
        section, block, iteration = key.split(".")
        slot = cache.get(section, int(block), int(iteration), "gdn")
        if not isinstance(slot, RecurrentState):
            continue
        slot.state = state.clone()
        conv = memory.conv_states.get(key)
        slot.conv_state = conv.clone() if conv is not None else None
        slot.seen = memory.tokens_seen
        restored += 1

    cache.position = memory.tokens_seen
    return restored
