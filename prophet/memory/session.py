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
import json
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
    def load(cls, path: str | Path) -> "SessionMemory":
        data = torch.load(path, map_location="cpu", weights_only=False)
        version = int(data.get("version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"session memory format v{version}, expected v{FORMAT_VERSION}; "
                "the state layout changed and loading it would be meaningless"
            )
        return cls(
            states=data["states"],
            conv_states=data.get("conv_states", {}),
            tokens_seen=int(data.get("tokens_seen", 0)),
            model_fingerprint=str(data.get("model_fingerprint", "")),
            version=version,
        )


def model_fingerprint(model: torch.nn.Module, *, n_sampled: int = 8) -> str:
    """A cheap identity for a set of weights.

    Hashes shapes plus a few sampled values, which is enough to notice a different
    checkpoint without walking gigabytes of parameters.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for name, param in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(str(tuple(param.shape)).encode())
        flat = param.detach().reshape(-1).float()
        if flat.numel():
            step = max(flat.numel() // n_sampled, 1)
            # .cpu() first: on a CUDA model .numpy() raises, which made session
            # extraction CPU-only on exactly the hardware it is meant for.
            hasher.update(flat[::step][:n_sampled].cpu().numpy().tobytes())
    return hasher.hexdigest()


def _key(section: str, block: int, iteration: int) -> str:
    return f"{section}.{block}.{iteration}"


def extract_session(cache: ProphetCache, *, fingerprint: str = "") -> SessionMemory:
    """Pull the persistable part out of a live cache.

    Only bounded-state slots are kept. Attention KV caches are deliberately dropped: they
    grow with context, so persisting them would reintroduce exactly the linear-memory
    problem the recurrent core exists to avoid.

    **What that means, stated plainly.** After :func:`restore_session` the recurrent
    core resumes with its accumulated state, but the prelude and coda attention layers --
    including the sliding-window sinks -- start from an empty cache at position
    ``tokens_seen``. "Resume where it stopped" holds for the bounded state and not for
    the attention context. Persisting the (bounded) windowed caches and sinks is the
    obvious next step and is not done here.
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
    if strict and fingerprint and memory.model_fingerprint:
        if fingerprint != memory.model_fingerprint:
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
