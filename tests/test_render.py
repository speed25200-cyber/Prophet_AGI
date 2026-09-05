"""Promoted episodes rendered back into the control-id stream, and served as a corpus
source whose targets the action heads can read."""

from __future__ import annotations

import torch

from prophet.agent.actions import ToolRegistry, ToolSchema
from prophet.agent.quarantine import Entry, Provenance, Quarantine
from prophet.agent.render import QuarantineSource, render_episode
from prophet.agent.verify import Tier
from prophet.data.corpus import TokenisedSource
from prophet.data.streaming import StreamingLoader
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.action import build_action_targets, special_ids

TOK = ProphetTokenizer(merges=[])
SID = special_ids()


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add(ToolSchema("read_file", "Read", {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
    }))
    return reg


def _trajectory() -> list[dict]:
    return [
        {"step": 0, "think": "look at notes.txt", "action": {"name": "read_file", "args": {"path": "notes.txt"}},
         "p_correct": 0.9, "tier": 1, "observation": "hello world"},
        {"step": 1, "think": "", "action": None, "gated": "malformed"},
        {"step": 2, "think": "done", "action": {"name": "done", "args": {}}, "p_correct": 0.95,
         "tier": 1, "observation": ""},
    ]


def test_render_mirrors_the_loop_and_drops_malformed_steps():
    text = render_episode("read notes.txt", _registry(), _trajectory())
    assert text.startswith("<|system|>Goal: read notes.txt\n<|tool_def|>")
    assert '<|call|>{"name":"read_file","args":{"path":"notes.txt"}}<|/call|>' in text
    assert "<|tool|>hello world<|assistant|>" in text
    assert text.count("<|call|>") == 2 and "<|nocall|>" not in text
    assert text.endswith("<|eos|>")


def test_rendered_episode_yields_action_targets():
    ids = TOK.encode(render_episode("read notes.txt", _registry(), _trajectory()), parse_special=True)
    t = build_action_targets(torch.tensor([ids]), TOK)
    assert t.counts["decisions"] == 2 and t.counts["calls_matched"] == 1
    # "notes.txt" is verbatim in the goal, so the value is a copy target.
    assert t.counts["copyable"] == 1
    assert t.selection.tolist() == [[1, 0]]  # read_file, then the reserved `done` = none


def _quarantine(tmp_path) -> Quarantine:
    q = Quarantine(tmp_path / "q.json")
    for i, tier in enumerate((Tier.GROUND_TRUTH, Tier.LEARNED, Tier.GROUND_TRUTH)):
        q.add(Entry(
            family="files", goal=f"goal {i}", trajectory=_trajectory(), outcome_passed=True,
            process_ok=True,
            provenance=Provenance(tier=int(tier), verifier_version="v0", p_correct=0.9,
                                  depth_disagreement=None, attempts=1),
        ))
    q.add(Entry(
        family="other", goal="no registry", trajectory=_trajectory(), outcome_passed=True,
        process_ok=True,
        provenance=Provenance(tier=int(Tier.GROUND_TRUTH), verifier_version="v0", p_correct=0.9,
                              depth_disagreement=None, attempts=1),
    ))
    return q


def test_source_serves_only_promoted_episodes_with_a_registry(tmp_path):
    src = QuarantineSource(_quarantine(tmp_path), {"files": _registry()})
    assert src.n_documents() == 2  # two ground-truth episodes; learned is never promoted
    docs = list(src.open(0))
    assert [d.split("\n")[0] for d in docs] == ["<|system|>Goal: goal 0", "<|system|>Goal: goal 2"]
    assert list(src.open(1)) == docs[1:]


def test_source_feeds_the_loader_with_control_ids(tmp_path):
    src = TokenisedSource(
        QuarantineSource(_quarantine(tmp_path), _registry()), TOK, max_epochs=None, parse_special=True
    )
    loader = StreamingLoader([src], seq_len=32)
    stream: list[int] = []
    for batch in loader.batches(8):
        stream += batch[0]
    assert SID["<|tool_def|>"] in stream and SID["<|call|>"] in stream
    # Without parse_special the control strings would be bytes and no anchors.
    plain = TokenisedSource(QuarantineSource(_quarantine(tmp_path), _registry()), TOK, max_epochs=None)
    assert SID["<|tool_def|>"] not in plain.next_document()
