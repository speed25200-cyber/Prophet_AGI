"""Typed action heads (A3): the heads, the targets derived from a token stream, the
loss terms, and the selection pointer's use in the agent loop."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from prophet.agent.actions import RESERVED_ACTIONS, ToolRegistry, ToolSchema
from prophet.agent.loop import AgentConfig, AgentLoop
from prophet.config import ProphetConfig
from prophet.data.streaming import StreamingLoader, sources_from_iterables
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.action import ActionHeads, build_action_targets, special_ids
from prophet.modeling.model import ProphetModel, ProphetOutput
from prophet.train.loop import TrainConfig, Trainer
from prophet.train.loss import compute_loss

TOK = ProphetTokenizer(merges=[])
SID = special_ids()


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add(ToolSchema("read_file", "Read a file", {
        "type": "object", "properties": {"path": {"type": "string"}, "mode": {"type": "string"}},
        "required": ["path"],
    }))
    reg.add(ToolSchema("count", "Count", {
        "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"],
    }))
    return reg


def _cfg(action: bool = True) -> ProphetConfig:
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg = dataclasses.replace(cfg, heads=dataclasses.replace(cfg.heads, action_head=action, action_dk=16))
    if action and cfg.copy_pointer_layer() is None:
        cfg = dataclasses.replace(
            cfg,
            mixer=dataclasses.replace(cfg.mixer, nope_layers=(1,)),
            recurrent=dataclasses.replace(cfg.recurrent, coda_pattern=["swa", "full_attn"]),
        )
    cfg.validate()
    return cfg


def _stream() -> tuple[list[int], dict]:
    """A tool-use example rendered with the control ids, plus the facts about it."""
    reg = _registry()
    prompt = "<|system|>Goal: read the file notes.txt\n" + reg.render() + "<|assistant|>"
    call = '<|call|>{"name":"read_file","args":{"path":"notes.txt","mode":"quick"}}<|/call|>'
    rest = "<|tool|>hello world<|assistant|><|nocall|>"
    text = prompt + "<|think|>ok<|/think|>" + call + rest
    ids = TOK.encode(text, parse_special=True)
    # Byte-level ids: one character per non-special token, so positions are easy.
    goal_at = text.index("notes.txt")
    return ids, {"text": text, "ids": ids, "value_char": goal_at}


# --------------------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------------------


def test_selection_masks_padded_anchors_and_keeps_the_null_option():
    heads = ActionHeads(8, 4, 4, 1e-5)
    x = torch.randn(2, 1, 8)
    anchors = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    logits = heads.select(x, anchors, mask)
    assert logits.shape == (2, 1, 4)
    assert torch.isfinite(logits[:, :, 0]).all()
    assert torch.isinf(logits[0, 0, 3]) and torch.isinf(logits[1, 0, 2])


def test_copy_pointer_respects_validity():
    heads = ActionHeads(8, 4, 4, 1e-5)
    x = torch.randn(1, 1, 8)
    keys = torch.randn(1, 5, 4)
    valid = torch.tensor([[[True, True, True, False, False]]])
    s, e = heads.copy(x, keys, valid)
    assert s.shape == (1, 1, 5) and torch.isinf(s[0, 0, 3:]).all() and torch.isfinite(e[0, 0, :3]).all()


# --------------------------------------------------------------------------------------
# Targets from the stream
# --------------------------------------------------------------------------------------


def test_targets_are_read_off_the_token_stream():
    ids, facts = _stream()
    t = build_action_targets(torch.tensor([ids]), TOK)
    row = ids
    anchors = [i for i, x in enumerate(row) if x == SID["<|/tool_def|>"]]
    assert t.anchor_positions.tolist() == [anchors]
    call_pos = row.index(SID["<|call|>"])
    nocall_pos = row.index(SID["<|nocall|>"])
    assert t.decision_positions.tolist() == [[call_pos, nocall_pos]]
    assert t.selection.tolist() == [[1, 0]]  # read_file is the first schema; nocall -> none
    # "notes.txt" is verbatim in the goal; "quick" is not.
    assert t.gate_target.tolist() == [[1.0, 0.0]]
    assert t.counts == {"decisions": 2, "calls_matched": 1, "values": 2, "copyable": 1}
    # The copy targets point at the goal's occurrence, token-aligned.
    start, end = int(t.copy_start[0, 0]), int(t.copy_end[0, 0])
    assert TOK.decode(row[start : end + 1]) == "notes.txt"
    assert end < call_pos
    # The copy position is the token before the value inside the call.
    cp = int(t.copy_positions[0, 0])
    assert TOK.decode(row[cp + 1 : cp + 10]) == "notes.txt"
    # Syntax and names are jumped; values are not; nothing outside the call is.
    jumped = t.jumped[0]
    assert bool(jumped[call_pos + 1]) and not bool(jumped[cp + 1]) and not bool(jumped[call_pos - 1])
    assert not bool(jumped[nocall_pos])


def test_unknown_tool_name_selects_none_and_a_truncated_call_is_unsupervised():
    text = '<|tool_def|>{"name":"a"}<|/tool_def|><|call|>{"name":"zzz","args":{}}<|/call|><|call|>{"name":"a"'
    t = build_action_targets(torch.tensor([TOK.encode(text, parse_special=True)]), TOK)
    assert t.selection.tolist() == [[0]]
    assert t.decision_positions.shape[1] == 1


def test_batch_rows_are_padded_independently():
    ids, _ = _stream()
    other = TOK.encode("plain text with no calls at all", parse_special=True)
    other += [SID["<|pad|>"]] * (len(ids) - len(other))
    t = build_action_targets(torch.tensor([ids, other]), TOK)
    assert t.anchor_positions[1].tolist() == [-1, -1]
    assert t.selection[1].tolist() == [-100, -100]
    assert not t.jumped[1].any()


# --------------------------------------------------------------------------------------
# Model and loss
# --------------------------------------------------------------------------------------


def test_heads_are_absent_when_off_and_positions_are_accepted_anyway():
    model = ProphetModel(_cfg(action=False)).eval()
    assert model.action is None
    with torch.no_grad():
        out = model(torch.randint(0, 2048, (1, 8)), loop_k=1, decision_positions=torch.tensor([[3]]))
    assert out.sel_logits is None and out.copy_gate is None


def test_model_scores_selection_and_copy_from_the_stream():
    ids, _ = _stream()
    batch = torch.tensor([ids])
    t = build_action_targets(batch, TOK)
    model = ProphetModel(_cfg()).eval()
    with torch.no_grad():
        out = model(batch, loop_k=1, **t.forward_kwargs())
    n_anchor = t.anchor_positions.shape[1]
    assert out.sel_logits.shape == (1, 2, n_anchor + 1)
    assert out.copy_start.shape == (1, 1, len(ids)) and out.copy_end.shape == out.copy_start.shape
    assert out.copy_gate.shape == (1, len(ids))
    # Causal validity: nothing after the copy position can be pointed at.
    cp = int(t.copy_positions[0, 0])
    assert torch.isinf(out.copy_start[0, 0, cp + 1 :]).all()
    assert torch.isfinite(out.copy_start[0, 0, : cp + 1]).all()
    assert torch.equal(out.copy_key_positions, torch.arange(len(ids)))


def test_action_terms_are_learnable_on_one_example():
    torch.manual_seed(0)
    ids, _ = _stream()
    batch = torch.tensor([ids])
    t = build_action_targets(batch, TOK)
    model = ProphetModel(_cfg()).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(40):
        out = model(batch, loop_k=1, **t.forward_kwargs())
        terms = compute_loss(out, batch, action_targets=t, sel_weight=1.0, ptr_weight=1.0,
                             gate_weight=1.0, jumped_lm_weight=0.1, z_loss_weight=0.0)
        if first is None:
            first = terms.metrics["loss/action"]
        opt.zero_grad()
        terms.total.backward()
        opt.step()
    assert terms.metrics["loss/action"] < 0.25 * first
    assert terms.metrics["action/sel_accuracy"] == 1.0
    with torch.no_grad():
        out = model(batch, loop_k=1, **t.forward_kwargs())
    assert int(out.copy_start[0, 0].argmax()) == int(t.copy_start[0, 0])
    assert int(out.copy_end[0, 0].argmax()) == int(t.copy_end[0, 0])


def test_jumped_tokens_are_down_weighted_in_the_lm_loss():
    ids, _ = _stream()
    batch = torch.tensor([ids])
    t = build_action_targets(batch, TOK)
    model = ProphetModel(_cfg()).eval()
    with torch.no_grad():
        out = model(batch, loop_k=1, **t.forward_kwargs())
        plain = compute_loss(out, batch, z_loss_weight=0.0).lm
        weighted = compute_loss(out, batch, action_targets=t, jumped_lm_weight=0.1, z_loss_weight=0.0).lm
    assert not torch.isclose(plain, weighted)


def test_trainer_needs_the_tokenizer_and_then_trains_a_step(tmp_path):
    cfg = _cfg()
    ids, _ = _stream()
    loader = StreamingLoader(sources_from_iterables({"a": (1.0, [ids] * 4)}), seq_len=len(ids))
    tc = TrainConfig(total_steps=1, seq_len=len(ids), checkpoint_dir=str(tmp_path), device="cpu",
                     activation_checkpointing=True)
    with pytest.raises(ValueError, match="tokenizer"):
        Trainer(ProphetModel(cfg), loader, tc, model_config=cfg)
    trainer = Trainer(ProphetModel(cfg), loader, tc, model_config=cfg, tokenizer=TOK)
    assert trainer.cfg.sel_weight == cfg.heads.sel_loss_weight
    history = trainer.train(max_steps=1)
    assert history and "loss/action" in history[-1].extra


# --------------------------------------------------------------------------------------
# The loop uses the selection pointer
# --------------------------------------------------------------------------------------


class _Selecting(nn.Module):
    """Emits a fixed script (see tests/test_agent.py) and a fixed selection choice."""

    def __init__(self, script: list[int], choice: int) -> None:
        super().__init__()
        self.script, self.choice = script, choice
        self.cursor, self.speaking = 0, False
        self.dummy = nn.Parameter(torch.zeros(1))
        self.action = object()  # "has action heads"
        self.seen_kwargs: list[dict] = []

    def _project(self, h):
        return torch.zeros(*h.shape[:-1], 600)

    def forward(self, ids, *, cache=None, loop_k=None, return_mtp=True, halt_threshold=None,
                modality_ids=None, **kw):
        self.seen_kwargs.append(kw)
        b, s = ids.shape
        if cache is not None:
            cache.position += s
        if int(ids[0, -1]) in (SID["<|think|>"], SID["<|call|>"]):
            self.speaking = True
        logits = torch.full((b, s, 600), -20.0)
        nxt = TOK.eos_id
        if self.speaking and self.cursor < len(self.script):
            nxt = self.script[self.cursor]
            self.cursor += 1
            if nxt in (SID["<|/think|>"], SID["<|/call|>"]):
                self.speaking = False
        logits[:, -1, nxt] = 20.0
        sel = None
        if "decision_positions" in kw:
            sel = torch.full((b, 1, 3), -5.0)
            sel[:, :, self.choice] = 5.0
        return ProphetOutput(logits=logits, hidden=torch.zeros(b, s, 8), loop_k=1,
                             confidence=torch.full((b, s), 5.0), sel_logits=sel)


def _script(*spans: str) -> list[int]:
    out: list[int] = []
    for span in spans:
        out += TOK.encode(span, parse_special=True)
    return out


def _loop(model) -> AgentLoop:
    reg = _registry()
    reg.bind("read_file", lambda path, mode="": f"contents of {path}")
    reg.bind("count", lambda n: n)
    return AgentLoop(model, TOK, reg, AgentConfig(max_steps=1, think_budget=4, action_budget=64, halt_threshold=None))


def test_loop_passes_anchor_and_decision_positions_to_the_model():
    model = _Selecting(_script("<|/think|>", '{"name":"count","args":{"n":1}}<|/call|>'), choice=2)
    _loop(model).run("g")
    first = model.seen_kwargs[0]
    assert "anchor_positions" in first and first["anchor_positions"].shape[1] == 2
    assert any("decision_positions" in kw and kw["decision_positions"].tolist() == [[0]] for kw in model.seen_kwargs)


def test_selection_narrows_the_grammar_to_the_chosen_tool():
    # The head picks `count` (index 2) while the LM tries to emit `read_file`: the
    # constrained decoder finds no viable token and the call is recorded as malformed.
    model = _Selecting(_script("<|/think|>", '{"name":"read_file","args":{"path":"x"}}<|/call|>'), choice=2)
    result = _loop(model).run("g")
    rec = result.steps[0]
    assert rec.gated == "malformed" and rec.selected == "count" and rec.sel_margin > 0.9
    # The same script with the head agreeing decodes fine.
    model = _Selecting(_script("<|/think|>", '{"name":"read_file","args":{"path":"x"}}<|/call|>'), choice=1)
    rec = _loop(model).run("g").steps[0]
    assert rec.action is not None and rec.action.name == "read_file" and rec.selected == "read_file"


def test_none_leaves_only_the_reserved_actions():
    model = _Selecting(_script("<|/think|>", '{"name":"done"}<|/call|>'), choice=0)
    result = _loop(model).run("g")
    assert result.finished and result.steps[0].selected == "none"
    assert set(RESERVED_ACTIONS) >= {"done"}


def test_budget_counts_the_action_heads_exactly():
    from prophet.budget import count_parameters

    cfg = _cfg()
    real = sum(p.numel() for p in ProphetModel(cfg).parameters())
    # The estimator carries a pre-existing ~1e-4 residual on this config (norm and
    # bias accounting); the claim here is that the heads' own count is exact.
    assert abs(count_parameters(cfg).total - real) / real < 2e-4
    off = _cfg(action=False)
    assert count_parameters(cfg).total - count_parameters(off).total == sum(
        p.numel() for p in ProphetModel(cfg).action.parameters()
    )
