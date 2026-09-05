"""Typed action heads (track A3): a selection pointer over schema anchors, a span-copy
pointer over context, and a copy gate -- plus the derivation of their targets from an
ordinary token stream.

Why heads and not more grammar. A3 measured how tool calls fail at 1-4B: syntax is rare
and a grammar already removes it; the dominant failures are **omission** (calling when
nothing should be called, or the reverse), the **wrong tool**, and **wrong argument
values** that were sitting verbatim in context. A wrong tool is one wrong argmax over
``n+1`` options, not a slightly wrong token string, so the selection head scores exactly
that decision; a copied value cannot drift digit by digit, so the copy head points at it.

Everything reads the coda output ``x`` (pre-``norm_out``, where the confidence head
reads) and owns no cache: anchors are hidden states already in the prompt, and the copy
pointer scores the *existing* keys of the coda's NoPE global-attention layer -- zero
extra cache bytes, which is the number that matters on a phone.

Targets come from the training stream itself. A tool-use SFT example rendered with the
control ids (``<|tool_def|>...<|/tool_def|>``, ``<|call|>{json}<|/call|>``,
``<|nocall|>``) already says which tool was chosen and which argument values were
verbatim in context; :func:`build_action_targets` reads that off the ids, so no second
data format exists to drift from the first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from prophet.data.tokenizer import N_BYTES, SPECIAL_TOKENS
from prophet.modeling.layers import make_norm

__all__ = ["ActionHeads", "ActionTargets", "build_action_targets", "special_ids"]


def special_ids() -> dict[str, int]:
    return {name: N_BYTES + i for i, name in enumerate(SPECIAL_TOKENS)}


class ActionHeads(nn.Module):
    """Selection pointer + span-copy pointer + copy gate, read from the coda output."""

    def __init__(
        self, d_model: int, d_k: int, head_dim: int, eps: float, *, norm_kind: str = "rmsnorm"
    ) -> None:
        super().__init__()
        self.norm = make_norm(norm_kind, d_model, eps)
        self.sel_q = nn.Linear(d_model, d_k, bias=False)
        self.sel_k = nn.Linear(d_model, d_k, bias=False)
        self.null_key = nn.Parameter(torch.zeros(d_k))
        """The "none of these" option: no call, or a reserved action with no schema."""
        self.copy_start_q = nn.Linear(d_model, head_dim, bias=False)
        self.copy_end_q = nn.Linear(d_model, head_dim, bias=False)
        self.gate = nn.Linear(d_model, 1)
        self.d_k, self.head_dim = d_k, head_dim

    def select(self, x_t: Tensor, anchors: Tensor, anchor_mask: Tensor) -> Tensor:
        """``x_t`` (b, c, d) at decision positions; ``anchors`` (b, n, d) at the
        ``<|/tool_def|>`` positions; ``anchor_mask`` (b, n). Returns (b, c, n+1) logits,
        index 0 = none."""
        q = self.sel_q(self.norm(x_t))
        k = self.sel_k(self.norm(anchors))
        null = self.null_key.expand(k.shape[0], 1, -1)
        k = torch.cat([null, k], dim=1)
        mask = torch.cat([anchor_mask.new_ones(k.shape[0], 1), anchor_mask], dim=1)
        logits = torch.einsum("bcd,bnd->bcn", q, k) / self.d_k**0.5
        return logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

    def copy(self, x_t: Tensor, keys: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """``x_t`` (b, m, d) at copy positions; ``keys`` (b, L, head_dim), one KV head of
        the coda NoPE layer; ``valid`` (b, m, L). Returns start and end logits over L."""
        h = self.norm(x_t)
        s = torch.einsum("bmd,bld->bml", self.copy_start_q(h), keys) / self.head_dim**0.5
        e = torch.einsum("bmd,bld->bml", self.copy_end_q(h), keys) / self.head_dim**0.5
        return s.masked_fill(~valid, float("-inf")), e.masked_fill(~valid, float("-inf"))

    def copy_gate(self, x: Tensor) -> Tensor:
        """(b, s, d) -> (b, s) logit of "the value starting here is verbatim in context"."""
        return self.gate(self.norm(x)).squeeze(-1)


# --------------------------------------------------------------------------------------
# Targets from the stream
# --------------------------------------------------------------------------------------


@dataclass
class ActionTargets:
    """Per-batch supervision for the heads, all derived from the token ids.

    Positions are absolute indices into the sequence; ``-1`` pads and ``-100`` is the
    ignore label. ``jumped`` marks tokens a typed runtime would emit for the model
    (syntax, tool name, parameter names), which the LM loss down-weights.
    """

    anchor_positions: Tensor
    """(b, n) positions of ``<|/tool_def|>``."""
    decision_positions: Tensor
    """(b, c) positions of ``<|call|>`` and ``<|nocall|>``."""
    selection: Tensor
    """(b, c) target index into ``[none, anchor_1..anchor_n]``."""
    copy_positions: Tensor
    """(b, m) value-start positions: the token *before* a value's first token."""
    copy_start: Tensor
    """(b, m) context position of the value's first token, or -100."""
    copy_end: Tensor
    """(b, m) context position of the value's last token, or -100."""
    gate_positions: Tensor
    """(b, g) every value-start position, copyable or not."""
    gate_target: Tensor
    """(b, g) 1.0 when the value is verbatim and token-aligned in context."""
    jumped: Tensor
    """(b, s) bool."""
    counts: dict[str, int] = field(default_factory=dict)

    def forward_kwargs(self) -> dict[str, Tensor]:
        return {
            "anchor_positions": self.anchor_positions,
            "decision_positions": self.decision_positions,
            "copy_positions": self.copy_positions,
        }


def _pad(rows: list[list[int]], fill: int, device) -> Tensor:
    width = max((len(r) for r in rows), default=0)
    out = torch.full((len(rows), max(width, 1)), fill, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
    return out


def _char_spans(ids: list[int], tokenizer) -> tuple[str, list[tuple[int, int]]]:
    """Decoded text and each token's ``[start, end)`` character span. Control ids
    decode to their name so they occupy characters and never match a value."""
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special=False)
        spans.append((pos, pos + len(piece)))
        text_parts.append(piece)
        pos += len(piece)
    return "".join(text_parts), spans


def _token_at(spans: list[tuple[int, int]], char: int, *, end: bool = False) -> int | None:
    """Index of the token whose span starts (or, with ``end``, ends) exactly at ``char``."""
    for i, (a, b) in enumerate(spans):
        if (b if end else a) == char:
            return i
    return None


def _walk_values(obj, path: str = "") -> list[tuple[str, object]]:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += _walk_values(v, f"{path}.{k}" if path else k)
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out += _walk_values(v, f"{path}[{i}]")
        return out
    return [(path, obj)]


def _value_char_span(text: str, start: int, value) -> tuple[int, int] | None:
    """Character span of ``value``'s literal inside the JSON text starting at ``start``.

    Strings are located as their quoted JSON form; the span excludes the quotes.
    Numbers and booleans are located as their JSON literal.
    """
    literal = json.dumps(value, ensure_ascii=False)
    at = text.find(literal, start)
    if at < 0:
        return None
    if isinstance(value, str):
        return at + 1, at + len(literal) - 1
    return at, at + len(literal)


def build_action_targets(ids: Tensor, tokenizer) -> ActionTargets:
    """Derive selection, copy and gate targets from a batch of token ids.

    For every ``<|call|>`` the JSON up to ``<|/call|>`` is parsed; its ``name`` is
    matched against the schemas opened by ``<|tool_def|>`` earlier in the sequence
    (index 0 when it matches none: a reserved action or a bad name), and each argument
    value is looked up as a **token-aligned, verbatim** occurrence in the context before
    the call -- the last such occurrence, so a fresh tool result wins over an old one.
    Values that occur but not on token boundaries are not copyable and train the gate
    to say so. ``<|nocall|>`` is a decision with target 0.
    """
    sid = special_ids()
    open_def, close_def = sid["<|tool_def|>"], sid["<|/tool_def|>"]
    call_open, call_close, nocall = sid["<|call|>"], sid["<|/call|>"], sid["<|nocall|>"]
    b, s = ids.shape
    rows = ids.tolist()
    anchors_rows, decision_rows, selection_rows = [], [], []
    copy_pos_rows, copy_start_rows, copy_end_rows = [], [], []
    gate_pos_rows, gate_rows = [], []
    jumped = torch.zeros(b, s, dtype=torch.bool, device=ids.device)
    counts = {"decisions": 0, "calls_matched": 0, "values": 0, "copyable": 0}

    for r, row in enumerate(rows):
        text, spans = _char_spans(row, tokenizer)
        # Schemas: (anchor position, name) in order of appearance.
        schemas: list[tuple[int, str]] = []
        i = 0
        while i < s:
            if row[i] == open_def:
                j = i + 1
                while j < s and row[j] != close_def:
                    j += 1
                if j < s:
                    body = text[spans[i][1] : spans[j][0]]
                    try:
                        name = str(json.loads(body).get("name", ""))
                    except (json.JSONDecodeError, AttributeError):
                        name = ""
                    schemas.append((j, name))
                    i = j
            i += 1
        anchors_rows.append([p for p, _ in schemas])

        decisions, selections = [], []
        copy_pos, copy_start, copy_end, gate_pos, gate = [], [], [], [], []
        for t, tid in enumerate(row):
            if tid == nocall:
                decisions.append(t)
                selections.append(0)
                counts["decisions"] += 1
                continue
            if tid != call_open:
                continue
            j = t + 1
            while j < s and row[j] != call_close:
                j += 1
            if j >= s:
                break  # truncated call: no supervision
            counts["decisions"] += 1
            decisions.append(t)
            body_start, body_end = spans[t][1], spans[j][0]
            body = text[body_start:body_end]
            try:
                call = json.loads(body)
            except json.JSONDecodeError:
                selections.append(-100)
                continue
            name = str(call.get("name", ""))
            index = next((k + 1 for k, (_, n) in enumerate(schemas) if n == name), 0)
            if index:
                counts["calls_matched"] += 1
            selections.append(index)
            # Everything inside the call is "jumped" except the argument values.
            jumped[r, t + 1 : j] = True
            for _, value in _walk_values(call.get("args", {})):
                if isinstance(value, (dict, list)) or value is None:
                    continue
                span = _value_char_span(text, body_start, value)
                if span is None:
                    continue
                first = _token_at(spans, span[0])
                last = _token_at(spans, span[1], end=True)
                if first is None or last is None or first < 1:
                    continue
                counts["values"] += 1
                jumped[r, first : last + 1] = False
                value_start = first - 1  # the position whose state emits <|copy|> or not
                literal = text[span[0] : span[1]]
                # Last token-aligned verbatim occurrence strictly before the call. A
                # word in prose is usually one token *with its leading space* (" anchor")
                # while the same word in JSON is bare ("anchor"), so an occurrence may
                # start at the space just before it; the decode path strips that space.
                found = None
                at = text.rfind(literal, 0, body_start)
                while at >= 0:
                    ce = _token_at(spans, at + len(literal), end=True)
                    cs = _token_at(spans, at)
                    if cs is None and at > 0 and text[at - 1] == " ":
                        cs = _token_at(spans, at - 1)
                    if cs is not None and ce is not None and ce < t:
                        found = (cs, ce)
                        break
                    at = text.rfind(literal, 0, at)
                gate_pos.append(value_start)
                gate.append(1 if found else 0)
                if found:
                    counts["copyable"] += 1
                    copy_pos.append(value_start)
                    copy_start.append(found[0])
                    copy_end.append(found[1])
        decision_rows.append(decisions)
        selection_rows.append(selections)
        copy_pos_rows.append(copy_pos)
        copy_start_rows.append(copy_start)
        copy_end_rows.append(copy_end)
        gate_pos_rows.append(gate_pos)
        gate_rows.append(gate)

    dev = ids.device
    return ActionTargets(
        anchor_positions=_pad(anchors_rows, -1, dev),
        decision_positions=_pad(decision_rows, -1, dev),
        selection=_pad(selection_rows, -100, dev),
        copy_positions=_pad(copy_pos_rows, -1, dev),
        copy_start=_pad(copy_start_rows, -100, dev),
        copy_end=_pad(copy_end_rows, -100, dev),
        gate_positions=_pad(gate_pos_rows, -1, dev),
        gate_target=_pad(gate_rows, -100, dev).float(),
        jumped=jumped,
        counts=counts,
    )
