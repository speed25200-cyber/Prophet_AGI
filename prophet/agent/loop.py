"""The agent loop: one model, one grammar, one head, one state, one offline step.

What is deliberately *not* here, and why: no planner module, no critic model, no
sub-agents, no summariser. Track A2's survey found that each of those is where a new
failure class appears -- inter-agent failures are over a third of multi-agent failures,
and summarisation is how constraint violations go from 0% to 30-59%. The loop is
single-threaded and every "agentic" decision happens in exactly one place, the gates.

The step, in order:

1. **Snapshot** the cache, so this step can be undone in O(1).
2. **Think** -- free text under learned halting, bounded by a token budget. Nothing
   constrains it; that is where exploration comes from.
3. **Act** -- a grammar-constrained call at a fixed deep depth. The constraint is active
   only between the call delimiters.
4. **Gate** -- the verifier reads the signals and the confidence head and decides: act,
   verify first (irreversible actions below threshold), refuse ``done`` until a check
   agrees, or ask the user with the action that would resolve the question.
5. **Loop-detect** -- the harness counts identical actions; past the cap it forces a
   reflection rather than letting the model repeat itself (step repetition is ~16% of
   failures, and models do not notice it from the inside).
6. **Execute and ingest** -- the observation enters at ``loop_k=1`` as a distinct
   modality, capped by the harness; the oldest observation past the window is evicted
   from the attention cache and survives only in the recurrent state.

At episode end nothing is written to memory. The trajectory goes to quarantine with its
provenance, and consolidation runs offline, per task family, behind a merge gate.

What this file is: the control flow, the gates and the bookkeeping, runnable against any
model with the ProphetModel interface and any tool set. What it is not: trained. The
think/act behaviour of an untrained model is noise, and the tests in ``tests/test_agent``
exercise the loop's *mechanics* -- gates fire, rollback restores, loops are caught,
quarantine records -- not its competence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from prophet.agent.actions import (
    Action,
    ActionGrammar,
    ConstrainedDecoder,
    ToolRegistry,
)
from prophet.agent.quarantine import Entry, Provenance, Quarantine
from prophet.agent.state import AgentState, Observation
from prophet.agent.verify import (
    SignalScorer,
    Tier,
    Verdict,
    VerifierConfig,
    decide,
    extract_signals,
)
from prophet.memory.session import extract_session, model_fingerprint, restore_session
from prophet.modeling.model import ProphetCache

__all__ = ["Tokenizer", "AgentConfig", "StepRecord", "EpisodeResult", "AgentLoop"]


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids, *, skip_special: bool = True) -> str: ...
    def special_id(self, name: str) -> int: ...


@dataclass
class AgentConfig:
    ledger_write: str = "all"
    """What the model's ledger layers keep of what leaves their window: ``"all"``, or
    ``"tool"`` -- only the tokens of tool observations (the trainer's
    ``TrainConfig.ledger_write`` must say the same)."""
    max_steps: int = 64
    think_budget: int = 128
    """Maximum think-span tokens per step."""
    action_budget: int = 96
    k_decide: int = 4
    """Recurrence depth for the action span and the pinned prompt: 4 on a 5090, 2 on
    the phone."""
    k_think: int | None = None
    """Depth ceiling for the think span; ``None`` takes the model's default depth."""
    k_ingest: int = 1
    """Depth at which tool observations enter. Cheap on purpose: the observation is
    read, not reasoned about, and attention in the prelude and coda still sees it."""
    depth_policy: Literal["auto", "fixed", "token"] = "auto"
    """Whether the three depths above may differ within one episode.

    ``token`` needs a model trained with ``recurrent.token_depth``; it is the
    ingest-cheap / think-deep schedule of docs/08_AGENT.md. ``fixed`` runs the whole
    episode at ``k_decide``, letting learned halting only lower it, which is the exact
    regime for a model trained at one depth per sequence. ``auto`` reads the model's
    config and picks ``token`` when the switch is on."""
    halt_threshold: float | None = 0.9
    """Passed to the model for the think span when learned halting is on."""
    use_selection_head: bool = True
    """When the model has action heads, let the selection pointer at ``<|call|>`` decide
    the tool (or "none", which leaves only the reserved actions) and restrict the
    grammar to it; the LM then only fills arguments. Its margin is recorded per step."""
    tau_act: float = 0.6
    """Confidence below which an irreversible action is verified first."""
    tau_done: float = 0.7
    """Confidence below which ``done`` is refused until a check agrees."""
    tau_ask: float = 0.4
    max_repeats: int = 3
    tool_output_cap_tokens: int = 512
    window_steps: int = 8
    sample_temperature: float = 0.7
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    verifier_version: str = "prior-0"
    family: str = "default"
    sample_actions: bool = False
    """Sample the action span at ``sample_temperature`` instead of decoding it greedily.
    Off for benches; on when the episode is training data for a policy-gradient method
    (docs/research/A5_klpo.md): a greedy sampler is a point mass and teaches nothing."""
    record_sampling: bool = False
    """Record, for every token the model draws, the sampler's log-probability as used
    (grammar-masked, tempered) and ``mc_draws`` auxiliary draws with theirs, in
    ``EpisodeResult.sampled``; the exact token stream goes to ``EpisodeResult.ids``."""
    mc_draws: int = 8
    sample_copy: bool = False
    """Sample the copy pointer's start and end from their softmax at ``sample_temperature``
    instead of taking the argmax (docs/31 amendment 12). Without it the pointer never
    explores, so a systematically misplaced copy can never yield a verified episode and
    the closed loop cannot correct it. Greedy (temperature 0) keeps the argmax."""
    no_repeat_action: bool = False
    """Forbid, at step *i*, the action name of step *i - 1* (docs/31 amendment 11). Canonical
    trajectories never repeat a step; without this the decoder could loop on ``note``
    and never reach ``done`` within the step budget."""
    copy_topk: int = 0
    """Draw the copy pointer's start uniformly among its ``copy_topk`` best positions
    (docs/31 amendment 15); 0 keeps the argmax (or the tempered draw of ``sample_copy``).
    On the tasks a checkpoint always fails, the right value was the pointer's *second*
    choice with a probability its softmax never reaches (docs/32 §14)."""
    copy_explore: str = "all"
    """Which copy events ``copy_topk`` explores: ``"all"`` of them, or only those whose
    argmax start lies in a tool ``"observations"`` (docs/31 amendment 17). What is copied
    from the goal is a constant of the task and not worth exploring; what is copied from
    an observation is the choice a loop can get systematically wrong."""
    copy_boundaries: str = "off"
    """Restrict the copy pointer's start to word boundaries -- positions whose previous
    token ends in whitespace, a quote or punctuation, or that open an observation --
    for the exploratory draw only (``"explore"``) or for the argmax as well
    (``"always"``); ``"off"`` leaves the pointer free (docs/31 amendment 19). A copied
    value is a whole word or field; a start inside a word is never right, and on the
    misplaced pointer of docs/32 §13 the right start ranked 11th and 15th, behind
    positions inside the same name."""


@dataclass
class StepRecord:
    step: int
    think: str
    action: Action | None
    verdict: Verdict | None
    observation: str
    gated: str = ""
    """What the gate did, if anything: 'verify_first', 'refused_done', 'ask', 'reflect'."""
    selected: str | None = None
    """What the selection head chose at ``<|call|>``: a tool name, 'none', or None when
    the model has no action heads."""
    sel_margin: float | None = None
    """Top-1 minus top-2 selection probability -- A3's ambiguity signal."""
    copied: int = 0
    """Argument values filled by the copy pointer rather than generated."""


@dataclass
class EpisodeResult:
    finished: bool
    reason: str
    steps: list[StepRecord]
    final_notes: str
    verified_before_done: bool
    asked_user: str | None = None
    session: Any | None = None
    """The bounded recurrent state at the end of the episode (``SessionMemory``), for
    the next episode to start from. Attention caches are not carried: see
    ``prophet.memory.session``."""
    tokens: int = 0
    """Every token the model processed in the episode: prompt, spans, observations and
    spliced values. The denominator of "tokens per success"."""
    ids: list[int] | None = None
    """With ``record_sampling``: the exact token stream the model was fed."""
    sampled: list[dict[str, Any]] | None = None
    """With ``record_sampling``: one record per token the sampler drew --
    ``position`` in ``ids``, ``token``, ``logq``, ``mc_ids``, ``mc_logq``."""


def choose_copy_span(
    s_logits: torch.Tensor, e_logits: torch.Tensor, *, temperature: float, topk: int = 0
) -> tuple[int, int]:
    """Start and end indices of the copy span among the key positions. ``temperature``
    zero (or negative) takes the argmax of each pointer; otherwise both are sampled from
    their tempered softmax, the end restricted to positions at or after the start.
    ``topk`` > 0 draws the start uniformly among the ``topk`` highest-scoring positions
    instead (docs/31 amendment 15): a confident pointer's second choice is then reached
    as often as its first, which no temperature achieves."""
    finite = int(torch.isfinite(s_logits).sum())
    if topk > 0 and finite > 0:
        candidates = torch.topk(s_logits, min(topk, finite)).indices
        start_i = int(candidates[torch.randint(candidates.numel(), (1,))].item())
    elif temperature > 0:
        start_i = int(torch.multinomial(torch.softmax(s_logits / temperature, -1), 1).item())
    else:
        start_i = int(s_logits.argmax())
    e_logits = e_logits.masked_fill(torch.arange(e_logits.numel()) < start_i, float("-inf"))
    if temperature > 0:
        end_i = int(torch.multinomial(torch.softmax(e_logits / temperature, -1), 1).item())
    else:
        end_i = int(e_logits.argmax())
    return start_i, end_i


class AgentLoop:
    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        tools: ToolRegistry,
        cfg: AgentConfig,
        *,
        scorer: SignalScorer | None = None,
        quarantine: Quarantine | None = None,
        verifier_tool: Callable[[AgentState], bool] | None = None,
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.tools = tools
        self.cfg = cfg
        self.scorer = scorer or SignalScorer()
        self.quarantine = quarantine
        self.verifier_tool = verifier_tool
        """An executable check for the task (tests, a checklist). When present, ``done``
        is accepted only if it passes; when absent the confidence head decides."""
        self.grammar = ActionGrammar(tools)
        self.decoder = ConstrainedDecoder(
            self.grammar,
            lambda tid: self.tok.decode([tid]),
            end_id=self._sid("<|/call|>"),
        )
        self._ids: list[int] = []
        self._observation_spans: list[tuple[int, int]] = []
        self._copied = 0
        self._has_action = getattr(model, "action", None) is not None
        recurrent = getattr(getattr(model, "cfg", None), "recurrent", None)
        if cfg.depth_policy == "auto":
            # A model without a config (a scripted stand-in) ignores depth anyway.
            self.variable_depth = recurrent is None or bool(
                getattr(recurrent, "token_depth", False)
            )
        else:
            self.variable_depth = cfg.depth_policy == "token"
        if self.variable_depth and recurrent is not None and not recurrent.token_depth:
            raise ValueError(
                "depth_policy='token' on a model whose recurrent.token_depth is off: "
                "varying the depth within one cache is undefined for it"
            )
        self.k_think = (
            cfg.k_think
            if cfg.k_think is not None
            else (int(recurrent.default_loop_k) if recurrent is not None else cfg.k_decide)
        )

    # -- ids ---------------------------------------------------------------------------

    def _sid(self, name: str) -> int | None:
        try:
            return self.tok.special_id(name)
        except KeyError:
            return None

    # -- decoding ----------------------------------------------------------------------

    @torch.no_grad()
    def _feed(
        self,
        ids: list[int],
        cache: ProphetCache,
        *,
        loop_k: int | None,
        halt_threshold: float | None = None,
        modality: int | None = None,
        positions: dict[str, list[int]] | None = None,
        observation: bool = False,
    ):
        if not ids:
            return None
        self._ids.extend(ids)  # absolute position == index; eviction never renumbers
        t = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        kw: dict[str, Any] = dict(cache=cache, loop_k=loop_k, return_mtp=True)
        if self.cfg.ledger_write == "tool":
            kw["ledger_write_mask"] = torch.full_like(t, observation, dtype=torch.bool)
        if halt_threshold is not None:
            kw["halt_threshold"] = halt_threshold
        if positions and getattr(self.model, "action", None) is not None:
            for name, values in positions.items():
                kw[name] = torch.tensor([values], dtype=torch.long, device=t.device)
        if modality is not None and getattr(self.model, "modality_embed", None) is not None:
            kw["modality_ids"] = torch.full_like(t, modality)
        return self.model(t, **kw)

    @torch.no_grad()
    def _decode(
        self,
        cache: ProphetCache,
        *,
        budget: int,
        loop_k: int | None,
        halt_threshold: float | None,
        greedy: bool,
        stop_ids: set[int],
        constrained: bool = False,
    ) -> tuple[str, Any]:
        """Generate up to ``budget`` tokens; return the text and the last output.

        In a constrained span with action heads, every fed token also scores the copy
        pointer at its own position, so that when the grammar reports a value start the
        decision to copy -- and where from -- is already in hand.
        """
        out = None
        pieces: list[int] = []
        prefix = ""
        last_id: int | None = None
        copy_kw = {"copy_positions": [0]} if constrained and self._has_action else None
        self._copied = 0
        for _ in range(budget):
            if last_id is not None:
                out = self._feed(
                    [last_id],
                    cache,
                    loop_k=loop_k,
                    halt_threshold=halt_threshold,
                    positions=copy_kw,
                )
            elif out is None:
                # First token of the span is produced from the cache's current logits;
                # the caller has already fed the span opener.
                out = self._last_output
            if constrained and copy_kw is not None:
                spliced = self._try_copy(prefix, out)
                if spliced:
                    ids = self.tok.encode(spliced)
                    pieces += ids
                    prefix = self.tok.decode(pieces)
                    # The value entered as if generated; the next token follows it.
                    out = self._feed(ids, cache, loop_k=loop_k, positions=copy_kw)
                    last_id = None
                    self._copied += 1
                    if self.grammar.check(prefix).complete:
                        break
            logits = out.logits[0, -1].float()
            if constrained:
                ranked = logits.topk(min(self.decoder.candidates, logits.numel())).indices.tolist()
                allowed = self.decoder.allowed(prefix, ranked)
                if not allowed:
                    break
                mask = torch.full_like(logits, float("-inf"))
                mask[allowed] = 0.0
                logits = logits + mask
            if greedy or self.cfg.sample_temperature <= 0:
                nxt = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / self.cfg.sample_temperature, -1)
                nxt = int(torch.multinomial(probs, 1).item())
                if self.cfg.record_sampling:
                    self._record_draw(logits, probs, nxt)
            if nxt in stop_ids:
                last_id = nxt
                break
            pieces.append(nxt)
            prefix = self.tok.decode(pieces)
            last_id = nxt
            if constrained and self.grammar.check(prefix).complete:
                break
        if last_id is not None:
            # Fold the terminal token into the cache so the next span starts after it.
            self._last_output = self._feed([last_id], cache, loop_k=loop_k) or self._last_output
        return prefix, out

    def _record_draw(self, logits: torch.Tensor, probs: torch.Tensor, token: int) -> None:
        """The sampler as it was actually used at this position, plus auxiliary draws.

        ``logits`` are already grammar-masked; the log-probabilities are those of the
        tempered distribution the token was drawn from. The position is where the token
        will land in ``self._ids``: it is fed next, nothing is fed in between.
        """
        log_probs = torch.log_softmax(logits / self.cfg.sample_temperature, -1)
        draws = torch.multinomial(probs, max(int(self.cfg.mc_draws), 1), replacement=True)
        self._sampled.append(
            {
                "position": len(self._ids),
                "token": int(token),
                "logq": float(log_probs[token]),
                "mc_ids": draws.tolist(),
                "mc_logq": log_probs[draws].tolist(),
            }
        )

    def _record_kw(self) -> dict[str, Any]:
        if not self.cfg.record_sampling:
            return {}
        return {"ids": list(self._ids), "sampled": list(self._sampled)}

    def _try_copy(self, prefix: str, out) -> str | None:
        """At a value start, ask the gate; if it says copy, read the span the pointers
        chose out of everything fed so far, render it as the JSON the schema expects,
        and accept it only if the grammar does. Otherwise generate as usual."""
        state = self.grammar.check(prefix)
        if not state.value_start:
            return None
        gate = getattr(out, "copy_gate", None)
        starts, ends = getattr(out, "copy_start", None), getattr(out, "copy_end", None)
        key_pos = getattr(out, "copy_key_positions", None)
        if gate is None or starts is None or ends is None or key_pos is None:
            return None
        if float(gate[0, -1]) <= 0.0:
            return None
        s_logits, e_logits = starts[0, -1].float(), ends[0, -1].float()
        topk = self.cfg.copy_topk
        if topk > 0 and self.cfg.copy_explore == "observations":
            preferred = int(key_pos[int(s_logits.argmax())])
            if not self._from_observation(preferred):
                topk = 0
        if self.cfg.copy_boundaries == "always" or (
            self.cfg.copy_boundaries == "explore" and topk > 0
        ):
            boundary = torch.tensor([self._word_start(int(k)) for k in key_pos])
            if bool(boundary.any()):
                s_logits = s_logits.masked_fill(~boundary, float("-inf"))
        start_i, end_i = choose_copy_span(
            s_logits,
            e_logits,
            temperature=self.cfg.sample_temperature if self.cfg.sample_copy else 0.0,
            topk=topk,
        )
        start, end = int(key_pos[start_i]), int(key_pos[end_i])
        if end < start or end >= len(self._ids):
            return None
        span = self.tok.decode(self._ids[start : end + 1])
        if span.startswith(" "):
            span = span[1:]  # a word's token carries its leading space; the value does not
        if not span or "\n" in span:
            return None
        if state.expected_type in (None, "string"):
            rendered = json.dumps(span)
        elif state.expected_type in ("integer", "number"):
            try:
                value = int(span) if state.expected_type == "integer" else float(span)
            except ValueError:
                return None
            rendered = json.dumps(value)
        elif state.expected_type == "boolean":
            if span not in ("true", "false"):
                return None
            rendered = span
        else:
            return None
        return rendered if self.grammar.check(prefix + rendered).viable else None

    def _from_observation(self, position: int) -> bool:
        """Whether an absolute position of ``self._ids`` lies in a tool observation fed
        this episode."""
        return any(start <= position < end for start, end in self._observation_spans)

    _BOUNDARY_CHARS = " \t\n\r\"'`:,;=(){}[]<>|"

    def _word_start(self, position: int) -> bool:
        """Whether a copied span may start at ``position``: the first token of the
        context or of an observation, or a token whose predecessor ends in whitespace,
        a quote or punctuation (a control token decodes to nothing and counts too)."""
        if position <= 0 or position >= len(self._ids):
            return position == 0
        if any(position == start for start, _ in self._observation_spans):
            return True
        previous = self.tok.decode([self._ids[position - 1]])
        return previous == "" or previous[-1] in self._BOUNDARY_CHARS

    # -- the episode -------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        goal: str,
        *,
        notes: str = "",
        modality_tool: int | None = None,
        session: Any | None = None,
    ) -> EpisodeResult:
        """Run one episode.

        ``session`` carries the recurrent core's bounded state from an earlier episode
        (track R03 applied to the agent): the model starts with what it accumulated,
        while the attention layers start empty at the carried position. The returned
        ``EpisodeResult.session`` is the state to pass next time.
        """
        self.model.eval()
        cache = ProphetCache()
        self._ids = []
        self._sampled = []
        self._observation_spans = []
        fingerprint = (
            model_fingerprint(self.model)
            if isinstance(self.model, torch.nn.Module) and hasattr(self.model, "cfg")
            else ""
        )
        # An episode without a session starts from empty ledgers: they are buffers on
        # the model, not on the cache, and would otherwise leak from one episode into the
        # next through the weights.
        for layer in self._ledger_layers():
            layer.ledger.reset()
        if session is not None:
            restore_session(session, cache, fingerprint=fingerprint, model=self._module())
            # Positions continue from the carried count; the id log must line up.
            self._ids = [self.tok.pad_id] * cache.position
        carried = len(self._ids)  # placeholders are not tokens this episode processed
        state = AgentState(goal=goal, notes=notes, window_steps=self.cfg.window_steps)

        # The pinned prompt is read at the deciding depth: it is what every later span
        # reasons over. Under a fixed depth policy this call also pins the cache, and
        # every later span passes ``loop_k=None`` to follow that pin (halting may only
        # lower it). Under per-token depth each span names its own ceiling.
        pinned_ids = self._pinned_ids(goal, notes)
        anchor_id = self._sid("<|/tool_def|>")
        anchors = [i for i, tid in enumerate(pinned_ids) if tid == anchor_id]
        tool_names = [s.name for s in self.tools.schemas()]  # the anchors' order
        self._last_output = self._feed(
            pinned_ids,
            cache,
            loop_k=self.cfg.k_decide,
            positions={"anchor_positions": anchors} if anchors else None,
        )
        k_think = self.k_think if self.variable_depth else None
        k_act = self.cfg.k_decide if self.variable_depth else None
        k_ingest = self.cfg.k_ingest if self.variable_depth else None

        records: list[StepRecord] = []
        verified_before_done = False
        think_open, think_close = self._sid("<|think|>"), self._sid("<|/think|>")
        call_open, call_close = self._sid("<|call|>"), self._sid("<|/call|>")

        for _ in range(self.cfg.max_steps):
            state.record_snapshot(cache)
            step = state.step

            # 1. think: free text, learned halting, budgeted.
            think = ""
            if think_open is not None:
                self._last_output = self._feed(
                    [think_open], cache, loop_k=k_think, halt_threshold=self.cfg.halt_threshold
                )
                think, _ = self._decode(
                    cache,
                    budget=self.cfg.think_budget,
                    loop_k=k_think,
                    halt_threshold=self.cfg.halt_threshold,
                    greedy=False,
                    stop_ids={think_close} if think_close is not None else set(),
                )

            # 2. act: grammar-constrained, greedy, at the deciding depth. With action
            # heads, the selection pointer decides the tool at <|call|> and the grammar
            # is narrowed to it; the LM fills arguments.
            self._last_output = self._feed(
                [call_open],
                cache,
                loop_k=k_act,
                positions={"decision_positions": [0]},
            )
            selected, sel_margin = self._selection(self._last_output, tool_names)
            exclude = frozenset()
            if self.cfg.no_repeat_action:
                previous = next(
                    (t["action"] for t in reversed(state.trajectory) if t.get("action")), None
                )
                if previous is not None:
                    exclude = frozenset({previous["name"]})
            if selected is not None and self.cfg.use_selection_head:
                self.grammar.restrict(set() if selected == "none" else {selected}, exclude=exclude)
            elif exclude:
                self.grammar.restrict(None, exclude=exclude)
            try:
                text, out = self._decode(
                    cache,
                    budget=self.cfg.action_budget,
                    loop_k=k_act,
                    halt_threshold=None,
                    greedy=not self.cfg.sample_actions,
                    stop_ids={call_close} if call_close is not None else set(),
                    constrained=True,
                )
            finally:
                self.grammar.restrict(None)
            action = self.grammar.complete(text)
            if action is None:
                # The grammar guarantees viability, not completion within budget.
                records.append(
                    StepRecord(
                        step,
                        think,
                        None,
                        None,
                        "",
                        gated="malformed",
                        selected=selected,
                        sel_margin=sel_margin,
                        copied=self._copied,
                    )
                )
                state.trajectory.append(
                    {"step": step, "think": think, "action": None, "gated": "malformed"}
                )
                state.step += 1
                continue

            # 3. gate.
            sig = extract_signals(out, project=getattr(self.model, "_project", None))
            p = self.scorer.prob(sig)
            verdict = decide(sig, p, self.cfg.verifier, attempts=state.attempts_on_current)
            gated = ""

            if action.name == "done":
                if self.verifier_tool is not None:
                    ok = self.verifier_tool(state)
                    verified_before_done = ok
                    if not ok:
                        gated = "refused_done"
                        action = Action("verify", {"what": "verifier failed; keep working"})
                elif p < self.cfg.tau_done:
                    gated = "refused_done"
                    action = Action("verify", {"what": f"confidence {p:.2f} below tau_done"})
                if action.name == "done":
                    records.append(
                        StepRecord(
                            step,
                            think,
                            action,
                            verdict,
                            "",
                            gated,
                            selected=selected,
                            sel_margin=sel_margin,
                            copied=self._copied,
                        )
                    )
                    state.trajectory.append(self._traj(step, action, verdict, "", think))
                    self._close(state, passed=True, verified=verified_before_done)
                    return EpisodeResult(
                        True,
                        "done",
                        records,
                        state.notes,
                        verified_before_done,
                        session=self._session(cache, fingerprint),
                        tokens=len(self._ids) - carried,
                        **self._record_kw(),
                    )

            elif action.name == "ask" or (p < self.cfg.tau_ask and self._needs_user(action)):
                q = action.args.get("question", "clarification needed")
                records.append(
                    StepRecord(
                        step,
                        think,
                        action,
                        verdict,
                        "",
                        "ask",
                        selected=selected,
                        sel_margin=sel_margin,
                        copied=self._copied,
                    )
                )
                state.trajectory.append(self._traj(step, action, verdict, "", think))
                self._close(state, passed=False, verified=False)
                return EpisodeResult(
                    False,
                    "ask",
                    records,
                    state.notes,
                    False,
                    asked_user=q,
                    session=self._session(cache, fingerprint),
                    tokens=len(self._ids) - carried,
                    **self._record_kw(),
                )

            elif self.tools.is_irreversible(action.name) and p < self.cfg.tau_act:
                gated = "verify_first"
                action = Action(
                    "verify", {"what": f"dry-run of {action.name} at confidence {p:.2f}"}
                )

            # 4. loop detector.
            repeats = state.note_action(action.hash())
            if repeats > self.cfg.max_repeats:
                gated = "reflect"
                action = Action(
                    "note", {"text": state.notes + f"\n[stuck: repeated {action.name} x{repeats}]"}
                )

            # 5. execute / reserved actions.
            observation = self._execute(action, state, cache)

            # 6. ingest the observation cheaply, as its own modality; evict past the window.
            if observation:
                obs_ids = self._observation_ids(observation)
                start = cache.position
                self._last_output = (
                    self._feed(
                        obs_ids, cache, loop_k=k_ingest, modality=modality_tool, observation=True
                    )
                    or self._last_output
                )
                self._observation_spans.append((start, cache.position))
                obs = Observation(
                    step, action.name, observation, len(obs_ids), start, cache.position
                )
                for old in state.push_observation(obs):
                    state.evict_from_attention(cache, old)

            records.append(
                StepRecord(
                    step,
                    think,
                    action,
                    verdict,
                    observation,
                    gated,
                    selected=selected,
                    sel_margin=sel_margin,
                    copied=self._copied,
                )
            )
            state.trajectory.append(self._traj(step, action, verdict, observation, think))
            state.step += 1

        self._close(state, passed=False, verified=False)
        return EpisodeResult(
            False,
            "max_steps",
            records,
            state.notes,
            False,
            session=self._session(cache, fingerprint),
            tokens=len(self._ids) - carried,
            **self._record_kw(),
        )

    # -- helpers -------------------------------------------------------------------------

    def _session(self, cache: ProphetCache, fingerprint: str):
        """The carried state: the recurrent slots and, when the model has them, the
        attention ledgers (what fell out of the window, kept by key)."""
        if not cache.slots and not self._ledger_layers():
            return None
        return extract_session(cache, fingerprint=fingerprint, model=self._module())

    def _module(self) -> torch.nn.Module | None:
        return self.model if isinstance(self.model, torch.nn.Module) else None

    def _ledger_layers(self) -> list:
        module = self._module()
        if module is None:
            return []
        from prophet.modeling.layers import LedgerAttention

        return [m for m in module.modules() if isinstance(m, LedgerAttention)]

    @staticmethod
    def _selection(out, tool_names: list[str]) -> tuple[str | None, float | None]:
        """Read the selection head at the <|call|> just fed: (choice, margin)."""
        logits = getattr(out, "sel_logits", None)
        if logits is None:
            return None, None
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        top = probs.topk(min(2, probs.numel()))
        margin = float(top.values[0] - (top.values[1] if probs.numel() > 1 else 0.0))
        index = int(top.indices[0])
        if index == 0 or index - 1 >= len(tool_names):
            return "none", margin
        return tool_names[index - 1], margin

    def _pinned_ids(self, goal: str, notes: str) -> list[int]:
        """Control ids are spliced in explicitly; the goal, the tool schemas and the
        notes are encoded as plain text, so none of them can mint a control token."""
        ids = [self.tok.bos_id]
        ids += self._with_control("<|system|>")
        ids += self.tok.encode(f"Goal: {goal}\n")
        for schema in self.tools.schemas():
            # Real control ids around each schema: the closing one is the anchor the
            # selection head reads. Encoded as text they would be bytes, and no anchor.
            ids += self._with_control("<|tool_def|>")
            ids += self.tok.encode(schema.body())
            ids += self._with_control("<|/tool_def|>")
        ids += self.tok.encode(f"\nNotes:\n{notes}\n")
        ids += self._with_control("<|assistant|>")
        return ids

    def _observation_ids(self, text: str) -> list[int]:
        """``<|tool|>`` then the observation as plain text -- the text itself is
        untrusted and a literal ``"<|assistant|>"`` inside it stays seven characters."""
        body = self.tok.encode(text)[: self.cfg.tool_output_cap_tokens]
        return self._with_control("<|tool|>") + body

    def _with_control(self, name: str) -> list[int]:
        sid = self._sid(name)
        return [] if sid is None else [sid]

    def _needs_user(self, action: Action) -> bool:
        return action.name in ("done", "submit", "send", "purchase")

    def _traj(
        self, step: int, action: Action | None, verdict: Verdict | None, obs: str, think: str = ""
    ) -> dict:
        """One serialised step. The think span and the full (already capped)
        observation are kept: a promoted episode is rendered back into training text
        by ``prophet.agent.render``, and a summary cannot be un-summarised."""
        return {
            "step": step,
            "think": think,
            "action": None if action is None else {"name": action.name, "args": action.args},
            "p_correct": None if verdict is None else verdict.p_correct,
            "tier": None if verdict is None else int(verdict.tier),
            "observation": obs,
        }

    def _execute(self, action: Action, state: AgentState, cache: ProphetCache) -> str:
        if action.name == "note":
            text = str(action.args.get("text", ""))
            ids = self.tok.encode(text)
            state.notes = self.tok.decode(ids[: state.notes_cap_tokens])
            return ""
        if action.name == "rollback":
            target = int(action.args.get("step", 0))
            ok = state.rollback(cache, target)
            return f"rolled back to step {target}" if ok else f"no snapshot for step {target}"
        if action.name == "verify":
            if self.verifier_tool is not None:
                ok = self.verifier_tool(state)
                return "verification passed" if ok else "verification failed"
            return "no verifier available; re-check against the notes"
        if action.name in ("ask", "done"):
            return ""
        try:
            result = self.tools.run(action)
        except (KeyError, ValueError, TypeError) as exc:
            return f"error: {exc}"
        return "" if result is None else str(result)

    def _close(self, state: AgentState, *, passed: bool, verified: bool) -> None:
        if self.quarantine is None:
            return
        last = next((t for t in reversed(state.trajectory) if t.get("p_correct") is not None), None)
        tier = (
            Tier.GROUND_TRUTH
            if (verified and passed)
            else (Tier.LEARNED if last is not None else Tier.UNVERIFIED)
        )
        self.quarantine.add(
            Entry(
                family=self.cfg.family,
                goal=state.goal,
                trajectory=state.trajectory,
                outcome_passed=passed,
                process_ok=verified,
                provenance=Provenance(
                    tier=int(tier),
                    verifier_version=self.cfg.verifier_version,
                    p_correct=float(last["p_correct"]) if last else 0.0,
                    depth_disagreement=None,
                    attempts=state.attempts_on_current,
                ),
            )
        )
