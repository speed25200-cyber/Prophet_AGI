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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from prophet.agent.actions import Action, ActionGrammar, ConstrainedDecoder, ToolRegistry
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
from prophet.modeling.model import ProphetCache

__all__ = ["Tokenizer", "AgentConfig", "StepRecord", "EpisodeResult", "AgentLoop"]


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids, *, skip_special: bool = True) -> str: ...
    def special_id(self, name: str) -> int: ...


@dataclass
class AgentConfig:
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


@dataclass
class StepRecord:
    step: int
    think: str
    action: Action | None
    verdict: Verdict | None
    observation: str
    gated: str = ""
    """What the gate did, if anything: 'verify_first', 'refused_done', 'ask', 'reflect'."""


@dataclass
class EpisodeResult:
    finished: bool
    reason: str
    steps: list[StepRecord]
    final_notes: str
    verified_before_done: bool
    asked_user: str | None = None


class AgentLoop:
    def __init__(self, model, tokenizer: Tokenizer, tools: ToolRegistry, cfg: AgentConfig,
                 *, scorer: SignalScorer | None = None, quarantine: Quarantine | None = None,
                 verifier_tool: Callable[[AgentState], bool] | None = None) -> None:
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
            self.grammar, lambda tid: self.tok.decode([tid]),
            end_id=self._sid("<|/call|>"),
        )
        recurrent = getattr(getattr(model, "cfg", None), "recurrent", None)
        if cfg.depth_policy == "auto":
            # A model without a config (a scripted stand-in) ignores depth anyway.
            self.variable_depth = recurrent is None or bool(getattr(recurrent, "token_depth", False))
        else:
            self.variable_depth = cfg.depth_policy == "token"
        if self.variable_depth and recurrent is not None and not recurrent.token_depth:
            raise ValueError(
                "depth_policy='token' on a model whose recurrent.token_depth is off: "
                "varying the depth within one cache is undefined for it"
            )
        self.k_think = cfg.k_think if cfg.k_think is not None else (
            int(recurrent.default_loop_k) if recurrent is not None else cfg.k_decide
        )

    # -- ids ---------------------------------------------------------------------------

    def _sid(self, name: str) -> int | None:
        try:
            return self.tok.special_id(name)
        except KeyError:
            return None

    # -- decoding ----------------------------------------------------------------------

    @torch.no_grad()
    def _feed(self, ids: list[int], cache: ProphetCache, *, loop_k: int | None,
              halt_threshold: float | None = None, modality: int | None = None):
        if not ids:
            return None
        t = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        kw: dict[str, Any] = dict(cache=cache, loop_k=loop_k, return_mtp=True)
        if halt_threshold is not None:
            kw["halt_threshold"] = halt_threshold
        if modality is not None and getattr(self.model, "modality_embed", None) is not None:
            kw["modality_ids"] = torch.full_like(t, modality)
        return self.model(t, **kw)

    @torch.no_grad()
    def _decode(self, cache: ProphetCache, *, budget: int, loop_k: int | None,
                halt_threshold: float | None, greedy: bool, stop_ids: set[int],
                constrained: bool = False) -> tuple[str, Any]:
        """Generate up to ``budget`` tokens; return the text and the last output."""
        out = None
        pieces: list[int] = []
        prefix = ""
        last_id: int | None = None
        for _ in range(budget):
            if last_id is not None:
                out = self._feed([last_id], cache, loop_k=loop_k, halt_threshold=halt_threshold)
            elif out is None:
                # First token of the span is produced from the cache's current logits;
                # the caller has already fed the span opener.
                out = self._last_output
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

    # -- the episode -------------------------------------------------------------------

    @torch.no_grad()
    def run(self, goal: str, *, notes: str = "", modality_tool: int | None = None) -> EpisodeResult:
        self.model.eval()
        cache = ProphetCache()
        state = AgentState(goal=goal, notes=notes, window_steps=self.cfg.window_steps)

        # The pinned prompt is read at the deciding depth: it is what every later span
        # reasons over. Under a fixed depth policy this call also pins the cache, and
        # every later span passes ``loop_k=None`` to follow that pin (halting may only
        # lower it). Under per-token depth each span names its own ceiling.
        self._last_output = self._feed(self._pinned_ids(goal, notes), cache, loop_k=self.cfg.k_decide)
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
                self._last_output = self._feed([think_open], cache, loop_k=k_think,
                                               halt_threshold=self.cfg.halt_threshold)
                think, _ = self._decode(
                    cache, budget=self.cfg.think_budget, loop_k=k_think,
                    halt_threshold=self.cfg.halt_threshold, greedy=False,
                    stop_ids={think_close} if think_close is not None else set(),
                )

            # 2. act: grammar-constrained, greedy, at the deciding depth.
            self._last_output = self._feed([call_open], cache, loop_k=k_act)
            text, out = self._decode(
                cache, budget=self.cfg.action_budget, loop_k=k_act,
                halt_threshold=None, greedy=True,
                stop_ids={call_close} if call_close is not None else set(), constrained=True,
            )
            action = self.grammar.complete(text)
            if action is None:
                # The grammar guarantees viability, not completion within budget.
                records.append(StepRecord(step, think, None, None, "", gated="malformed"))
                state.trajectory.append({"step": step, "action": None, "gated": "malformed"})
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
                    records.append(StepRecord(step, think, action, verdict, "", gated))
                    state.trajectory.append(self._traj(step, action, verdict, ""))
                    self._close(state, passed=True, verified=verified_before_done)
                    return EpisodeResult(True, "done", records, state.notes, verified_before_done)

            elif action.name == "ask" or (p < self.cfg.tau_ask and self._needs_user(action)):
                q = action.args.get("question", "clarification needed")
                records.append(StepRecord(step, think, action, verdict, "", "ask"))
                state.trajectory.append(self._traj(step, action, verdict, ""))
                self._close(state, passed=False, verified=False)
                return EpisodeResult(False, "ask", records, state.notes, False, asked_user=q)

            elif self.tools.is_irreversible(action.name) and p < self.cfg.tau_act:
                gated = "verify_first"
                action = Action("verify", {"what": f"dry-run of {action.name} at confidence {p:.2f}"})

            # 4. loop detector.
            repeats = state.note_action(action.hash())
            if repeats > self.cfg.max_repeats:
                gated = "reflect"
                action = Action("note", {"text": state.notes + f"\n[stuck: repeated {action.name} x{repeats}]"})

            # 5. execute / reserved actions.
            observation = self._execute(action, state, cache)

            # 6. ingest the observation cheaply, as its own modality; evict past the window.
            if observation:
                obs_ids = self._observation_ids(observation)
                start = cache.position
                self._last_output = self._feed(obs_ids, cache, loop_k=k_ingest, modality=modality_tool) or self._last_output
                obs = Observation(step, action.name, observation, len(obs_ids), start, cache.position)
                for old in state.push_observation(obs):
                    state.evict_from_attention(cache, old)

            records.append(StepRecord(step, think, action, verdict, observation, gated))
            state.trajectory.append(self._traj(step, action, verdict, observation))
            state.step += 1

        self._close(state, passed=False, verified=False)
        return EpisodeResult(False, "max_steps", records, state.notes, False)

    # -- helpers -------------------------------------------------------------------------

    def _pinned_ids(self, goal: str, notes: str) -> list[int]:
        """Control ids are spliced in explicitly; the goal, the tool schemas and the
        notes are encoded as plain text, so none of them can mint a control token."""
        ids = [self.tok.bos_id]
        ids += self._with_control("<|system|>")
        ids += self.tok.encode(f"Goal: {goal}\n{self.tools.render()}\nNotes:\n{notes}\n")
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

    def _traj(self, step: int, action: Action | None, verdict: Verdict | None, obs: str) -> dict:
        return {
            "step": step,
            "action": None if action is None else {"name": action.name, "args": action.args},
            "p_correct": None if verdict is None else verdict.p_correct,
            "tier": None if verdict is None else int(verdict.tier),
            "observation": obs[:200],
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
        tier = Tier.GROUND_TRUTH if (verified and passed) else (
            Tier.LEARNED if last is not None else Tier.UNVERIFIED
        )
        self.quarantine.add(Entry(
            family=self.cfg.family, goal=state.goal, trajectory=state.trajectory,
            outcome_passed=passed, process_ok=verified,
            provenance=Provenance(
                tier=int(tier), verifier_version=self.cfg.verifier_version,
                p_correct=float(last["p_correct"]) if last else 0.0,
                depth_disagreement=None, attempts=state.attempts_on_current,
            ),
        ))
