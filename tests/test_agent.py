"""Tests for the agentic pillar.

These exercise the loop's *mechanics*, not its competence: an untrained model's thinking
is noise, so the gate, rollback, loop-detection and quarantine tests drive the loop with
a scripted model that emits a chosen token sequence at a chosen confidence. A final smoke
test runs the real tiny ProphetModel through the loop end to end to prove the plumbing
holds -- nothing about the actions it takes is asserted, because nothing about them is
meaningful yet.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from prophet.agent.actions import (
    RESERVED_ACTIONS,
    Action,
    ActionGrammar,
    ConstrainedDecoder,
    ToolRegistry,
    ToolSchema,
)
from prophet.agent.loop import AgentConfig, AgentLoop, choose_copy_span
from prophet.agent.quarantine import Entry, Provenance, Quarantine
from prophet.agent.state import AgentState, Observation, snapshot_cache
from prophet.agent.verify import (
    Signals,
    SignalScorer,
    Tier,
    VerifierConfig,
    auroc,
    decide,
    extract_signals,
)
from prophet.config import ProphetConfig
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.model import ProphetCache, ProphetModel, ProphetOutput

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def registry() -> ToolRegistry:
    fs: dict[str, str] = {"a.py": "print(1)\n"}
    reg = ToolRegistry(
        [
            ToolSchema(
                "read_file",
                "read a file",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            ToolSchema(
                "write_file",
                "write a file",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"],
                },
                irreversible=True,
            ),
            ToolSchema(
                "count",
                "count",
                {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            ),
        ]
    )
    reg.bind("read_file", lambda path: fs.get(path, "error: no such file"))
    reg.bind("write_file", lambda path, text: fs.__setitem__(path, text) or "ok")
    reg.bind("count", lambda n: str(n * 2))
    reg._fs = fs  # type: ignore[attr-defined]
    return reg


TOK = ProphetTokenizer(merges=[])  # byte-level plus specials; needs no training


class ScriptedModel(nn.Module):
    """Emits a fixed token script with a fixed confidence, through a real ProphetCache
    interface, so the loop's control flow can be tested deterministically.

    The loop feeds tokens it never samples from (the pinned prompt, a span's terminal
    token, an observation), so a script cannot simply advance on every call. Instead the
    model *speaks* only inside a span: a fed ``<|think|>`` or ``<|call|>`` opens one, each
    call then emits the next script token, and emitting a closer ends it. Outside a span
    it emits ``<|eos|>``, which nothing reads.
    """

    OPENERS = (TOK.special_id("<|think|>"), TOK.special_id("<|call|>"))
    CLOSERS = (TOK.special_id("<|/think|>"), TOK.special_id("<|/call|>"))

    def __init__(self, script: list[int], *, confidence: float = 3.0, vocab: int = 600) -> None:
        super().__init__()
        self.script = script
        self.confidence = confidence
        self.vocab = vocab
        self.cursor = 0
        self.speaking = False
        self.dummy = nn.Parameter(torch.zeros(1))
        self.modality_embed = None

    def _project(self, h):
        return torch.zeros(*h.shape[:-1], self.vocab)

    def forward(
        self,
        ids,
        *,
        cache=None,
        loop_k=None,
        return_mtp=True,
        halt_threshold=None,
        modality_ids=None,
    ):
        b, s = ids.shape
        if cache is not None:
            cache.position += s
            if cache.loop_k is None:
                cache.loop_k = loop_k or 1
        if int(ids[0, -1]) in self.OPENERS:
            self.speaking = True
        logits = torch.full((b, s, self.vocab), -20.0)
        nxt = TOK.eos_id
        if self.speaking and self.cursor < len(self.script):
            nxt = self.script[self.cursor]
            self.cursor += 1
            if nxt in self.CLOSERS:
                self.speaking = False
        logits[:, -1, nxt] = 20.0
        return ProphetOutput(
            logits=logits,
            hidden=torch.zeros(b, s, 8),
            loop_k=loop_k or 1,
            confidence=torch.full((b, s), self.confidence),
        )


def script_for(*spans: str) -> list[int]:
    """Token ids for a sequence of spans the scripted model should emit."""
    out: list[int] = []
    for span in spans:
        out += TOK.encode(span, parse_special=True)
    return out


def call(action: dict) -> str:
    return json.dumps(action, separators=(",", ":")) + "<|/call|>"


# --------------------------------------------------------------------------------------
# Actions and grammar
# --------------------------------------------------------------------------------------


def test_grammar_accepts_prefixes_and_rejects_dead_ends():
    g = ActionGrammar(registry())
    assert g.check('{"name":"rea').viable
    assert g.check('{"name":"read_file","args":{"path":"a.py"}}').complete
    assert not g.check('{"name":"nope').viable
    assert not g.check('{"name":"count","args":{"n":"x"').viable  # wrong type
    assert not g.check('{"name":"read_file","args":{"pth"').viable  # unknown key
    assert not g.check('{"name":"read_file","args":{}}').viable  # missing required


def test_grammar_never_claims_viable_for_an_uncompletable_string():
    g = ActionGrammar(registry())
    for bad in ['{"nam":', '{"name":"read_file","x":', '{"name":"read_file"}}', '["name"']:
        assert not g.check(bad).viable, bad


def test_reserved_actions_are_always_available():
    g = ActionGrammar(registry())
    for name in RESERVED_ACTIONS:
        assert g.check(f'{{"name":"{name}"').viable


def test_registry_validates_arguments():
    reg = registry()
    with pytest.raises(ValueError, match="missing required"):
        reg.run(Action("read_file", {}))
    with pytest.raises(TypeError):
        reg.run(Action("count", {"n": "three"}))
    assert reg.run(Action("count", {"n": 3})) == "6"


def test_action_hash_is_canonical():
    assert Action("f", {"a": 1, "b": 2}).hash() == Action("f", {"b": 2, "a": 1}).hash()
    assert Action("f", {"a": 1}).hash() != Action("f", {"a": 2}).hash()


def test_constrained_decoder_keeps_only_viable_tokens():
    g = ActionGrammar(registry())
    dec = ConstrainedDecoder(g, lambda t: TOK.decode([t]), end_id=TOK.special_id("<|/call|>"))
    prefix = '{"name":"'
    ranked = [ord("r"), ord("w"), ord("c"), ord("z"), ord("d"), TOK.special_id("<|/call|>")]
    allowed = dec.allowed(prefix, ranked)
    assert (
        ord("r") in allowed and ord("w") in allowed and ord("c") in allowed and ord("d") in allowed
    )
    assert ord("z") not in allowed
    assert TOK.special_id("<|/call|>") not in allowed  # not complete yet


def test_constrained_decoder_allows_end_only_when_complete():
    g = ActionGrammar(registry())
    end = TOK.special_id("<|/call|>")
    dec = ConstrainedDecoder(g, lambda t: TOK.decode([t]), end_id=end)
    assert end in dec.allowed('{"name":"done"}', [end])
    assert dec.allowed('{"name":"nope"}', [end]) == []


# --------------------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------------------


def test_execution_overrides_every_learned_signal():
    sc = SignalScorer()
    assert sc.prob(Signals(confidence_logit=-10.0, execution=True)) > 0.9
    assert sc.prob(Signals(confidence_logit=10.0, execution=False)) < 0.1


def test_learned_tier_never_consolidates():
    """A head may gate acting, never remembering."""
    v = decide(Signals(), 0.99, VerifierConfig(threshold=0.5), attempts=0)
    assert v.decision == "act" and v.tier == Tier.LEARNED and v.consolidate is False


def test_consensus_consolidates_and_ground_truth_consolidates():
    assert decide(Signals(), 0.9, VerifierConfig(), attempts=0, agreements=3).consolidate
    assert decide(Signals(execution=True), 0.1, VerifierConfig(), attempts=0).consolidate


def test_free_verifier_permits_many_retries_then_asks():
    cfg = VerifierConfig(max_attempts_free=3)
    assert decide(Signals(execution=False), 0.1, cfg, attempts=0).decision == "retry_sample"
    assert decide(Signals(execution=False), 0.1, cfg, attempts=3).decision == "ask"


def test_depth_retry_needs_a_measured_gain():
    sig = Signals(depth_disagreement=0.4)
    assert (
        decide(sig, 0.2, VerifierConfig(depth_gain_points=0.0), attempts=0).decision
        != "retry_depth"
    )
    assert (
        decide(sig, 0.2, VerifierConfig(depth_gain_points=6.0), attempts=0).decision
        == "retry_depth"
    )


def test_learned_check_permits_one_extra_attempt_then_asks():
    cfg = VerifierConfig(max_attempts_learned=2, verifier_cost_ratio=0.0)
    assert decide(Signals(), 0.2, cfg, attempts=1).decision == "retry_sample"
    assert decide(Signals(), 0.2, cfg, attempts=2).decision == "ask"


def test_scorer_fits_and_calibrates_on_program_labels():
    torch.manual_seed(0)
    x = torch.randn(600, 10)
    y = (x[:, 0] - 0.7 * x[:, 4] + 0.3 * torch.randn(600) > 0).float()
    sc = SignalScorer()
    nll = sc.fit(x[:400], y[:400])
    assert sc.fitted and nll < 0.5
    t = sc.calibrate_temperature(x[400:], y[400:])
    assert 0.25 <= t <= 4.0


def test_auroc_by_rank():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_extract_signals_reads_every_free_signal_from_a_real_output():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg.recurrent.halting = "ponder"
    cfg.heads.confidence_head = True
    model = ProphetModel(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 2048, (1, 8)), loop_k=3)
    sig = extract_signals(out, project=model._project)
    assert sig.depth_disagreement is not None and 0.0 <= sig.depth_disagreement <= 1.0
    assert sig.mtp_disagreement is not None
    assert sig.expected_depth is not None
    assert sig.mean_entropy_bits > 0


# --------------------------------------------------------------------------------------
# Quarantine
# --------------------------------------------------------------------------------------


def _entry(fam, tier, ok=True, version="v1"):
    return Entry(fam, "goal", [], ok, True, Provenance(int(tier), version, 0.8, 0.1, 1))


def test_quarantine_promotion_rules(tmp_path):
    q = Quarantine(tmp_path / "q.json")
    assert not q.add(_entry("a", Tier.UNVERIFIED))
    q.add(_entry("a", Tier.LEARNED))
    assert q.promoted() == []
    for _ in range(2):
        q.add(_entry("b", Tier.CONSENSUS))
    assert q.promoted("b") == []
    q.add(_entry("b", Tier.CONSENSUS))
    assert len(q.promoted("b")) == 3
    q.add(_entry("c", Tier.GROUND_TRUTH))
    assert len(q.promoted("c")) == 1


def test_quarantine_persists_and_revokes(tmp_path):
    path = tmp_path / "q.json"
    q = Quarantine(path)
    q.add(_entry("c", Tier.GROUND_TRUTH, version="bad"))
    reloaded = Quarantine(path)
    assert reloaded.promoted("c")
    assert reloaded.revoke("bad") == 1
    assert Quarantine(path).promoted("c") == []


# --------------------------------------------------------------------------------------
# State: snapshots and rollback
# --------------------------------------------------------------------------------------


def test_rollback_restores_the_state_before_the_step_and_replays_exactly():
    torch.manual_seed(0)
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg).eval()
    cache, state = ProphetCache(), AgentState(goal="g")
    chunks = [torch.randint(0, 2048, (1, n)) for n in (12, 5, 7)]
    with torch.no_grad():
        model(chunks[0], cache=cache, loop_k=2)
        state.record_snapshot(cache)  # before step 0 acts
        first = model(chunks[1], cache=cache, loop_k=2).logits
        state.step = 1
        state.record_snapshot(cache)  # before step 1 acts
        model(chunks[2], cache=cache, loop_k=2)
        state.step = 2
        assert state.rollback(cache, 0)
        assert cache.position == 12 and state.step == 0
        again = model(chunks[1], cache=cache, loop_k=2).logits
    assert torch.allclose(first, again, atol=1e-6)


def test_rollback_to_an_unknown_step_is_refused():
    state = AgentState(goal="g")
    assert not state.rollback(ProphetCache(), 3)


def test_snapshot_cost_is_independent_of_episode_length():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg).eval()
    sizes = []
    for total in (16, 64, 160):
        cache = ProphetCache()
        with torch.no_grad():
            model(torch.randint(0, 2048, (1, total)), cache=cache, loop_k=2)
        sizes.append(snapshot_cache(cache).n_bytes())
    # Windowed attention (window 256 here) grows until the window; recurrent state does
    # not. Assert the recurrent part is flat.
    from prophet.modeling.layers import RecurrentState

    rec = []
    for total in (16, 64, 160):
        cache = ProphetCache()
        with torch.no_grad():
            model(torch.randint(0, 2048, (1, total)), cache=cache, loop_k=2)
        rec.append(sum(s.n_bytes() for s in cache.slots.values() if isinstance(s, RecurrentState)))
    assert len(set(rec)) == 1


def test_evicting_an_observation_drops_only_its_span():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg).eval()
    cache, state = ProphetCache(), AgentState(goal="g")
    with torch.no_grad():
        model(torch.randint(0, 2048, (1, 20)), cache=cache, loop_k=2)
    dropped = state.evict_from_attention(cache, Observation(0, "x", "", 5, 5, 10))
    assert dropped > 0
    from prophet.modeling.layers import AttentionCache

    for slot in cache.slots.values():
        if isinstance(slot, AttentionCache):
            assert not bool(((slot.positions >= 5) & (slot.positions < 10)).any())


# --------------------------------------------------------------------------------------
# Loop mechanics, driven by a scripted model
# --------------------------------------------------------------------------------------


def _loop(script, *, confidence=3.0, cfg=None, quarantine=None, verifier_tool=None):
    model = ScriptedModel(script, confidence=confidence)
    cfg = cfg or AgentConfig(max_steps=6, think_budget=4, action_budget=64, halt_threshold=None)
    return AgentLoop(
        model, TOK, registry(), cfg, quarantine=quarantine, verifier_tool=verifier_tool
    )


def test_done_with_high_confidence_finishes():
    loop = _loop(script_for("<|/think|>", call({"name": "done"})), confidence=5.0)
    result = loop.run("finish")
    assert result.finished and result.reason == "done"


def test_done_with_low_confidence_is_refused_without_a_verifier():
    loop = _loop(
        script_for("<|/think|>", call({"name": "done"}), "<|/think|>", call({"name": "done"})),
        confidence=-5.0,
    )
    result = loop.run("finish")
    assert not result.finished
    assert any(r.gated == "refused_done" for r in result.steps)


def test_done_is_accepted_only_when_the_verifier_passes():
    calls = {"n": 0}

    def verifier(state):
        calls["n"] += 1
        return calls["n"] >= 2

    loop = _loop(
        script_for("<|/think|>", call({"name": "done"}), "<|/think|>", call({"name": "done"})),
        confidence=-5.0,
        verifier_tool=verifier,
    )
    result = loop.run("finish")
    assert result.finished and result.verified_before_done
    assert result.steps[0].gated == "refused_done"


def test_irreversible_action_below_threshold_is_verified_first():
    loop = _loop(
        script_for(
            "<|/think|>",
            call({"name": "write_file", "args": {"path": "a.py", "text": "x"}}),
            "<|/think|>",
            call({"name": "done"}),
        ),
        confidence=-5.0,
    )
    result = loop.run("edit")
    assert result.steps[0].gated == "verify_first"
    assert result.steps[0].action.name == "verify"
    assert loop.tools._fs["a.py"] == "print(1)\n"  # nothing was written


def test_irreversible_action_above_threshold_executes():
    loop = _loop(
        script_for(
            "<|/think|>",
            call({"name": "write_file", "args": {"path": "a.py", "text": "x"}}),
            "<|/think|>",
            call({"name": "done"}),
        ),
        confidence=5.0,
    )
    result = loop.run("edit")
    assert result.steps[0].gated == ""
    assert loop.tools._fs["a.py"] == "x"


def test_ask_returns_the_question_to_the_user():
    loop = _loop(
        script_for("<|/think|>", call({"name": "ask", "args": {"question": "which file?"}}))
    )
    result = loop.run("edit")
    assert not result.finished and result.reason == "ask" and result.asked_user == "which file?"


def test_repeated_identical_action_triggers_reflection():
    same = call({"name": "read_file", "args": {"path": "a.py"}})
    loop = _loop(
        script_for(*(["<|/think|>", same] * 5)),
        cfg=AgentConfig(max_steps=5, think_budget=4, halt_threshold=None, max_repeats=3),
    )
    result = loop.run("read")
    gated = [r.gated for r in result.steps]
    assert "reflect" in gated
    assert "[stuck: repeated read_file" in result.final_notes


def test_observations_are_ingested_and_windowed():
    reads = ["<|/think|>", call({"name": "read_file", "args": {"path": "a.py"}})] * 3
    loop = _loop(
        script_for(*reads),
        cfg=AgentConfig(max_steps=3, think_budget=4, halt_threshold=None, window_steps=2),
    )
    result = loop.run("read")
    assert all(r.observation == "print(1)\n" for r in result.steps)


def test_rollback_action_restores_position():
    loop = _loop(
        script_for(
            "<|/think|>",
            call({"name": "count", "args": {"n": 2}}),
            "<|/think|>",
            call({"name": "rollback", "args": {"step": 0}}),
            "<|/think|>",
            call({"name": "done"}),
        ),
        confidence=5.0,
    )
    result = loop.run("go")
    assert result.steps[1].action.name == "rollback"
    assert "rolled back to step 0" in result.steps[1].observation


def test_malformed_call_within_budget_is_recorded_not_crashed():
    loop = _loop(
        script_for("<|/think|>", '{"name":"read_file","args":{"path":"a.py"'),  # never closes
        cfg=AgentConfig(max_steps=1, think_budget=4, action_budget=8, halt_threshold=None),
    )
    result = loop.run("read")
    assert result.steps and result.steps[0].gated == "malformed"
    # The span the budget cut is kept for reading (docs/33 amendment 7).
    span = result.steps[0].span
    assert span and '{"name":"read_file","args":{"path":"a.py"'.startswith(span)


def test_episode_lands_in_quarantine_with_provenance(tmp_path):
    q = Quarantine(tmp_path / "q.json")
    loop = _loop(
        script_for("<|/think|>", call({"name": "done"})),
        confidence=5.0,
        quarantine=q,
        verifier_tool=lambda s: True,
    )
    loop.run("finish")
    assert q.summary()["entries"] == 1
    e = q.entries[0]
    assert e.provenance.tier == Tier.GROUND_TRUTH and e.promoted and e.process_ok


def test_unverified_success_is_learned_tier_and_not_promoted(tmp_path):
    q = Quarantine(tmp_path / "q.json")
    loop = _loop(script_for("<|/think|>", call({"name": "done"})), confidence=5.0, quarantine=q)
    loop.run("finish")
    assert q.entries[0].provenance.tier == Tier.LEARNED and not q.entries[0].promoted


def test_real_model_runs_through_the_loop_end_to_end():
    """Plumbing only: an untrained model's actions mean nothing, and none are asserted."""
    torch.manual_seed(0)
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg.heads.confidence_head = True
    model = ProphetModel(cfg).eval()
    loop = AgentLoop(
        model,
        TOK,
        registry(),
        AgentConfig(max_steps=2, think_budget=6, action_budget=24, halt_threshold=None),
    )
    result = loop.run("read a.py")
    assert len(result.steps) <= 2


def test_session_state_carries_across_episodes():
    """The recurrent core's bounded state persists between episodes; positions and the
    id log continue from where the carried session stopped."""
    from prophet.modeling.layers import RecurrentState

    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg).eval()
    loop = AgentLoop(
        model,
        TOK,
        registry(),
        AgentConfig(max_steps=1, think_budget=3, action_budget=12, halt_threshold=None),
    )
    first = loop.run("one")
    assert first.session is not None and first.session.tokens_seen > 0
    saved = {k: v.clone() for k, v in first.session.states.items()}
    second = loop.run("two", session=first.session)
    assert second.session.tokens_seen > first.session.tokens_seen
    # The second episode started from the saved state: its first feed read it.
    assert any(isinstance(s, RecurrentState) for s in ProphetCache().slots.values()) is False
    assert set(saved) <= set(second.session.states)
    assert not all(torch.equal(saved[k], second.session.states[k]) for k in saved)


def test_episode_tokens_exclude_the_carried_prefix():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg).eval()
    loop = AgentLoop(
        model,
        TOK,
        registry(),
        AgentConfig(max_steps=1, think_budget=3, action_budget=12, halt_threshold=None),
    )
    first = loop.run("one")
    second = loop.run("two", session=first.session)
    assert 0 < second.tokens < 2 * first.tokens


# --------------------------------------------------------------------------------------
# Compact grammar: what the renderer writes is exactly what the decoder admits
# --------------------------------------------------------------------------------------


def test_grammar_rejects_whitespace_outside_strings_by_default():
    g = ActionGrammar(registry())
    assert not g.check(' {"name"').viable
    assert not g.check('{"name": "read_file"').viable
    assert not g.check('{"name":"note","args": {').viable
    # Inside strings, spaces are data; a tab is written escaped, as json.dumps does
    # (a raw one is a call json.loads refuses: docs/33 amendment 8).
    assert g.check('{"name":"note","args":{"text":"a b\\tc"}}').complete
    tolerant = ActionGrammar(registry(), compact=False)
    assert tolerant.check(' {"name"').viable and tolerant.check('{"name": "read_file"').viable


def test_grammar_refuses_inside_strings_what_json_refuses():
    """docs/33 amendment 8: a raw control character or an unknown escape inside a string
    makes a call json.loads refuses, yet the grammar called such a prefix viable. A newline
    sampled into a proposal's file name kept the span "viable" to the end of its budget:
    27 of 30 proposals malformed at the first calibration rung. Viable must mean
    completable."""
    g = ActionGrammar(registry())
    for tolerant in (False, True):
        grammar = ActionGrammar(registry(), compact=not tolerant)
        head = '{"name":"write_file","args":{"path":"a.py","text":"'
        for bad in ("a\nb", "a\tb", "a\x00", "a\\y", "a\\u00zz"):
            assert not grammar.check(head + bad).viable, repr(bad)
            with pytest.raises(json.JSONDecodeError):
                json.loads(head + bad + '"}}')
        for good in ("a\\nb", "a\\tb", 'a\\"b', "a\\\\b", "a\\/b", "caf\\u00e9"):
            full = head + good + '"}}'
            assert grammar.check(full).complete and grammar.complete(full) is not None, good
        for partial in ("a\\", "a\\u", "a\\u00", "a\\u00e"):  # ends inside an escape
            state = grammar.check(head + partial)
            assert state.viable and state.in_string, partial
    # Keys go through the same scanner.
    assert not g.check('{"name":"read_file","args":{"pa\nth').viable
    # Inside an array or object value, the same rules.
    nested = ActionGrammar(
        ToolRegistry(
            [
                ToolSchema(
                    "tag",
                    "tag",
                    {
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                        "required": ["items"],
                    },
                )
            ]
        )
    )
    assert not nested.check('{"name":"tag","args":{"items":["a\nb').viable
    assert not nested.check('{"name":"tag","args":{"items":["a\\qb').viable
    assert nested.check('{"name":"tag","args":{"items":["a\\nb"]}}').complete
    assert nested.check('{"name":"tag","args":{"items":["a\\u00').viable


def test_nested_arrays_and_objects_are_scanned_as_strict_json():
    """docs/33 amendment 9: the container scanner only counted brackets and strings, so
    {"a""b"}, [1 2] or a mismatched bracket were "complete" calls json.loads refuses.
    Nested values now follow JSON exactly, and value-only sampling sees the open strings
    inside them (the object proposal format draws its keys and values there)."""
    nested = ActionGrammar(
        ToolRegistry(
            [
                ToolSchema(
                    "put",
                    "put",
                    {
                        "type": "object",
                        "properties": {"obj": {"type": "object"}, "arr": {"type": "array"}},
                        "required": ["obj", "arr"],
                    },
                )
            ]
        )
    )
    head = '{"name":"put","args":{"obj":'
    for bad in ('{"a""b"}', '{"a":"b",}', '{"a":"b"]', '{"a":[1 2]}', "{1:2}"):
        assert not nested.check(head + bad).viable, bad
        with pytest.raises(json.JSONDecodeError):
            json.loads(head + bad + ',"arr":[]}}')
    # Whitespace JSON would allow is refused in the compact grammar, as everywhere else.
    assert not nested.check(head + '{"a" :1}').viable
    assert ActionGrammar(nested.registry, compact=False).check(head + '{"a" :1}').viable
    good = head + '{"a":"b","c":{"d":[1,"x",true,null]}},"arr":[{},[],"y"]}}'
    assert nested.check(good).complete and nested.complete(good) is not None
    assert nested.check(head + "{}").viable and nested.check(head + '{"a":').viable
    assert nested.check(head + '{"a":"b"').viable  # a comma or a brace may follow
    # Where value-only sampling draws: inside keys and values of the object, not between.
    assert nested.check(head + '{"ke').in_string
    assert nested.check(head + '{"key":"va').in_string
    assert not nested.check(head + '{"key":"value"').in_string
    assert not nested.check(head + '{"key":').in_string


def test_the_decoder_refuses_a_newline_inside_a_string_value():
    """The same defect as the decoder met it: a newline token inside an open string was a
    viable candidate; it is now masked like any dead end."""
    g = ActionGrammar(registry())
    dec = ConstrainedDecoder(g, lambda t: TOK.decode([t]), end_id=TOK.special_id("<|/call|>"))
    newline, letter = TOK.encode("\n")[0], TOK.encode("b")[0]
    prefix = '{"name":"read_file","args":{"path":"a'
    assert dec.allowed(prefix, [newline, letter]) == [letter]


def test_constrained_decoder_cannot_open_a_call_with_whitespace():
    """The first closed-loop pilot died here: a drifting model opened the span with
    spaces, the tolerant grammar let it, and no candidate was viable two tokens later."""
    g = ActionGrammar(registry())
    dec = ConstrainedDecoder(g, lambda t: TOK.decode([t]), end_id=TOK.special_id("<|/call|>"))
    space, newline, brace = TOK.encode(" ")[0], TOK.encode("\n")[0], TOK.encode("{")[0]
    assert dec.allowed("", [space, newline, brace]) == [brace]
    assert dec.allowed("  ", [space, newline, brace]) == []  # already dead, as the loop sees it


def test_rendered_calls_are_exactly_what_the_compact_grammar_accepts():
    """Train/decode agreement: every call the renderer writes into a training row is a
    complete string for the grammar the loop decodes with, and contains no whitespace
    the grammar would refuse."""
    import re

    from prophet.agent import tasks as task_families
    from prophet.agent.render import render_episode

    for family in ("lookup", "calc", "files"):
        for task in task_families.make_tasks(2, family=family, seed=3):
            tools = task_families.tools_for(task)
            grammar = ActionGrammar(tools)
            text = render_episode(task.goal, tools, task_families.perfect_trajectory(task))
            bodies = re.findall(r"<\|call\|>(.*?)<\|/call\|>", text, flags=re.S)
            assert bodies, family
            for body in bodies:
                assert grammar.check(body).complete, body
                assert grammar.complete(body) is not None, body


def test_done_takes_no_arguments_so_the_grammar_closes_its_body_at_once():
    """docs/31 amendment 8: the renderer only ever writes {"name":"done","args":{}}; the old
    optional "summary" key let a drifting model open a key after "args":{ and die there
    (mode (c) of docs/32). With no properties, "}" is the only viable continuation."""
    g = ActionGrammar(registry())
    assert g.check('{"name":"done","args":{').viable
    assert not g.check('{"name":"done","args":{"').viable
    assert g.check('{"name":"done","args":{}}').complete
    assert g.complete('{"name":"done","args":{"summary":"x"}}') is None


def test_restrict_can_exclude_a_reserved_action_for_no_repeat():
    g = ActionGrammar(registry())
    g.restrict({"read_file"}, exclude=frozenset({"note"}))
    assert "read_file" in g.names and "done" in g.names and "note" not in g.names
    g.restrict(None, exclude=frozenset({"done"}))
    assert "note" in g.names and "done" not in g.names
    g.restrict(None)
    assert "note" in g.names and "done" in g.names


def test_choose_copy_span_is_argmax_when_greedy_and_explores_when_sampled():
    import torch as _t

    s = _t.tensor([0.0, 0.1, 0.0, -50.0])  # start: two close candidates, 1 and 0
    e = _t.tensor([0.0, 0.0, 3.0, 0.0])  # end: 2 preferred
    assert choose_copy_span(s, e, temperature=0.0) == (1, 2)
    _t.manual_seed(0)
    starts = {choose_copy_span(s, e, temperature=1.0)[0] for _ in range(200)}
    assert starts >= {0, 1} and 3 not in starts  # explores the close pair, never the -50
    for _ in range(50):
        st, en = choose_copy_span(s, e, temperature=1.0)
        assert en >= st  # the end never precedes the start


def test_choose_copy_span_topk_reaches_every_top_candidate_equally():
    """docs/31 amendment 15: a confident pointer's second choice is drawn as often as
    its first, which a tempered softmax never does."""
    import torch as _t

    s = _t.tensor([0.0, 6.0, -1.0, -50.0])  # the softmax at 0.7 puts ~1e-4 on position 0
    e = _t.tensor([0.0, 0.0, 3.0, 0.0])
    assert choose_copy_span(s, e, temperature=0.0, topk=1) == (1, 2)
    _t.manual_seed(0)
    draws = [choose_copy_span(s, e, temperature=0.0, topk=3)[0] for _ in range(300)]
    counts = {i: draws.count(i) for i in set(draws)}
    assert set(counts) == {0, 1, 2}, counts  # the three best, never the -50
    assert all(60 <= c <= 140 for c in counts.values()), counts  # roughly uniform
    for st, en in (choose_copy_span(s, e, temperature=0.0, topk=3) for _ in range(50)):
        assert en >= st
    # The loop reads the option: a config with copy_topk set changes the choice.
    assert AgentConfig(copy_topk=3).copy_topk == 3 and AgentConfig().copy_topk == 0


def test_copy_exploration_can_be_limited_to_spans_read_from_observations(monkeypatch):
    """docs/31 amendment 17: with ``copy_explore="observations"`` the top-k draw applies
    only when the pointer's preferred start lies in a tool observation; a span copied
    from the goal keeps the argmax."""
    import types

    import torch as _t

    from prophet.agent import loop as loop_module

    seen = []

    def spy(s_logits, e_logits, *, temperature, topk=0):
        seen.append(topk)
        return int(s_logits.argmax()), int(s_logits.argmax())

    monkeypatch.setattr(loop_module, "choose_copy_span", spy)
    reg = ToolRegistry()
    reg.add(
        ToolSchema("say", "Say", {"type": "object", "properties": {"text": {"type": "string"}}})
    )
    tok = ProphetTokenizer(merges=[])
    for explore, expect in (("observations", [2, 0]), ("all", [2, 2])):
        loop = AgentLoop(None, tok, reg, AgentConfig(copy_topk=2, copy_explore=explore))
        loop._ids = list(range(40))
        loop._observation_spans = [(20, 30)]  # one observation fed at positions 20..29
        prefix = '{"name":"say","args":{"text":'
        for preferred in (1, 0):  # index 1 -> position 25 (observation); 0 -> 3 (goal)
            out = types.SimpleNamespace(
                copy_gate=_t.tensor([[1.0]]),
                copy_start=_t.tensor([[[0.0, 0.0]]]),
                copy_end=_t.tensor([[[0.0, 0.0]]]),
                copy_key_positions=_t.tensor([3, 25]),
            )
            out.copy_start[0, 0, preferred] = 5.0
            loop._try_copy(prefix, out)
        assert seen == expect, (explore, seen)
        seen.clear()
    assert AgentLoop(None, tok, reg, AgentConfig())._from_observation(0) is False


def test_copy_starts_can_be_restricted_to_word_boundaries(monkeypatch):
    """docs/31 amendment 19: a copied value is a whole word; starts inside a word are
    masked for the exploratory draw ("explore") or for every choice ("always")."""
    import types

    import torch as _t

    from prophet.agent import loop as loop_module

    seen = []

    def spy(s_logits, e_logits, *, temperature, topk=0):
        seen.append((topk, [float(v) for v in s_logits]))
        return int(s_logits.argmax()), int(s_logits.argmax())

    monkeypatch.setattr(loop_module, "choose_copy_span", spy)
    reg = ToolRegistry()
    reg.add(
        ToolSchema("say", "Say", {"type": "object", "properties": {"text": {"type": "string"}}})
    )
    tok = ProphetTokenizer(merges=[])
    text = 'goal x\n"be" beacon_0.txt\n'
    ids = [ord(c) for c in text]  # the bare tokenizer is byte-level
    obs_start = text.index('"be"')
    inside = text.index("acon")  # inside beacon_0.txt
    after_nl = text.index("beacon") - 0  # follows a space
    positions = [0, obs_start, inside, after_nl]
    for mode, topk, expect_masked in (
        ("off", 2, []),
        ("explore", 2, [inside]),
        ("explore", 0, []),
        ("always", 0, [inside]),
    ):
        loop = AgentLoop(None, tok, reg, AgentConfig(copy_topk=topk, copy_boundaries=mode))
        loop._ids = list(ids)
        loop._observation_spans = [(obs_start, len(ids))]
        assert loop._word_start(0) and loop._word_start(obs_start) and loop._word_start(after_nl)
        assert not loop._word_start(inside)
        out = types.SimpleNamespace(
            copy_gate=_t.tensor([[1.0]]),
            copy_start=_t.tensor([[[0.0, 1.0, 5.0, 2.0]]]),
            copy_end=_t.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
            copy_key_positions=_t.tensor(positions),
        )
        loop._try_copy('{"name":"say","args":{"text":', out)
        got_topk, logits = seen.pop()
        masked = [positions[i] for i, v in enumerate(logits) if v == float("-inf")]
        assert masked == expect_masked and got_topk == topk, (mode, topk, masked)
    # Only finite candidates are drawn among.
    s = _t.tensor([1.0, float("-inf"), 3.0, float("-inf")])
    e = _t.tensor([0.0, 0.0, 0.0, 0.0])
    _t.manual_seed(0)
    assert {choose_copy_span(s, e, temperature=0.0, topk=3)[0] for _ in range(100)} == {0, 2}


def test_grammar_reports_the_inside_of_a_string_value_and_the_decoder_can_widen():
    """docs/33 amendment 2: a sampled sub-word inside a key leaves the decoder's top
    candidates without a viable continuation; sampling is limited to string values and
    the decoder may check the whole vocabulary before giving up."""
    reg = ToolRegistry()
    reg.add(
        ToolSchema(
            "propose",
            "P",
            {
                "type": "object",
                "properties": {"keys": {"type": "string"}, "ask": {"type": "string"}},
            },
        )
    )
    grammar = ActionGrammar(reg, compact=True)
    assert not grammar.check('{"name":"prop').in_string
    assert not grammar.check('{"name":"propose","args":{"ke').in_string
    assert not grammar.check('{"name":"propose","args":{"keys":').in_string
    assert grammar.check('{"name":"propose","args":{"keys":"').in_string
    assert grammar.check('{"name":"propose","args":{"keys":"city,ye').in_string
    assert not grammar.check('{"name":"propose","args":{"keys":"city"').in_string
    assert not grammar.check('{"name":"propose","args":{"keys":"city",').in_string
    tok = ProphetTokenizer(merges=[])
    decoder = ConstrainedDecoder(grammar, lambda tid: tok.decode([tid]), candidates=2)
    prefix = '{"name":"propose","args":{"k'
    ranked = [ord("z"), ord("q"), ord("e")]  # the viable "e" sits outside the top 2
    assert decoder.allowed(prefix, ranked) == []
    assert decoder.allowed(prefix, ranked, limit=None) == [ord("e")]
    loop = AgentLoop(None, tok, reg, AgentConfig(sample_actions=True, sample_scope="values"))
    assert not loop._sample_here('{"name":"pro', constrained=True, greedy=False)
    assert loop._sample_here('{"name":"propose","args":{"keys":"ci', constrained=True, greedy=False)
    assert loop._sample_here("free text", constrained=False, greedy=False)
    assert not loop._sample_here("free text", constrained=False, greedy=True)
    span = AgentLoop(None, tok, reg, AgentConfig(sample_actions=True))
    assert span._sample_here('{"name":"pro', constrained=True, greedy=False)


def test_a_partial_key_must_open_a_parameter_not_given_yet():
    """A decoder walking a prefix of a key already given reaches a dead end at the
    closing quote; the grammar refuses the prefix itself (docs/33 amendment 3)."""
    reg = ToolRegistry()
    reg.add(
        ToolSchema(
            "propose",
            "P",
            {
                "type": "object",
                "properties": {"ask": {"type": "string"}, "keys": {"type": "string"}},
            },
        )
    )
    grammar = ActionGrammar(reg, compact=True)
    assert grammar.check('{"name":"propose","args":{"ask":"code","k').viable
    assert grammar.check('{"name":"propose","args":{"ask":"code","ke').viable
    dead = grammar.check('{"name":"propose","args":{"ask":"code","a')
    assert not dead.viable and "unseen" in dead.reason
    assert not grammar.check('{"name":"propose","args":{"ask":"code","ask":"x"}').viable
    # A fresh key is still fine, and so is the first key.
    assert grammar.check('{"name":"propose","args":{"a').viable


def test_once_every_parameter_is_given_only_the_closing_brace_is_viable():
    """docs/33 amendment 4: a comma after the last parameter led the decoder into a key
    that cannot exist; the grammar refuses the comma itself."""
    reg = ToolRegistry()
    reg.add(
        ToolSchema(
            "propose",
            "P",
            {
                "type": "object",
                "properties": {"ask": {"type": "string"}, "keys": {"type": "string"}},
            },
        )
    )
    grammar = ActionGrammar(reg, compact=True)
    assert grammar.check('{"name":"propose","args":{"ask":"code",').viable
    assert grammar.check('{"name":"propose","args":{"ask":"code","keys":"x"}}').complete
    dead = grammar.check('{"name":"propose","args":{"ask":"code","keys":"x",')
    assert not dead.viable and "all parameters given" in dead.reason
    assert grammar.check('{"name":"propose","args":{"ask":"code","keys":"x"').viable


def test_sampling_can_be_kept_to_the_likeliest_tokens():
    import torch as _t

    tok = ProphetTokenizer(merges=[])
    reg = ToolRegistry()
    logits = _t.tensor([0.1, 3.0, 2.0, -1.0, 2.5])
    loop = AgentLoop(None, tok, reg, AgentConfig(sample_topk=2))
    kept = loop._sampling_logits(logits)
    assert kept.isfinite().sum() == 2 and kept[1] == 3.0 and kept[4] == 2.5
    assert AgentLoop(None, tok, reg, AgentConfig())._sampling_logits(logits).equal(logits)
    assert (
        AgentLoop(None, tok, reg, AgentConfig(sample_topk=9))._sampling_logits(logits).equal(logits)
    )


def test_copy_can_be_switched_off_for_episodes_that_invent_their_values(monkeypatch):
    """docs/33 amendment 5: with ``allow_copy`` off the pointer is never consulted, even
    when the gate is open at a value start."""
    import types

    import torch as _t

    from prophet.agent import loop as loop_module

    calls = []
    monkeypatch.setattr(loop_module, "choose_copy_span", lambda *a, **k: calls.append(1) or (0, 0))
    reg = ToolRegistry()
    reg.add(
        ToolSchema("say", "Say", {"type": "object", "properties": {"text": {"type": "string"}}})
    )
    tok = ProphetTokenizer(merges=[])
    out = types.SimpleNamespace(
        copy_gate=_t.tensor([[1.0]]),
        copy_start=_t.tensor([[[0.0, 5.0]]]),
        copy_end=_t.tensor([[[0.0, 0.0]]]),
        copy_key_positions=_t.tensor([3, 25]),
    )
    prefix = '{"name":"say","args":{"text":'
    on = AgentLoop(None, tok, reg, AgentConfig())
    on._ids = list(range(40))
    on._try_copy(prefix, out)
    assert calls == [1]
    off = AgentLoop(None, tok, reg, AgentConfig(allow_copy=False))
    off._ids = list(range(40))
    assert off._try_copy(prefix, out) is None and calls == [1]


def test_ordered_grammar_requires_the_keys_in_schema_order():
    """docs/33 amendment 6: the renderer writes argument keys in the schema's order; an
    ordered grammar refuses any other, also as a partial key."""
    reg = ToolRegistry()
    reg.add(
        ToolSchema(
            "propose",
            "P",
            {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "keys": {"type": "string"},
                    "ask": {"type": "string"},
                },
            },
        )
    )
    free, ordered = ActionGrammar(reg, compact=True), ActionGrammar(reg, compact=True, ordered=True)
    in_order = '{"name":"propose","args":{"file":"a.json","keys":"x","ask":"x"}}'
    out_of_order = '{"name":"propose","args":{"ask":"x","file":"a.json","keys":"x"}}'
    assert free.check(in_order).complete and free.check(out_of_order).complete
    assert ordered.check(in_order).complete
    dead = ordered.check(out_of_order)
    assert not dead.viable and "schema order" in dead.reason
    # Strict: the next key is the first one not given yet, no skipping ahead.
    assert not ordered.check('{"name":"propose","args":{"keys":"x","f').viable
    assert not ordered.check('{"name":"propose","args":{"file":"a.json","a').viable
    assert not ordered.check('{"name":"propose","args":{"file":"a.json","ask":"x"').viable
    assert ordered.check('{"name":"propose","args":{"file":"a.json","k').viable
    assert ordered.check('{"name":"propose","args":{"file":"a.json","keys":"x","a').viable
    # The loop builds its grammar from the option.
    tok = ProphetTokenizer(merges=[])
    assert AgentLoop(None, tok, reg, AgentConfig(ordered_keys=True)).grammar.ordered
    assert not AgentLoop(None, tok, reg, AgentConfig()).grammar.ordered
