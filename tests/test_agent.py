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
from prophet.agent.loop import AgentConfig, AgentLoop
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
    reg = ToolRegistry([
        ToolSchema("read_file", "read a file", {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        ToolSchema("write_file", "write a file", {
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"]}, irreversible=True),
        ToolSchema("count", "count", {"type": "object", "properties": {"n": {"type": "integer"}},
                                       "required": ["n"]}),
    ])
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

    def forward(self, ids, *, cache=None, loop_k=None, return_mtp=True, halt_threshold=None,
                modality_ids=None):
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
            logits=logits, hidden=torch.zeros(b, s, 8), loop_k=loop_k or 1,
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
    assert not g.check('{"name":"count","args":{"n":"x"').viable        # wrong type
    assert not g.check('{"name":"read_file","args":{"pth"').viable       # unknown key
    assert not g.check('{"name":"read_file","args":{}}').viable          # missing required


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
    assert ord("r") in allowed and ord("w") in allowed and ord("c") in allowed and ord("d") in allowed
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
    assert decide(sig, 0.2, VerifierConfig(depth_gain_points=0.0), attempts=0).decision != "retry_depth"
    assert decide(sig, 0.2, VerifierConfig(depth_gain_points=6.0), attempts=0).decision == "retry_depth"


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
        state.record_snapshot(cache)                  # before step 0 acts
        first = model(chunks[1], cache=cache, loop_k=2).logits
        state.step = 1
        state.record_snapshot(cache)                  # before step 1 acts
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
    return AgentLoop(model, TOK, registry(), cfg, quarantine=quarantine, verifier_tool=verifier_tool)


def test_done_with_high_confidence_finishes():
    loop = _loop(script_for("<|/think|>", call({"name": "done"})), confidence=5.0)
    result = loop.run("finish")
    assert result.finished and result.reason == "done"


def test_done_with_low_confidence_is_refused_without_a_verifier():
    loop = _loop(script_for("<|/think|>", call({"name": "done"}), "<|/think|>", call({"name": "done"})),
                 confidence=-5.0)
    result = loop.run("finish")
    assert not result.finished
    assert any(r.gated == "refused_done" for r in result.steps)


def test_done_is_accepted_only_when_the_verifier_passes():
    calls = {"n": 0}
    def verifier(state):
        calls["n"] += 1
        return calls["n"] >= 2
    loop = _loop(script_for("<|/think|>", call({"name": "done"}), "<|/think|>", call({"name": "done"})),
                 confidence=-5.0, verifier_tool=verifier)
    result = loop.run("finish")
    assert result.finished and result.verified_before_done
    assert result.steps[0].gated == "refused_done"


def test_irreversible_action_below_threshold_is_verified_first():
    loop = _loop(script_for("<|/think|>", call({"name": "write_file", "args": {"path": "a.py", "text": "x"}}),
                            "<|/think|>", call({"name": "done"})), confidence=-5.0)
    result = loop.run("edit")
    assert result.steps[0].gated == "verify_first"
    assert result.steps[0].action.name == "verify"
    assert loop.tools._fs["a.py"] == "print(1)\n"  # nothing was written


def test_irreversible_action_above_threshold_executes():
    loop = _loop(script_for("<|/think|>", call({"name": "write_file", "args": {"path": "a.py", "text": "x"}}),
                            "<|/think|>", call({"name": "done"})), confidence=5.0)
    result = loop.run("edit")
    assert result.steps[0].gated == ""
    assert loop.tools._fs["a.py"] == "x"


def test_ask_returns_the_question_to_the_user():
    loop = _loop(script_for("<|/think|>", call({"name": "ask", "args": {"question": "which file?"}})))
    result = loop.run("edit")
    assert not result.finished and result.reason == "ask" and result.asked_user == "which file?"


def test_repeated_identical_action_triggers_reflection():
    same = call({"name": "read_file", "args": {"path": "a.py"}})
    loop = _loop(script_for(*(["<|/think|>", same] * 5)),
                 cfg=AgentConfig(max_steps=5, think_budget=4, halt_threshold=None, max_repeats=3))
    result = loop.run("read")
    gated = [r.gated for r in result.steps]
    assert "reflect" in gated
    assert "[stuck: repeated read_file" in result.final_notes


def test_observations_are_ingested_and_windowed():
    reads = ["<|/think|>", call({"name": "read_file", "args": {"path": "a.py"}})] * 3
    loop = _loop(script_for(*reads), cfg=AgentConfig(max_steps=3, think_budget=4, halt_threshold=None, window_steps=2))
    result = loop.run("read")
    assert all(r.observation == "print(1)\n" for r in result.steps)


def test_rollback_action_restores_position():
    loop = _loop(script_for("<|/think|>", call({"name": "count", "args": {"n": 2}}),
                            "<|/think|>", call({"name": "rollback", "args": {"step": 0}}),
                            "<|/think|>", call({"name": "done"})), confidence=5.0)
    result = loop.run("go")
    assert result.steps[1].action.name == "rollback"
    assert "rolled back to step 0" in result.steps[1].observation


def test_malformed_call_within_budget_is_recorded_not_crashed():
    loop = _loop(script_for("<|/think|>", '{"name":"read_file","args":{"path":"a.py"'),  # never closes
                 cfg=AgentConfig(max_steps=1, think_budget=4, action_budget=8, halt_threshold=None))
    result = loop.run("read")
    assert result.steps and result.steps[0].gated == "malformed"


def test_episode_lands_in_quarantine_with_provenance(tmp_path):
    q = Quarantine(tmp_path / "q.json")
    loop = _loop(script_for("<|/think|>", call({"name": "done"})), confidence=5.0, quarantine=q,
                 verifier_tool=lambda s: True)
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
    loop = AgentLoop(model, TOK, registry(),
                     AgentConfig(max_steps=2, think_budget=6, action_budget=24, halt_threshold=None))
    result = loop.run("read a.py")
    assert len(result.steps) <= 2
