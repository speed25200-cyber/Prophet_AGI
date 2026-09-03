"""Tests for the data pipeline.

The properties asserted here are the ones whose failure is silent: a mixture that does
not sum to one still trains, a contaminated corpus still shows a falling loss, and a
loader that restarts a source on resume still produces plausible batches. None of these
announce themselves, so each gets an assertion.
"""

from __future__ import annotations

import pytest

from prophet.data.decontaminate import Decontaminator, normalise, ngrams
from prophet.data.mixture import Mixture, MixtureError, Phase, Source
from prophet.data.recipes import prophet_v1_mixture
from prophet.data.streaming import (
    LoaderState,
    MixtureSampler,
    SequencePacker,
    StreamingLoader,
    sources_from_iterables,
)

B = 1e9


# --------------------------------------------------------------------------------------
# Mixture
# --------------------------------------------------------------------------------------


def _mix(**kw) -> Mixture:
    base = dict(
        name="m",
        total_tokens=10 * B,
        phases=[
            Phase("a", 0.6, [Source("s1", "x", "web", 0.7, available_tokens=100 * B),
                             Source("s2", "y", "code", 0.3, available_tokens=100 * B)]),
            Phase("b", 0.4, [Source("s3", "z", "math", 1.0, available_tokens=100 * B)]),
        ],
    )
    base.update(kw)
    return Mixture(**base)


def test_valid_mixture_passes():
    _mix().validate()


def test_phase_weights_must_sum_to_one():
    m = _mix(phases=[Phase("a", 0.6, [Source("s", "x", "web", 1.0)])])
    with pytest.raises(MixtureError, match="phase weights sum"):
        m.validate()


def test_source_weights_must_sum_to_one():
    m = _mix(phases=[Phase("a", 1.0, [Source("s1", "x", "web", 0.4),
                                      Source("s2", "y", "code", 0.4)])])
    with pytest.raises(MixtureError, match="source weights sum"):
        m.validate()


def test_excessive_repetition_is_rejected():
    """Repetition stops paying past roughly four epochs, and it is easy to do by accident
    when a small high-quality corpus is given a large share."""
    m = Mixture(
        name="m",
        total_tokens=100 * B,
        phases=[Phase("a", 1.0, [Source("small", "x", "math", 1.0, available_tokens=2 * B)])],
    )
    with pytest.raises(MixtureError, match="epochs"):
        m.validate()


def test_unverified_sources_are_reported_not_trusted():
    m = Mixture(
        name="m",
        total_tokens=10 * B,
        phases=[Phase("a", 1.0, [Source("unknown", "x", "web", 1.0)])],
    )
    m.validate()  # no size, so no epoch claim can be made
    assert m.unverified_sources() == ["a/unknown"]


def test_domain_shares_sum_to_one():
    shares = _mix().domain_shares()
    assert sum(shares.values()) == pytest.approx(1.0)


def test_rescaling_preserves_proportions():
    small = prophet_v1_mixture(10 * B)
    large = prophet_v1_mixture(300 * B)
    for domain, share in small.domain_shares().items():
        assert large.domain_shares()[domain] == pytest.approx(share, abs=1e-9)


def test_yaml_roundtrip(tmp_path):
    m = prophet_v1_mixture(40 * B)
    path = tmp_path / "mix.yaml"
    m.to_yaml(path)
    restored = Mixture.from_yaml(path)
    assert restored.domain_shares() == pytest.approx(m.domain_shares())
    restored.validate()


# --------------------------------------------------------------------------------------
# The shipped recipe
# --------------------------------------------------------------------------------------


def test_prophet_v1_recipe_is_valid_at_every_plausible_budget():
    for budget in (10 * B, 25 * B, 40 * B, 60 * B, 300 * B):
        prophet_v1_mixture(budget).validate()


def test_prophet_v1_matches_the_r06_domain_targets():
    """The mixture is the research finding; drift in it is a silent regression."""
    shares = prophet_v1_mixture(40 * B).domain_shares()
    expected = {
        "web": 0.442, "code": 0.144, "synthetic": 0.132, "math": 0.123,
        "reference": 0.072, "instruction": 0.048, "multilingual": 0.025,
        "long_context": 0.014,
    }
    for domain, target in expected.items():
        assert shares[domain] == pytest.approx(target, abs=0.005), domain


def test_maths_and_code_are_over_weighted_versus_published_small_model_recipes():
    """R06 decision D4, asserted so it cannot be diluted by a later edit."""
    shares = prophet_v1_mixture(40 * B).domain_shares()
    assert shares["math"] + shares["code"] > 0.25


def test_instruction_data_is_absent_from_the_stable_phase():
    """R06's recency play: instruction data is deliberately saved for late phases."""
    mixture = prophet_v1_mixture(40 * B)
    phase_a = next(p for p in mixture.phases if p.name.startswith("A"))
    assert phase_a.domain_shares().get("instruction", 0.0) == 0.0
    phase_c = next(p for p in mixture.phases if p.name.startswith("C"))
    assert phase_c.domain_shares()["instruction"] > 0.25


def test_every_source_declares_a_licence():
    for phase in prophet_v1_mixture().phases:
        for source in phase.sources:
            assert source.license != "unknown", f"{phase.name}/{source.name}"


def test_context_length_grows_across_phases():
    lens = [p.context_len for p in prophet_v1_mixture().phases]
    assert lens == sorted(lens) and lens[0] < lens[-1]


# --------------------------------------------------------------------------------------
# Decontamination
# --------------------------------------------------------------------------------------


def test_normalise_strips_case_punctuation_and_accents():
    assert normalise("Héllo,   WORLD!!") == "hello world"


def test_ngrams_counts_are_correct():
    assert list(ngrams("a b c d", 3)) == ["a b c", "b c d"]


def test_detects_a_reformatted_benchmark_item():
    """Contamination rarely arrives verbatim; it arrives reformatted."""
    d = Decontaminator(n=5, threshold=0.5)
    d.add_benchmark(
        "gsm8k",
        ["Natalia sold clips to 48 of her friends in April and then sold half as many in May"],
    )
    assert d.is_contaminated(
        "Q: NATALIA sold clips, to 48 of her friends in April!!! "
        "and then sold half as many in May. Answer?"
    )


def test_leaves_unrelated_documents_alone():
    d = Decontaminator(n=5, threshold=0.5)
    d.add_benchmark("gsm8k", ["Natalia sold clips to 48 of her friends in April"])
    assert not d.is_contaminated("Belgian autumn weather is unremarkable but persistent")


def test_short_examples_are_matched_exactly():
    """N-gram containment is meaningless for a three-word answer, so short items fall
    back to exact substring matching rather than being silently skipped."""
    d = Decontaminator(n=13, threshold=0.5)
    d.add_benchmark("tiny", ["the mitochondria"])
    assert d.is_contaminated("As we know, the mitochondria is the powerhouse of the cell")
    assert not d.is_contaminated("Unrelated text entirely")


def test_threshold_governs_partial_overlap():
    d_strict = Decontaminator(n=3, threshold=0.99)
    d_loose = Decontaminator(n=3, threshold=0.2)
    example = "alpha beta gamma delta epsilon zeta eta theta"
    for d in (d_strict, d_loose):
        d.add_benchmark("b", [example])
    partial = "alpha beta gamma delta plus a lot of unrelated material here"
    assert d_loose.is_contaminated(partial)
    assert not d_strict.is_contaminated(partial)


def test_report_lists_per_benchmark_counts():
    d = Decontaminator(n=4, threshold=0.5)
    d.add_benchmark("bench_a", ["one two three four five six"])
    d.add_benchmark("bench_b", ["seven eight nine ten eleven twelve"])
    d.is_contaminated("one two three four five six")
    d.is_contaminated("nothing to see here at all")
    report = d.report()
    assert "bench_a | 1" in report
    assert d.documents_seen == 2 and d.documents_rejected == 1


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------


def _docs(n_sources: int = 3, per_source: int = 400):
    weights = [0.5, 0.3, 0.2][:n_sources]
    return {
        f"s{i}": (w, [[i * 1000 + j] * 7 for j in range(per_source)])
        for i, w in enumerate(weights)
    }


def test_sampler_is_deterministic_and_stateless():
    a = MixtureSampler([0.5, 0.5], seed=7)
    b = MixtureSampler([0.5, 0.5], seed=7)
    # Evaluated out of order, which a stateful RNG could not survive.
    assert [a.source_for_step(s) for s in (100, 5, 99999, 0)] == [
        b.source_for_step(s) for s in (100, 5, 99999, 0)
    ]


def test_sampler_realises_the_requested_mixture():
    shares = MixtureSampler([0.5, 0.3, 0.2], seed=1).empirical_shares(200_000)
    for observed, target in zip(shares, (0.5, 0.3, 0.2)):
        assert observed == pytest.approx(target, abs=0.005)


def test_different_seeds_give_different_orders():
    a = MixtureSampler([0.5, 0.5], seed=1)
    b = MixtureSampler([0.5, 0.5], seed=2)
    assert [a.source_for_step(s) for s in range(50)] != [
        b.source_for_step(s) for s in range(50)
    ]


def test_packer_emits_exact_length_sequences():
    packer = SequencePacker(8, separator=0)
    for _ in range(10):
        packer.add([1, 2, 3, 4, 5])
    seqs = []
    while packer.ready():
        seqs.append(packer.pop())
    assert all(len(s) == 8 for s in seqs)


def test_resume_reproduces_the_uninterrupted_stream():
    """The property that makes a multi-week interrupted run coherent."""
    docs = _docs()
    uninterrupted = StreamingLoader(
        sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=0
    )
    expected = list(uninterrupted.batches(12))

    interrupted = StreamingLoader(
        sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=0
    )
    head = list(interrupted.batches(5))
    state = interrupted.state()

    resumed = StreamingLoader(
        sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=0
    )
    resumed.load_state(state)
    tail = list(resumed.batches(7))

    assert head + tail == expected


def test_carry_tokens_survive_a_checkpoint():
    """Dropping the partial-document buffer would shift the stream on every resume — a
    small, permanent corruption that no metric would reveal."""
    loader = StreamingLoader(
        sources_from_iterables(_docs()), seq_len=10, batch_size=1, seed=3, separator=0
    )
    list(loader.batches(3))
    state = loader.state()
    assert state.carry, "expected leftover tokens with a non-divisible sequence length"

    restored = LoaderState.from_dict(state.to_dict())
    assert restored.carry == state.carry
    assert restored.cursors == state.cursors


def test_checkpoint_state_is_small():
    loader = StreamingLoader(
        sources_from_iterables(_docs()), seq_len=64, batch_size=1, seed=3
    )
    list(loader.batches(50))
    assert len(str(loader.state().to_dict())) < 4096


def test_unknown_source_in_checkpoint_is_rejected():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    with pytest.raises(KeyError, match="unknown source"):
        loader.load_state(LoaderState(step=0, cursors={"nonexistent": 3}))


def test_exhausted_source_wraps_instead_of_stalling():
    docs = {"only": (1.0, [[1, 2, 3]] * 4)}
    loader = StreamingLoader(sources_from_iterables(docs), seq_len=6, seed=0)
    assert len(list(loader.batches(5))) == 5
