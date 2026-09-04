"""Tests for the data pipeline.

The properties asserted here are the ones whose failure is silent: a mixture that does
not sum to one still trains, a contaminated corpus still shows a falling loss, and a
loader that restarts a source on resume still produces plausible batches. None of these
announce themselves, so each gets an assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.verify_datasets as verify_datasets
from prophet.data.decontaminate import Decontaminator, ngrams, normalise
from prophet.data.mixture import Mixture, MixtureError, Phase, Source
from prophet.data.recipes import prophet_v1_mixture
from prophet.data.streaming import (
    LOADER_STATE_FORMAT_VERSION,
    IterableSource,
    LoaderState,
    MixtureSampler,
    SequencePacker,
    StreamingLoader,
    sources_from_iterables,
)

B = 1e9
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# Mixture
# --------------------------------------------------------------------------------------


def _mix(**kw) -> Mixture:
    base = dict(
        name="m",
        total_tokens=10 * B,
        phases=[
            Phase("a", 0.6, [
                Source("s1", "x", "web", 0.7, available_tokens=100 * B, license="Apache-2.0"),
                Source("s2", "y", "code", 0.3, available_tokens=100 * B, license="Apache-2.0"),
            ]),
            Phase("b", 0.4, [
                Source("s3", "z", "math", 1.0, available_tokens=100 * B, license="Apache-2.0"),
            ]),
        ],
    )
    base.update(kw)
    return Mixture(**base)


def test_valid_mixture_passes():
    _mix().validate()


def test_phase_weights_must_sum_to_one():
    m = _mix(phases=[Phase("a", 0.6, [Source("s", "x", "web", 1.0, license="MIT")])])
    with pytest.raises(MixtureError, match="phase weights sum"):
        m.validate()


def test_source_weights_must_sum_to_one():
    m = _mix(phases=[Phase("a", 1.0, [Source("s1", "x", "web", 0.4, license="MIT"),
                                      Source("s2", "y", "code", 0.4, license="MIT")])])
    with pytest.raises(MixtureError, match="source weights sum"):
        m.validate()


def test_excessive_repetition_is_rejected():
    """Repetition stops paying past roughly four epochs, and it is easy to do by accident
    when a small high-quality corpus is given a large share."""
    m = Mixture(
        name="m",
        total_tokens=100 * B,
        phases=[Phase("a", 1.0, [Source("small", "x", "math", 1.0, available_tokens=2 * B,
                                        license="MIT")])],
    )
    with pytest.raises(MixtureError, match="epochs"):
        m.validate()


def test_epoch_limit_is_cumulative_across_phase_aliases():
    m = Mixture(
        name="m",
        total_tokens=8,
        phases=[
            Phase("a", 0.5, [Source("first", "same/corpus", "math", 1.0,
                                    available_tokens=1, license="MIT")]),
            Phase("b", 0.5, [Source("alias", "same/corpus", "math", 1.0,
                                    available_tokens=1, license="MIT")]),
        ],
    )

    with pytest.raises(MixtureError, match="cumulative planned 8.00 epochs"):
        m.validate()


def test_unverified_sources_are_reported_not_trusted():
    m = Mixture(
        name="m",
        total_tokens=10 * B,
        phases=[Phase("a", 1.0, [Source("unknown", "x", "web", 1.0, license="Apache-2.0")])],
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
    m = _mix()
    path = tmp_path / "mix.yaml"
    m.to_yaml(path)
    restored = Mixture.from_yaml(path)
    assert restored.domain_shares() == pytest.approx(m.domain_shares())
    restored.validate()


# --------------------------------------------------------------------------------------
# The shipped recipe
# --------------------------------------------------------------------------------------


def test_prophet_v1_recipe_is_blocked_until_mixed_licences_are_reviewed():
    for budget in (10 * B, 25 * B, 40 * B, 60 * B, 300 * B):
        with pytest.raises(MixtureError, match="requires REVIEW"):
            prophet_v1_mixture(budget).validate()


def test_documentation_override_allows_only_explicit_pending_reviews():
    prophet_v1_mixture().validate(allow_pending_license_review=True)

    for licence, message in (
        ("All Rights Reserved", "restrictive licence"),
        ("Some new permissive licence", "not on the release allowlist"),
    ):
        unapproved = Mixture(
            name="m", total_tokens=B,
            phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0,
                                            license=licence)])],
        )
        with pytest.raises(MixtureError, match=message):
            unapproved.validate(allow_pending_license_review=True)


def test_shipped_yaml_exactly_matches_the_recipe():
    from_yaml = Mixture.from_yaml(ROOT / "configs" / "data_mixture_v1.yaml")
    assert from_yaml.to_dict() == prophet_v1_mixture().to_dict()


def test_dataset_verifier_treats_licence_mismatch_as_a_problem(monkeypatch, capsys):
    mixture = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "org/data", "web", 1.0, license="MIT")])],
    )
    monkeypatch.setattr(verify_datasets, "hub_reachable", lambda: True)
    monkeypatch.setattr(verify_datasets, "prophet_v1_mixture", lambda _tokens: mixture)
    monkeypatch.setattr(
        verify_datasets,
        "fetch",
        lambda hf_id: verify_datasets.Check(
            source="", hf_id=hf_id, exists=True, hub_license="Apache-2.0"
        ),
    )
    monkeypatch.setattr("sys.argv", ["verify_datasets.py"])

    assert verify_datasets.main() == 1
    assert "does not match" in capsys.readouterr().out


def test_dataset_verifier_reuses_fail_closed_hub_licence_policy():
    check = verify_datasets.Check(
        source="a/s",
        hf_id="org/data",
        exists=True,
        declared_license="CC BY NC 4.0",
        hub_license="CC‑BY‑NC‑4.0",
    )
    assert check.license_matches
    assert "restrictive licence" in check.hub_license_problem


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


def test_blocked_licence_is_rejected():
    """R10 read the Gemma terms: a model trained on Gemma-generated synthetic data is a
    Model Derivative. One row would bind the whole project, so this is an error, not a
    warning -- discovering it after training cannot be fixed except by retraining."""
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "synthetic", 1.0,
                                        license="Gemma Terms of Use")])],
    )
    with pytest.raises(MixtureError, match="Model Derivative"):
        m.validate()


def test_non_commercial_licence_is_rejected():
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "instruction", 1.0,
                                        license="CC-BY-NC-4.0")])],
    )
    with pytest.raises(MixtureError, match="restrictive licence"):
        m.validate()


@pytest.mark.parametrize(
    "licence",
    [
        "CC BY NC 4.0",
        "CC_BY_NC_4.0",
        "CC‑BY‑NC‑4.0",
        "NonCommercial",
        "ALL RIGHTS RESERVED",
        "Proprietary",
    ],
)
def test_restrictive_licence_spellings_are_rejected(licence):
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0, license=licence)])],
    )
    with pytest.raises(MixtureError, match="restrictive licence"):
        m.validate()


@pytest.mark.parametrize(
    "licence",
    ["Apache-2.0", "MIT", "CC-BY-4.0", "ODC-By-1.0", "BSD-3-Clause"],
)
def test_reviewed_commercial_licences_are_allowed(licence):
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0, license=licence)])],
    )
    m.validate()


def test_unreviewed_licence_is_rejected_fail_closed():
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0,
                                        license="Some new permissive licence")])],
    )
    with pytest.raises(MixtureError, match="not on the release allowlist"):
        m.validate()


def test_missing_licence_is_rejected():
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0)])],
    )
    with pytest.raises(MixtureError, match="licence not established"):
        m.validate()


def test_mixed_licences_are_explicitly_blocked_for_review():
    m = Mixture(
        name="m", total_tokens=B,
        phases=[Phase("a", 1.0, [Source("s", "x", "web", 1.0,
                                        license="REVIEW: mixed (per-subset)")])],
    )
    with pytest.raises(MixtureError, match="requires REVIEW"):
        m.validate()
    assert any("cannot train yet" in warning for warning in m.license_warnings())


def test_shipped_recipe_identifies_every_pending_licence_review():
    warnings = prophet_v1_mixture().license_warnings()
    assert len(warnings) == 3
    assert all("REVIEW" in warning for warning in warnings)


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


def test_zero_threshold_is_rejected_instead_of_matching_every_document():
    with pytest.raises(ValueError, match="above 0"):
        Decontaminator(n=3, threshold=0.0)


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
    for observed, target in zip(shares, (0.5, 0.3, 0.2), strict=True):
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
    assert restored.format_version == LOADER_STATE_FORMAT_VERSION
    assert restored.manifest_fingerprint == state.manifest_fingerprint


def test_checkpoint_state_is_small():
    loader = StreamingLoader(
        sources_from_iterables(_docs()), seq_len=64, batch_size=1, seed=3
    )
    list(loader.batches(50))
    assert len(str(loader.state().to_dict())) < 4096


def test_unknown_source_in_checkpoint_is_rejected():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    state = loader.state()
    state.cursors["nonexistent"] = 3
    with pytest.raises(KeyError, match="unknown source"):
        loader.load_state(state)


def test_missing_source_cursor_in_checkpoint_is_rejected():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    state = loader.state()
    state.cursors.pop("s1")

    with pytest.raises(KeyError, match="missing source cursor"):
        loader.load_state(state)


def test_out_of_range_source_cursor_is_rejected_without_mutation():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    list(loader.batches(2))
    before = loader.state().to_dict()
    invalid = loader.state()
    invalid.cursors["s0"] = len(loader.sources[0].documents)

    with pytest.raises(ValueError, match="exceeds"):
        loader.load_state(invalid)
    assert loader.state().to_dict() == before


def test_validate_state_is_non_mutating_and_accepts_sparse_carry():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    state = loader.state()
    before = loader.state().to_dict()

    validated = loader.validate_state(state.to_dict())

    assert validated.carry == {}
    assert loader.state().to_dict() == before


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(
            lambda docs: StreamingLoader(
                sources_from_iterables(docs), seq_len=16, batch_size=2, seed=43, separator=0
            ),
            id="seed",
        ),
        pytest.param(
            lambda docs: StreamingLoader(
                list(reversed(sources_from_iterables(docs))),
                seq_len=16,
                batch_size=2,
                seed=42,
                separator=0,
            ),
            id="source-order",
        ),
        pytest.param(
            lambda docs: StreamingLoader(
                [
                    IterableSource(source.name, source.weight + (0.1 if index == 0 else 0.0),
                                   source.documents)
                    for index, source in enumerate(sources_from_iterables(docs))
                ],
                seq_len=16,
                batch_size=2,
                seed=42,
                separator=0,
            ),
            id="source-weight",
        ),
        pytest.param(
            lambda docs: StreamingLoader(
                sources_from_iterables(docs), seq_len=8, batch_size=2, seed=42, separator=0
            ),
            id="sequence-length",
        ),
        pytest.param(
            lambda docs: StreamingLoader(
                sources_from_iterables(docs), seq_len=16, batch_size=1, seed=42, separator=0
            ),
            id="batch-size",
        ),
        pytest.param(
            lambda docs: StreamingLoader(
                sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=99
            ),
            id="separator",
        ),
    ],
)
def test_resume_rejects_changed_stream_parameters(replacement):
    docs = _docs()
    original = StreamingLoader(
        sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=0
    )
    list(original.batches(3))

    with pytest.raises(ValueError, match="manifest fingerprint"):
        replacement(docs).load_state(original.state())


def test_resume_rejects_changed_source_contents():
    docs = _docs()
    original = StreamingLoader(
        sources_from_iterables(docs), seq_len=16, batch_size=2, seed=42, separator=0
    )
    state = original.state()
    changed = _docs()
    changed["s0"][1][0][0] += 1
    resumed = StreamingLoader(
        sources_from_iterables(changed), seq_len=16, batch_size=2, seed=42, separator=0
    )

    with pytest.raises(ValueError, match="manifest fingerprint"):
        resumed.load_state(state)


def test_resume_rejects_an_unversioned_state_mapping():
    loader = StreamingLoader(sources_from_iterables(_docs()), seq_len=8, seed=0)
    legacy = loader.state().to_dict()
    legacy.pop("format_version")

    with pytest.raises(ValueError, match="format_version"):
        loader.load_state(legacy)


def test_exhausted_source_wraps_instead_of_stalling():
    docs = {"only": (1.0, [[1, 2, 3]] * 4)}
    loader = StreamingLoader(sources_from_iterables(docs), seq_len=6, seed=0)
    assert len(list(loader.batches(5))) == 5
