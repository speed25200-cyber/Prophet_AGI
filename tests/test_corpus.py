"""The real-corpus layer: local files, decontamination in the path, epoch caps at draw
time, and phased loaders -- all resumable exactly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prophet.data.corpus import (
    EpochCapExceeded,
    HubSource,
    LocalTextSource,
    PhasedLoader,
    TokenisedSource,
    build_loader,
    build_sources,
    phase_steps,
    row_passes,
)
from prophet.data.decontaminate import Decontaminator
from prophet.data.mixture import Mixture, Phase, Source
from prophet.data.streaming import StreamingLoader, sources_from_iterables
from prophet.data.tokenizer import ProphetTokenizer

TOK = ProphetTokenizer(merges=[])

WEB = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "the mitochondria is the powerhouse of the cell, as every student knows",
    "rain in spain falls mainly on the plain",
    "to be or not to be that is the question",
    "all that glitters is not gold",
]
CODE = ["def f():\n    return 1", "x = 2", "print(x)", "import os"]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "web.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "score": i}) for i, t in enumerate(WEB)) + "\n"
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "part0.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in CODE[:2]) + "\n\n"
    )
    (tmp_path / "code" / "part1.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in CODE[2:]) + "\n"
    )
    return tmp_path


# --------------------------------------------------------------------------------------
# Local sources
# --------------------------------------------------------------------------------------


def test_local_source_reads_in_order_and_seeks(corpus):
    web = LocalTextSource.from_root(corpus, "web", 1.0)
    assert web.n_documents() == 6
    assert list(web.open(0)) == WEB
    assert list(web.open(4)) == WEB[4:]
    assert list(web.open(6)) == []


def test_local_source_spans_files_and_skips_blank_lines(corpus):
    code = LocalTextSource.from_root(corpus, "code", 1.0)
    assert code.n_documents() == 4
    assert list(code.open(0)) == CODE
    assert list(code.open(1)) == CODE[1:]
    assert list(code.open(2)) == CODE[2:]  # first document of the second file
    assert list(code.open(3)) == CODE[3:]


def test_line_index_is_cached_beside_the_file_and_reused(corpus):
    web = LocalTextSource.from_root(corpus, "web", 1.0)
    web.n_documents()
    cache = corpus / "web.jsonl.lines"
    assert cache.exists()
    stamp = cache.stat().st_mtime_ns
    again = LocalTextSource.from_root(corpus, "web", 1.0)
    assert list(again.open(3)) == WEB[3:]
    assert cache.stat().st_mtime_ns == stamp


def test_missing_source_names_the_file_it_wanted(corpus):
    with pytest.raises(FileNotFoundError, match="maths.jsonl"):
        LocalTextSource.from_root(corpus, "maths", 1.0)


def test_row_filters():
    assert row_passes({"score": 4}, {"score_min": 4})
    assert not row_passes({"score": 3}, {"score_min": 4})
    assert not row_passes({}, {"score_min": 4})
    assert row_passes({"lang": "en"}, {"lang_in": ["en", "fr"]})
    assert row_passes({"lang": "en", "n": 2}, {"lang": "en", "n_max": 2})
    assert not row_passes({"lang": "de"}, {"lang": "en"})


def test_hub_source_reports_unknown_size():
    src = HubSource("web", 1.0, "org/dataset")
    assert src.n_documents() is None


# --------------------------------------------------------------------------------------
# Tokenised sources: rejection and epoch cap
# --------------------------------------------------------------------------------------


def _decon() -> Decontaminator:
    d = Decontaminator(n=13, threshold=0.5)
    d.add_benchmark("bio", ["the mitochondria is the powerhouse of the cell"])
    return d


def test_rejected_documents_yield_empty_and_keep_the_cursor_counting(corpus):
    src = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK, decontaminator=_decon())
    docs = [src.next_document() for _ in range(6)]
    assert docs[2] == [] and all(len(d) for i, d in enumerate(docs) if i != 2)
    assert src.cursor == 6 and src.stats.rejected == 1
    assert docs[0][-1] == TOK.eos_id


def test_loader_drops_empties_without_a_stray_separator(corpus):
    src = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK, decontaminator=_decon(), max_epochs=None)
    loader = StreamingLoader([src], seq_len=16, separator=TOK.special_id("<|pad|>"))
    stream: list[int] = []
    for batch in loader.batches(6):
        stream += batch[0]
    clean = [t for i, t in enumerate(WEB) if i != 2]
    expected: list[int] = []
    for i, t in enumerate(clean):
        if i:
            expected.append(TOK.special_id("<|pad|>"))
        expected += TOK.encode(t, add_eos=True)
    assert len(stream) == 96 and stream == expected[: len(stream)]


def test_resume_is_exact_with_rejections_in_the_stream(corpus):
    def make():
        src = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK,
                              decontaminator=_decon(), max_epochs=None)
        return StreamingLoader([src], seq_len=8)

    a = make()
    list(a.batches(3))
    saved = a.state().to_dict()
    rest = list(a.batches(4))
    b = make()
    b.restore(saved)
    assert list(b.batches(4)) == rest


def test_epoch_cap_raises_at_the_limit_with_a_monotone_cursor(corpus):
    src = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK, max_epochs=2)
    for _ in range(12):
        src.next_document()
    assert src.cursor == 12 and src.epochs() == 2.0
    with pytest.raises(EpochCapExceeded, match="2.00 epochs"):
        src.next_document()


def test_uncapped_source_wraps_and_the_cursor_survives_a_checkpoint(corpus):
    src = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK, max_epochs=None)
    first = [src.next_document() for _ in range(7)]
    assert first[6] == first[0] and src.cursor == 7
    loader = StreamingLoader([src], seq_len=8)
    saved = loader.state().to_dict()
    fresh = TokenisedSource(LocalTextSource.from_root(corpus, "web", 1.0), TOK, max_epochs=None)
    other = StreamingLoader([fresh], seq_len=8)
    other.restore(saved)
    assert fresh.cursor == 7 and fresh.next_document() == first[1]


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


def _phase_loader(seed: int = 0) -> PhasedLoader:
    a = StreamingLoader(sources_from_iterables({"a": (1.0, [[1, 1, 1, 1]] * 50)}), seq_len=4, seed=seed)
    b = StreamingLoader(sources_from_iterables({"b": (1.0, [[2, 2, 2, 2]] * 50)}), seq_len=4, seed=seed)
    return PhasedLoader([("A", 3, a), ("B", 2, b)])


def test_phased_loader_switches_at_the_boundary_and_runs_the_last_phase_open_ended():
    loader = _phase_loader()
    batches = [b[0] for b in loader.batches(7)]
    assert batches[:3] == [[1, 1, 1, 1]] * 3
    assert batches[3:] == [[2, 2, 2, 2]] * 4
    assert loader.phase == 1 and loader.total_steps() == 5
    assert "open-ended" in loader.describe()


def test_phased_loader_resumes_inside_a_phase():
    a = _phase_loader()
    list(a.batches(4))  # one batch into phase B
    saved = a.state().to_dict()
    rest = list(a.batches(3))
    b = _phase_loader()
    b.restore(saved)
    assert b.phase == 1 and b.step_in_phase == 1
    assert list(b.batches(3)) == rest


def test_phased_loader_rejects_a_checkpoint_with_the_wrong_shape():
    with pytest.raises(ValueError, match="phases"):
        _phase_loader().restore({"phase": 0, "step_in_phase": 0, "loaders": [{}]})


def _mixture() -> Mixture:
    return Mixture(
        name="t", total_tokens=4000.0,
        phases=[
            Phase("A", 0.75, [Source("web", "x/web", "web", 0.5, license="apache-2.0"),
                              Source("code", "x/code", "code", 0.5, license="mit")]),
            Phase("B", 0.25, [Source("web", "x/web", "web", 1.0, license="apache-2.0")]),
        ],
    )


def test_phase_steps_follow_the_token_shares():
    assert phase_steps(_mixture(), batch_tokens=100) == [30, 10]
    assert phase_steps(_mixture(), batch_tokens=10_000) == [1, 1]  # never zero


def test_build_loader_from_local_corpora(corpus):
    loader = build_loader(_mixture(), tokenizer=TOK, seq_len=8, batch_size=2, local_root=corpus, max_epochs=None)
    assert loader.steps == [188, 62]
    batch = next(loader.batches(1))
    assert len(batch) == 2 and all(len(row) == 8 for row in batch)
    assert loader.seq_len == 8
    assert "phase A" in loader.describe()


def test_build_loader_names_the_missing_source_when_the_hub_is_off(corpus):
    (corpus / "web.jsonl").unlink()
    (corpus / "web.jsonl.lines").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="'web'.*--hub"):
        build_loader(_mixture(), tokenizer=TOK, seq_len=8, batch_size=1, local_root=corpus)


def test_build_sources_falls_back_to_the_hub_when_allowed(corpus):
    (corpus / "web.jsonl").unlink()
    (corpus / "web.jsonl.lines").unlink(missing_ok=True)
    sources = build_sources(_mixture().phases[0], tokenizer=TOK, local_root=corpus, allow_hub=True)
    assert isinstance(sources[0].source, HubSource) and sources[0].source.hf_id == "x/web"
    assert isinstance(sources[1].source, LocalTextSource)


def test_overrides_win_over_files(corpus):
    override = LocalTextSource.from_root(corpus, "code", 0.5)
    sources = build_sources(_mixture().phases[0], tokenizer=TOK, local_root=corpus, overrides={"web": override})
    assert sources[0].source is override


def test_extra_sources_join_only_the_named_phases(corpus):
    extra = sources_from_iterables({"episodes": (0.5, [[7, 7, 7, 7]] * 20)})
    loader = build_loader(_mixture(), tokenizer=TOK, seq_len=8, batch_size=1, local_root=corpus,
                          max_epochs=None, extra_sources=extra)
    assert [s.name for s in loader.loaders[0].sources] == ["web", "code"]
    assert [s.name for s in loader.loaders[1].sources] == ["web", "episodes"]
    with pytest.raises(ValueError, match="unknown phases"):
        build_loader(_mixture(), tokenizer=TOK, seq_len=8, batch_size=1, local_root=corpus,
                     max_epochs=None, extra_sources=extra, extra_phases=["Z"])
