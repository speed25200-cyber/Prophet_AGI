"""The corpus builder on a synthetic stream: filters, exclusions, split, sharding, manifest."""

import hashlib
import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from prophet.data.decontaminate import normalise
from scripts.prepare_loop_core_corpus import (
    FILTERS,
    ShardWriter,
    build,
    estimate_tokens,
    passes_filters,
)
from scripts.prepare_pilot import content_hash, gram_hashes, split_for

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def paragraph(rng: random.Random, words: int) -> str:
    """Random letter words: normalisation keeps them distinct, unlike digits."""
    return " ".join("".join(rng.choices(LETTERS, k=7)) for _ in range(words))


def row(text: str, *, url: str, score: int = 4, ident: str | None = None) -> dict:
    return {"text": text, "int_score": score, "url": url, "id": ident or url}


def make_prior(root: Path, docs: list[dict]) -> Path:
    prior = root / "prior"
    path = prior / "train" / "fineweb-edu" / "part-00000.jsonl"
    path.parent.mkdir(parents=True)
    data = b"".join(
        (json.dumps({**d, "content_sha256": content_hash(normalise(d["text"]))}) + "\n").encode()
        for d in docs
    )
    path.write_bytes(data)
    manifest = {
        "complete": True,
        "artifacts": {
            "train/fineweb-edu/part-00000.jsonl": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "documents": len(docs),
            }
        },
    }
    (prior / "manifest.json").write_text(json.dumps(manifest))
    return prior


def make_args(out: Path, **overrides) -> Namespace:
    base = dict(
        out=out,
        revision="rev",
        target_train_bytes=20_000,
        validation_docs=3,
        validation_per_mille=300,
        max_scan_rows=10_000,
        shard_bytes=4_000,
        seed=5,
        prior=[],
        benchmark_cache=None,
        composition_seed=9,
        composition_per_level=4,
        composition_test_per_level=2,
        tokenizer=None,
        token_sample_docs=2,
    )
    base.update(overrides)
    return Namespace(**base)


def test_filters_follow_the_pilot_policy():
    long = "w " * 300
    assert passes_filters(row(long, url="u", score=2)) == "quality"
    assert passes_filters(row("short", url="u")) == "length"
    assert passes_filters(row("x" * (FILTERS["max_bytes"] + 1), url="u")) == "length"
    assert passes_filters(row(long, url="u")) is None
    with pytest.raises(ValueError, match="schema"):
        passes_filters({"text": "x"})


def test_shard_writer_bounds_shards_and_hashes_each(tmp_path):
    writer = ShardWriter(tmp_path / "s", shard_bytes=120)
    for i in range(10):
        writer.write({"text": "t" * 40, "i": i})
    writer.close()
    assert len(writer.artifacts) > 1
    for name, meta in writer.artifacts.items():
        data = (tmp_path / "s" / name).read_bytes()
        assert meta["bytes"] == len(data) <= 120
        assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert sum(m["documents"] for m in writer.artifacts.values()) == 10


def test_build_applies_exclusions_split_protection_and_writes_a_complete_manifest(tmp_path):
    rng = random.Random(0)
    base = [row(paragraph(rng, 120), url=f"https://site{i}.org/p/{i}") for i in range(40)]
    seed, per_mille = 5, 300
    validation_like = [
        r
        for r in base
        if split_for(r["url"], content_hash(normalise(r["text"])), seed, per_mille) == "validation"
    ]
    assert len(validation_like) >= 4, "the synthetic split must produce validation documents"
    leak_source = validation_like[0]["text"]
    span = " ".join(leak_source.split()[10:30])
    leaked_text = paragraph(rng, 40) + " " + span + " " + paragraph(rng, 40)
    leaked = next(
        candidate
        for candidate in (row(leaked_text, url=f"https://leak{i}.org/x") for i in range(100))
        if split_for(candidate["url"], content_hash(normalise(leaked_text)), seed, per_mille)
        == "train"
    )

    prior_doc = row(paragraph(rng, 120), url="https://prior.org/a")
    prior = make_prior(tmp_path, [prior_doc])
    same_content = row(prior_doc["text"], url="https://elsewhere.org/b")
    same_url = row(paragraph(rng, 120), url="https://prior.org/a")

    bench_text = paragraph(rng, 30)
    cache = tmp_path / "bench"
    cache.mkdir()
    (cache / "arc.jsonl").write_text(json.dumps({"text": bench_text}) + "\n")
    contaminated = row(
        paragraph(rng, 50) + " " + bench_text + " " + paragraph(rng, 50), url="https://c.org/z"
    )

    duplicate = row(base[3]["text"], url="https://dup.org/q")
    low_quality = row(paragraph(rng, 120), url="https://lq.org/q", score=1)
    short = row("too short", url="https://short.org/q")
    # Specials sit in the middle so the byte target cannot stop the scan before them;
    # every validation-split document is accepted so the leaked span is protected.
    stream = (
        base[:5]
        + [leaked, same_content, same_url, contaminated, duplicate, low_quality, short]
        + base[5:]
    )

    out = tmp_path / "corpus"
    args = make_args(
        out,
        prior=[prior],
        benchmark_cache=cache,
        seed=seed,
        validation_per_mille=per_mille,
        validation_docs=len(validation_like),
        target_train_bytes=10**7,
    )
    manifest = build(iter(stream), args, source_sha="abc")

    assert manifest["complete"] and (out / "manifest.json").exists()
    assert not out.with_name(out.name + ".building").exists()
    stats = manifest["stats"]
    assert stats["excluded_prior_exact_content"] == 1
    assert stats["excluded_prior_url"] == 1
    assert stats["excluded_protected_13_word_span"] == 1
    assert stats["rejected_duplicate"] == 1
    assert stats["rejected_quality"] == 1 and stats["rejected_length"] == 1
    assert stats["train_rejected_validation_span"] >= 1
    assert stats["validation_docs"] == len(validation_like)
    assert manifest["exclusions"]["prior"][0]["documents"] == 1
    assert manifest["exclusions"]["benchmarks"]["arc.jsonl"]["rows"] == 1

    # Every artifact hashes as recorded; shards respect the byte bound.
    train_docs, validation_docs = [], []
    for name, meta in manifest["artifacts"].items():
        data = (out / name).read_bytes()
        assert meta["sha256"] == hashlib.sha256(data).hexdigest() and meta["bytes"] == len(data)
        assert len(data) <= args.shard_bytes
        rows = [json.loads(line) for line in data.decode().splitlines()]
        assert len(rows) == meta["documents"]
        (train_docs if name.startswith("train/") else validation_docs).extend(rows)
    assert len(validation_docs) == len(validation_like) and len(train_docs) == stats["train_docs"]
    assert len(manifest["artifacts"]) > 2, "small shard bound must produce several shards"

    # Independent check: no train document shares a 13-word span with validation, and
    # no train document is the leaked one, the prior one or the contaminated one.
    protected = set()
    for doc in validation_docs:
        protected |= gram_hashes(normalise(doc["text"]))
    for doc in train_docs:
        assert not protected & gram_hashes(normalise(doc["text"]))
        assert doc["url"] not in {leaked["url"], "https://prior.org/a", "https://c.org/z"}
        assert doc["content_sha256"] == content_hash(normalise(doc["text"]))
    train_urls = {d["url"] for d in train_docs}
    assert not train_urls & {d["url"] for d in validation_docs}

    comp = manifest["composition"]
    train_comp = out / comp["train"]["path"]
    test_comp = out / comp["test"]["path"]
    assert comp["train"]["sha256"] == hashlib.sha256(train_comp.read_bytes()).hexdigest()
    first = json.loads(train_comp.read_text().splitlines()[0])
    assert set(first) == {"text"}
    test_rows = [json.loads(line) for line in test_comp.read_text().splitlines()]
    assert all({"prompt", "answer", "choices", "hops", "kind"} <= set(r) for r in test_rows)
    assert comp["train"]["documents"] == 4 * 9 and comp["test"]["documents"] == 2 * 9


def test_build_refuses_an_existing_corpus_or_partial_build(tmp_path):
    out = tmp_path / "c"
    out.mkdir()
    with pytest.raises(FileExistsError):
        build(iter([]), make_args(out), source_sha="x")
    out.rmdir()
    out.with_name("c.building").mkdir()
    with pytest.raises(FileExistsError, match="partial"):
        build(iter([]), make_args(out), source_sha="x")


def test_build_fails_loudly_when_the_stream_cannot_fill_a_split(tmp_path):
    rng = random.Random(1)
    stream = [row(paragraph(rng, 120), url=f"https://only{i}.org/p") for i in range(3)]
    with pytest.raises(ValueError, match="empty split"):
        build(iter(stream), make_args(tmp_path / "c", validation_per_mille=1), source_sha="x")


class FakeTokenizer:
    def encode(self, text, *, add_eos=False):
        return text.split() + (["<eos>"] if add_eos else [])


def test_token_estimate_scales_a_sampled_ratio_to_the_split(tmp_path):
    out = tmp_path
    writer = ShardWriter(out / "train" / "fineweb-edu", shard_bytes=10**6)
    for i in range(6):
        writer.write({"text": " ".join(["word"] * (10 + i))})
    writer.close()
    artifacts = {f"train/fineweb-edu/{n}": m for n, m in writer.artifacts.items()}
    report = estimate_tokens(out, artifacts, FakeTokenizer(), docs_per_shard=2)
    split = report["splits"]["train"]
    assert split["sampled_tokens"] == (10 + 1) + (11 + 1)
    assert split["estimated_tokens"] == int(
        artifacts["train/fineweb-edu/part-00000.jsonl"]["bytes"] / split["bytes_per_token"]
    )
    assert "validation" not in report["splits"]
