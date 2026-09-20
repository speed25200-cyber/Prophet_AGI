"""Cross-campaign leakage and bounded-source checks before further GPU work."""

import json
from collections import Counter

import pytest

from prophet.data.decontaminate import normalise
from scripts.prepare_fresh_pilot import Exclusions, fresh_rows, load_exclusions, sha256
from scripts.prepare_pilot import (
    DATASET,
    REVISION,
    SUBSET,
    content_hash,
    partition_rows,
    write_pilot,
)


def document(name):
    text = " ".join(f"{name}word{i}" for i in range(70))
    return {
        "text": text,
        "content_sha256": content_hash(normalise(text)),
        "url": f"https://example.org/{name}",
        "int_score": 3,
    }


def fixture(tmp_path):
    prior = tmp_path / "prior"
    manifest = write_pilot(
        {"train": [document("oldtrain")], "validation": [document("oldval")]},
        prior,
        {
            "format_version": 1,
            "source": DATASET,
            "source_revision": REVISION,
            "subset": SUBSET,
            "stats": {"rows_seen": 2},
        },
    )
    arc = tmp_path / "arc.jsonl"
    arc.write_text(
        json.dumps(
            {
                "prompt": "Question: What happens when warm ocean water freezes?\nAnswer:",
                "choices": [
                    "Salt concentration increases in the surrounding liquid water.",
                    "Nothing happens.",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return prior, arc, manifest


def test_prior_train_validation_and_all_arc_choices_protect_new_splits(tmp_path):
    prior, arc, _ = fixture(tmp_path)
    index, _, receipt = load_exclusions(prior, arc, sha256(arc))
    for name in ("oldtrain", "oldval"):
        original = document(name)
        assert (
            index.reason({**original, "text": original["text"].upper(), "url": ""})
            == "prior_exact_content"
        )
        assert (
            index.reason({**document("new"), "url": original["url"] + "?version=2"}) == "prior_url"
        )
        span = " ".join(original["text"].split()[15:28])
        assert (
            index.reason({"text": f"Changed prefix {span} changed suffix"})
            == "protected_13_word_span"
        )
    for fragment in (
        "what happens when warm ocean water freezes",
        "salt concentration increases in the surrounding liquid water",
    ):
        assert (
            index.reason({"text": "begin " + fragment.upper() + " end"}) == "protected_short_phrase"
        )
    assert index.reason({"text": "begin nothing happens end"}) is None
    assert receipt["protected"]["benchmark_fragments_under_five_words"] == 1
    assert receipt["protected"]["prior_documents"] == 2
    assert receipt["arc_questions"] == 1


def test_short_phrases_match_whole_words_and_long_spans_are_conservative():
    index = Exclusions()
    index.add("one two three four five", benchmark=True)
    assert index.reason({"text": "someone two three four five"}) is None
    assert index.reason({"text": "one two three four fives"}) is None
    assert index.reason({"text": "ONE, two three four five! more"}) == "protected_short_phrase"
    index.add(" ".join(f"word{i}" for i in range(30)), benchmark=True)
    assert (
        index.reason({"text": " ".join(f"word{i}" for i in range(5, 18))})
        == "protected_13_word_span"
    )


def test_raw_scan_cap_applies_even_if_every_document_is_rejected():
    index = Exclusions()
    index.add("forbidden")
    visited = []

    def source():
        for i in range(100):
            visited.append(i)
            yield {"text": "forbidden"}

    stats = Counter()
    assert list(fresh_rows(source(), index, skip_rows=3, max_scan_rows=7, stats=stats)) == []
    assert visited == list(range(10))
    assert stats == {"raw_rows_scanned": 7, "last_source_ordinal": 9, "prior_exact_content": 7}


def test_pipeline_is_deterministic_and_rejects_cross_campaign_and_new_heldout_overlap():
    from scripts.prepare_pilot import split_for

    index = Exclusions()
    index.add(document("oldtrain")["text"])
    source = [document(f"new{i}") for i in range(300)]
    val = next(r for r in source if split_for(r["url"], "", 42, 200) == "validation")
    train = next(r for r in source if split_for(r["url"], "", 42, 200) == "train")
    source += [{**train, "text": train["text"] + " " + val["text"]}, document("oldtrain")]

    def run():
        counters = Counter()
        accepted = fresh_rows(source, index, skip_rows=0, max_scan_rows=1000, stats=counters)
        groups, stats = partition_rows(
            accepted, max_docs=1000, max_bytes=1000000, validation_per_mille=200
        )
        return groups, stats, counters

    first = run()
    assert first == run()
    assert first[1]["heldout_overlap_rejected"] == 1
    assert first[2]["prior_exact_content"] == 1
    assert all(
        row["text"] != document("oldtrain")["text"] for rows in first[0].values() for row in rows
    )


@pytest.mark.parametrize(
    "corruption", ["prior_bytes", "prior_count", "arc_bytes", "source", "path", "cursor_format"]
)
def test_changed_exclusion_inputs_fail_closed(tmp_path, corruption):
    prior, arc, manifest = fixture(tmp_path)
    expected_arc = sha256(arc)
    key = "train/fineweb-edu/part-00000.jsonl"
    if corruption == "prior_bytes":
        with (prior / key).open("ab") as stream:
            stream.write(b" ")
    elif corruption == "prior_count":
        manifest["artifacts"][key]["documents"] += 1
    elif corruption == "arc_bytes":
        arc.write_text("changed", encoding="utf-8")
    elif corruption == "source":
        manifest["source_revision"] = "0" * 40
    elif corruption == "cursor_format":
        manifest["format_version"] = 2
    else:
        manifest["artifacts"]["train/../../../outside.jsonl"] = manifest["artifacts"].pop(key)
    (prior / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_exclusions(prior, arc, expected_arc)


def test_bad_arc_wrapper_is_rejected_even_when_file_hash_matches(tmp_path):
    prior, arc, _ = fixture(tmp_path)
    arc.write_text(
        json.dumps({"prompt": "changed wrapper", "choices": ["a", "b"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wrapper"):
        load_exclusions(prior, arc, sha256(arc))
