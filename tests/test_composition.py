"""Composition tasks: the generator is checked by an independent parser, the evaluator
against the single-candidate scorer it must reproduce."""

import random
import re

import pytest

from prophet.data.composition import (
    ADDITION_CHOICES,
    addition_example,
    generate,
    pseudowords,
    read_jsonl,
    split_vocabulary,
    table_example,
    write_jsonl,
)
from prophet.eval.choices import continuation_nats, encode_choice, rank_choices
from prophet.eval.composition import evaluate_composition, paired_accuracy_difference
from tests.test_choice_eval import CharacterTokenizer, FixedDistribution


def follow_chain(text: str) -> tuple[str, str]:
    """Independent parser: rebuild the mapping from the text and walk the hops."""
    mapping = dict(re.findall(r"^(\w+) -> (\w+)$", text, flags=re.M))
    start = re.search(r"^start: (\w+)$", text, flags=re.M).group(1)
    hops = int(re.search(r"^hops: (\d+)$", text, flags=re.M).group(1))
    answer = re.search(r"^answer: (\w+)$", text, flags=re.M).group(1)
    value = start
    for _ in range(hops):
        value = mapping[value]
    return value, answer


def test_pseudowords_are_distinct_and_seeded():
    words = pseudowords(500, seed=1)
    assert len(set(words)) == 500 and words == pseudowords(500, seed=1)
    assert words != pseudowords(500, seed=2)
    assert all(
        re.fullmatch(r"[bdfgklmnprstvz][aeiou]{1}(?:[bdfgklmnprstvz][aeiou]){2}", w) for w in words
    )


def test_train_and_test_vocabularies_are_disjoint():
    halves = split_vocabulary(200, seed=3)
    assert len(halves["train"]) == len(halves["test"]) == 100
    assert not set(halves["train"]) & set(halves["test"])


def test_table_chains_are_correct_have_no_fixed_points_and_answer_is_a_candidate():
    rng = random.Random(0)
    names = pseudowords(60, seed=0)
    for hops in (1, 2, 3, 4):
        for _ in range(50):
            example = table_example(rng, names, hops=hops, size=12)
            walked, written = follow_chain(example["text"])
            assert walked == written == example["answer"]
            assert example["answer"] in example["choices"] and len(example["choices"]) == 12
            mapping = dict(re.findall(r"^(\w+) -> (\w+)$", example["text"], flags=re.M))
            assert all(k != v for k, v in mapping.items())
            assert example["text"] == example["prompt"] + " " + example["answer"] + "\n"
            assert example["prompt"].endswith("answer:")


def test_addition_examples_have_distinct_near_misses_and_the_true_sum():
    rng = random.Random(1)
    for digits in (2, 3, 4, 5, 6):
        for _ in range(30):
            example = addition_example(rng, digits=digits)
            a, b = map(int, re.match(r"add: (\d+) \+ (\d+)", example["prompt"]).groups())
            assert len(str(a)) == len(str(b)) == digits
            assert example["answer"] == str(a + b)
            assert len(example["choices"]) == ADDITION_CHOICES
            assert len(set(example["choices"])) == ADDITION_CHOICES
            assert example["choices"].count(example["answer"]) == 1


def test_generate_is_deterministic_balanced_and_split_specific(tmp_path):
    train = list(generate("train", per_level=3, seed=7, vocabulary_size=100))
    again = list(generate("train", per_level=3, seed=7, vocabulary_size=100))
    test = list(generate("test", per_level=3, seed=7, vocabulary_size=100))
    assert train == again
    assert len(train) == 3 * (4 + 5)
    train_words = {w for e in train if e["kind"] == "table" for w in e["choices"]}
    test_words = {w for e in test if e["kind"] == "table" for w in e["choices"]}
    assert not train_words & test_words, "test tables must not share a pseudoword with training"
    levels = sorted({(e["kind"], e["hops"]) for e in train})
    assert levels == [("addition", d) for d in (2, 3, 4, 5, 6)] + [
        ("table", h) for h in (1, 2, 3, 4)
    ]

    stats = write_jsonl(iter(train), tmp_path / "train.jsonl", fields=("text",))
    rows = read_jsonl(tmp_path / "train.jsonl")
    assert stats["documents"] == len(rows) == len(train)
    assert all(set(row) == {"text"} for row in rows)
    write_jsonl(iter(test), tmp_path / "test.jsonl")
    assert read_jsonl(tmp_path / "test.jsonl")[0]["choices"]


class PaddedCharacterTokenizer(CharacterTokenizer):
    pad_id = 0


def test_evaluation_reproduces_single_candidate_scoring_and_pairs_items():
    tokenizer = PaddedCharacterTokenizer()
    model = FixedDistribution()
    examples = list(generate("test", per_level=2, seed=11, vocabulary_size=60, table_size=4))
    report = evaluate_composition(model, examples, tokenizer, batch_size=4, seq_len=256)
    assert report["examples"] == len(examples)
    assert set(report["levels"]) == {f"{e['kind']}:{e['hops']}" for e in examples}
    for example, item in zip(examples, report["items"], strict=True):
        nats = [
            continuation_nats(
                model, *encode_choice(tokenizer, example["prompt"], c, max_tokens=257)
            )
            for c in example["choices"]
        ]
        assert item["nats"] == pytest.approx(nats, rel=1e-5, abs=1e-4)
        # Summation order differs between the batched and single paths by ~1e-6, which
        # can flip an exact tie between candidates; the prediction must be a minimum.
        ranking = rank_choices(nats, [len(c) for c in example["choices"]])
        assert item["nats"][item["prediction"]] <= min(nats) + 1e-4
        assert item["ties"] >= 1 and ranking["ties"] >= 1
        assert item["gold"] == example["choices"].index(example["answer"])
    for level in report["levels"].values():
        assert 0.0 <= level["accuracy"] <= 1.0
        expected_chance = 1 / 4 if level["kind"] == "table" else 1 / ADDITION_CHOICES
        assert level["chance"] == pytest.approx(expected_chance)
    other = evaluate_composition(model, examples, tokenizer, batch_size=4, seq_len=256, loop_k=2)
    contrast = paired_accuracy_difference(report, other, kind="table", level=1)
    assert contrast["count"] == 2 and contrast["difference"] == 0.0
    with pytest.raises(ValueError, match="not paired"):
        paired_accuracy_difference(
            report, {"items": list(reversed(other["items"]))}, kind="table", level=1
        )


def test_evaluation_rejects_answers_outside_the_candidates():
    tokenizer = PaddedCharacterTokenizer()
    bad = [{"kind": "table", "hops": 1, "prompt": "x:", "answer": "zz", "choices": ["a", "b"]}]
    with pytest.raises(ValueError, match="one of the candidates"):
        evaluate_composition(FixedDistribution(), bad, tokenizer, seq_len=64)
