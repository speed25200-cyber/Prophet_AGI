"""Tests for Prophet-Tok v1.

Each test corresponds to a specific tokenizer pathology that track R01 identified as
costing us points on the target benchmarks. They exist so that a later change to the
pre-tokenisation pattern -- which is easy to make and impossible to notice -- cannot
silently reintroduce the pathology it was written to fix.
"""

from __future__ import annotations

import pytest

from prophet.data.tokenizer import (
    N_BYTES,
    N_RESERVED,
    SPECIAL_TOKENS,
    BPETrainer,
    ProphetTokenizer,
    pre_tokenize,
)

CORPUS = [
    "the quick brown fox jumps over the lazy dog. " * 40,
    "def function(x):\n    return x + 1\n" * 40,
    "les nombres 12345 et 67890 sont grands. " * 30,
    "import os\nfor i in range(10):\n    print(i)\n" * 30,
]


@pytest.fixture(scope="module")
def tokenizer() -> ProphetTokenizer:
    merges = BPETrainer(vocab_size=N_BYTES + N_RESERVED + 400).train(CORPUS)
    return ProphetTokenizer(merges=merges, vocab_size=N_BYTES + N_RESERVED + 400)


# --------------------------------------------------------------------------------------
# Repair 1 — arithmetic
# --------------------------------------------------------------------------------------


def test_digits_are_always_separate_units():
    """Inconsistent digit grouping is a large part of why models fail arithmetic they
    otherwise have the capacity for."""
    units = pre_tokenize("The total is 1234567 euros")
    digits = [u for u in units if u.isdigit()]
    assert digits == list("1234567")


def test_no_digit_ever_merges_with_a_neighbour(tokenizer):
    for problem in tokenizer.check_invariants():
        assert "digit" not in problem


def test_the_same_digit_gets_the_same_id_everywhere(tokenizer):
    """The property that makes column-wise arithmetic learnable at all."""
    def digit_ids(text: str) -> list[int]:
        return [tokenizer.encode(c)[0] for c in text if c.isdigit()]

    assert digit_ids("7") == digit_ids("1237")[-1:] == digit_ids("999997")[-1:]


def test_numbers_of_any_length_cost_one_token_per_digit(tokenizer):
    for number in ("5", "42", "1234", "999999999"):
        assert len(tokenizer.encode(number)) == len(number)


# --------------------------------------------------------------------------------------
# Repair 2 and 3 — code structure
# --------------------------------------------------------------------------------------


def test_no_merge_spans_a_newline(tokenizer):
    """A merge crossing a line break makes indentation depend on the previous line's
    content, which is exactly backwards for code."""
    assert tokenizer.check_invariants() == []


def test_a_line_break_is_always_its_own_unit():
    units = pre_tokenize("a = 1\nb = 2\n")
    assert units.count("\n") == 2
    assert all(u == "\n" or "\n" not in u for u in units)


def test_indentation_is_a_unit_of_its_own():
    units = pre_tokenize("def f():\n    if x:\n        return 1")
    assert "    " in units
    assert "        " in units


def test_indentation_depth_is_distinguishable():
    """Four spaces and eight spaces must not be the same unit, or the model cannot see
    nesting depth."""
    units = pre_tokenize("if a:\n    if b:\n        pass")
    indents = [u for u in units if u.strip() == "" and u != "\n"]
    assert len(set(indents)) >= 2


# --------------------------------------------------------------------------------------
# Repair 4 — byte fallback and totality
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "x = 1234 + 5678",
        "def f():\n    return 1\n",
        "Ünïcodé accents",
        "日本語のテキスト",
        "emoji \U0001f600 and \u2713 symbols",
        "\t\ttabs\tand   spaces  ",
        "mixed\r\nline\rendings\n",
        "",
        " ",
        "\n\n\n",
    ],
)
def test_encoding_round_trips_exactly(tokenizer, text):
    """There is no unknown token, so encoding is total and lossless. Silent corruption
    here would be invisible in every downstream metric."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_every_byte_value_is_representable(tokenizer):
    raw = bytes(range(256)).decode("latin-1")
    assert tokenizer.decode(tokenizer.encode(raw)) == raw


def test_decode_is_total_for_model_generated_byte_ids(tokenizer):
    assert isinstance(tokenizer.decode(range(N_BYTES)), str)
    assert tokenizer.decode([0xFF]) == "\N{REPLACEMENT CHARACTER}"


def test_unused_in_range_ids_are_safe_but_exposed_for_logit_masking(tokenizer):
    unused = next(token_id for token_id in range(len(tokenizer))
                  if token_id not in tokenizer.valid_token_ids)
    assert tokenizer.decode([unused]) == ""
    assert tokenizer.decode([unused], skip_special=False) == f"<|unused_{unused}|>"
    with pytest.raises(ValueError, match="unknown or unused"):
        tokenizer.decode([len(tokenizer)])


def test_unknown_token_is_never_emitted(tokenizer):
    unk = tokenizer.special_id("<|unk|>")
    for text in ("\x00\x01\x02", "🜁🜂🜃", "日本"):
        assert unk not in tokenizer.encode(text)


# --------------------------------------------------------------------------------------
# Vocabulary layout
# --------------------------------------------------------------------------------------


def test_byte_ids_occupy_the_first_256_slots(tokenizer):
    for b in range(256):
        assert tokenizer._token_to_id[bytes([b])] == b


def test_reserved_ids_exist_for_later_modalities(tokenizer):
    """Track R12 hook H1: adding image or audio tokens later must not change the
    embedding table, which would force a retrain."""
    assert len(SPECIAL_TOKENS) < N_RESERVED
    highest_special = max(tokenizer.special_id(s) for s in SPECIAL_TOKENS)
    assert highest_special < N_BYTES + N_RESERVED
    # No learned merge may occupy a reserved slot.
    for token_id in tokenizer._token_to_id.values():
        assert not (N_BYTES + len(SPECIAL_TOKENS) <= token_id < N_BYTES + N_RESERVED)


def test_special_tokens_have_stable_ids(tokenizer):
    """Their order is frozen: changing it invalidates every trained checkpoint."""
    assert tokenizer.special_id("<|pad|>") == N_BYTES
    assert tokenizer.special_id("<|bos|>") == N_BYTES + 1
    assert tokenizer.special_id("<|eos|>") == N_BYTES + 2


def test_abstention_and_modality_tokens_are_present(tokenizer):
    for name in ("<|idk|>", "<|retrieve|>", "<|img|>", "<|think|>"):
        assert tokenizer.special_id(name) >= N_BYTES


def test_bos_and_eos_can_be_added_and_skipped(tokenizer):
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello"
    assert "<|bos|>" in tokenizer.decode(ids, skip_special=False)


def test_trainer_rejects_a_vocabulary_with_no_room_for_merges():
    with pytest.raises(ValueError, match="no room for merges"):
        BPETrainer(vocab_size=N_BYTES + N_RESERVED)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_save_and_load_preserve_segmentation(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    reloaded = ProphetTokenizer.load(path)
    for text in ("def f():\n    return 1", "x = 42", "Ünïcodé"):
        assert reloaded.encode(text) == tokenizer.encode(text)


def test_loading_a_vocabulary_built_with_a_different_pattern_is_refused(tokenizer, tmp_path):
    """Loading it would silently change how text is segmented -- a change that trains
    fine and produces a subtly worse model."""
    import json

    path = tmp_path / "tok.json"
    tokenizer.save(path)
    data = json.loads(path.read_text())
    data["pre_tokenize_pattern"] = r"\w+"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="different pre-tokenisation"):
        ProphetTokenizer.load(path)


# --------------------------------------------------------------------------------------
# The parameter argument
# --------------------------------------------------------------------------------------


def test_a_32k_vocabulary_is_a_fraction_of_a_128k_one():
    """R01's headline argument, measured on our own configuration rather than quoted.

    At d_model=1024 over 24 layers with tied embeddings, a 128k vocabulary spends about a
    third of the model on a lookup table that performs no computation; 32k spends under a
    tenth. The recovered parameters buy depth and width instead, which is why the
    vocabulary choice is the single largest quality-per-byte win in track R01.
    """
    from prophet.budget import count_parameters
    from prophet.config import FrontendConfig, ProphetConfig

    def embedding_share(vocab: int) -> float:
        cfg = ProphetConfig(d_model=1024, n_layers=24,
                            frontend=FrontendConfig(vocab_size=vocab))
        p = count_parameters(cfg)
        return p.embedding / p.total

    big, small = embedding_share(131072), embedding_share(32768)
    assert big > 0.30, f"expected a 128k vocabulary to dominate, measured {big:.1%}"
    assert small < 0.12, f"expected a 32k vocabulary to be minor, measured {small:.1%}"

    # The number that matters is the absolute recovery, not the ratio: at this width the
    # swap frees roughly 100M parameters to spend on computation instead of lookup.
    cfg = ProphetConfig(d_model=1024, n_layers=24, frontend=FrontendConfig(vocab_size=131072))
    small_cfg = ProphetConfig(d_model=1024, n_layers=24,
                              frontend=FrontendConfig(vocab_size=32768))
    recovered = count_parameters(cfg).embedding - count_parameters(small_cfg).embedding
    assert recovered > 90e6, f"only {recovered / 1e6:.0f}M recovered"
