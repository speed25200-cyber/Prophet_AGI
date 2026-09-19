import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prophet.data.tokenizer import ProphetTokenizer
from prophet.eval.text import evaluate_documents


class UniformModel(nn.Module):
    def forward(self, ids, **kwargs):
        torch.rand(1)  # Prove that evaluation isolates RNG even if a model draws.
        return SimpleNamespace(logits=torch.zeros(*ids.shape, 512, device=ids.device))


@pytest.mark.parametrize("seq_len", [2, 4, 11])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_every_target_and_utf8_byte_counted_once(seq_len, batch_size):
    tokenizer = ProphetTokenizer([], vocab_size=512)
    docs = ["alpha", "é日本🦊 xyz", "", "a", "\nsecond document" * 3]
    model = UniformModel().train()
    before = torch.get_rng_state().clone()
    result = evaluate_documents(model, docs, tokenizer, seq_len=seq_len, batch_size=batch_size)
    assert model.training and torch.equal(before, torch.get_rng_state())
    expected_tokens = 0
    expected_bytes = 0
    for text, score in zip(docs, result["documents"], strict=True):
        ids = tokenizer.encode(text, add_eos=True)
        assert score["scored_tokens"] == len(ids) - 1
        # Byte fallback tokenizer: the first unscored byte is exactly one payload byte.
        assert score["scored_bytes"] == max(0, len(text.encode("utf-8")) - 1)
        expected_tokens += len(ids) - 1
        expected_bytes += score["scored_bytes"]
    assert result["scored_tokens"] == expected_tokens
    assert result["scored_bytes"] == expected_bytes
    assert result["nats_per_token"] == pytest.approx(math.log(512))
    assert result["bits_per_byte"] == pytest.approx(expected_tokens * 9 / expected_bytes)


def test_document_context_is_never_joined_and_padding_never_scored():
    tokenizer = ProphetTokenizer([], vocab_size=512)
    class ContextModel(nn.Module):
        def forward(self, ids, **kwargs):
            # The prediction depends on *every* preceding token in this row.
            predicted = ids.cumsum(-1) % 512
            logits = torch.zeros(*ids.shape, 512)
            logits.scatter_(-1, predicted.unsqueeze(-1), 5.0)
            return SimpleNamespace(logits=logits)
    model = ContextModel()
    docs = ["x" * 12, "ABC", "y" * 20]
    combined = evaluate_documents(model, docs, tokenizer, seq_len=8, batch_size=3)
    separate = [evaluate_documents(model, [text], tokenizer, seq_len=8, batch_size=1) for text in docs]
    assert combined["total_nats"] == pytest.approx(sum(r["total_nats"] for r in separate))


def test_byte_length_does_not_round_trip_partial_unicode():
    tokenizer = ProphetTokenizer([], vocab_size=512)
    ids = tokenizer.encode("🦊")
    assert tokenizer.byte_length(ids[1:]) == 3
    assert len(tokenizer.decode(ids[1:]).encode("utf-8")) != 3
    assert tokenizer.byte_length([tokenizer.eos_id]) == 0


def test_empty_evaluation_refused_and_mode_restored():
    model = UniformModel().train()
    with pytest.raises(ValueError, match="no scored"):
        evaluate_documents(model, [""], ProphetTokenizer([], vocab_size=512), seq_len=4)
    assert model.training
