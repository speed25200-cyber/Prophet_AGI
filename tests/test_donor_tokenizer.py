import json
import unicodedata

import pytest

from prophet.data.corpus import LocalTextSource, TokenisedSource
from prophet.data.donor_tokenizer import DonorByteTokenizer
from prophet.data.streaming import StreamingLoader
from prophet.data.tokenizer import ProphetTokenizer

tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture
def tokenizer_path(tmp_path):
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.normalizer = tokenizers.normalizers.NFC()
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.add_special_tokens(["<|endoftext|>", "<|im_end|>"])
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return path


def adapter(path):
    return DonorByteTokenizer(path, eos_id=257, pad_id=256, vocab_size=260)


@pytest.mark.parametrize("text", ["Bonjour à tous!", "e\u0301 🍋 漢字\n\t", "", "<|im_end|> literal <|endoftext|>"])
def test_raw_text_roundtrip_and_exact_payload_bytes(tokenizer_path, text):
    tok = adapter(tokenizer_path)
    ids = tok.encode(text, add_eos=True)
    normalized = unicodedata.normalize("NFC", text)
    assert ids[-1] == 257 and 257 not in ids[:-1] and 256 not in ids
    assert tok.decode(ids) == normalized
    assert tok.byte_length(ids) == len(normalized.encode())
    assert sum(tok.byte_length([i]) for i in ids) == tok.byte_length(ids)


def test_rejects_control_protocol_and_unknown_ids(tokenizer_path):
    tok = adapter(tokenizer_path)
    with pytest.raises(ValueError, match="raw text"):
        tok.encode("hello", parse_special=True)
    with pytest.raises(ValueError, match="unknown token"):
        tok.byte_length([259])
    with pytest.raises(ValueError, match="separate"):
        DonorByteTokenizer(tokenizer_path, eos_id=1, pad_id=256, vocab_size=260)
    fingerprint = tok.fingerprint()
    fingerprint["normalizer"]["type"] = "NFD"
    assert tok.fingerprint()["normalizer"] == {"type": "NFC"}


def test_donor_loader_resume_binds_tokenizer_contents(tmp_path, tokenizer_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "abcdefgh " * 40}) + "\n", encoding="utf-8")

    def loader():
        source = TokenisedSource(LocalTextSource("text", 1, [corpus]), adapter(tokenizer_path))
        return StreamingLoader([source], seq_len=8, batch_size=1)

    original = loader()
    list(original.batches(3))
    state = original.state().to_dict()
    resumed = loader()
    resumed.load_state(resumed.validate_state(state))
    assert list(original.batches(2)) == list(resumed.batches(2))
    specification = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    specification["normalizer"] = None
    tokenizer_path.write_text(json.dumps(specification), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        loader().validate_state(state)


def test_legacy_tokenizer_fingerprint_is_unchanged(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("abc\n")
    tok = ProphetTokenizer(merges=[])
    source = LocalTextSource("text", 1, [corpus])
    actual = TokenisedSource(source, tok).fingerprint()
    assert actual == {"source": source.fingerprint(), "max_epochs": 4, "parse_special": False,
                      "vocab_size": tok.vocab_size, "merges": [],
                      "special_tokens": tok._special_to_id, "decontamination": None}
