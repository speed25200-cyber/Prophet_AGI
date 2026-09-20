"""Causal continuation scoring for raw-text, zero-shot choice evaluations.

No chat template, BOS, EOS, truncation or KV-cache reuse. Joint tokenization must
preserve the complete prompt prefix; otherwise the scoring boundary is ambiguous.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch
import torch.nn.functional as F


def encode_choice(tokenizer, prompt: str, choice: str, *, max_tokens: int):
    if not prompt or not choice or not choice.strip():
        raise ValueError("prompt and choice must be nonempty")
    prefix = tokenizer.encode(prompt, add_eos=False)
    ids = tokenizer.encode(prompt + " " + choice, add_eos=False)
    if not prefix or ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError("joint tokenization changes the prompt boundary")
    if len(ids) > max_tokens:
        raise ValueError("candidate exceeds context; truncation is forbidden")
    return ids, len(prefix)


def token_ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


@torch.inference_mode()
def continuation_nats(model, ids, start: int, *, device="cpu", loop_k=None):
    if not 1 <= start < len(ids):
        raise ValueError("continuation needs preceding context and at least one target")
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    options = {"return_mtp": False}
    if loop_k is not None:
        options["loop_k"] = loop_k
    output = model(tokens[:, :-1], **options)
    logits = output.logits
    if logits.shape[:2] != tokens[:, :-1].shape or logits.dtype != torch.float32:
        raise ValueError("choice scoring requires full FP32 causal logits")
    scores = logits[0, start - 1:]
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite continuation logits")
    loss = F.cross_entropy(scores, tokens[0, start:], reduction="sum").item()
    if not math.isfinite(loss):
        raise ValueError("nonfinite continuation loss")
    return loss


def rank_choices(nats, character_lengths):
    if len(nats) < 2 or len(nats) != len(character_lengths):
        raise ValueError("invalid choice counts")
    if any(not math.isfinite(x) or x < 0 for x in nats) or any(n <= 0 for n in character_lengths):
        raise ValueError("invalid choice scores or lengths")
    normalized = [x / n for x, n in zip(nats, character_lengths, strict=True)]
    raw_min, norm_min = min(nats), min(normalized)
    return {"prediction": nats.index(raw_min), "prediction_character_normalized": normalized.index(norm_min),
            "ties": sum(x == raw_min for x in nats),
            "ties_character_normalized": sum(x == norm_min for x in normalized)}
