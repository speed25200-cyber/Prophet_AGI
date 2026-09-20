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
    if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
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
    scores = logits[0, start - 1 :]
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite continuation logits")
    loss = F.cross_entropy(scores, tokens[0, start:], reduction="sum").item()
    if not math.isfinite(loss):
        raise ValueError("nonfinite continuation loss")
    return loss


@torch.inference_mode()
def batched_continuation_nats(
    model,
    candidates,
    *,
    pad_id: int,
    batch_size: int,
    seq_len: int,
    device="cpu",
    loop_k=None,
    progress=None,
):
    """Same answer-only FP32 scores, with causal right-padding and fixed shapes.

    Every candidate must fit whole. Padding and prompt targets are never scored;
    the last partial batch retains the same shape to avoid kernel specializations.
    """
    if batch_size < 1 or seq_len < 1 or not candidates:
        raise ValueError("positive shape and nonempty candidates required")
    for ids, start in candidates:
        if not 1 <= start < len(ids):
            raise ValueError("continuation needs preceding context and targets")
        if len(ids) - 1 > seq_len:
            raise ValueError("candidate exceeds context; truncation is forbidden")
    options = {"return_mtp": False}
    if loop_k is not None:
        options["loop_k"] = loop_k
    was_training = model.training
    model.eval()
    results = []
    try:
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            inputs = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
            targets = torch.full_like(inputs, -100)
            for row, (ids, start) in enumerate(batch):
                inputs[row, : len(ids) - 1] = torch.tensor(ids[:-1], device=device)
                targets[row, start - 1 : len(ids) - 1] = torch.tensor(ids[start:], device=device)
            logits = model(inputs, **options).logits
            if logits.shape[:2] != inputs.shape or logits.dtype != torch.float32:
                raise ValueError("choice scoring requires full FP32 causal logits")
            # Inspect only scored positions, matching the single-candidate oracle.
            selected = targets != -100
            scores = logits[selected]
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("nonfinite continuation logits")
            # Keep the same [targets, vocabulary] reduction layout as the
            # single-candidate scorer, and avoid softmax work on ignored positions.
            ce = F.cross_entropy(scores, targets[selected], reduction="none")
            lengths = [len(ids) - start for ids, start in batch]
            nats = torch.stack([part.sum() for part in ce.split(lengths)]).cpu().tolist()
            if any(not math.isfinite(value) for value in nats):
                raise ValueError("nonfinite continuation loss")
            results.extend(nats)
            if progress is not None:
                progress(len(results), len(candidates))
    finally:
        model.train(was_training)
    return results


def continuation_bucket(tokens: int, *, max_seq_len: int):
    """Select a bounded power-of-two input shape from length alone."""
    if tokens < 2 or max_seq_len < 1 or max_seq_len & (max_seq_len - 1):
        raise ValueError("positive power-of-two shape and preceding context required")
    if tokens - 1 > max_seq_len:
        raise ValueError("candidate exceeds context; truncation is forbidden")
    return max(min(32, max_seq_len), 1 << (tokens - 2).bit_length())


def rank_choices(nats, character_lengths):
    if len(nats) < 2 or len(nats) != len(character_lengths):
        raise ValueError("invalid choice counts")
    if any(not math.isfinite(x) or x < 0 for x in nats) or any(n <= 0 for n in character_lengths):
        raise ValueError("invalid choice scores or lengths")
    normalized = [x / n for x, n in zip(nats, character_lengths, strict=True)]
    raw_min, norm_min = min(nats), min(normalized)
    return {
        "prediction": nats.index(raw_min),
        "prediction_character_normalized": normalized.index(norm_min),
        "ties": sum(x == raw_min for x in nats),
        "ties_character_normalized": sum(x == norm_min for x in normalized),
    }
