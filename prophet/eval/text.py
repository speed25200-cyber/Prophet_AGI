"""Document-preserving held-out CE/BPB with exact target and byte accounting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch

from prophet.data.tokenizer import ProphetTokenizer
from prophet.train.chunked_loss import shifted_token_losses


@dataclass
class DocumentScore:
    index: int
    sha256: str
    total_nats: float = 0.0
    scored_tokens: int = 0
    scored_bytes: int = 0


@torch.no_grad()
def evaluate_documents(model, documents: Iterable[str], tokenizer: ProphetTokenizer, *,
                       seq_len: int = 2048, batch_size: int = 8, device: str = "cpu",
                       loss_chunk_tokens: int = 512, loop_k: int | None = None) -> dict:
    """Score all tokens after the first token of each document, including EOS.

    Windows overlap by one context token; every target occurs exactly once. Rows
    never cross document boundaries. Padding is ignored by CE and byte accounting.
    Only window-local context is available, and positions reset for each window.
    Per-document sums permit paired comparisons without treating tokens as IID.
    """
    if seq_len < 2 or batch_size < 1:
        raise ValueError("seq_len must be >=2 and batch_size positive")
    device = torch.device(device)
    scores: list[DocumentScore] = []
    pending: list[tuple[int, list[int]]] = []
    was_training = model.training

    def flush():
        if not pending:
            return
        # Fixed shapes avoid hundreds of FLA specializations for variable document tails.
        inputs = torch.full((batch_size, seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
        targets = torch.full_like(inputs, -100)
        for row, (_, ids) in enumerate(pending):
            inputs[row, :len(ids)] = torch.tensor(ids, device=device)
            targets[row, :len(ids)] = inputs[row, :len(ids)]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(inputs, return_mtp=False, **({"loop_k": loop_k} if loop_k is not None else {}))
        logits = output.logits if hasattr(output, "logits") else output
        ce, _ = shifted_token_losses(logits, targets, 1, loss_chunk_tokens)
        nats = ce.double().sum(-1).cpu().tolist()
        for row, (index, ids) in enumerate(pending):
            if not math.isfinite(nats[row]):
                raise ValueError(f"non-finite evaluation loss in document {index}")
            scores[index].total_nats += nats[row]
            scores[index].scored_tokens += len(ids) - 1
            scores[index].scored_bytes += tokenizer.byte_length(ids[1:])
        pending.clear()

    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        # Evaluation must not change subsequent randomized recurrence initialization.
        with torch.random.fork_rng(devices=devices):
            for index, text in enumerate(documents):
                scores.append(DocumentScore(index, hashlib.sha256(text.encode("utf-8")).hexdigest()))
                ids = tokenizer.encode(text, add_eos=True)
                for start in range(0, len(ids) - 1, seq_len - 1):
                    pending.append((index, ids[start:start + seq_len]))
                    if len(pending) == batch_size:
                        flush()
            flush()
    finally:
        model.train(was_training)
    total_nats = sum(s.total_nats for s in scores)
    tokens = sum(s.scored_tokens for s in scores)
    byte_count = sum(s.scored_bytes for s in scores)
    if not tokens or not byte_count:
        raise ValueError("evaluation has no scored text bytes or tokens")
    return {
        "total_nats": total_nats, "scored_tokens": tokens, "scored_bytes": byte_count,
        "nats_per_token": total_nats / tokens, "bits_per_byte": total_nats / byte_count / math.log(2),
        "documents": [asdict(s) for s in scores], "seq_len": seq_len, "batch_size": batch_size,
        "loop_k": loop_k, "precision": "bf16 autocast" if device.type == "cuda" else "fp32",
        "protocol": "window-local positions; one-token overlap; first token unscored; EOS scored with zero payload bytes",
    }
