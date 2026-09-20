"""Paired document uncertainty for fixed trained models, not seed uncertainty."""

from __future__ import annotations

import math
import random
from statistics import mean


def paired_document_difference(left: dict, right: dict, *, draws: int = 10000,
                               seed: int = 0) -> dict:
    """Return left-minus-right CE/BPB and a paired document bootstrap interval.

    Each draw resamples whole documents jointly in both arms. Ratios use total
    loss / total tokens or bytes, so long documents retain their corpus weight.
    This interval conditions on these two models; it cannot replace training seeds.
    """
    if draws < 100:
        raise ValueError("use at least 100 bootstrap draws")
    for key in ("seq_len", "loop_k", "precision", "protocol"):
        if left[key] != right[key]:
            raise ValueError(f"evaluation setting differs: {key}")
    a, b = left["documents"], right["documents"]
    if len(a) != len(b) or not a:
        raise ValueError("document sets differ or are empty")
    deltas, tokens, payload = [], [], []
    for x, y in zip(a, b, strict=True):
        for key in ("index", "sha256", "scored_tokens", "scored_bytes"):
            if x[key] != y[key]:
                raise ValueError(f"paired document mismatch: {key}")
        if not all(math.isfinite(z["total_nats"]) and z["total_nats"] >= 0 for z in (x, y)):
            raise ValueError("invalid document loss")
        if x["scored_tokens"] <= 0 or x["scored_bytes"] <= 0:
            raise ValueError("documents need positive token and byte counts")
        deltas.append(x["total_nats"] - y["total_nats"])
        tokens.append(x["scored_tokens"])
        payload.append(x["scored_bytes"])
    for report in (left, right):
        for key in ("total_nats", "scored_tokens", "scored_bytes"):
            if not math.isclose(report[key], sum(doc[key] for doc in report["documents"]),
                                rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"report aggregate differs from documents: {key}")
    rng = random.Random(seed)
    ce, bpb = [], []
    for _ in range(draws):
        indices = [rng.randrange(len(a)) for _ in a]
        loss = sum(deltas[i] for i in indices)
        ce.append(loss / sum(tokens[i] for i in indices))
        bpb.append(loss / sum(payload[i] for i in indices) / math.log(2))

    def interval(samples):
        samples.sort()
        # Empirical inverse CDF, explicitly using nearest-rank endpoints.
        return [samples[math.ceil(0.025 * draws) - 1], samples[math.ceil(0.975 * draws) - 1]]

    return {"direction": "left minus right; negative favors left",
            "ce_delta": sum(deltas) / sum(tokens),
            "bpb_delta": sum(deltas) / sum(payload) / math.log(2),
            "ce_document_bootstrap_95": interval(ce),
            "bpb_document_bootstrap_95": interval(bpb),
            "document_win_fraction": mean(d < 0 for d in deltas),
            "documents": len(a), "draws": draws, "bootstrap_seed": seed,
            "uncertainty_scope": "paired document resampling conditional on these models; excludes training-seed uncertainty"}
