import copy
import math

import pytest

from prophet.eval.paired import paired_document_difference


def report(losses):
    documents = [{"index": i, "sha256": str(i), "total_nats": n,
                  "scored_tokens": (i + 1) * 10, "scored_bytes": (i + 1) * 20}
                 for i, n in enumerate(losses)]
    return {"documents": documents, "seq_len": 2048, "loop_k": None,
            "precision": "fp32", "protocol": "test",
            **{k: sum(d[k] for d in documents)
               for k in ("total_nats", "scored_tokens", "scored_bytes")}}


def test_weighted_constant_improvement_has_exact_interval_and_no_rng_dependency():
    left, right = report([10., 20.]), report([20., 40.])
    result = paired_document_difference(left, right, draws=100)
    assert result["ce_delta"] == -1
    assert result["ce_document_bootstrap_95"] == [-1, -1]
    assert result["bpb_delta"] == pytest.approx(-0.5 / math.log(2))
    assert result["document_win_fraction"] == 1
    assert result == paired_document_difference(left, right, draws=100)


@pytest.mark.parametrize("change", ["document", "context", "aggregate", "nan"])
def test_incompatible_or_corrupt_report_is_rejected(change):
    left = report([10., 40.])
    right = copy.deepcopy(left)
    if change == "document":
        right["documents"][0]["sha256"] = "other"
    elif change == "context":
        right["seq_len"] = 256
    elif change == "aggregate":
        right["scored_tokens"] += 1
    else:
        right["documents"][0]["total_nats"] = float("nan")
    with pytest.raises(ValueError):
        paired_document_difference(left, right, draws=100)
