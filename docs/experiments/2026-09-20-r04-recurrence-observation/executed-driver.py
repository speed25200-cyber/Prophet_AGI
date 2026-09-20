"""Observe frozen R04 core states; descriptive measurements, not an adoption test."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def state_metrics(states):
    """CPU float64 reductions over unpadded [tokens, hidden] core outputs."""
    import torch

    result = []
    previous = None
    for step, state in enumerate(states, 1):
        value = state.detach().cpu().double()
        if value.ndim != 2 or value.shape[0] < 2 or not torch.isfinite(value).all():
            raise ValueError("finite unpadded token matrix with at least two positions required")
        lengths = value.norm(dim=-1)
        if (lengths == 0).any():
            raise ValueError("cosine is undefined for a zero token state")
        unit = value / lengths[:, None]
        n = len(value)
        energy = value.square().sum()
        centered = value - value.mean(dim=0)
        row = {
            "loop": step,
            "tokens": n,
            "state_rms": float(value.square().mean().sqrt()),
            "token_centered_energy_fraction": float(centered.square().sum() / energy),
            # Exact ordered off-diagonal pair mean without a T-by-T allocation.
            "mean_distinct_token_cosine": float(
                (unit.sum(dim=0).square().sum() - n) / (n * (n - 1))
            ),
            "previous_loop_cosine": None,
            "relative_previous_loop_change": None,
        }
        if previous is not None:
            if previous.shape != value.shape:
                raise ValueError("loop shapes differ")
            row["previous_loop_cosine"] = float(
                (unit * (previous / previous.norm(dim=-1)[:, None])).sum(dim=-1).mean()
            )
            row["relative_previous_loop_change"] = float(
                (value - previous).norm() / previous.norm()
            )
        result.append(row)
        previous = value
    if not result:
        raise ValueError("no recurrent states captured")
    return result


def observe(model, inputs, valid_tokens, depth):
    """Copy core outputs with a removable observer; never replace a forward value."""
    import torch

    if model.training or inputs.shape[0] != 1 or not 2 <= valid_tokens <= inputs.shape[1]:
        raise ValueError("evaluation mode, batch one and valid token count required")
    if depth < 1:
        raise ValueError("positive recurrence depth required")
    states = []

    def capture(_module, _args, output):
        states.append(output[0, :valid_tokens].detach().cpu().clone())

    handle = model.sections["core"][-1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            output = model(inputs, loop_k=depth, return_mtp=False)
        if len(states) != depth:
            raise ValueError("observed core count differs from requested depth")
    finally:
        handle.remove()
    return output, state_metrics(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh output required")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.model_repo, text=True
    ).strip()
    if revision != "f5a7d71be1723324e36a7be2546c940a69de9fdf":
        raise ValueError("requires the frozen adaptation model source")
    subprocess.run(["git", "diff", "HEAD", "--exit-code"], cwd=args.model_repo, check=True)
    sys.path.insert(0, str(args.model_repo.resolve()))
    import torch

    from prophet.config import ProphetConfig
    from prophet.data.corpus import LocalTextSource
    from prophet.data.tokenizer import ProphetTokenizer
    from prophet.modeling.model import ProphetModel
    from scripts.adapt_r04_depth import configure_numerics, runtime_for
    from scripts.audit_r04_restart import load_run
    from scripts.run_r04_pilot import verify_pilot, write_json

    started = time.monotonic()
    state, report = load_run(args.run, 512)
    identity = report["identity"]
    if not report["complete"] or identity["code"]["revision"] != revision:
        raise ValueError("completed frozen adaptation required")
    if (
        configure_numerics("cuda") != identity["numerical_policy"]
        or runtime_for("cuda") != identity["runtime"]
    ):
        raise ValueError("numerical policy or runtime differs")
    if verify_pilot(args.corpus) != identity["data"]:
        raise ValueError("corpus differs")
    cfg = ProphetConfig.from_dict(state["config"])
    if (
        cfg.recurrent.eval_state_init != "zeros"
        or cfg.recurrent.halting != "none"
        or cfg.recurrent.token_depth
    ):
        raise ValueError("requires deterministic fixed-depth inference")
    model = ProphetModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.cuda().eval()
    tokenizer = ProphetTokenizer.load(args.corpus / "tokenizer.json")
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)
    output = {
        "protocol": "r04-recurrence-observation-v1",
        "complete": False,
        "identity": identity,
        "checkpoint": report["checkpoint"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "First 16 validation documents in original order, first 2048 token IDs including EOS when reached; batch one padded to 2048, padding excluded from metrics; eight loops.",
        "metric_precision": "core inference uses recorded BF16 autocast; detached state reductions use CPU float64",
        "documents": [],
        "scope": "Descriptive internal states only. Token cosine and centered energy do not by themselves establish information loss, causal failure, reasoning, extrapolation quality or architecture adoption. This subset does not replace the full-validation primary screen.",
    }
    for index, text in enumerate(itertools.islice(validation.open(), 16)):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != report["results"]["4"]["documents"][index]["sha256"]:
            raise ValueError("document order differs")
        ids = tokenizer.encode(text, add_eos=True)[:2048]
        inputs = torch.full((1, 2048), tokenizer.pad_id, device="cuda", dtype=torch.long)
        inputs[0, : len(ids)] = torch.tensor(ids, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model(inputs, loop_k=8, return_mtp=False).hidden.detach().cpu()
            observed, metrics = observe(model, inputs, len(ids), 8)
        if not torch.equal(reference, observed.hidden.detach().cpu()):
            raise ValueError("observer changed the model output")
        output["documents"].append(
            {
                "index": index,
                "sha256": digest,
                "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "observer_hidden_equal": True,
                "loops": metrics,
            }
        )
        del observed, reference
        write_json(args.out, output)
    if len(output["documents"]) != 16:
        raise ValueError("fewer than sixteen documents")
    output["complete"] = True
    output["elapsed_seconds"] = time.monotonic() - started
    write_json(args.out, output)
    print("RECURRENCE_OBSERVATION_COMPLETE", output["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
