"""Exact real-shape BF16 equality before training the zero-initialized adapter."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts import adapt_r04_depth as engine  # noqa: E402
from scripts.adapt_r04_reinjection import PLAN_DIR, configuration, load_weights  # noqa: E402


def compare_initial(control, candidate, ids, depths=(4, 6)):
    results = {}
    control.eval()
    candidate.eval()
    with (
        torch.no_grad(),
        torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda"),
    ):
        for depth in depths:
            left = control(ids, loop_k=depth, return_mtp=False)
            right = candidate(ids, loop_k=depth, return_mtp=False)
            results[str(depth)] = {
                "hidden_equal": torch.equal(left.hidden, right.hidden),
                "logits_equal": torch.equal(left.logits, right.logits),
                "finite": bool(
                    torch.isfinite(left.hidden).all() and torch.isfinite(left.logits).all()
                ),
            }
            del left, right
    return {"passed": all(all(r.values()) for r in results.values()), "depths": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine.require(not args.out.exists(), "use a fresh gate report")
    numerics = engine.configure_numerics("cuda")
    runtime = engine.runtime_for("cuda")
    plan = json.loads((PLAN_DIR / "protocol.json").read_bytes())
    state, parent, _ = engine.load_parent(args.parent_run, plan)
    engine.require(runtime == parent["runtime"], "runtime differs from parent")
    engine.require(engine.verify_pilot(args.corpus) == parent["data"], "pilot differs from parent")
    engine.require(
        engine.sha256(args.parent_run / f"evaluation-step-{state['step']:06d}.json")
        == plan["parent_report_sha256"],
        "parent report bytes differ",
    )
    tokenizer = ProphetTokenizer.load(args.corpus / "tokenizer.json")
    source = TokenisedSource(
        LocalTextSource.from_root(args.corpus / "train", "fineweb-edu", 1.0), tokenizer
    )
    loader = StreamingLoader(
        [source], seq_len=plan["seq_len"], batch_size=plan["batch_size"], seed=parent["seed"]
    )
    loader.load_state(state["loader"])
    ids = torch.tensor(next(iter(loader.batches(1))), dtype=torch.long, device="cuda")
    models = []
    for arm in ("fixed_sum", "learned_mix"):
        cfg = configuration(parent["config"], arm)
        model = ProphetModel(cfg)
        load_weights(model, state["model"])
        models.append(model.cuda())
    del state
    torch.cuda.reset_peak_memory_stats()
    report = compare_initial(*models, ids)
    report.update(
        protocol="r04-input-adapter-initial-equality-v1",
        plan_sha256=engine.sha256(PLAN_DIR / "protocol.json"),
        parent_checkpoint=plan["parent_checkpoint"],
        code={**engine.code_identity(), "gate_sha256": engine.sha256(Path(__file__))},
        numerical_policy=numerics,
        runtime=runtime,
        input_shape=list(ids.shape),
        input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        loader_step_after=loader.state().step,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope="Exact hidden and full-logit equality on the same next packed training batch at k4 and k6, CUDA BF16 autocast. No optimizer updates. Each arm must separately reproduce all original k4 validation documents before training.",
    )
    engine.write_json(args.out, report)
    engine.require(report["passed"], "zero adapter changes initial predictions")
    print("R04_INPUT_ADAPTER_INITIAL_EQUAL", report["input_shape"], flush=True)


if __name__ == "__main__":
    main()
