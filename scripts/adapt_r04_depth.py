"""Matched R04 depth-policy warm starts, with a separate real-shape CUDA preflight.

The published plan fixes weights, data, schedule and decision criteria. This
driver does not change model topology or resume the parent's optimizer schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from prophet.train.loss import compute_loss  # noqa: E402
from scripts.run_r04_pilot import sha256, verify_pilot, write_json  # noqa: E402

PLAN_DIR = ROOT / "docs/experiments/2026-09-20-r04-depth-adaptation-plan"
DEPTHS = [4, 1, 2, 6, 8]


@dataclass(frozen=True)
class Experiment:
    """An explicit component experiment using the same audited training engine."""

    plan_dir: Path
    arms: tuple[str, str]
    protocol: str
    config_factory: Callable
    load_weights: Callable
    entrypoint: Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_numerics(device):
    """Bind the deterministic warm start to an explicit numerical policy.

    This is a new experiment contract. The historical R04 checkpoints are used
    only as warm-start weights; previous adaptation outputs cannot be resumed
    under changed arithmetic. External CUDA kernels still need real-shape gates.
    """
    cuda = torch.device(device).type == "cuda"
    if cuda:
        require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python",
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if cuda:
        # Keep the original R04 precision choices; select deterministic kernels.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return {
        "name": "r04-strict-determinism-v1",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if cuda else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "sdpa_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        }
        if cuda
        else None,
    }


def runtime_for(device):
    cuda = torch.device(device).type == "cuda"
    if cuda:
        require(torch.cuda.is_available() and HAS_FLA, "recorded CUDA/FLA runtime required")
        require(os.environ.get("TRITON_F32_DEFAULT") == "tf32x3", "set TRITON_F32_DEFAULT=tf32x3")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda if cuda else None,
        "fla": version("fla-core") if cuda else None,
        "triton": version("triton") if cuda else None,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "triton_f32_default": "tf32x3" if cuda else None,
    }


def code_identity():
    subprocess.run(["git", "diff", "HEAD", "--exit-code"], cwd=ROOT, check=True)
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver_sha256": sha256(Path(__file__)),
    }


def adaptation_config(parent_config, arm):
    require(arm in ("fixed4", "uniform2to6"), "unknown adaptation arm")
    cfg = ProphetConfig.from_dict(parent_config)
    r = cfg.recurrent
    require(
        r.enabled and r.train_loop_min == r.train_loop_max == r.default_loop_k == 4,
        "parent must be the fixed-four-loop model",
    )
    require(
        not r.token_depth
        and r.halting == "none"
        and not r.iteration_readout
        and not r.iteration_embedding
        and not cfg.heads.action_head
        and not cfg.heads.confidence_head
        and not cfg.heads.n_multi_token_predict,
        "plain R04 policy without auxiliary heads required",
    )
    r.train_loop_dist = "uniform"
    r.train_loop_min = 4 if arm == "fixed4" else 2
    r.train_loop_max = 4 if arm == "fixed4" else 6
    r.truncated_backprop_steps = 6
    cfg.validate()
    return cfg


def load_parent(run, plan):
    expected = plan["parent_checkpoint"]
    protocol = json.loads((run / "protocol.json").read_bytes())
    evaluation = json.loads((run / f"evaluation-step-{expected['step']:06d}.json").read_bytes())
    require(
        evaluation["run_protocol"] == protocol and evaluation["checkpoint"] == expected,
        "parent publication differs from the planned checkpoint",
    )
    require(
        expected["slot"] in (0, 1) and protocol["variant"] == "loop", "invalid parent slot or arm"
    )
    manifest = json.loads((run / "checkpoints/manifest.json").read_bytes())
    require(expected in manifest["checkpoints"], "parent checkpoint rotated")
    path = run / f"checkpoints/ckpt_slot{expected['slot']}.pt"
    require(
        path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
        "parent checkpoint bytes differ",
    )
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(state["step"] == evaluation["step"] == expected["step"], "parent step differs")
    require(json.loads(json.dumps(state["config"])) == protocol["config"], "parent config differs")
    require(
        state["tokens_seen"]
        == evaluation["train_tokens"]
        == expected["step"] * protocol["batch_size"] * protocol["seq_len"],
        "parent token count differs",
    )
    require(
        state["skipped_nonfinite"] == 0
        and all(torch.isfinite(t).all() for t in state["model"].values()),
        "nonfinite parent or skipped parent updates",
    )
    return state, protocol, evaluation


class DepthTrainer(Trainer):
    """Record the depth actually used and bind its history to the checkpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require(
            self.cfg.grad_accum_steps == 1, "this depth experiment uses one microbatch per step"
        )
        self.depth_history = []

        def observe(model, inputs, output):
            if model.training:
                self.depth_history.append(output.loop_k)

        self.depth_hook = self.model.register_forward_hook(observe)

    def state_dict(self):
        require(
            len(self.depth_history) == self.step, "depth history does not match completed steps"
        )
        return {**super().state_dict(), "adaptation_depth_history": list(self.depth_history)}

    def load_state_dict(self, state):
        history = state.get("adaptation_depth_history")
        r = self.model_config.recurrent
        require(
            isinstance(history, list)
            and len(history) == state["step"]
            and all(type(k) is int and r.train_loop_min <= k <= r.train_loop_max for k in history),
            "invalid recorded depth history",
        )
        super().load_state_dict(state)
        self.depth_history = list(history)


def profile_max_depth(trainer, *, steps=3):
    """Destructive on this disposable preflight trainer; never used in a train run."""
    require(trainer.device.type == "cuda", "real-shape preflight requires CUDA")
    trainer.model.train()
    torch.cuda.reset_peak_memory_stats()
    durations, losses = [], []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        ids = trainer._batch()
        for opt in trainer.optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = trainer.model(ids, loop_k=6)
        terms = compute_loss(
            output,
            ids,
            project=trainer.model._project,
            loss_chunk_tokens=trainer.cfg.loss_chunk_tokens,
            z_loss_weight=trainer.cfg.z_loss_weight,
            mtp_weight=trainer.cfg.mtp_weight,
            confidence_weight=trainer.cfg.confidence_weight,
        )
        require(torch.isfinite(terms.total).item(), "nonfinite preflight loss")
        terms.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), 1.0, error_if_nonfinite=True
        )
        for opt in trainer.optimizers:
            opt.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append({"loss": float(terms.total.detach()), "gradient_norm": float(norm)})
    peak = torch.cuda.max_memory_allocated()
    capacity = torch.cuda.get_device_properties(0).total_memory
    require(
        all(torch.isfinite(p).all() for p in trainer.model.parameters()),
        "nonfinite weights after preflight optimizer updates",
    )
    return {
        "passed": peak < 0.9 * capacity,
        "maximum_loop_k": 6,
        "full_backprop": True,
        "actual_depths": trainer.depth_history,
        "steps": steps,
        "step_seconds": durations,
        "losses": losses,
        "peak_allocated_bytes": peak,
        "device_capacity_bytes": capacity,
        "memory_limit_fraction": 0.9,
        "scope": "Disposable model/optimizer, real next training batches; original weights and loader untouched. Allocated peak excludes non-PyTorch allocations.",
    }


def main(*, experiment: Experiment | None = None):
    plan_dir = PLAN_DIR if experiment is None else experiment.plan_dir
    arms = ("fixed4", "uniform2to6") if experiment is None else experiment.arms
    config_factory = adaptation_config if experiment is None else experiment.config_factory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=arms, required=True)
    parser.add_argument("--mode", choices=["preflight", "train"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-session-steps", type=int, default=64)
    parser.add_argument("--session-minutes", type=float, default=20)
    args = parser.parse_args()
    manifest_path = args.out / "adaptation.json"
    if not manifest_path.exists():
        require(
            not args.out.exists() or not any(args.out.iterdir()),
            "new adaptation needs an empty output directory",
        )
    require(
        args.max_session_steps > 0
        and math.isfinite(args.session_minutes)
        and args.session_minutes > 0,
        "positive session limits required",
    )
    numerical_policy = configure_numerics(args.device)
    plan = json.loads((plan_dir / "protocol.json").read_bytes())
    source_identity = code_identity()
    if experiment is not None:
        require(plan["experiment"] == experiment.protocol, "experiment protocol differs")
        source_identity["entrypoint_sha256"] = sha256(experiment.entrypoint)
    state, parent, original = load_parent(args.parent_run, plan)
    if "parent_report_sha256" in plan:
        require(
            sha256(args.parent_run / f"evaluation-step-{state['step']:06d}.json")
            == plan["parent_report_sha256"],
            "parent report bytes differ",
        )
    provenance = verify_pilot(args.corpus)
    require(provenance == parent["data"], "pilot data differs from parent")
    runtime = runtime_for(args.device)
    require(runtime == parent["runtime"], "runtime differs from original R04")
    cfg = config_factory(parent["config"], args.arm)
    require(
        cfg.to_dict()
        == ProphetConfig.from_dict(
            json.loads((plan_dir / f"{args.arm}.json").read_bytes())
        ).to_dict(),
        "configuration differs from plan",
    )
    steps, batch, length = plan["steps_each"], plan["batch_size"], plan["seq_len"]
    require(
        (batch, length) == (parent["batch_size"], parent["seq_len"]),
        "parent loader shape must be retained",
    )
    require(
        state["tokens_seen"] + steps * batch * length
        <= 4 * plan["original_training_corpus_tokens"],
        "four-pass corpus limit exceeded",
    )
    tokenizer = ProphetTokenizer.load(args.corpus / "tokenizer.json")
    source = TokenisedSource(
        LocalTextSource.from_root(args.corpus / "train", "fineweb-edu", 1.0), tokenizer
    )
    loader = StreamingLoader([source], seq_len=length, batch_size=batch, seed=parent["seed"])
    loader.load_state(state["loader"])
    parent_loader_sha = hashlib.sha256(
        json.dumps(state["loader"], sort_keys=True).encode()
    ).hexdigest()
    parent_loader_step = state["loader"]["step"]
    torch.manual_seed(parent["seed"])
    model = ProphetModel(cfg)
    if experiment is None:
        model.load_state_dict(state["model"], strict=True)
    else:
        experiment.load_weights(model, state["model"])
    parameters = plan["parameters_each"]
    if isinstance(parameters, dict):
        parameters = parameters[args.arm]
    require(
        sum(p.numel() for p in model.parameters()) == parameters,
        "parameter count differs",
    )
    del state
    identity = {
        "protocol": "r04-depth-adaptation-v2" if experiment is None else experiment.protocol,
        "numerical_policy": numerical_policy,
        "plan_sha256": sha256(plan_dir / "protocol.json"),
        "parent_checkpoint": plan["parent_checkpoint"],
        "parent_loader_sha256": parent_loader_sha,
        "data": provenance,
        "runtime": runtime,
        "code": source_identity,
        "arm": args.arm,
        "mode": args.mode,
        "evaluation_depths": DEPTHS,
        "warm_start": "parent weights and loader, new optimizer/schedule, RNG reset to parent seed",
    }
    recipe = plan["recipe"]
    train_cfg = TrainConfig(
        total_steps=steps,
        batch_size=batch,
        seq_len=length,
        peak_lr_muon=recipe["muon_lr"],
        peak_lr_adamw=recipe["adamw_lr"],
        weight_decay=recipe["weight_decay"],
        grad_clip=recipe["grad_clip"],
        loss_chunk_tokens=parent["loss_chunk_tokens"],
        max_consecutive_nonfinite=1,
        checkpoint_dir=str(args.out / "checkpoints"),
        checkpoint_every=128,
        log_every=1,
        max_wall_seconds=args.session_minutes * 60,
        seed=parent["seed"],
        device=args.device,
    )

    def log(metrics):
        record = asdict(metrics)
        record["loop_k"] = trainer.depth_history[-1]
        record["loader_step"] = trainer.loader.state().step
        with (args.out / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if metrics.step % 16 == 0:
            print(metrics.format(), "k", record["loop_k"], flush=True)

    trainer = DepthTrainer(
        model,
        loader,
        train_cfg,
        model_config=cfg,
        tokenizer=tokenizer,
        run_identity=identity,
        on_log=log,
    )
    torch.manual_seed(parent["seed"])
    manifest = json.loads(
        json.dumps(
            {
                "identity": identity,
                "config": cfg.to_dict(),
                "training_contract": trainer.training_contract(),
            }
        )
    )
    resumed_meta = None
    if manifest_path.exists():
        require(
            json.loads(manifest_path.read_bytes()) == manifest,
            "output belongs to another adaptation experiment",
        )
        require(
            args.mode == "train" and trainer.ckpt.has_checkpoint(),
            "no resumable adaptation checkpoint",
        )
        restored, resumed_meta = trainer.ckpt.load_latest()
        trainer.load_state_dict(restored)
        del restored
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        if args.mode == "train":
            trainer.ckpt.save(trainer.state_dict(), 0)

    require(
        0 <= trainer.step <= steps and trainer.tokens_seen == trainer.step * batch * length,
        "adaptation checkpoint step or token count differs",
    )
    require(
        trainer.loader.state().step == parent_loader_step + trainer.step * batch,
        "adaptation loader no longer continues the parent cursor",
    )

    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)

    def evaluate(k):
        return evaluate_documents(
            model,
            validation.open(),
            tokenizer,
            device=args.device,
            seq_len=length,
            batch_size=batch,
            loss_chunk_tokens=train_cfg.loss_chunk_tokens,
            loop_k=k,
        )

    if trainer.step == 0:
        baseline = evaluate(4)
        same_inputs = [
            [d[k] for k in ("sha256", "scored_tokens", "scored_bytes")]
            for d in baseline["documents"]
        ] == [
            [d[k] for k in ("sha256", "scored_tokens", "scored_bytes")]
            for d in original["documents"]
        ]
        passed = same_inputs and math.isclose(
            baseline["nats_per_token"], original["nats_per_token"], rel_tol=0, abs_tol=1e-5
        )
        write_json(
            args.out / "baseline.json",
            {"passed": passed, "evaluation": baseline, "identity": identity},
        )
        require(passed, "initial k4 does not reproduce the parent validation")
    else:
        baseline = json.loads((args.out / "baseline.json").read_bytes())
        require(
            baseline["passed"] and baseline["identity"] == identity,
            "missing successful initial baseline",
        )

    if args.mode == "preflight":
        profile = profile_max_depth(trainer)
        write_json(args.out / "preflight.json", {**profile, "identity": identity})
        require(profile["passed"], "k6 memory exceeds preflight limit")
        print("DEPTH_PREFLIGHT_PASSED", profile, flush=True)
        return

    destination = args.out / f"evaluation-step-{steps:06d}.json"
    unfinished = None
    if trainer.step == steps and destination.exists():
        existing = json.loads(destination.read_bytes())
        require(
            existing["identity"] == identity
            and existing["step"] == steps
            and existing["checkpoint"] == resumed_meta.to_dict()
            and existing["depth_history"] == trainer.depth_history,
            "completed report identity differs",
        )
        if existing["complete"]:
            require(
                set(existing["results"]) == {str(k) for k in DEPTHS},
                "incomplete final depth results",
            )
            print("DEPTH_ADAPTATION_ALREADY_COMPLETE", flush=True)
            return
        unfinished = existing
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    old_handlers = {
        sig: signal.signal(sig, lambda *_: setattr(trainer, "stop_requested", True))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if trainer.step < steps:
            trainer.train(max_steps=args.max_session_steps)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    require(trainer.skipped_nonfinite == 0, "adaptation has skipped updates")
    checkpoint = (
        resumed_meta
        if resumed_meta is not None and resumed_meta.step == trainer.step
        else trainer.ckpt.save(trainer.state_dict(), trainer.step)
    )
    training_peak = torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None
    report = unfinished or {
        "complete": False,
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "checkpoint": checkpoint.to_dict(),
        "identity": identity,
        "depth_history": trainer.depth_history,
        "depth_counts": dict(Counter(trainer.depth_history)),
        "skipped_nonfinite": trainer.skipped_nonfinite,
        "loader_step": trainer.loader.state().step,
        "session_training_peak_allocated_bytes": training_peak,
        "results": {},
    }
    target = args.out / f"evaluation-step-{trainer.step:06d}.json"
    require(
        unfinished is not None or not target.exists(), "preserve previous adaptation evaluation"
    )
    for k in DEPTHS if trainer.step == steps else [4]:
        if str(k) not in report["results"]:
            report["results"][str(k)] = evaluate(k)
        write_json(target, report)
    report["complete"] = trainer.step == steps
    write_json(target, report)
    print("DEPTH_ADAPTATION_SESSION_COMPLETE", trainer.step, report["depth_counts"], flush=True)


if __name__ == "__main__":
    main()
