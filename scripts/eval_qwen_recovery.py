#!/usr/bin/env python3
"""Evaluate the unchanged donor or an audited initialization on full development data.

Uses exactly the document/window/byte policy used after recovery training. This is
development evaluation, not an untouched benchmark or cached-decoding equivalence gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts import recover_qwen as recovery  # noqa: E402


class DonorEvaluationAdapter(torch.nn.Module):
    """Forward-only donor interface; no cache survives a document window."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, *, return_mtp=False):
        if return_mtp:
            raise ValueError("the unchanged donor has no Prophet MTP head")
        return self.model(input_ids=ids, use_cache=False, return_dict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "validation", "data-audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("donor", "initialization"), required=True)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loop-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if (
        args.seq_len < 2
        or min(args.batch_size, args.loop_k, args.chunk_tokens, args.cpu_threads) < 1
    ):
        parser.error("invalid sequence length, batch size, depth, chunk size or thread count")
    if args.arm == "initialization" and (args.initialization is None or args.audit is None):
        parser.error("initialization evaluation requires --initialization and --audit")
    if args.arm == "donor" and (args.initialization is not None or args.audit is not None):
        parser.error("donor evaluation does not accept initialization artifacts")
    if args.out.exists():
        raise FileExistsError("preserve the previous evaluation")
    data_audit = json.loads(args.data_audit.read_bytes())
    validation_sha = recovery.digest(args.validation)
    if (
        not data_audit.get("complete")
        or data_audit["splits"]["validation"]["sha256"] != validation_sha
    ):
        raise ValueError("validation differs from the prepared development corpus")
    documents = list(recovery.read_documents(args.validation))
    if len(documents) != data_audit["splits"]["validation"]["retained_documents"]:
        raise ValueError("validation document count differs from the data audit")
    source_config, tokenizer = recovery.load_source(args.source)
    device = torch.device(args.device)
    donor_dtype = recovery.recovery_precision(args.precision, device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this evaluation protocol supports CPU or CUDA only")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    import transformers
    from transformers import AutoModelForCausalLM

    initialization_sha, config, loop_k = None, None, None
    if args.arm == "donor":
        if args.seq_len > source_config["max_position_embeddings"]:
            raise ValueError("sequence length exceeds donor context")
        model = DonorEvaluationAdapter(
            AutoModelForCausalLM.from_pretrained(
                args.source,
                local_files_only=True,
                trust_remote_code=False,
                dtype=donor_dtype,
                attn_implementation="sdpa",
            )
        )
    else:
        payload, initialization_sha = recovery.load_initialization(args.initialization, args.audit)
        cfg = recovery.recovery_config(payload["config"], args.loop_k, args.seq_len)
        if cfg.frontend.vocab_size != tokenizer.vocab_size:
            raise ValueError("student vocabulary differs from the pinned donor")
        if device.type == "cuda" and "gdn" in cfg.recurrent.core_pattern and not HAS_FLA:
            raise RuntimeError("CUDA hybrid evaluation requires the pinned FLA kernel")
        model = ProphetModel(cfg)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        config, loop_k = cfg.to_dict(), args.loop_k
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    def progress_documents():
        for index, document in enumerate(documents):
            if index % 20 == 0:
                print(
                    "EVALUATION_PROGRESS",
                    index,
                    len(documents),
                    round(time.perf_counter() - started, 2),
                    flush=True,
                )
            yield document

    evaluation = evaluate_documents(
        model,
        progress_documents(),
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        loss_chunk_tokens=args.chunk_tokens,
        loop_k=loop_k,
        precision=args.precision,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "complete": True,
        "arm": args.arm,
        "trained_steps": 0,
        "donor_revision": recovery.REVISION,
        "donor_weights_sha256": recovery.WEIGHTS_SHA256,
        "donor_config": source_config,
        "initialization_sha256": initialization_sha,
        "initialization_audit_sha256": recovery.digest(args.audit) if args.audit else None,
        "config": config,
        "tokenizer": tokenizer.fingerprint(),
        "validation_sha256": validation_sha,
        "data_audit_sha256": recovery.digest(args.data_audit),
        "evaluation": evaluation,
        "unique_parameters": sum(p.numel() for p in model.parameters()),
        "evaluation_seconds": time.perf_counter() - started,
        "timing_scope": "full document evaluation including tokenization, CE and any kernel compilation",
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": str(device),
            "cpu_threads": args.cpu_threads,
            "cuda": torch.version.cuda,
            "fla": version("fla-core") if device.type == "cuda" and HAS_FLA else None,
            "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
            "requested_precision": args.precision,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "scope": "full recovery development validation; not an untouched test or cached-decoding gate",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recovery.write_report(args.out, report)
    print(
        "EVALUATION_COMPLETE",
        args.arm,
        evaluation["nats_per_token"],
        evaluation["bits_per_byte"],
        flush=True,
    )


if __name__ == "__main__":
    main()
