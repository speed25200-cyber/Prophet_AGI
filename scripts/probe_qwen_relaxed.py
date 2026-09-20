"""CPU research probe: cyclic shared weights with depth-specific SVD residuals.

This deliberately keeps the donor's serial attention, norms, positions and depth.
It isolates a compression mechanism; it is not an adopted Prophet architecture,
training recipe, variable-depth model or deployable exported checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_recovery_cache_suite import select_prefixes  # noqa: E402
from scripts.recover_qwen import (  # noqa: E402
    REVISION,
    WEIGHTS_SHA256,
    digest,
    load_source,
    read_documents,
)

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class ResidualLinear(nn.Module):
    """One shared base plus one depth's rank-limited residual, with no biases.

    Capacity remains allocated during rank sweeps. Active-rank parameter counts
    are therefore accounting projections, not measured resident-memory savings.
    The active rank is experiment state and must be set explicitly after loading.
    """

    def __init__(self, base: nn.Parameter, target: Tensor, capacity: int):
        super().__init__()
        if base.ndim != 2 or target.shape != base.shape or not 0 < capacity <= min(base.shape):
            raise ValueError("incompatible base/target shape or residual capacity")
        if base.dtype != torch.float32 or target.dtype != torch.float32:
            raise ValueError("the research initialization requires FP32")
        if not torch.isfinite(base).all() or not torch.isfinite(target).all():
            raise ValueError("nonfinite source matrix")
        self.base = base
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(target - base, full_matrices=False)
            scale = s[:capacity].sqrt()
            self.up = nn.Parameter((u[:, :capacity] * scale).contiguous())
            self.down = nn.Parameter((scale[:, None] * vh[:capacity]).contiguous())
        self.capacity = capacity
        self.rank = capacity

    def set_rank(self, rank: int):
        if not isinstance(rank, int) or not 0 <= rank <= self.capacity:
            raise ValueError("active rank must fit the allocated residual")
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        output = F.linear(x, self.base)
        if self.rank:
            output = output + F.linear(F.linear(x, self.down[: self.rank]), self.up[:, : self.rank])
        return output


def install_residuals(model, capacity: int):
    """Modify only the twenty middle layers; preserve the eight outer layers."""
    layers = model.model.layers
    if len(layers) != 28:
        raise ValueError("this frozen Qwen probe requires 28 layers")
    modules = []
    for slot in range(4):
        group = list(range(4 + slot, 24, 4))
        for path in PROJECTIONS:
            originals = [layers[index].get_submodule(path) for index in group]
            if any(not isinstance(m, nn.Linear) or m.bias is not None for m in originals):
                raise ValueError("expected unmodified bias-free donor linear modules")
            base = nn.Parameter(torch.stack([m.weight.detach() for m in originals]).mean(0))
            for index, original in zip(group, originals, strict=True):
                replacement = ResidualLinear(base, original.weight.detach(), capacity)
                parent, name = path.rsplit(".", 1)
                setattr(layers[index].get_submodule(parent), name, replacement)
                modules.append(replacement)
            assert len({id(layers[index].get_submodule(path).base) for index in group}) == 1
            print("FACTORIZED", len(modules), 140, flush=True)
    return modules


def score_prefixes(model, prefixes):
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for document_sha, ids in prefixes:
            batch = torch.tensor([ids], dtype=torch.long)
            logits = model(batch, use_cache=False).logits[:, :-1].float()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite prefix logits")
            nats = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
                )
            )
            rows.append(
                {
                    "document_sha256": document_sha,
                    "input_ids_sha256": hashlib.sha256(batch.numpy().tobytes()).hexdigest(),
                    "targets": len(ids) - 1,
                    "total_nats": nats,
                }
            )
    total_nats = sum(r["total_nats"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    return {
        "nats_per_token": total_nats / targets,
        "total_nats": total_nats,
        "targets": targets,
        "documents": rows,
        "seconds": time.perf_counter() - started,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--documents", type=int, default=16)
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 64, 128, 256])
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior probe results")
    if args.documents < 1 or not 2 <= args.length <= 512:
        p.error("require positive documents and a prefix length of 2 to 512")
    if args.ranks != sorted(set(args.ranks)) or args.ranks[0] != 0 or not 0 < args.ranks[-1] <= 256:
        p.error("use unique ascending ranks starting at zero, capacity at most 256")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    source_config, tokenizer = load_source(args.source)
    validation_sha = digest(args.validation)
    if validation_sha != "06903e9925fb4e2ceed27eedc09f5d4b33b32e3e2ce1bfe6e1deed62a491e8af":
        raise ValueError("probe requires the frozen recovery development corpus")
    selected = select_prefixes(
        read_documents(args.validation), tokenizer, args.documents, [args.length]
    )
    prefixes = [(sha, ids[: args.length]) for sha, ids in selected]
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    donor_count = sum(p.numel() for p in model.parameters())
    assert donor_count == 596049920
    args.out.mkdir(parents=True)
    result = {
        "complete": False,
        "protocol": "qwen-serial-depth-residual-probe-v1",
        "source_revision": REVISION,
        "source_weights_sha256": WEIGHTS_SHA256,
        "source_config": source_config,
        "validation_sha256": validation_sha,
        "tokenizer": tokenizer.fingerprint(),
        "ranks_declared_before_scores": args.ranks,
        "prefix_count": args.documents,
        "prefix_length": args.length,
        "selection": "first distinct eligible development-document SHA256 values in ascending order, no EOS",
        "device": "cpu",
        "torch": str(torch.__version__),
        "threads": 2,
        "precision": "float32",
        "script_sha256": digest(Path(__file__)),
        "donor_parameters": donor_count,
        "results": {},
        "scope": "Exploratory deterministic development prefixes; same donor norms, positional policy and fixed serial depth. All middle linear weights share cyclic bases and have separate SVD residuals. No reinjection, GDN, auxiliary heads, training, variable depth, capability score or deployment claim. This prefix CE is not comparable to the full-document recovery CE.",
    }

    def record():
        (args.out / "report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    record()
    result["results"]["donor"] = score_prefixes(model, prefixes)
    record()
    print("PROBE_DONOR", result["results"]["donor"]["nats_per_token"], flush=True)
    start = time.perf_counter()
    modules = install_residuals(model, args.ranks[-1])
    result["factorization_seconds"] = time.perf_counter() - start
    assert len(modules) == 140
    resident = sum(p.numel() for p in model.parameters())
    residual_capacity = sum(m.up.numel() + m.down.numel() for m in modules)
    base_unique = resident - residual_capacity
    assert base_unique == 344391680 and resident <= 500000000
    result["allocated_unique_parameters"] = resident
    result["allocated_residual_capacity"] = args.ranks[-1]
    result["unique_base_and_preserved_norm_parameters"] = base_unique
    result["middle_base_parameter_aliases_verified"] = 28
    for rank in args.ranks:
        for module in modules:
            module.set_rank(rank)
        evaluation = score_prefixes(model, prefixes)
        active = base_unique + sum(rank * (m.up.shape[0] + m.down.shape[1]) for m in modules)
        evaluation["active_rank_parameter_accounting"] = active
        evaluation["allocation_scope"] = (
            "Capacity remains allocated at the maximum rank for this sweep."
        )
        result["results"][str(rank)] = evaluation
        record()
        print("PROBE_RANK", rank, active, evaluation["nats_per_token"], flush=True)
    result["complete"] = True
    record()
    print("PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
