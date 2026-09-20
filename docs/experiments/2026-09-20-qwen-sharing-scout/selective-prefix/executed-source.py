"""CPU scout for selective sharing versus a shallower unchanged-weight control.

No architecture adoption, training, cached inference or benchmark selection.
All variants and development prefixes are declared before scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_recovery_cache_suite import select_prefixes  # noqa: E402
from scripts.probe_qwen_relaxed import PROJECTIONS, score_prefixes  # noqa: E402
from scripts.recover_qwen import (  # noqa: E402
    REVISION,
    WEIGHTS_SHA256,
    digest,
    load_source,
    read_documents,
)


@contextmanager
def selective_sharing(layers, family):
    """Temporarily share cyclic means, restoring the original module objects."""
    if len(layers) != 28 or family not in ("self_attn", "mlp"):
        raise ValueError("requires 28 layers and a known projection family")
    originals = []
    try:
        for slot in range(4):
            group = list(range(4 + slot, 24, 4))
            for path in PROJECTIONS:
                if not path.startswith(family + "."):
                    continue
                modules = [layers[i].get_submodule(path) for i in group]
                if any(not isinstance(m, nn.Linear) or m.bias is not None for m in modules):
                    raise ValueError("requires bias-free donor projections")
                base = torch.stack([m.weight.detach() for m in modules]).mean(0)
                replacement = nn.Linear(base.shape[1], base.shape[0], bias=False)
                replacement.weight = nn.Parameter(base)
                for index, module in zip(group, modules, strict=True):
                    parent, name = path.rsplit(".", 1)
                    owner = layers[index].get_submodule(parent)
                    originals.append((owner, name, module))
                    setattr(owner, name, replacement)
                assert len({id(layers[i].get_submodule(path)) for i in group}) == 1
        yield
    finally:
        for owner, name, module in originals:
            setattr(owner, name, module)


@contextmanager
def remove_middle_eight(model):
    """Keep original layers 0..11 and 20..27; no repeated layers or copied weights."""
    original = model.model.layers
    if len(original) != 28 or model.config.num_hidden_layers != 28:
        raise ValueError("requires the original 28-layer donor")
    indices = list(range(12)) + list(range(20, 28))
    prior_indices = [layer.self_attn.layer_idx for layer in original]
    try:
        model.model.layers = nn.ModuleList([original[i] for i in indices])
        model.config.num_hidden_layers = len(indices)
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = index
        yield
    finally:
        model.model.layers = original
        model.config.num_hidden_layers = 28
        for layer, index in zip(original, prior_indices, strict=True):
            layer.self_attn.layer_idx = index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior results")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config, tokenizer = load_source(args.source)
    validation_sha = digest(args.validation)
    if validation_sha != "06903e9925fb4e2ceed27eedc09f5d4b33b32e3e2ce1bfe6e1deed62a491e8af":
        raise ValueError("requires frozen recovery development data")
    prefixes = [
        (sha, ids[:512])
        for sha, ids in select_prefixes(read_documents(args.validation), tokenizer, 16, [512])
    ]
    variants = {
        "share-mlp": 445054976,
        "share-attention": 495386624,
        "remove-middle-eight": 470202368,
    }
    result = {
        "complete": False,
        "protocol": "qwen-selective-sharing-prefix-v1",
        "script_sha256": digest(Path(__file__)),
        "scoring_script_sha256": digest(Path(__file__).with_name("probe_qwen_relaxed.py")),
        "donor_revision": REVISION,
        "donor_weights_sha256": WEIGHTS_SHA256,
        "validation_sha256": validation_sha,
        "source_config": config,
        "tokenizer": tokenizer.fingerprint(),
        "device": "cpu",
        "precision": "float32",
        "torch": str(torch.__version__),
        "threads": 2,
        "variants_declared_before_scores": variants,
        "selection": "First 16 distinct eligible development-document SHA256 values ascending; first 512 tokens, no EOS.",
        "scope": "Development-only untrained initialization scout, same prefixes as the residual probe. Cyclic middle-layer sharing preserves separate original norms; pruning is a fixed-depth control. Not an adopted recurrent architecture, cache test, capacity benchmark, training or deployment claim.",
        "results": {},
    }
    args.out.mkdir(parents=True)

    def record():
        (args.out / "report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    record()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    original_parameters = list(model.parameters())
    assert sum(p.numel() for p in original_parameters) == 596049920
    result["results"]["donor"] = score_prefixes(model, prefixes)
    record()
    print("SELECTIVE_DONOR", result["results"]["donor"]["nats_per_token"], flush=True)
    for name, expected in variants.items():
        context = (
            remove_middle_eight(model)
            if name == "remove-middle-eight"
            else selective_sharing(
                model.model.layers, "mlp" if name == "share-mlp" else "self_attn"
            )
        )
        with context:
            count = sum(p.numel() for p in model.parameters())
            assert count == expected and count <= 500000000, (name, count, expected)
            evaluation = score_prefixes(model, prefixes)
            evaluation["registered_unique_parameters"] = count
            evaluation["allocation_scope"] = (
                "Original donor modules retained for restoration; this is registered model size, not resident memory."
            )
        assert [id(p) for p in model.parameters()] == [id(p) for p in original_parameters]
        result["results"][name] = evaluation
        record()
        print("SELECTIVE_VARIANT", name, count, evaluation["nats_per_token"], flush=True)
    result["complete"] = True
    record()
    print("SELECTIVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
