"""CPU full-donor identity/cache/gradient gate; not a quality or training result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.convert.retained_recurrence import (  # noqa: E402
    RetainedRecurrenceConfig,
    RetainedRecurrentQwen,
)

DONOR = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHTS = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TRAIN_SHA = "41b7c3179322e87174a4e252028b261c857327def204a456163c08e733e87570"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weights_digest(model):
    return {
        name: hashlib.sha256(parameter.detach().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in model.named_parameters()
    }


def compare(actual, expected, *, exact=False):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    if exact:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    return {
        "exact": torch.equal(actual, expected),
        "max_absolute_error": float((actual - expected).abs().max()),
        "argmax_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    root = Path(__file__).resolve().parent.parent
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("run from a clean frozen checkout")
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    source_hashes = {
        name: digest(root / name)
        for name in (
            "prophet/convert/retained_recurrence.py",
            "scripts/gate_retained_recurrence.py",
        )
    }
    manifest_path = args.donor / "donor-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["complete"] and manifest["model"] == DONOR and manifest["revision"] == REVISION
    assert manifest["license"] == "apache-2.0" and manifest["gated"] is False
    for name, metadata in manifest["artifacts"].items():
        path = (args.donor / name).resolve()
        assert path.is_relative_to(args.donor.resolve())
        assert path.stat().st_size == metadata["bytes"] and digest(path) == metadata["sha256"]
    assert digest(args.donor / "model.safetensors") == WEIGHTS
    assert digest(args.train_file) == TRAIN_SHA
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers import (
        __version__ as transformers_version,
    )

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.donor, local_files_only=True, trust_remote_code=False
    )
    prefixes = []
    with args.train_file.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if len(ids) >= 32:
                prefixes.append((row["content_sha256"], ids[:32]))
            if len(prefixes) == 4:
                break
    assert len(prefixes) == 4
    tokens = torch.tensor([ids for _, ids in prefixes])
    print("LOADING_VERIFIED_DONOR", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.donor,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    original_weights = weights_digest(base)
    with torch.no_grad():
        reference = base(tokens, use_cache=False).logits
    model = RetainedRecurrentQwen(base, RetainedRecurrenceConfig()).eval()
    resident = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert resident <= 500_000_000
    with torch.no_grad():
        identity = compare(model(tokens).logits, reference, exact=True)
    print("FULL_DONOR_IDENTITY", identity, resident, trainable, flush=True)
    cache_results = []
    with torch.no_grad():
        for loops in (1, 2, 4):
            sample = tokens[:, :16]
            expected = model(sample, loop_k=loops).logits
            first = model(sample[:, :12], loop_k=loops, use_cache=True)
            cache = first.past_key_values
            pieces = [first.logits]
            for position in range(12, 16):
                pieces.append(
                    model(
                        sample[:, position : position + 1],
                        loop_k=loops,
                        use_cache=True,
                        past_key_values=cache,
                    ).logits
                )
            record = {
                "loops": loops,
                "batch": 4,
                "sequence": 16,
                "comparison": compare(torch.cat(pieces, 1), expected),
                "cache_bytes": cache.n_bytes(),
            }
            effective_layers = 24 + (loops - 1) * 12
            assert cache.n_bytes() == 2 * 4 * 16 * 2 * 64 * 4 * effective_layers
            cache_results.append(record)
            print("CACHE", record, flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.bridge.parameters(), lr=1e-3)
    sample = tokens[:1, :8]
    output = model(sample, loop_k=2)
    loss = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].flatten(0, 1), sample[:, 1:].flatten()
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in model.bridge.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradients[name] = {
            "nonzero": int(torch.count_nonzero(parameter.grad)),
            "norm": float(parameter.grad.norm()),
        }
        assert gradients[name]["nonzero"] > 0
    assert all(p.grad is None for p in base.parameters())
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.bridge.parameters())
    assert weights_digest(base) == original_weights
    with torch.no_grad():
        retained = compare(model(tokens).logits, reference, exact=True)
    print("BRIDGE_UPDATE_RETAINS_BASE", gradients, flush=True)
    report = {
        "status": "passed",
        "scope": "CPU identity/cache/one-update plumbing only; not quality, "
        "useful depth, CUDA validation, full-size restart or an architecture adoption",
        "source_revision": source_revision,
        "source_files": source_hashes,
        "donor_manifest_sha256": digest(manifest_path),
        "donor": manifest,
        "train_file_sha256": TRAIN_SHA,
        "inputs": [
            {
                "content_sha256": sha,
                "token_ids_sha256": hashlib.sha256(
                    torch.tensor(ids, dtype=torch.int64).numpy().tobytes()
                ).hexdigest(),
            }
            for sha, ids in prefixes
        ],
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": "cpu",
            "dtype": "float32",
            "attention": "sdpa",
            "threads": 2,
        },
        "recurrence_contract": model.get_extra_state(),
        "resident_parameters": resident,
        "trainable_parameters": trainable,
        "weight_bytes_fp32": resident * 4,
        "gradient_and_adam_moment_bytes_fp32": trainable * 12,
        "identity": identity,
        "cache": cache_results,
        "bridge_gradients": gradients,
        "post_update_identity": retained,
        "all_unique_base_parameter_hashes_unchanged": True,
        "seconds": time.monotonic() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print("GATE_COMPLETE", args.out, report["seconds"], flush=True)


if __name__ == "__main__":
    main()
