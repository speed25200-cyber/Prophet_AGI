#!/usr/bin/env python3
"""The first weights: a CPU-sized Prophet trained on real text, end to end.

    python scripts/first_run_cpu.py --work /tmp/prophet-first-run --stage all --minutes 40

No GPU, no network: the corpus is whatever real text the machine holds -- this
repository's own documentation and the operating system's, as prose; installed Python
sources, as code. Nothing is downloaded and nothing leaves the ``--work`` directory
(rule 4: no data in git). What the run proves is not a capability claim -- at seven
million parameters there is none to make -- but that every stage the A100 run will use
works on real data and produces a number that can be checked:

1. ``corpus``     build ``corpus/prose.jsonl``, ``corpus/code.jsonl``, a held-out
                  ``benchmarks/heldout.jsonl`` cut from both *before* training, and the
                  two-phase ``mixture.yaml`` that describes them.
2. ``tokenizer``  train Prophet-Tok on a sample (``scripts/train_tokenizer.py``).
3. ``train``      ``scripts/train.py`` with the real path: decontamination against the
                  held-out set, phased loader, wall-clock budget, then a second launch
                  that resumes from the checkpoint to prove the resume.
4. ``eval``       bits per byte on the held-out set for the untrained model and for the
                  checkpoint, plus tokens seen and throughput, written to ``report.json``
                  and ``docs``-ready markdown.

The blockwise delta-rule scan is what makes this affordable on CPU; the same
configuration under the reference scan would take fifty times longer.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.mixture import Mixture, Phase, Source  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.harness import evaluate_bpb  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.checkpoint import CheckpointManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "prophet_cpu_first_run.json"
LICENSE = "local machine text, training only, never redistributed"


# --------------------------------------------------------------------------------------
# 1. corpus
# --------------------------------------------------------------------------------------


def _paragraphs(text: str, *, min_chars: int) -> list[str]:
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        block = block.strip()
        if len(block) >= min_chars and sum(c.isalpha() for c in block) > 0.5 * len(block):
            out.append(block)
    return out


def _read(path: Path, limit: int = 2_000_000) -> str | None:
    try:
        data = path.read_bytes()[:limit]
        return data.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def build_corpus(work: Path, *, seed: int, max_prose: int, max_code: int, heldout_every: int) -> dict:
    rng = random.Random(seed)
    prose: list[str] = []
    for md in sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md", ROOT / "CLAUDE.md"]:
        text = _read(md)
        if text:
            prose += _paragraphs(text, min_chars=200)
    for doc in sorted(Path("/usr/share/doc").rglob("*")):
        if not doc.is_file() or doc.stat().st_size < 5_000 or doc.suffix in (".gz", ".bz2", ".xz", ".zip"):
            continue
        text = _read(doc, limit=200_000)
        if text:
            prose += _paragraphs(text, min_chars=300)
    code: list[str] = []
    site = Path("/usr/local/lib/python3.11/dist-packages")
    files = [p for p in site.rglob("*.py") if 2_000 <= p.stat().st_size <= 60_000]
    rng.shuffle(files)
    for py in files[: max_code * 2]:
        text = _read(py)
        if text and "\x00" not in text:
            code.append(text)
        if len(code) >= max_code:
            break
    rng.shuffle(prose)
    prose = prose[:max_prose]

    # The held-out set is cut *before* training and is also what the decontaminator
    # indexes: a training document that repeats a held-out paragraph is rejected.
    heldout = [p for i, p in enumerate(prose) if i % heldout_every == 0]
    heldout += [c for i, c in enumerate(code) if i % heldout_every == 0]
    prose = [p for i, p in enumerate(prose) if i % heldout_every != 0]
    code = [c for i, c in enumerate(code) if i % heldout_every != 0]

    (work / "corpus").mkdir(parents=True, exist_ok=True)
    (work / "benchmarks").mkdir(parents=True, exist_ok=True)
    for name, docs in (("prose", prose), ("code", code)):
        with (work / "corpus" / f"{name}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps({"text": d}) + "\n")
    with (work / "benchmarks" / "heldout.jsonl").open("w") as f:
        for d in heldout:
            f.write(json.dumps({"text": d}) + "\n")

    def tokens_estimate(docs: list[str]) -> float:
        return sum(len(d.encode("utf-8")) for d in docs) / 3.5

    mixture = Mixture(
        name="cpu-first-run", total_tokens=1.0, description="local machine text, two phases",
        phases=[
            Phase("A-stable", 0.8, [
                Source("prose", "local", "web", 0.5, available_tokens=tokens_estimate(prose), license=LICENSE),
                Source("code", "local", "code", 0.5, available_tokens=tokens_estimate(code), license=LICENSE),
            ], purpose="broad mix"),
            Phase("C-anneal", 0.2, [
                Source("prose", "local", "web", 0.6, available_tokens=tokens_estimate(prose), license=LICENSE),
                Source("code", "local", "code", 0.4, available_tokens=tokens_estimate(code), license=LICENSE),
            ], purpose="prose-heavier anneal"),
        ],
    )
    mixture.validate()
    mixture.to_yaml(work / "mixture.yaml")
    stats = {
        "prose_docs": len(prose), "code_docs": len(code), "heldout_docs": len(heldout),
        "prose_bytes": sum(len(d.encode()) for d in prose),
        "code_bytes": sum(len(d.encode()) for d in code),
        "heldout_bytes": sum(len(d.encode()) for d in heldout),
    }
    (work / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# --------------------------------------------------------------------------------------
# 4. eval
# --------------------------------------------------------------------------------------


def _batches(tokenizer: ProphetTokenizer, docs: list[str], *, seq_len: int, batch_size: int):
    """``(ids, n_bytes)`` batches with no padding: full-length rows are batched, the
    short tail of each document is scored on its own, so every scored token is real."""
    full: list[list[int]] = []
    full_bytes = 0
    for doc in docs:
        ids = tokenizer.encode(doc, add_eos=True)
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len]
            if len(chunk) < 2:
                continue
            n_bytes = len(tokenizer.decode(chunk[1:]).encode("utf-8"))  # the predicted part
            if len(chunk) == seq_len:
                full.append(chunk)
                full_bytes += n_bytes
                if len(full) == batch_size:
                    yield torch.tensor(full, dtype=torch.long), full_bytes
                    full, full_bytes = [], 0
            else:
                yield torch.tensor([chunk], dtype=torch.long), n_bytes
    if full:
        yield torch.tensor(full, dtype=torch.long), full_bytes


def evaluate(work: Path, *, seq_len: int, max_docs: int) -> dict:
    cfg = ProphetConfig.from_json(CONFIG)
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    heldout = [json.loads(l)["text"] for l in (work / "benchmarks" / "heldout.jsonl").read_text().splitlines() if l.strip()]
    heldout = heldout[:max_docs]
    torch.manual_seed(0)
    fresh = ProphetModel(cfg).eval()
    with torch.no_grad():
        untrained = evaluate_bpb(fresh, _batches(tokenizer, heldout, seq_len=seq_len, batch_size=8))
    ckpt = CheckpointManager(work / "checkpoints")
    state, meta = ckpt.load_latest()
    trained = ProphetModel(cfg).eval()
    trained.load_state_dict(state["model"])
    with torch.no_grad():
        after = evaluate_bpb(trained, _batches(tokenizer, heldout, seq_len=seq_len, batch_size=8))
    report = {
        "config": cfg.name,
        "parameters": sum(p.numel() for p in trained.parameters()),
        "step": meta.step,
        "tokens_seen": int(state.get("tokens_seen", 0)),
        "heldout_docs": len(heldout),
        "heldout_tokens": after.n_tokens,
        "heldout_bytes": after.n_bytes,
        "bpb_untrained": untrained.bits_per_byte,
        "bpb_trained": after.bits_per_byte,
        "nats_per_token_untrained": untrained.nats_per_token,
        "nats_per_token_trained": after.nats_per_token,
    }
    (work / "report.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", choices=["corpus", "tokenizer", "train", "eval", "all"], default="all")
    ap.add_argument("--minutes", type=float, default=30.0, help="wall-clock budget of the first launch")
    ap.add_argument("--resume-minutes", type=float, default=5.0, help="budget of the resuming launch")
    ap.add_argument("--tokens", type=float, default=20e6, help="token budget the schedule is sized for")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-prose", type=int, default=20_000)
    ap.add_argument("--max-code", type=int, default=3_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    stages = ["corpus", "tokenizer", "train", "eval"] if args.stage == "all" else [args.stage]
    started = time.time()

    if "corpus" in stages:
        stats = build_corpus(work, seed=args.seed, max_prose=args.max_prose, max_code=args.max_code, heldout_every=50)
        print(f"corpus     {stats}")
    if "tokenizer" in stages:
        _run([sys.executable, ROOT / "scripts" / "train_tokenizer.py", "--data-root", work / "corpus",
              "--out", work / "tokenizer.json", "--vocab-size", args.vocab_size, "--max-docs", 3000])
    if "train" in stages:
        common = [
            sys.executable, ROOT / "scripts" / "train.py", "--config", CONFIG,
            "--tokenizer", work / "tokenizer.json", "--data-root", work / "corpus",
            "--benchmarks", work / "benchmarks", "--mixture", work / "mixture.yaml",
            "--tokens", args.tokens, "--batch-size", args.batch_size, "--seq-len", args.seq_len,
            "--checkpoint-dir", work / "checkpoints", "--checkpoint-every", 100, "--log-every", 20,
            "--device", "cpu", "--allow-slow-scan", "--seed", args.seed,
        ]
        _run(common + ["--session-minutes", args.minutes])
        # Prove the resume: a second launch continues the same run and the same stream.
        _run(common + ["--session-minutes", args.resume_minutes])
    if "eval" in stages:
        report = evaluate(work, seq_len=args.seq_len, max_docs=400)
        report["wall_minutes_this_invocation"] = (time.time() - started) / 60
        (work / "report.json").write_text(json.dumps(report, indent=2))
        print("\n| Quantité | Valeur |\n|---|---:|")
        for key, value in report.items():
            print(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
