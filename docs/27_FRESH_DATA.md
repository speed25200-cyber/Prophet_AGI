# Fresh data after the negative depth screens

The completed variable-depth and learned-reinjection screens do not establish
useful extra inference depth. Native ARC likewise does not show that six loops
improve on four. The original pilot has reached 3.8877 training passes per final
arm, near its four-pass ceiling. Its development scores have also guided multiple
research decisions. More updates on that same pilot are not an unrestricted next
experiment. The original results and failed thresholds remain unchanged.

The next implemented prerequisite is CPU-only preparation of a fresh bounded
English corpus. This is not a new adopted architecture, a training launch, or a
claim that another small web pilot will suffice for an assistant.

## Source and exclusions

`scripts/prepare_fresh_pilot.py` reuses the immutable FineWeb-Edu source and quality
filters from `prepare_pilot.py`. It verifies every artifact of the original v1
pilot against its manifest and normalized document hashes, then starts after the
original raw-source cursor. Skipping a prefix alone is insufficient: both new
splits additionally reject prior train AND validation exact normalized content,
URL host/path variants, and any shared normalized 13-word span. This conservative
rule also rejects common boilerplate.

The script verifies the caller-specified SHA256 of the existing frozen ARC items.
It protects the question and every answer choice separately, removing only the
known prompt wrapper. A match to any 13-word span rejects a candidate. For shorter
fragments, entire 5–12-word phrases are matched at word boundaries. Fragments of
fewer than five words are counted explicitly and only excluded on whole-document
equality; common short answers cannot establish benchmark leakage by themselves.

After these exclusions, the existing deterministic URL split, normalized-content
deduplication and new train-versus-validation 13-word screen apply. The raw-source
scan has its own strict cap, including rejected candidates. Output publication is
atomic and refuses existing output or staging directories. Source identities,
exclusion-input hashes, coverage limits, counters and output artifact hashes are
retained in the manifest. Text remains outside Git.

This does not detect semantic overlap or cover every benchmark. It cannot remove
knowledge or contamination already present in a parent checkpoint. In particular,
the manifest does **not** claim that the model or corpus is benchmark-clean.

## Execution and use

```bash
python scripts/prepare_fresh_pilot.py \
  --prior data/fineweb-pilot-v1 \
  --arc-items data/arc-recovery-eval-v1/items.jsonl \
  --arc-sha256 5011ec1abef00263b75101c484acdd4e8e2a1e51f8d6895059e87126ccdc6b3b \
  --out data/fineweb-fresh-v1 \
  --max-docs 20000 --max-bytes 200000000 --max-scan-rows 200000
```

The parent tokenizer must be reused; this command neither fits nor copies one.
Count actual train tokens before setting an update budget. Freeze a future paired
recipe and its decision rule before scoring the new validation set. Treat that
set as consumed once it informs a decision, rather than repeatedly calling it a
fresh holdout. Training, its numerical/restart gates and new-seed confirmation
remain separate work.

The memory index holds hashes of old-corpus spans; its RAM use scales with the
protected corpus. This bounded-pilot implementation is not a scalable web-corpus
index. Failed or interrupted preparation must be inspected before a new output
path is chosen; it does not resume partial preparation.

## Validation and compute scope

Eighteen targeted tests pass across fresh preparation and the existing pilot.
They cover prior train/validation protection, all answer choices, short phrase
boundaries, raw scan exhaustion with every candidate rejected, deterministic
partitioning, new-heldout overlap, altered hashes/counts/source identity, path
escape and changed prompt wrappers. Local CPU preparation has no GPU allocation.

Before revisiting architecture, the budget, design search and allocation tools
were run. The current control has 374,689,648 resident parameters. The design
search finds no full-training candidate at a one-A100-hour illustrative budget;
the plan likewise funds no complete track. That illustration is not a statement
of purchased credit. The budget command's default 300-hour token projection and
80-GB memory heading are theoretical defaults, not available compute or the
measured 40-GB device. Actual training limits must use fresh Colab availability
and measured throughput.

Published recurrence curricula and retained pretrained paths remain candidates
for a separately costed protocol. McLeish et al. report staged recurrence and a
healing/data curriculum, chiefly evaluating mathematical improvements
([primary paper](https://arxiv.org/html/2511.07384v1)). Shapiro reports an
identity-preserving one-loop path and intermediate supervision on controlled
iterative tasks, while explicitly reporting limited zero-shot verbal transfer
and a continual-interference failure
([primary abstract](https://arxiv.org/abs/2608.11233)). Neither result establishes
that our failed component will improve with more credit or another 512 updates.
