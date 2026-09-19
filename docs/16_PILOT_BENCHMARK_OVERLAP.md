# Retrospective pilot / benchmark overlap screen

The frozen R04 pilot was screened on 2026-09-19 against ten public benchmark
splits at pinned Hub revisions: LAMBADA OpenAI, ARC Easy/Challenge, CommonsenseQA,
MMLU, MMLU-Pro, GSM8K, HumanEval, MBPP and IFEval. Downloaded text remains under
ignored `data/benchmark-audit-v1`; no benchmark examples enter git or training.

The screen indexed 38,520 rows and 2,122,506 distinct normalized 13-word sequences.
It found **9 of 19,569 training documents** with at least one shared sequence and
**0 of 376 validation documents**. Six training documents overlap MMLU, six
MMLU-Pro and two HumanEval; these groups overlap each other, hence nine unique
documents. The exact corpus file hashes match the frozen pilot.

An independent exhaustive comparison of every reported document against every
item in its matching benchmark confirmed all 14 document/benchmark pairs. The
largest item containment was **0.40**; none reached the existing decontaminator's
0.50 threshold. These are lexical overlaps, not proof that a whole question or
answer was memorized. The pilot corpus and paired training protocol stay fixed.

This screen is incomplete by design: **359 rows shorter than 13 normalized words
were not tested**, and seven listed sources were not fetched (SciQ, PIQA,
HellaSwag, Social IQA, OpenBookQA, WinoGrande and gated GPQA). The fetch allowlist
uses public card metadata advertising MIT, Apache-2.0, CC-BY-4.0 or CC-BY-SA-4.0;
it does not constitute a complete license review. Missing card metadata is not a
finding that a dataset has no usable license. Semantic overlap remains untested.

No benchmark accuracy or benchmark-clean status follows from these results.
Before a production corpus or benchmark claim, complete the missing coverage,
check short items, and apply the selected filtering policy before freezing that
new corpus. The current pilot supports its internal held-out language-loss
comparison only.

## Evidence and reproduction

- [Pinned source metadata](experiments/2026-09-19-benchmark-sources.json)
- [Full overlap counts and document hashes](experiments/2026-09-19-pilot-benchmark-overlap.json)
- [Independent item-containment checks](experiments/2026-09-19-pilot-overlap-containment.json)

The run used Python 3.14, datasets 5.0.1 and huggingface_hub 1.32.0. No gated
access was requested or terms accepted. Source IDs and revisions in the metadata
identify each public Hub dataset and the fetched split; output file hashes bind
the exact rows used.

```bash
python scripts/audit_pilot_overlap.py \
  --sources docs/experiments/2026-09-19-benchmark-sources.json \
  --corpus data/fineweb-pilot-v1 --cache data/benchmark-audit-v1/text \
  --out /tmp/pilot-overlap.json
```
