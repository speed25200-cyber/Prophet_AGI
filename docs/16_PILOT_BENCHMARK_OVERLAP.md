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

## Supplemental WinoGrande screen

The original Hub card had no license metadata. The upstream
[WinoGrande repository](https://github.com/allenai/winogrande/tree/727e837f77521ef38bcc56df3b275c8da43f45af)
distinguishes the code license from the dataset license; the README inside its
[official v1.1 archive](https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip)
specifies **CC BY 2.0** for the dataset. The supplement records the upstream commit,
download-script, archive, README and development-file hashes instead of inferring
the data license from the code's Apache license. The audit allowlist now includes
CC BY 2.0. Attribution: WinoGrande v1.1, Keisuke Sakaguchi, Ronan Le Bras, Chandra
Bhagavatula and Yejin Choi, *An Adversarial Winograd Schema Challenge at Scale*
(2019); [license](https://creativecommons.org/licenses/by/2.0/).

The pinned Hub validation split contains **1,267 rows**. All extracted sentence and
option fields match the official archive's development rows exactly, in order.
Joining those fields and applying the unchanged normalization yields 9,625 unique
13-word phrases; none of these joined rows is too short for the screen.
The supplemental audit finds **zero matching training or validation documents**.
An independent fixed-string search using ripgrep, with explicit word boundaries on
normalized document lines, also finds zero matches in all 19,569 training and 376
validation documents. Corpus file hashes match the frozen pilot exactly.

This extends coverage to eleven sources / 39,787 rows, retaining the original
result of nine potentially overlapping training documents and zero validation
documents. It does not modify R04 or the prepared recovery corpus. Six listed
sources remain outside coverage: SciQ, PIQA, HellaSwag, Social IQA, OpenBookQA and
GPQA. The original 359 short rows and semantic overlap remain untested. No data
license or benchmark-clean claim follows for those missing sources.

The OpenBookQA repository's code license was not treated as a dataset license.
Its [data-license issue](https://github.com/allenai/OpenBookQA/issues/7), checked on
2026-09-19, is still open without comments; this check did not establish a separate
affirmative data-license statement, so that source remains excluded from this
allowlist-based audit. This records the limits of the review, not a finding that
the dataset has no usable license.

Evidence: [supplemental source and attribution](experiments/2026-09-19-winogrande-source.json),
[source equality](experiments/2026-09-19-winogrande-source-parity.json),
[overlap report](experiments/2026-09-19-winogrande-overlap.json),
[independent negative check](experiments/2026-09-19-winogrande-independent-check.json).
Examples and downloaded source files remain in ignored local storage. This use is
an exclusion screen, not a benchmark score or addition to training data.

```bash
python scripts/audit_pilot_overlap.py \
  --sources docs/experiments/2026-09-19-winogrande-source.json \
  --corpus data/fineweb-pilot-v1 --cache data/benchmark-audit-v1/winogrande-supplement \
  --out /tmp/winogrande-overlap.json
```

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
