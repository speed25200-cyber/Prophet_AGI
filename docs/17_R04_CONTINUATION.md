# R04 continuation on real text

Training retains source `e5720d0b774977455b1920a6f67b5df078377f11`, seed 0,
the frozen corpus and vocabulary, and the original 4,096-step optimizer schedule.
Both seed-0 arms have completed step 4,096. The endpoint favors the unshared
control in language loss; the shared model uses fewer parameters. Seeds 1 and 2
remain untrained, so the three-seed architecture gate remains open.

## Verified shared checkpoint at step 576

The shared model resumed from step 160 and reached 576: **9,437,184 cumulative
training tokens**. Full held-out evaluation of the same 376 documents gives
**5.199319 nats/token** and **1.684119 bits/byte**, compared with 6.080348 and
1.969494 at step 160. The target counts remain 393,040 tokens and 1,750,592 bytes.
No matched unshared comparison at step 576 has been run.

The checkpoint has 3,233,901,515 bytes and SHA256
`b5212049bad429968fede2f412ed6fd463e869c97c25536891e32be8f8e4876c`.
After reconnecting Drive, it was copied to Colab's local disk, checksum-verified,
loaded with the restricted tensor loader, and audited: **313 model/optimizer
tensors, all finite, zero skipped non-finite steps**.

The evidence archive was downloaded and verified on the local workstation:
58,493 bytes, SHA256
`28259443a5ed75c80e00ff79addd49682e49ddb35ddb7498c413c9b43b423376`.
Its 36 training log rows end at 576; report aggregates were independently checked
against all document records. [Raw reports](experiments/2026-09-19-r04-continuation/loop-seed0).

## Storage interruption and recovery

This session initially targeted step 1,024. Reads from the mounted Drive journal
became intermittently blocked after the periodic step-512 save. Interrupting the
notebook monitoring cell also delivered a stop request to the training child,
which completed step 576, saved, evaluated and exited with code 0.

A 60-second `flush_and_unmount` attempt failed. A subsequent call found Drive
already unmounted; its successful return was a no-op, not proof of a completed
flush. The existing cache and VM were preserved. Remounting succeeded, and the
step-576 files passed the local copy and full checkpoint audit described above.

The next continuation uses local working files and stdout, a separate process
session (`start_new_session=True`), and the same numerical training contract.
Only an explicit signal to the training child requests an early stop. A verified
snapshot will be copied to a new Drive directory after the session, preserving
the earlier checkpoints. This also follows Colab's advice to reduce mounted-Drive
I/O; the exact cause of the transient Drive stalls was not established.
[Colab FAQ](https://research.google.com/colaboratory/faq.html#drive-timeout).

## Shared model at step 1,024

The local continuation from 576 to 1,024 completed and evaluated all 376 held-out
documents: **16,777,216 cumulative training tokens**, **4.708506826 nats/token**,
**1.525139450 bits/byte**. The checkpoint audit inspected 313 model/optimizer
tensors: all finite, zero skipped non-finite steps. Target counts remain 393,040
tokens and 1,750,592 bytes.

The published checkpoint is slot 1, 3,233,913,291 bytes, SHA256
`920f8a3338b33de0e5835756b8ec0f94c34e2af18cee9ac8edbc97835534e726`.
The evidence ZIP was downloaded and verified locally: 162,395 bytes, SHA256
`60dfd5abca15928cc0066f3553eaaec33b69809fa5f375a366554f20b282a91b`.
All document sums, target identities, and 64 training log rows were independently
checked. [Validation](experiments/2026-09-19-r04-continuation/loop-seed0/evaluation-step-001024.json),
[checkpoint audit](experiments/2026-09-19-r04-continuation/loop-seed0/checkpoint-audit-step-001024.json).

## Matched seed-0 comparison at step 1,024

The unshared model completed its continuation from 128 to 1,024 using the same
frozen numerical protocol. Both arms have seen **16,777,216 training tokens** and
are evaluated on the same 376 complete documents and exact next-token targets.

| Measurement | Shared k=4 | Unshared k=1 |
|---|---:|---:|
| Parameters | 374,688,512 | 920,675,072 |
| Held-out nats/token | 4.708506826 | 4.682119010 |
| Held-out bits/byte | 1.525139450 | 1.516592134 |
| Median logged step time, 64 samples | 1.9675 s | 2.1544 s |
| Skipped non-finite steps | 0 | 0 |

Shared minus unshared is **+0.026387816 nats/token**, with a 95% paired document
bootstrap interval **[+0.021561723, +0.031465560]**, using 10,000 seeded resamples.
The BPB difference is +0.008547317, interval [+0.006977449, +0.010206979].
The shared model has lower document loss on 31.38% of the documents.
**This point favors the unshared model.** The intervals condition on these weights
and exclude training-seed uncertainty; the 4,096-step, three-seed pilot remains
unfinished. The logged step times exclude compilation, checkpointing and validation
and do not establish an end-to-end speed advantage.
[Reproducible paired summary](experiments/2026-09-19-r04-step1024-summary.json).

The unshared published checkpoint is slot 1, 7,605,985,915 bytes, SHA256
`19e7fef7005c561f96b8670c8daf82135ac61eb3ceda693889fcaf31953b4ee3`.
Its restricted CPU audit checked 865 model/optimizer tensors, all finite, with
zero skipped non-finite steps. Its evidence ZIP was downloaded and verified:
60,072 bytes, SHA256
`b369495e1deb321f416d206b704145bf1ce0f075ae4b5311e926c93de88cad14`.
All document aggregates and 64 training rows were checked independently.
[Unshared reports](experiments/2026-09-19-r04-continuation/plain-seed0).

```bash
python scripts/summarize_r04.py \
  --root docs/experiments/2026-09-19-r04-continuation \
  --step 1024 --seed 0 --out /tmp/r04-step1024-summary.json
```

## Shared model at step 2,048

The shared continuation completed with exit code 0 at **33,554,432 training tokens**.
Full validation on the same 376 documents gives **4.357405080 nats/token** and
**1.411413562 bits/byte**. Counts and document identities match the earlier evaluation:
393,040 scored targets and 1,750,592 payload bytes. Its unchanged numerical training
contract and all 313 model/optimizer tensors were audited, with zero non-finite
steps skipped.

The exact evaluated checkpoint is slot 0, 3,233,910,859 bytes, SHA256
`90239173054b59a0d06532ca5ff80c1ec5414b6d838d8c8c3c09b731c096cdd1`.
The downloaded evidence ZIP contains 62,621 bytes, SHA256
`9d85c442f6848200013af709ffe09d11e16a14c257ed853b459c136cfb1851b6`.
All per-document sums, target identities and 128 training log rows were independently
checked on the workstation. The previous step-1,024 evidence is preserved separately.
[Evaluation and checkpoint audit](experiments/2026-09-19-r04-step2048/loop-seed0).

## Matched seed-0 comparison at step 2,048

The unshared continuation also completed with exit code 0 under the same frozen
protocol. Both arms have seen **33,554,432 training tokens**. Full validation still
scores exactly the same 376 documents, 393,040 targets and 1,750,592 payload bytes.

| Measurement | Shared k=4 | Unshared k=1 |
|---|---:|---:|
| Parameters | 374,688,512 | 920,675,072 |
| Held-out nats/token | 4.357405080 | 4.367619376 |
| Held-out bits/byte | 1.411413562 | 1.414722089 |
| Median logged step time, 128 samples | 1.9674 s | 2.1556 s |
| Skipped non-finite steps | 0 | 0 |

Shared minus unshared is **-0.010214295 nats/token**, with a 95% paired document
bootstrap interval **[-0.014071717, -0.006317731]** (10,000 resamples, seed 0).
The BPB difference is -0.003308528, interval [-0.004559218, -0.002053992]. Shared
has lower document loss on 56.65% of documents. **This checkpoint favors sharing**,
reversing the step-1,024 ordering. The result is conditional on these two trained
models; the interval excludes training-seed uncertainty and does not establish an
architecture-wide advantage. The fixed 4,096-step, three-seed gate remains open.
Timing medians exclude setup, compilation, checkpointing and evaluation.
[Paired summary](experiments/2026-09-19-r04-step2048-summary.json).

The unshared checkpoint is slot 0, 7,605,983,483 bytes, SHA256
`268b685cc47ac90728eb0072e631250ab20ea1864c619b1e1f544bde6969eeb7`.
All 865 model/optimizer tensors are finite. The evidence ZIP is 62,669 bytes,
SHA256 `5e9167b1737f1d8146e35696b8129de4df942ad7eb7eb85292eccd7f692d15a5`.
Its 376 document sums/identities, CE/BPB denominators, checkpoint metadata and all
128 log rows were independently validated after download.
[Unshared reports](experiments/2026-09-19-r04-step2048/plain-seed0).

```bash
python scripts/summarize_r04.py \
  --root docs/experiments/2026-09-19-r04-step2048 \
  --step 2048 --seed 0 --out /tmp/r04-step2048-summary.json
```

Both fresh step-2,048 Drive snapshots passed their filesystem copy/checkpoint audits.
Drive was confirmed mounted, flushed successfully in 93.84 seconds, then remounted.
Both complete tensor audits matched their local originals exactly (313 shared,
865 unshared tensors). The downloaded persistence proof has SHA256
`0512f90a28088a63ac3fcead39e4c61fce9c1355603165e406e2aeffc62a00dd`.
[Post-remount persistence evidence](experiments/2026-09-19-r04-step2048-persistence.json).

The exact published checkpoints were also staged into separate local directories
with one-entry manifests and full audits. A bounded queue now continues seed 0
toward the original 4,096-step endpoint, shared first, then unshared, with an audit
and a fresh snapshot between arms. Each process uses local files/stdout and its
own process session. Its operational limit is 2,048 additional steps or 90 minutes;
the original total schedule and all numerical settings stay fixed. Unexpected early
exit, missing evaluation or failed audit stops the queue for inspection. Final
snapshot flushing remains a separate step after those future runs complete.

## Shared seed 0 completes step 4,096

The shared arm completed the original schedule with **67,108,864 training tokens**,
zero skipped non-finite updates, and all 313 checkpoint model/optimizer tensors
finite. Full validation scores the same 376 documents, 393,040 targets and
1,750,592 UTF-8 payload bytes: **3.855913359 nats/token** and **1.248974632 bits/byte**.
The median of the final 128 logged step durations is 1.9679 seconds, excluding
checkpointing, evaluation and compilation. All 256 log rows and document aggregates
were independently checked after download.

The checkpoint is slot 1, 3,233,905,547 bytes, SHA256
`e623941c9db93f792f02c263c79f4e01ec25d8a9afc9ccb032defd5323551cb4`.
Its fresh Drive snapshot is under `snapshots/step-004096-seed0/loop-seed0`.
After a successful 2.89-second flush with Drive mounted and a remount, the complete
checkpoint audit matched the local original. The same persistence check verified
all nine files of the separate Colab donor-recovery initialization pair.
[Evaluation, checkpoint audit and logs](experiments/2026-09-19-r04-step4096/loop-seed0),
[post-remount persistence proof](experiments/2026-09-19-r04-step4096/loop-persistence.json).
The downloaded evidence ZIP is 70,667 bytes, SHA256
`3593420b49f73bd97b9571cb5812de98fc663b1d6640be3366dd9d886ec1f02d`.

## Matched seed-0 endpoint at step 4,096

The unshared continuation also completed successfully under the frozen training
implementation. Both arms have processed **67,108,864 tokens**, with all 376
validation documents, 393,040 targets and 1,750,592 scored payload bytes unchanged.

| Measurement | Shared k=4 | Unshared k=1 |
|---|---:|---:|
| Parameters | 374,688,512 | 920,675,072 |
| Held-out nats/token | 3.855913359 | 3.844324022 |
| Held-out bits/byte | 1.248974632 | 1.245220713 |
| Median logged step time, 256 samples | 1.9677 s | 2.1567 s |
| Skipped non-finite steps | 0 | 0 |

Shared minus unshared is **+0.011589337 nats/token**, with paired document
bootstrap 95% interval **[+0.008332563, +0.014891672]**. The BPB difference is
+0.003753919, interval [+0.002707264, +0.004815489]. Sharing wins on 32.18% of
documents. **The final endpoint favors the unshared control**, reversing the
intermediate step-2,048 ordering again. Sharing stores 59.30% fewer parameters,
with 0.3015% higher BPB on this seed. The intervals condition on these models and
exclude training-seed uncertainty; no multi-seed superiority or noninferiority
claim follows. Logged step times exclude setup, checkpointing and evaluation.
[Paired endpoint summary](experiments/2026-09-19-r04-step4096-summary.json).

The unshared checkpoint is slot 1, 7,605,978,171 bytes, SHA256
`ca2f5f6decd29aa49d1301e4980b64052c2b1134851b6b360208fb72ecc5c31a`.
Both snapshots were remotely flushed while Drive was mounted (101.23 seconds),
then remounted and fully audited again. The post-remount audits exactly match the
local originals: 313 shared and 865 unshared model/optimizer tensors, all finite,
and no skipped updates. The paired report also matched its hash after remount.
[Full persistence proof](experiments/2026-09-19-r04-step4096-persistence.json),
[unshared reports](experiments/2026-09-19-r04-step4096/plain-seed0).

The 140,432-byte evidence ZIP has SHA256
`2fe9443c2ebab54cb8c9bd1bd7f7a9453415acb0eb7fad602e74f7138f54dc77`.
After download, every document identity, denominator, aggregate loss and all 256
training log rows per arm were checked. Independently rerunning the paired
summary on the workstation reproduced the complete Colab result, including all
10,000 bootstrap draws. Seeds 1 and 2 remain pending.

```bash
python scripts/summarize_r04.py \
  --root docs/experiments/2026-09-19-r04-step4096 \
  --step 4096 --seed 0 --out /tmp/r04-step4096-summary.json
```

## Inference depth sensitivity at step 1,024

`scripts/eval_r04_depth.py`, from revision
`04b744b195a54a28026021fb63fb25bec19ee501`, loaded the exact step-1,024 checkpoint while
retaining the frozen training implementation and runtime. Four complete evaluations
used the same documents and targets. The first, at the trained depth of four loops,
reproduced the saved validation CE exactly.

| Inference loops | Nats/token | Bits/byte | BPB change against four loops |
|---:|---:|---:|---:|
| 1 | 5.632993405 | 1.824591273 | +19.63% |
| 2 | 4.950292106 | 1.603456480 | +5.14% |
| 4, trained depth | 4.708506826 | 1.525139450 | reference |
| 8 | 4.939149559 | 1.599847281 | +4.90% |

**Eight inference loops hurt language loss at this checkpoint.** These fixed-k=4
training results do not establish useful extra inference computation, nor do they
test variable-depth training or reasoning accuracy. No depth-generalization or
architecture-adoption claim follows. The distinction from the variable-depth
training recipe in Huginn is documented in the [source audit](15_R04_SOURCE_AUDIT.md).

Recorded evaluation times were 27.64, 9.86, 14.94 and 45.38 seconds in execution
order k=4,1,2,8. They include I/O and possible compilation and are not controlled
latency benchmarks. Peak allocated GPU memory was 5.21–5.27 GiB for the standalone
model and evaluation workspace, excluding training optimizer and gradients.
[Full depth results](experiments/2026-09-19-r04-continuation/loop-seed0/depth-step-001024.json).

The [completed final step-4,096 sweep](23_R04_DEPTH_ADAPTATION.md) reproduces
the final k=4 result exactly and scores all 376 documents at k=1,2,4,6,8.
At k=6 CE is 4.424840339; at k=8 it is 4.479179774, versus 3.855913359
at k=4. Eight loops increase BPB by 16.1639% at the final checkpoint too.
The next matched variable-depth adaptation is budgeted but not yet trained.

Parameter-count note: the historical budget omitted existing GDN gate biases
and output normalization. Corrected exact counts are 374,689,648 shared and
920,679,616 unshared; historical source reports are retained unchanged.
The model weights, topology, loss results and rounded 59.3% saving are unaffected.

## Snapshot workflow

Both step-1,024 snapshots are stored below
`MyDrive/Prophet_AGI/R04/snapshots/step-001024-seed0`, in `loop-seed0` and
`plain-seed0`. Each filesystem copy passed its checksum and full checkpoint audit.
Drive was confirmed mounted before each successful flush. After remounting, both
checkpoints were read and audited again: the entire audits matched their local
sources, including all 313 shared and 865 unshared model/optimizer tensors.
The step-1,024 and step-2,048 snapshots remain available. With user authorization,
three obsolete checkpoints from the earlier Drive runs (shared steps 512 and 576,
unshared step 128) were permanently removed, releasing 14,073,783,633 bytes.
Their manifests were archived and retired. Training uses local files throughout
these storage operations.
[Persistence evidence](experiments/2026-09-19-r04-step1024-persistence.json),
downloaded with SHA256
`af529d2545186fb43ef639a9685aec02b6a9e0eb8d58f4a5fcc86ec5b07758c5`.

The revised [notebook](../notebooks/r04_pilot.ipynb) follows this sequence:
explicit resume source, audit and stage into local storage, local training and
monitoring, a new verified Drive snapshot, remote flush, then a separate VM-release
cell. An interrupted restore is left under `.restoring` and cannot become the
working run before its audit succeeds. The analysis helpers are independently
pinned to `a08a15fb493beabb9969c405f01380b28ca185ec`; training stays at `e5720d0`.
Before a local resume, the notebook audits the published checkpoint and selects
its manifest entry over any older save at the same step, retaining a manifest
backup and lower-step fallback entries. It refuses to silently discard a newer,
unevaluated checkpoint. This compensates for the manifest tie ordering in the
frozen training revision without changing its numerical implementation.

`scripts/snapshot_r04.py` copies only the checkpoint referenced by the completed
validation into a fresh directory, audits the destination independently, then
writes `SNAPSHOT_COMPLETE.json`. It never overwrites an existing snapshot or
inherits an old completion marker. A partial copy must not be used as a completed
snapshot. The audited step-1,024 and step-2,048 snapshots remain fallback sources;
remote flushing is a separate requirement before releasing the VM.
