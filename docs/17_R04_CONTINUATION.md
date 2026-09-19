# R04 continuation on real text

Training retains source `e5720d0b774977455b1920a6f67b5df078377f11`, seed 0,
the frozen corpus and vocabulary, and the original 4,096-step optimizer schedule.
The checkpoint is intermediate; the full paired, three-seed R04 gate remains open.

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

After this audit, the bounded Colab queue launched the unshared continuation from
its exact evaluated step-1,024 checkpoint toward 2,048. Shared snapshot copying runs
separately from the unshared GPU training, which uses local files. The queue stops
on errors, audits the unshared result after exit code 0, then creates its fresh
snapshot. Remote flush/remount verification is a separate remaining action.
**No matched comparison at step 2,048 is available yet.** The original 4,096-step
schedule and training source remain frozen; the full three-seed gate remains open.

## Inference depth sensitivity on the same weights

`scripts/eval_r04_depth.py`, from revision
`04b744b195a54a28026021fb63fb25bec19ee501`, loaded the exact checkpoint above while
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

## Snapshot workflow

Both step-1,024 snapshots are stored below
`MyDrive/Prophet_AGI/R04/snapshots/step-001024-seed0`, in `loop-seed0` and
`plain-seed0`. Each filesystem copy passed its checksum and full checkpoint audit.
Drive was confirmed mounted before each successful flush. After remounting, both
checkpoints were read and audited again: the entire audits matched their local
sources, including all 313 shared and 865 unshared model/optimizer tensors.
The earlier Drive runs were preserved. The current shared continuation uses local
files throughout these operations.
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
snapshot. The previous Drive run remains available as a fallback; remote flushing
is a separate requirement before releasing the VM.
