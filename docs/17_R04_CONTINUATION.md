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

The unshared seed-0 model is continuing from step 128 toward the same step-1,024
boundary. There is no matched comparison at 1,024 yet.

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

The shared step-1,024 snapshot is stored at
`MyDrive/Prophet_AGI/R04/snapshots/step-001024-seed0/loop-seed0`.
Its filesystem copy passed the checksum and full checkpoint audit. Drive was
confirmed mounted before `flush_and_unmount(timeout_ms=300000)`, which completed
successfully. After remounting, the checkpoint was read and audited again; the
entire audit matched the local source. The earlier Drive run was preserved.
The unshared continuation uses local files throughout these operations.

The revised [notebook](../notebooks/r04_pilot.ipynb) follows this sequence:
explicit resume source, audit and stage into local storage, local training and
monitoring, a new verified Drive snapshot, remote flush, then a separate VM-release
cell. An interrupted restore is left under `.restoring` and cannot become the
working run before its audit succeeds. The analysis helpers are independently
pinned to `a08a15fb493beabb9969c405f01380b28ca185ec`; training stays at `e5720d0`.

`scripts/snapshot_r04.py` copies only the checkpoint referenced by the completed
validation into a fresh directory, audits the destination independently, then
writes `SNAPSHOT_COMPLETE.json`. It never overwrites an existing snapshot or
inherits an old completion marker. A partial copy must not be used as a completed
snapshot. The previous Drive run remains available as a fallback; remote flushing
is a separate requirement before releasing the VM.
