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

The continuation toward 1,024 is in progress; no result at that boundary is
claimed by this report yet.
