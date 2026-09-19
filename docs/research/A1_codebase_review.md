# A1 — Independent codebase review

> **Two snapshots.** The findings below were made against the working tree at
> **2026-09-04 23:52 → 2026-09-05 00:09 UTC** ("snapshot 1"). The tree was then edited continuously
> while this file existed — `tests/test_review_fixes.py` is keyed to the B-numbers below — so every
> finding was **re-verified against the tree at 2026-09-05 02:10 UTC** ("snapshot 2", 436 tests
> collected) and carries a status: **open**, **fixed** (re-measured, not inferred from a diff), or
> **not re-checked**. Line numbers in the tables are snapshot-1 numbers; where a locus moved, the
> status cell gives the current one. Code added after snapshot 1 (`prophet/agent/`,
> `prophet/data/corpus.py`, action heads, `token_depth`, `tests/test_gpu.py`,
> `tests/test_hygiene.py`) is **outside the scope of this review** and has not been read.
> Every number marked *(measured)* comes from code I ran (§2.1, §2.2).

## 1. Verdict in one paragraph

At snapshot 1 the repository was a well-written **design document with executable illustrations**,
not a training system. The pieces that only have to be *consistent* (config schema, budget
arithmetic, allocator, mixture validation, tokenizer, checkpoint rotation) were careful and agreed
with each other to 1e-4. The pieces that have to be *correct under load* were not: the model every
shipped config built silently zeroed the gated-delta forget-gate bias at init (α = 0.5 instead of
≈0.95), multiplied every residual branch by 0.11 at forward time (destroying the "direct copy" of a
converted donor), and trained its halting head with a loss whose gradient was identical at every
position, so the input-dependent depth that W1 turned from option into requirement could not be
learned by construction. Attention with a cache was correct only for the two shapes the tests
exercised; the trainer ran fp32 with no autocast, no activation checkpointing, no CUDA RNG in the
checkpoint and no fused delta-rule kernel; `scripts/train.py` exited with code 2 before building a
model unless `--smoke` was passed; ~20 configuration fields were accepted, serialised and never read;
and the docs said "235 tests", "0.97B / 89 %", "session state under a megabyte", "BF16 with FP32
master weights" — none of which was true. **At snapshot 2, 19 of the 23 bugs and most of the claim
mismatches are fixed and covered by regression tests** (§2.2). What remains open is small in code
and large in meaning: the FLA path — the only path that can train on the A100 — has still never been
executed anywhere; the halting probe / incremental-decode equivalence holds at 1e-2 but not at 1e-6
and nobody knows why; the session-restore contract still drops attention context; three doc lines
still contradict the code. The engineering honesty now matches the scientific honesty; the remaining
risk is that the entire post-snapshot layer (corpus loader, agent loop, action heads) has had one
author and no adversarial read.

## 2. Bugs

Severity: **S1** silent — trains normally, model is wrong; **S2** wrong output/number in a path the
project relies on; **S3** loud or blocking; **S4** minor. **Status** is against snapshot 2.

| # | Where (snapshot 1) | What is wrong | How it manifests | Sev | Suggested fix | Status (snapshot 2) |
|---|---|---|---|---|---|---|
| B1 | `prophet/modeling/model.py:319-322`, `prophet/modeling/layers.py:438-439` | `ProphetModel._init_weights` zeroes every `nn.Linear` bias unless `_prophet_keep_bias` is set — and nothing ever set it. `GatedDeltaNet.__init__` sets `a_proj.bias = 3.0` (α = σ(3) ≈ 0.95, "otherwise the layer is untrainable") and the model init overwrote it. *(measured: `a_proj.bias.unique() == [0.0]`, α = 0.5 on `prophet_500m_probe.json`)* | Every recurrent layer started with a state half-life of one token; the loss still fell on the smoke corpus. All shipped configs and the converted donor affected. | **S1** | Keep layer-local inits; test `sigmoid(bias) > 0.9` on a built model. | **fixed** *(measured: bias 3.0, α = 0.953)*; `test_b1_forget_gate_starts_near_one_inside_a_built_model` |
| B2 | `prophet/train/loss.py:136-141`; `model.py:437-438`, `loss.py:139` | (a) "Expected LM loss over stopping times" was `Σ_i mean(p[...,i]) · mean(loss_i)` — its gradient w.r.t. `p[b,t,i]` is the **same value at every position** *(measured: 1 unique gradient value per step; per-token gives 10)*. (b) `hidden_per_step` was the coda output **before** `norm_out` *(measured: rms 0.100 vs 0.999)*, so the per-step losses were computed on near-uniform logits. | The halting head, which D4b/W1 make "Requis", could only learn one constant distribution. | **S1** | Per-token expectation; normalise before projecting; a learnability test. | **fixed** *(measured: 16 unique gradient values per step; step read-outs rms 0.9995 = `hidden`)*; `loss.py:60,172`; tests `test_b2_*` |
| B3 | `model.py:238-239, 156-157`; `config.py:337` | `residual_scaling` documented as "at init" was a **forward-time** multiplier on every branch, forever: 0.112 (main), 0.151 (mini), **0.102 for the Qwen3-1.7B conversion** *(measured)*. Copied donor blocks computed `x + 0.10·f(x)`. | Conversion did not reproduce the donor's function in any layer; "85 % coverage" was a tensor count. | **S1** | Init-time scaling; converter sets it off; donor-block ≡ converted-block test. | **fixed** — multiplier removed, init-time scaling on output projections, `convert/plan.py:184 residual_scaling=False`; tests `test_b3_*` |
| B4 | `layers.py:330-335` | With a cache and `s > 1`, full attention used `is_causal=True`, whose mask is top-left aligned. *(measured: max |full − chunked| = 1.30 for a 4-token chunk after 8 cached tokens)* | Any multi-token continuation with a cache silently wrong in every full-attention layer. | **S2** | Explicit mask on absolute positions when a cache is present. | **fixed** *(measured: 0.0)* — `_position_mask` on absolute positions; `test_b4_chunked_continuation_matches_full_forward_in_full_attention` |
| B5 | `layers.py:321, 195-219, 342-350` | SWA evicted **before** attending and masked on buffer indices. *(measured: 0.80 for a 24-token prefill, window 8 + 2 sinks)* | Every prompt longer than `sliding_window + 1` (2049 on the shipped configs) prefilled with a cache wrong in every SWA layer. | **S2** | Attend, then evict; absolute positions. | **fixed** *(measured: 0.0 prefill, 1.1e-7 chunk across an eviction)* — `AttentionCache.append`/`evict` split; tests `test_b5_*` |
| B6 | `model.py:431-446, 482` | (a) Probe coda passes were cache-free → at decode time the halting decision came from a coda that saw only the current token. (b) Early exit left core slots for skipped iterations stale *(measured: `seen=5` vs `position=6`)*. (c) `loop_k=k` reported after `break`. (d) Threshold on `.mean()` over batch **and** sequence. | `halt_threshold` at inference → fluent, wrong output; the 07_WALLS "vérifié par test" covered the no-threshold case only. | **S2** | Probes read the real context; pin or advance slots after an early exit; report real k; per-sequence threshold. | **mostly fixed** — probes read shallow copies of the slots; `loop_k` = iterations run *(measured: 1)*; the cache now carries a depth ceiling that "may only shrink", so the stale deeper slots are retired by contract (`model.py:82-88`, `test_b6_a_cache_refuses_a_deeper_call…`); threshold is `.all()` per position/sequence. **Open residual:** full-forward vs incremental `halt_probs` still differ by **3.5e-3** *(measured)* while logits agree to 4e-7; the regression test (`test_review_fixes.py:240`) passes only because its tolerance is 1e-2. Something in the probe copy path is not equivalent and should be understood, not tolerated. |
| B7 | `config.py:509-513`, `budget.py:256`, `model.py:391-394` | `memory.layers` validated against the global depth, budgeted at the global index, read at the **section-local** index in `trunk`/`coda` only. *(measured: `layers=(5,)` on a 2/2/2 model validates, allocates, and is never read)*. `kind="fast_weight"` validated + budgeted, built nothing. | A headline feature could be "enabled" as a no-op. | **S2** | Key ledgers by `(section, idx)`; validate against the layout; drop `fast_weight`. | **fixed** *(measured: `ledgers == ["output"]`, writing to it changes the logits)* — `memory.mount ∈ {output, coda}`, coda indices coda-local and validated, `fast_weight` refused; tests `test_b7_*`, `test_b8_fast_weight_is_refused_as_unimplemented` |
| B8 | `consolidate.py:84, 134, 391`; `model.py:272-273, 392-394`; `ledger.py:90, 112` | Three views of one ledger: in-model read on the pre-norm residual; `consolidate` targets in post-`norm_out` space; model-built ledger with `write_lr=0.01, decay=0.999` vs `LedgerConfig` defaults `1.0/1.0` used for every published number; `query` a trainable Linear routed to Muon while the docstring says keys are frozen. | Offline-consolidated ledgers could not be mounted; integrated ledger took 1 % of the "exact" step; training moved the addressing under stored associations. | **S2** | One read point; aligned defaults; frozen `query`. | **fixed** *(measured: `write_lr=1.0, decay=1.0`, no trainable parameter under `ledgers.*`; `output` mount reads after `norm_out`, the space `consolidate` writes in)*; tests `test_b7_hidden_stays_ledger_free…`, `test_b8_*` |
| B9 | `decontaminate.py:131-134, 94, 145-149` | (a) Occurrences counted, not distinct n-grams *(measured: one trigram repeated 5× → containment 0.625)*. (b) Items under 15 words substring-matched *(measured: "Yes" / "The answer is 4" rejected innocuous sentences)*. | Rule 5 routes every source through this; clean data silently discarded. | **S2** | Distinct hashes; word boundaries; minimum item length. | **fixed** *(measured: repeated trigram → no hit; "yes we can" and "the answer is 42" pass; a document that literally contains "What is the capital of France?" is still rejected, which is the intended behaviour)*; tests `test_b9_*` |
| B10 | `scripts/train.py:89-91`; `loop.py:63, 194-237`; `optim.py:131`; `budget.py:341-349` | Trainer fp32 end to end (no autocast, `TrainConfig.dtype` unread, fp32 grads, fp32 Muon momentum, no checkpointing, no TF32) while the gate assumed bf16 + fp32 master + 8-bit optimiser + checkpointing. *(measured: gate printed 37.1 GB "fits"; real static state 42.8 GB before activations; reference scan alone 144 GB at 8k tokens)* | "Fits" for a run that did not; MFU plan 16× off. | **S2** | Autocast/TF32/checkpointing; gate with real dtypes. | **fixed** — `TrainConfig.dtype="bfloat16"` autocast, `allow_tf32`, `gradient_checkpointing` honoured by `ProphetBlock` (`model.py:174-184`), `training_memory` defaults describe the trainer, probe passes and the MTP block counted in activations; `test_b10_training_memory_defaults_describe_the_trainer_as_built`. `train.py` now refuses a non-smoke run without `fla` unless `--allow-slow-scan`. |
| B11 | `loop.py:170, 181-182`; `model.py:413, 403` | Only the CPU RNG checkpointed; `randn_like` state init and depth sampling on the device generator. | Resume bit-identical on CPU only. | **S2** | Save CUDA RNG; per-step seeded generator. | **fixed** — `cuda_rng` in the state dict (`loop.py:228, 241-242`), explicit generator threaded (`model.py:448-454`); tests `test_b11_*`, `test_b22_*` |
| B12 | `budget.py:513, 517-521`; `build_configs.py:88`; `model.py:437` | `avg_loop_k = 4.5` vs log-uniform *(measured E[k] = 3.38)*; halting probe passes and the MTP block not counted → "22.4B tokens" ≈35–40 % too high. | The numbers that set the plan wrong in the flattering direction. | **S2** | E[k] from the distribution; count probe + MTP passes. | **fixed** *(measured: `avg_loop_k = 3.35`, tokens = **16.1B**, `activation_layers = 33`)*; README/01_ARCHITECTURE now say 16.1B; tests `test_b12_*` |
| B13 | `convert/weights.py:165-182`; `convert/plan.py:158-191` | GDN seed paired head `h`'s queries with donor heads `2h, 2h+1`'s values/outputs (whole-matrix tiling); `rope_theta` 500k vs donor 1e6; `swa(2048)` + sink on layers the donor ran as full attention; `residual_scaling=True`. | "As close to the donor's attention as possible" was false. | **S3** | Per-head tiling; donor `rope_theta`; match attention kinds; scaling off. | **mostly fixed** — per-head `repeat_interleave` (`weights.py:170-190`), `rope_theta=donor.rope_theta`, `residual_scaling=False`; `test_b13_gdn_seed_keeps_each_head_with_its_own_values`. **Open:** `prophet_config_for_donor` still puts `swa(2048)`+1 sink on every odd copied layer (`plan.py` pattern `["swa","full_attn"]`); the donor ran all of them as full attention, so half the copied prelude/coda blocks change function on prompts > 2048 tokens. |
| B14 | `scripts/train.py:37-48, 143-144` | Handler set `_STOP`, which nothing read; installing it removed `KeyboardInterrupt`. *(observed: `timeout 500` sent SIGTERM; the process ignored it, ran its remaining steps and wrote `ckpt_slot0.pt` ~40 s later)* | Only SIGKILL stopped a run. | **S3** | Read the flag per step. | **fixed** — `Trainer.stop_requested` read once per step (`loop.py:153, 372`); `test_b14_trainer_honours_a_stop_request` |
| B15 | `layers.py:478-484, 512` | The only path that can train (`fla`) was `pragma: no cover`, passed `head_first=False` (removed upstream), fp32 gates next to bf16 q/k/v, and expected the scan's `(b,h,dv,dk)` layout where FLA uses `[B,H,K,V]`. | `TypeError` on the first A100 step, or a silently transposed state across devices. | **S3** | Pin `fla`; explicit transpose; GPU equivalence test. | **partly fixed, still unexecuted** — `head_first` gone, dtypes cast, state transposed both ways with the layout stated (`layers.py:545-558`), `scale=1.0`; `tests/test_gpu.py` has the equivalence test under `skipif(not HAS_FLA)`. The code's own comment: "This path has not been executed in this repository". **Open until it runs on a GPU.** |
| B16 | `session.py:111` | `.numpy()` on a tensor still on the model's device. | `TypeError` for any CUDA model. | **S3** | `.cpu()`. | **fixed** (`session.py:113`) |
| B17 | `streaming.py:66-81`; `mixture.py`; `scripts/train.py:116-123` | No phases in the loader; silent wrap past the 4-epoch cap; nothing turned a dataset or a file into a `TokenSource`; no tokenizer artifact or training script; decontamination in no path; `train.py` returned 2 for any non-`--smoke` run. | Nothing trained on real data. | **S3** | Build the pipeline. | **addressed, not reviewed** — `prophet/data/corpus.py` (files/Hub, phased loaders, epoch cap raised at draw time, decontamination in the path), `train.py --data-root/--hub`, `tests/test_corpus.py`. Post-snapshot code; not read here. |
| B18 | `session.py:119-135, 166` | `extract_session` drops every attention cache; restore sets `cache.position` → prelude/coda attention (and sinks) start empty at position N. | "Resume where it stopped" false for the attention context. | **S3** | Persist the bounded windows + sinks, or document. | **open, now documented** (`session.py:126-134`: "Persisting the (bounded) windowed caches and sinks is the obvious next step and is not done here"). |
| B19 | `model.py:388-390` inside probe passes | Coda MoE routers got `k+1` bias nudges and `(k+1)×` aux loss per step. | Coda routers balanced 4× faster; aux dominated by coda. | **S4** | Suppress stats/updates in probes. | **fixed** — `probe_mode` suppresses stats and bias steps; bias steps are recorded and applied after backward (`moe.py:55-73`), which also fixed a `CheckpointError` the author found under activation checkpointing. |
| B20 | `optim.py:187` | `Linear(d,1)` heads routed to Muon. | Harmless, unintended. | **S4** | Route `min(shape)==1` to AdamW. | **fixed** (`optim.py:189-190`); `test_b20_*` |
| B21 | `layers.py:107-108` | `rope_scaling="linear"` scaled `theta` (NTK), not positions. | Wrong recipe if used. | **S4** | Position interpolation. | **fixed** (`layers.py:118-120`) |
| B22 | `model.py:343, 403` | `"poisson"` ignored the generator; none was passed. | Depth sampling on global RNG. | **S4** | Thread a generator. | **fixed** (`model.py:448-454`); `test_b22_*` |
| B23 | `moe.py:165-171`; `streaming.py:161-164`; `tokenizer.py:255` | 128 `nonzero` syncs per MoE layer; O(n) packer pop; unbounded encode cache. | Throughput only. | **S4** | Sort-and-split dispatch; deque; LRU. | **open** (`moe.py:216` still loops `nonzero` per expert) |

### 2.1 Minimal reproductions (snapshot 1)

These reproduced the S1/S2 bugs on snapshot 1; on snapshot 2 they print the "fixed" values quoted in
the status column. Run from the repository root with `python3` (CPU, torch 2.14, no `fla`).

```python
# B1 — forget gate zeroed by model init (snapshot 1: tensor([0.]); snapshot 2: tensor([3.]))
from prophet.config import ProphetConfig; from prophet.modeling.model import ProphetModel
m = ProphetModel(ProphetConfig.from_json("configs/prophet_500m_probe.json"))
print(m.sections["core"][0].mixer.a_proj.bias.unique())

# B4 — chunked prefill, full attention (1.297 → 0.0)
import torch; from prophet.modeling.layers import CausalSelfAttention, RotaryEmbedding, AttentionCache
a = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16).eval(); r = RotaryEmbedding(16, theta=1e4)
x = torch.randn(1, 12, 64); cos, sin = r(torch.arange(12)[None])
with torch.no_grad():
    full = a(x, cos=cos, sin=sin); c = AttentionCache()
    a(x[:, :8], cos=cos[:, :8], sin=sin[:, :8], cache=c)
    chunk = a(x[:, 8:], cos=cos[:, 8:], sin=sin[:, 8:], cache=c)
print((full[:, 8:] - chunk).abs().max())

# B5 — SWA prefill longer than the window, with a cache (0.801 → 0.0)
s = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16, window=8, sink_tokens=2).eval()
x = torch.randn(1, 24, 64); cos, sin = r(torch.arange(24)[None])
with torch.no_grad(): print((s(x, cos=cos, sin=sin) - s(x, cos=cos, sin=sin, cache=AttentionCache())).abs().max())

# B2 — the ponder gradient was the same at every position ([1,1,1] → [16,16,16,16] on the real loss)
B, T, K = 2, 5, 3; p = torch.rand(B, T, K).softmax(-1).requires_grad_(); L = [torch.rand(B, T) for _ in range(K)]
g = torch.autograd.grad(sum(p[..., i].mean() * L[i].mean() for i in range(K)), p)[0]   # snapshot-1 loss.py:136-141
print([g[..., i].unique().numel() for i in range(K)])

# B9 — decontaminator double counting (containment 0.625 → no hit)
from prophet.data.decontaminate import Decontaminator
d = Decontaminator(n=3, threshold=0.5); d.add_benchmark("b", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"])
print(d.check("alpha beta gamma. " * 5 + "unrelated weather"))
```

### 2.2 Re-verification on snapshot 2 (2026-09-05 02:10 UTC), measured

| Check | Snapshot 1 | Snapshot 2 |
|---|---|---|
| `count_parameters` / `num_parameters()` on the four shipped configs + Qwen3-1.7B conversion | ≤1.2e-4 (after the mid-review budget fix) | ≤1.2e-4 |
| GDN `a_proj.bias` in a built model | 0.0 (α = 0.5) | 3.0 (α = 0.953) |
| Forward-time residual multiplier | 0.112 / 0.151 / 0.102 | attribute gone; init-time scaling |
| Full-attention 4-token chunk after 8 cached tokens | 1.297 | 0.0 |
| SWA 24-token prefill vs no cache (window 8, 2 sinks); chunk across an eviction | 0.801 / — | 0.0 / 1.1e-7 |
| Ponder gradient: unique values per step across positions | 1 | 16 (all positions differ) |
| `hidden_per_step` rms vs `hidden` rms | 0.100 vs 0.999 | 0.9995 vs 0.9995 |
| `halt_probs` full forward vs incremental decode (untrained 64-d model) | 4.7e-3 | **3.5e-3 (still non-equivalent; test tolerance 1e-2)** |
| Early exit: iterations run / `loop_k` reported / deeper slots | 1 / 4 / stale | 1 / 1 / retired by cache ceiling |
| `memory.layers=(5,)` on a 2/2/2 recurrent model | validates, unread | `mount="output"` read; coda mounts validated coda-local |
| Model-built ledger `write_lr` / `decay` / trainable `query` | 0.01 / 0.999 / yes | 1.0 / 1.0 / no |
| Decontaminator: one trigram ×5; "yes we can"; "the answer is 42" | 0.625 hit; rejected; rejected | no hit; passes; passes |
| Session state, `prophet_mini`, k = 2 | 10.98 MB | 10.98 MB (06_MEMORY now says ~11 Mo; `session.py:9` still says "under a megabyte") |
| `tokens_affordable(prophet_main)`: `avg_loop_k`, tokens | 4.5, 22.2B | 3.35, 16.1B |
| Trainer: autocast / TF32 / activation checkpointing / CUDA RNG / stop flag | no / no / no / no / no | yes / yes / yes / yes / yes |
| Converter value seeding | whole-matrix `repeat` | per-head `repeat_interleave` |
| FLA call: `head_first` / state transpose / executed anywhere | yes / no / no | no / yes / **no** |
| Smoke path (`--smoke`, tiny config, 4×128 tokens, 1 thread) | 11.7 / 8.1 / 2.8 s per step, loss 7.67 → 7.48; resume from checkpoint works | not re-timed |
| Tests collected | 295 | 436 (`tests/test_gpu.py` skipped without CUDA) |
| Tests run here | 109/109 passed on `test_modeling`, `test_config_budget`, `test_convert` (7 min under CPU contention); full suite killed by a 10-minute timeout | full suite: see the note at the end of this file |

## 3. Claims the code does not honour

| # | Claim | Where it is made | What the code actually did (snapshot 1) | Status (snapshot 2) |
|---|---|---|---|---|
| C1 | "masque d'attention comme objet de première classe supportant les segments bidirectionnels, chemin `inputs_embeds`" | `docs/01_ARCHITECTURE.md:213-216`, `docs/00_PROBLEM_LANDSCAPE.md:317` | No mask object; `forward` had no `inputs_embeds`; `bidirectional_spans`/`adapter_mount_points` never read. | not re-checked (`test_hygiene.py::test_removed_fields_are_gone` suggests the flags were removed) |
| C2 | "Scale residual branches … **at init**" | `config.py:337` | Forward-time multiplier (B3). | fixed |
| C3 | "Initialise the forget gate near 1" | `layers.py:435-439` | Zeroed by model init (B1). | fixed |
| C4 | "the halting head learns which iterations were actually good enough to stop at" / "profondeur dépendant de l'entrée" | `loss.py:132-135`; `07_WALLS.md` §A.3, §B; D4b | Position-independent gradient (B2). | fixed |
| C5 | "For Prophet-mini the whole state is under a megabyte" / "~0.6 Mo" | `session.py:9`; `06_MEMORY.md` §2 | 10.98 MB at k = 2 *(measured)*. | **open in `session.py:9`**; `06_MEMORY.md:35` now says "~11 Mo (mesuré, mini)" |
| C6 | "registre ~33 Mo (mini)" | `06_MEMORY.md` §2 | 33.5 MB is `LedgerConfig()` with `dim=256`; mini's config gives 10.5 MB nominal / 21 MB fp32; `n_bytes(dtype_bytes=2)` reports half the buffer. | **open** (`06_MEMORY.md:36`) |
| C7 | "cible à 1e-7 près en une seule écriture" | `06_MEMORY.md` §3 | With `write_lr=1.0`; the model built 0.01. | fixed (defaults aligned) |
| C8 | "Keys are **frozen after initialisation**" | `ledger.py:90` | `query` trainable, routed to Muon. | fixed |
| C9 | `t = m(h⁻) + λ(h⁺ − h⁻)` | `consolidate.py:12-16` docstring | Code uses the absolute target; `06_MEMORY.md` §4 calls the docstring's formula wrong. | not re-checked |
| C10 | "Muon: **2 bytes/param of state**" / "2 octets/paramètre" | `optim.py:15`; `docs/03_TRAINING.md:30` | fp32 momentum, 4 bytes *(measured)*. | **open**: `optim.py:15` and `03_TRAINING.md:30` still say 2; `03_TRAINING.md:150` now says 4 — the document contradicts itself |
| C11 | "Entraînement : **BF16 avec poids maîtres FP32**" | `docs/03_TRAINING.md:127` | No autocast anywhere. | fixed (`03_TRAINING.md:148`: autocast BF16, fp32 params, TF32) |
| C12 | "training memory 37.1 GB … fits" | `scripts/train.py`, `python -m prophet.budget` | Assumed an 8-bit optimiser and checkpointing that did not exist. | fixed (`training_memory` defaults describe the trainer; 01_ARCHITECTURE now 47.1 GB) |
| C13 | "1.72B → **0.97B**, **89 %** hérités" | `README.md:127,132`; `01_ARCHITECTURE.md:248-250`; `05_ROADMAP.md:15` | `convert_donor.py --plan-only`: 1.02B, 85 %. | **open in `README.md:137`** ("~970M … 89 %"); `README.md:142` and `01_ARCHITECTURE.md:248` now say 1.02B |
| C14 | "Configurations … produites par `scripts/design_search.py`" | `README.md:59`, `01_ARCHITECTURE.md:123` | Shipped JSONs come from `build_configs.py` with `n_kv_heads=2` and `halting="ponder"`; the search used `n_kv_heads=1`, no halting. | not re-checked |
| C15 | "**235 tests passent**" | `README.md:76,111` | 295 collected. | fixed ("~430"; 436 collected) |
| C16 | "a resumed run … identical weights" / "byte-identical" | `scripts/train.py:5-7`; `streaming.py:14-16`; CLAUDE.md | CPU only (B11). | fixed |
| C17 | "harnais d'évaluation à trois niveaux" | `README.md`; `docs/04_EVAL.md:39-41` | `run_suite` runs caller-supplied lambdas; no runner for any named task; only `evaluate_bpb` executes a model. | not re-checked (`eval/` has grown an agentic benchmark since) |
| C18 | "**One code path.** Training, prefill and single-token decoding share the same forward" | `layers.py:13-14` | Chunked prefill wrong (B4/B5). | fixed |
| C19 | "Checkpoint and exit cleanly if the platform gives us any warning" | `scripts/train.py:41-44` | Flag never read; Ctrl-C disabled (B14). | fixed |
| C20 | "201 Mo, contre **158 Mo de poids**" | `docs/07_WALLS.md` §D.4 | Matches no config. | not re-checked |
| C21 | "40B is what `prophet.scaling` says 300 A100-hours buys" | `recipes.py:172-177`; `docs/02_DATA.md` | Budget said 22.2B / 52.1B; now 16.1B / 52.1B. | not re-checked |
| C22 | "configs/ … **jamais à la main**" | `CLAUDE.md:54` | `prophet_tiny_smoke.json` not in `build_configs.CONFIGS`. | not re-checked |
| C23 | "`n_kv_heads` = `n_heads / 8`" | `01_ARCHITECTURE.md` §4 | Shipped 12/2, 10/2, 12/3. | not re-checked |
| C24 | `kv_compression="mla"` shrinks the KV cache | `config.py:111-113`, `budget.py` | Estimator only; model built plain GQA. | not re-checked |
| C25 | `activation`, `norm_kind`, `dropout`, `router_dtype`, `halting="entropy"` are switches | `config.py:217,335,343,240,202` | None honoured. | `norm_kind`, `dropout`, `max_seq_len` now honoured and tested (`test_hygiene.py`); **`FeedForwardConfig.activation` still unread** *(measured by field scan)* |
| C26 | "Un champ de configuration que rien ne lit est un bug" | `CLAUDE.md` | Twenty such fields at snapshot 1. | one left (`activation`) |

**Honoured (checked at snapshot 1, still true):** budget ≡ model parameter count on all shipped
configs and the converted donor; `linear_beta_max` reaches `GatedDeltaNet`; the delta-rule scan
matches its docstring formula; NoPE reaches `CausalSelfAttention`; the probe-pass cache fix (10
positions for 10 tokens); checkpoint rotation, mixture sums, tokenizer invariants and the allocator
behave as documented.

## 4. Dead and unread code

**Config fields no code outside `prophet/config.py` read at snapshot 1** (grep over `prophet/` and
`scripts/`, excluding each field's definition line). Snapshot-2 status in the last column.

| Field | Line (s1) | Status s1 | Status s2 |
|---|---|---|---|
| `FrontendConfig.patch_target_bytes`, `patch_entropy_threshold`, `local_window` | 56, 59, 65 | unread; byte frontend `NotImplementedError` yet fully budgeted | removed or read (field scan: no longer flagged) |
| `FrontendConfig.patch_max_bytes` | 58 | read only by `validate()` | idem |
| `MixerConfig.kv_compression`, `kv_lora_rank` | 111-113 | budget only; model has no MLA | not re-checked |
| `RecurrentCoreConfig.halting="entropy"` | 202 | accepted; only `"ponder"` builds a head | not re-checked |
| `FeedForwardConfig.activation` | 217 | unread; SwiGLU hard-wired | **still unread** |
| `FeedForwardConfig.router_dtype` | 240 | never passed to `MoERouter` | no longer flagged |
| `MemoryConfig.kind="fast_weight"` | 259 | validates, budgeted, builds nothing | refused (`test_b8_fast_weight_is_refused_as_unimplemented`) |
| `MemoryConfig.update_rule`, `surprise_threshold`, `persist_across_sessions`, `max_persisted_writes` | 265-274 | unread | `surprise_threshold` now drives gating (`test_hygiene.py`), `max_writes` enforced; others no longer flagged |
| `ModalityConfig.bidirectional_spans`, `adapter_mount_points` | 312, 315 | unread (C1) | removed (`test_removed_fields_are_gone`) |
| `ProphetConfig.norm_kind`, `max_seq_len`, `dropout` | 335, 342, 343 | unread | honoured and tested (`test_hygiene.py`) |
| `TrainConfig.dtype`, `seed` | `loop.py:63, 61` | unread | `dtype` drives autocast; `seed` not re-checked |
| `TrainConfig.confidence_weight` | `loop.py:48, 142-145` | resolved, never passed to `compute_loss`; no targets anywhere → confidence head never trained | not re-checked |
| `MixerConfig.nope_layers` | 135 | fixed during snapshot 1 | fixed |

**Dead functions, parameters and objects at snapshot 1** (not re-checked unless stated):

- `count_parameters(cfg, loop_k=...)` — `loop_k` accepted, unused (`budget.py:227`).
- `build_schedule` (`schedule.py:127`) never called; `CosineSchedule` tests only.
- `SparseMoE.aux_loss()` (`moe.py:178`) never called.
- `Mixture.license_warnings` (`mixture.py:202`) never called by any script.
- `ProphetTokenizer.fertility`, `Decontaminator.index_bytes`, `MixtureSampler.empirical_shares` — tests or nothing.
- `CheckpointManager.verify` ("run at startup") — `train.py` never called it.
- `consolidate.py:123, 141` — `slots` set filled and discarded.
- `scripts/train.py:37-48` — `_STOP` (fixed, B14).
- `GatedDeltaNet.allow_fused` never set `False`; `HAS_FLA` exported, reported by nothing (s2: `train.py` now reads it).
- `prophet/kernels/` — empty directory, no `__init__.py`, listed in `CLAUDE.md` as a package (s2: still empty).
- `RecurrentState.seen` written, never read; `AttentionCache.seen` read by one test (s2: `seen` now drives absolute positions — live).

## 5. Doc drift

| Where | Said (s1) | Was | Status (s2) |
|---|---|---|---|
| `README.md:76,111` | 235 tests | 295 collected | fixed ("~430"; 436) |
| `README.md:127,132`; `01_ARCHITECTURE.md:248-250`; `05_ROADMAP.md:15` | 0.97B, 89 % | 1.02B, 85 % | **`README.md:137` still "~970M … 89 %"**; the other two lines fixed |
| `README.md:65-69`, `01_ARCHITECTURE.md:129-133` | 22.4B tokens, "produites par design_search.py" | halting/MTP passes uncounted; search config ≠ shipped config | tokens now 16.1B; provenance claim not re-checked |
| `01_ARCHITECTURE.md` §4 | GQA `n_heads/8`, window 2048 "identical on every attention layer" | shipped 12/2, 10/2, 12/3; schema default 4096 | not re-checked |
| `01_ARCHITECTURE.md` §6 | mask object, `inputs_embeds` | absent | not re-checked |
| `06_MEMORY.md` §2 | 0.6 MB session, 33 MB ledger | 11 MB; 10.5/21 MB | session fixed (~11 Mo); **ledger line still 33 Mo** |
| `06_MEMORY.md` §3, §5 | write exact to 1e-7 | with `write_lr=1.0` only | fixed (defaults aligned) |
| `docs/03_TRAINING.md:30,127` | Muon 2 B/param; BF16 + FP32 master | 4 B; fp32 only | **line 30 still "2 octets"** while line 150 says 4; line 127/148 fixed |
| `prophet/train/optim.py:15` | "2 bytes/param" | 4 | **open** |
| `prophet/memory/session.py:9` | "under a megabyte" | 11 MB | **open** |
| `docs/02_DATA.md`, `recipes.py:172-177` | 40B tokens | 22.2B → 16.1B / 52.1B | not re-checked |
| `docs/07_WALLS.md` §D.4 | 158 MB of weights | matches no config | not re-checked |
| `docs/07_WALLS.md` §A.3 | halting implemented, head receives gradient | position-independent gradient | fixed in code; doc not re-checked |
| `CLAUDE.md:54` | configs never hand-written | `prophet_tiny_smoke.json` is | not re-checked |
| `CLAUDE.md` | "173 tests" → "~300" → "~430" during the review | 295 → 436 | current |
| `prophet/train/optim.py:18-20` | routers/heads keep AdamW | `Linear(d,1)` heads went to Muon | fixed |

## 6. What breaks on a real run first

**Snapshot 1** — walking `python scripts/train.py --config configs/prophet_main.json --steps 40000` on an A100:

1. **Exit code 2 before any model is built.** The gate printed "37.1 GB, fits" (wrong), then "Real-corpus streaming is not wired up yet" — no dataset → `TokenSource` path, no tokenizer artifact, no phase schedule, decontamination in no path.
2. **If a source were wired:** 3.83B fp32 parameters allocated on the **host** first (15.3 GB; a standard Colab VM has 12.7 GB → host OOM), then GPU: ≈43 GB static with Muon momentum.
3. **First core iteration:** `HAS_FLA=False` → the Python reference scan, `seq_len × 4 layers × k` sequential steps with every `S_t` retained for backward: **144 GB for 8 192 tokens** *(computed)* → CUDA OOM. With `fla`: `head_first` TypeError or a silently transposed state.
4. **If that were fixed:** fp32 without TF32 (19.5 TFLOPS, 6 % of the bf16 peak the 35 % MFU plan assumed); a device sync per expert per MoE layer; halting running the coda `k` extra times with grad.
5. **Training would look healthy** while α = 0.5 (B1), branches ×0.112 (B3), the halting head fitting a constant (B2), coda routers balancing 4× faster (B19), the confidence head receiving no gradient.
6. **Step 200:** ≈31 GB `torch.save` + SHA-256 per checkpoint to Drive; 60+ GB of Drive for two slots.
7. **Session dies, rerun:** resume not bit-identical (B11); Ctrl-C did nothing (B14).
8. **First inference:** prompts > 2 049 tokens wrong in every SWA layer (B5); block continuations wrong in every full-attention layer (B4); `halt_threshold` → confident nonsense (B6); `model_fingerprint` crashes on CUDA (B16).

For calibration, the path that did run: `--smoke` on `prophet_tiny_smoke.json` (6.4M parameters,
4×128 tokens/step) took **11.7 s, 8.1 s, 2.8 s** for its first three steps single-threaded *(measured;
the spread is k drawn from log-uniform 1..4)* — the reference scan makes even the toy path two to
three orders of magnitude slower than a dense model of that size; loss 7.67 → 7.48; resume from the
checkpoint worked ("resumed from step 1", checkpoint at step 3).

**Snapshot 2** — the same walk, as far as I can see without a GPU: the gate now describes the trainer
as built and `train.py` refuses a real run without `fla` unless `--allow-slow-scan`; steps 2, 4, 5,
7 and 8 are fixed in code with regression tests. **What breaks first now is B15**: the FLA path is the
first thing a real run executes and it has never been executed anywhere — the GPU test exists but has
only ever been skipped. After that: whatever the unreviewed corpus loader does on the first real
shard, then the 31 GB checkpoints (unchanged), then the session-restore contract (B18).

## 7. Test gaps (ranked, snapshot 2 status in brackets)

1. **The training path has never run.** `tests/test_gpu.py` exists but is `skipif(not CUDA)` and `skipif(not HAS_FLA)`; until it runs green on an A100 the FLA layout, dtype and `scale` contract (B15) is a belief. [open]
2. **Halting probe equivalence is asserted at 1e-2** (`test_review_fixes.py:240`) while logits agree to 4e-7 and `halt_probs` differ by 3.5e-3 *(measured)*. A 1e-2 tolerance on a probability hides whatever is not equivalent in the probe-copy path. [open]
3. Cache equivalence for chunked prefill / prefill-beyond-window at the attention and model level. [fixed: `test_b4_*`, `test_b5_*`]
4. Halting learnability on a synthetic task with token-dependent optimal depth. [partly: `test_b2_ponder_gradient_differs_across_positions` checks the gradient, not learning]
5. Layer-local init preserved through model init. [fixed: `test_b1_*`]
6. Resume determinism with the device generator. [fixed: `test_b11_*`, `test_b22_*`; real CUDA run still pending]
7. MoE + recurrence + halting together. [fixed indirectly: probe suppression and post-backward router updates are tested; a combined config test is not re-checked]
8. Conversion tested function-wise (donor block ≡ converted block on the same input). [partly: `test_b3_*`, `test_b13_*` test tensors and scaling; the swa/full_attn kind mismatch on copied layers (B13 residual) has no test]
9. `training_memory` vs a real allocation. [partly: `test_b10_*` checks the defaults describe the trainer, not that the estimate matches allocated bytes]
10. Memory integration on a recurrent model; consolidated ledger read through the model. [fixed: `test_b7_*`]
11. Decontaminator false positives. [fixed: `test_b9_*`]
12. Tokenizer at corpus scale; a shipped vocabulary. [not re-checked; `test_corpus.py` exists]
13. Streaming: long documents across a resume; multi-phase; epoch cap at draw time. [not re-checked; `corpus.py` claims all three]
14. Tests that test the mock (harness lambdas, random-token BPB, `*_renders`). [unchanged]
15. Session restore after attention-context loss: no test states what a restored session is allowed to differ by (B18). [open]

## 8. The ten things to fix first (snapshot 2)

1. **Run `tests/test_gpu.py` on an A100 with `fla` installed** and keep the transcript in the repo — B15 is the only thing standing between the tree and a first real step, and it is unverified by construction.
2. **Explain the 3.5e-3 halt_probs mismatch** between full forward and incremental decode (B6 residual) and tighten `test_review_fixes.py:240` to 1e-6; if the probe copy is not equivalent, halting decisions at decode time are still not the trained ones.
3. **B13 residual:** copied donor layers should keep the donor's attention kind; `swa(2048)` + 1 sink on layers Qwen3 ran as full attention changes their function on any prompt over 2 048 tokens.
4. **B18:** persist the bounded SWA windows and sinks in `SessionMemory`, or make `restore_session` refuse to set `cache.position` without them.
5. **Fix the three stale doc lines:** `README.md:137` (0.97B / 89 % → 1.02B / 85 %), `prophet/train/optim.py:15` + `docs/03_TRAINING.md:30` (2 → 4 bytes/param), `prophet/memory/session.py:9` and `docs/06_MEMORY.md:36` (session 11 MB; ledger 10.5 MB nominal / 21 MB fp32 for mini's config).
6. **`FeedForwardConfig.activation`:** the last unread field — implement `geglu`/`relu2` or delete it (CLAUDE.md's own rule).
7. **Re-check the "not re-checked" rows** of §3/§4/§5 (C1, C9, C14, C17, C20–C24; `kv_compression`, `halting="entropy"`, `TrainConfig.confidence_weight`, `prophet/kernels/`) — they were true at snapshot 1 and nothing in the fix tests names them.
8. **Independent read of the post-snapshot layer** (`prophet/data/corpus.py`, `prophet/agent/`, action heads, `token_depth`): it now carries the real-data path and the agent loop, it was written in the same sitting as the fixes, and no one has read it adversarially.
9. **B23:** sort-and-split MoE dispatch instead of 128 `nonzero` syncs per layer; a deque in `SequencePacker`; a bounded encode cache — none of it changes semantics, all of it decides whether the MFU plan is reachable.
10. **Keep the discipline that made this review short the second time:** every fix here landed with a test named after the finding. Make that the rule for the next review (A2) too — and pin the tree before starting it; this one was reviewed while it moved.

---

*Test runs performed here:* 109/109 passed on `tests/test_modeling.py tests/test_config_budget.py
tests/test_convert.py` at snapshot 1 (7 min 26 s under CPU contention); the full suite could not
complete inside the 10-minute tool limit with three test processes sharing four cores.

*Snapshot 2, full suite, run as four disjoint file groups with `OMP_NUM_THREADS=1`
(2026-09-05 02:2x UTC):* **433 passed, 3 skipped (`tests/test_gpu.py`, no CUDA), 0 failed — 436 of
436 collected** — A `test_modeling`+`test_review_fixes` 77 passed in 13 s; B `test_training`+`test_memory`
75 passed in 43 s; C data/corpus/tokenizer/eval/analysis/plan/budget/hygiene 173 passed in 7 s;
D convert/agent/action/render/token_depth/gpu 108 passed + 3 skipped in 40 s. Total ≈ 1 min 45 s
single-threaded: the earlier 10-minute overruns were torch thread oversubscription across concurrent
processes, not slow tests. Worth a `pytest.ini` / `conftest.py` setting `torch.set_num_threads(1)`.
