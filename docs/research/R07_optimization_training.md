# R07 — Optimization and Training Dynamics: Maximum Loss Reduction per A100-Hour

**Track owner:** R07 · **Status:** research complete, recipe proposed · **Date:** 2026-09-03

> **Environment caveat that shaped this report.** `arxiv.org`, `huggingface.co`, `openreview.net`,
> `alphaxiv.org`, `emergentmind.com` and most paper mirrors are blocked by this session's egress
> policy. Numbers below come from (a) search-engine extraction of paper abstracts/results,
> (b) `github.com` (reachable) source and READMEs, and (c) first-principles arithmetic done here.
> arXiv IDs marked **(†)** are recalled from training data and were *not* re-verified against
> arxiv.org in this session — check them before citing externally. Every *number* in the budget
> and memory tables was computed here and is reproducible from the formulas given.

---

## 1. Problem statement

### 1.1 The hardware, stated exactly

| Property | Value | Consequence for Prophet |
|---|---|---|
| GPU | NVIDIA A100 80GB **SXM4**, GA100, **compute capability sm_80 (Ampere)** | — |
| BF16/FP16 dense tensor-core peak | **312 TFLOP/s** (624 only with 2:4 structured sparsity — not usable for dense training) | All MFU math below uses 312e12 |
| TF32 tensor-core peak | 156 TFLOP/s | FP32 fallback paths cost 2× |
| FP8 tensor cores | **NONE.** FP8 was introduced with Hopper (sm_90). A100 also has no FP4/MXFP. | **No TransformerEngine FP8 training, no FP8 GEMMs, no `torch.float8_e4m3fn` speedup.** The modded-nanogpt FP8 records are H100-only and do not port. BF16 is our floor and our ceiling. |
| Memory | 80 GB HBM2e, **~2039 GB/s** (SXM4; the PCIe 80GB card is ~1935 GB/s) | Usable is ~79.2 GiB; subtract ~0.8–1.2 GiB CUDA context + allocator fragmentation ⇒ **budget 77 GiB**. |
| FlashAttention | **FA2 only.** FA2 supports Ampere/Ada/Hopper. **FA3 requires H100/H800 (CUDA ≥12.3)**; FA4 is Hopper+Blackwell. Confirmed from `Dao-AILab/flash-attention` README. | Attention kernel ceiling ≈ 203 TFLOP/s causal @16k (65% of A100 peak), 224–227 TFLOP/s fwd-only, per the FA2 paper (2307.08691). |
| Host | Colab A100 high-RAM ≈ 83 GB system RAM, PCIe Gen4 ×16 (~25 GB/s theoretical, ~18–22 GB/s achieved) | CPU offload of optimizer state is *possible* but costs seconds/step — see §5.4. |
| Session | preempted every ~12–24 h; local disk does **not** survive VM teardown | Checkpointing is a first-class design constraint, not an afterthought (§4.7). |

> ⚠️ **Verify before you build.** Colab has historically served `A100-SXM4-**40GB**` on Pro+ far more
> often than 80GB. Run `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` at the
> top of *every* session and branch the config on the result. If it is 40 GB, every static-memory
> number in §5 must fit in **37 GiB**, which rules out everything above ~3B total parameters.

### 1.2 FLOP arithmetic

Useful compute: `C = 312e12 × MFU × 3600 × A100-hours`.
Training cost per token: `≈ 6 · N_active` (fwd+bwd; MoE counts *active* params), plus a causal-attention
term `6 · L · d_model · s` per token which is **+15% at d=2048, L=24, s=4096** and is *not* in the 6N figure.

| A100-hours | MFU 25% | MFU 35% | MFU 45% |
|---|---|---|---|
| 150 | 4.21e19 | 5.90e19 | 7.58e19 |
| **250** | 7.02e19 | **9.83e19** | 1.26e20 |
| 300 | 8.42e19 | 1.18e20 | 1.52e20 |
| 400 | 1.12e20 | 1.57e20 | 2.02e20 |

**Take C = 9.83e19 FLOPs as the planning number** (250 A100-h of *pretraining* at 35% MFU, leaving
~50–100 h for ablations, restarts, anneals and SFT out of a "few hundred hour" budget).

### 1.3 Tokens we can actually buy

| N_active | Tokens D = C/(6N) | D/N | Chinchilla multiplier (D/20N) |
|---|---|---|---|
| 0.30 B | **54.6 B** | 182× | 9.1× |
| 0.50 B | **32.8 B** | 66× | 3.3× |
| 0.80 B | **20.5 B** | 26× | 1.3× |
| 1.00 B | 16.4 B | 16× | 0.8× |
| **1.30 B** | **12.6 B** | 10× | **0.5× — undertrained** |

Compute-optimal point for C = 9.83e19 (Chinchilla 2203.15556, D = 20N): **N ≈ 905M, D ≈ 18.1B tokens.**

**This is the single most important finding in this report and it is not an optimizer finding:**
the project's stated target of **~1.3B active parameters is already *above* compute-optimal for our
budget.** At 1.3B active we can only afford 12.6B tokens — *half* of Chinchilla — so the model would
be badly undertrained and would lose to a well-trained 500M model. And a **10B-total** MoE trained on
12.6B tokens sees **1.26 tokens per total parameter**, which is roughly 1/1000 of what any published
MoE has had. Sparse models need *more* tokens per total parameter than dense models, not fewer.

### 1.4 What we are up against

| Model | Pretrain FLOPs (6ND) | Equivalent A100-h @40% MFU | × our budget |
|---|---|---|---|
| Qwen3-1.7B (36T tok) | 3.67e23 | 817,000 | **3,700×** |
| Qwen3-4B (36T tok) | 8.64e23 | 1,923,000 | 8,800× |
| SmolLM3-3B (11T tok) | 1.98e23 | 441,000 | 2,000× |
| Llama-3.2-3B (9T tok) | 1.62e23 | 361,000 | 1,650× |
| Phi-4-mini (5T tok) | 1.14e23 | 254,000 | 1,160× |
| Gemma-3-4B (4T tok) | 9.60e22 | 214,000 | 980× |
| **Prophet (250 A100-h)** | **9.83e19** | 250 | 1× |

We are **~1,000–4,000× short on pretraining compute.** No optimizer closes that. The best-measured
optimizer speedup that survives a properly tuned AdamW baseline is **1.1–1.4×** (§2.4). Therefore:

**R07's actual job is threefold, in priority order:**
1. **Don't waste compute.** A diverged run, a bad LR, a lost checkpoint, or a 25%-MFU implementation
   costs 20–100% of the budget. This dominates optimizer choice.
2. **Take the free 1.2–1.4×** from a matrix-aware optimizer and good systems engineering.
   1.3× on 250 h = **75 A100-hours recovered** = 30% more tokens.
3. **Make the budget arithmetic drive the architecture**, i.e. push back on 1.3B active / 10B total.

Beating Qwen3-1.7B must come from data quality, distillation, and inference-time architecture
(other tracks), *not* from pretraining FLOPs. R07 supplies the multiplier, not the miracle.

---

## 2. State of the art

### 2.1 Optimizer comparison table

Memory column is **optimizer-state bytes per parameter**, excluding weights/grads/master.
"Real speedup" = measured against a *carefully tuned* AdamW baseline at 100M–1.2B scale.

| Optimizer | Paper | State (B/param) | Claimed speedup | **Speedup vs tuned AdamW** | Step-time overhead | HP transfer |
|---|---|---|---|---|---|---|
| **AdamW** (BF16 states) | — | 4 (bf16 m,v) / 8 (fp32) | baseline | 1.0× | — | poor without muP |
| **Muon** | Jordan et al. 2024 blog + 2502.16982 | **2 (bf16 mom) / 4 (fp32)** — *half of AdamW* | 1.8× on nanoGPT speedrun; **~2× (52% of FLOPs)** at 16B-A3B / 5.7T tok | **1.35× @130M → 1.1× @1.2B** (2509.02046) | NS-5 = **3.33·m/T** extra FLOPs (m = short matrix dim, T = tokens/opt-step) ⇒ **1.3% at d=2048, 512K-token batch**; 1.45× per-step in large-batch distributed settings (2607.20548) | **Excellent** — spectral-norm units are natively muP-consistent |
| **Muon + RMS-match + WD** ("Moonlight") | 2502.16982 | 2–4 | ~2× at compute-optimal | as above | as above | works "out of the box" per authors |
| **MuonClip** (Muon + QK-Clip) | Kimi K2, 2507.20534 | 2–4 | 15.5T tokens, **zero loss spikes**, 1.04T-A32B | stability, not speed | negligible | — |
| **SOAP** (Adam in Shampoo eigenbasis) | 2409.11321 | **~12–48** (see §2.3) | −40% iters, −35% wall-clock | **1.35× @130M → ~1.1× @1.2B**; *wins at ≥8× Chinchilla* (2509.02046) | 1.72× per-step (2607.20548); eigendecomp O(d³) every k steps | good; +1 HP (precond. freq.) |
| **Distributed Shampoo** | 2002.09018 †, Meta impl. | ~8–40 | **28% faster** — won AlgoPerf 2024 external-tuning track (2502.15015) | ~1.2–1.28× | high; `precondition_frequency` 100s–1000s | needs grafting from prior optimizer |
| **PSGD-Kron** | Li, `kron_torch` | 4–20 (tunable via `memory_save_mode`) | — | **competitive; best at ≥8× Chinchilla** (2509.02046) | precond. update prob. anneals 1.0→0.03 by ~4k steps | LR ≈ Adam/3, WD 3–10× larger |
| **Sophia** | 2305.14342 | 8 | **2×** over AdamW | **~1.0×.** Follow-ups found the original AdamW baseline was crippled by a non-randomizing dataloader; "no significant speedup for models under 0.5B" | Hessian est. every 10 steps | — |
| **Adam-mini** | 2406.16793 | **~4** (one v per block) | 49.6% throughput / 33% wall-clock (Llama2-7B, 2×A800) | ~1.0× *algorithmically*; the throughput win was a **memory/communication** win | none | — |
| **Lion** | 2302.06675 † | **2** (momentum only) | ~equal, half memory | ≤1.0×, needs LR ≈ Adam/10, WD ×10 | none | brittle |
| **AdEMAMix** | 2409.03137 | **12** (fp32 m1,m2,v) | 1.3B model @101B tok ≈ AdamW @197B tok (**1.95× token-efficiency**) | strong at 720M with 1M-token batches (2509.01440 ranks AdEMAMix/MARS top); **not reproduced as a wall-clock win in 2509.02046** | small | **β₃=0.9999 ⇒ ~10,000-step memory horizon — see §3.5, this is disqualifying for preemptible training** |
| **MARS** | 2411.10438 † | 8–12 | variance-reduced, top-2 at 720M (2509.01440) | modest | extra grad copy | — |
| **Scion / ClippedScion** | 2502.07529 | **2** (momentum only, can be fp16) | speedups on nanoGPT "without any reliance on Adam" | comparable to Muon | low | **norm choice per layer gives HP transfer across sizes** |
| **Schedule-Free AdamW** | 2405.15682 † | 8–12 | won AlgoPerf self-tuning track, **8% faster** (2502.15015) | ~1.08× | small | removes the schedule HP |
| **Adafactor** | 1804.04235 † | **~0.05** (factored second moment) | memory only | ~0.97× (slight loss penalty) | none | — |
| **AdamW 8-bit** (blockwise) | Dettmers 2110.02861 † | **2** (1B each for m,v) | memory only, "32-bit performance" | ~1.0× | small quant/dequant | — |
| **SPlus** (stable whitening) | 2506.07254 | ~12 | faster than Adam/Shampoo on transformers | not independently replicated at 1B | medium | — |

### 2.2 Muon, precisely (verified against source)

From `KellerJordan/Muon/muon.py` and `MoonshotAI/Moonlight/examples/toy_train.py`:

* Newton–Schulz **quintic** coefficients `a=3.4445, b=-4.7750, c=2.0315`, **5 iterations** (Moonshot's
  comment: "6 is probably always enough"). Coefficients chosen to maximise slope at zero.
* Keller's scale: `update *= max(1, rows/cols)**0.5`.
* **Moonshot's RMS-matching scale: `lr_eff = lr · 0.2 · sqrt(max(A, B))`** for an A×B matrix.
  Since an orthogonalised update `O` has `RMS(O) = sqrt(min(A,B)/(A·B))`, this makes the
  per-element update RMS **exactly `0.2·lr`, independent of shape** — i.e. it puts Muon and AdamW
  (whose update RMS ≈ lr) on a common LR scale, and it *is* the μP width rule for these matrices.
* Momentum `μ=0.95`, **Nesterov on** (`grad.lerp_(momentum, β)` before orthogonalisation).
* **Decoupled weight decay applied before the update**: `p.mul_(1 - lr·wd)`. Moonshot: "weight decay
  plays a crucial role in Muon's scalability" — un-decayed Muon *does not scale*.
* **Parameter split (both repos agree):** Muon on `p.ndim >= 2` **excluding `embed_tokens` and
  `lm_head`**; AdamW on everything else (embeddings, unembedding, all 1-D gains/biases).
* Defaults: `lr=0.02`, `momentum=0.95`, `weight_decay` must be tuned (0 in Keller's default).

### 2.3 Why SOAP/Shampoo memory is disqualifying at our total size

For an n×m matrix SOAP stores `GGᵀ (n×n)`, `GᵀG (m×m)`, eigenbases `Q_L (n×n)`, `Q_R (m×m)`, plus
Adam moments in the eigenbasis (`2nm`). For a square d×d matrix that is `4d² + 2d² = 6d²` vs AdamW's
`2d²` ⇒ **3× AdamW state**. For an MLP matrix (d, 4d) it is `2(d²+16d²) + 8d² = 42d²` vs AdamW's `8d²`
⇒ **5.25×**. In fp32 that is **~24–48 bytes/param**. On 80 GB with a 4B-total model that is 96–192 GB
of optimizer state alone. **SOAP and Shampoo are viable only for the ≤600M dense "mini" model.**

Muon is the *only* matrix-aware optimizer that costs **less** memory than AdamW (one momentum buffer,
i.e. SGD-momentum footprint). For a single-80GB-GPU MoE, that is decisive independent of speed.

### 2.4 How much of the claimed speedup survives careful tuning

Three 2025–2026 studies converge:

* **Wen et al., "Fantastic Pretraining Optimizers and Where to Find Them" (2509.02046).**
  Ten optimizers (AdamW, NAdamW, Mars, Cautious, Lion, Adam-mini, Muon, Scion, Kron/PSGD, Soap) ×
  four scales (130M / 300M / 520M / 1.2B), seq 4096, three-phase tuning (coordinate-descent sweeps
  over LR, WD, warmup, β₁, β₂, ε, clip, batch; then re-sweep only scale-sensitive HPs; then fit
  scaling laws). Results:
  - **No optimizer reached the claimed 2×.** Max observed ≈ **1.4×**.
  - Speedup **decays with scale: 1.4× @130M → 1.1× @1.2B**.
  - **Muon wins at 1×–4× Chinchilla; SOAP and Kron take over at ≥8× Chinchilla and in the
    over-trained 16× regime.**
  - Matrix preconditioners (Muon/Soap/Kron) consistently beat scalar methods (Lion/Adam-mini/Signum).
  - Early-training loss curves are *misleading* — rankings flip late.
* **Semenov et al., "Benchmarking Optimizers for LLM Pretraining" (2509.01440, `epfml/llm-optimizer-benchmark`).**
  11–24 optimizers, Llama-style 124M/210M/720M dense + 520M MoE on FineWeb, 1M-token batches.
  **AdEMAMix and MARS best at 720M**, ahead of Muon. The authors attribute the disagreement with
  Wen et al. to **batch size**: at 1M-token batches AdamW-family variance reduction matters more.
* **"SOAP, Muon, and Beyond" (2607.20548, Jul 2026).** Scales to 72B MoE. **AdamW degrades past its
  critical batch size, while Muon and SOAP retain token efficiency at global batches up to 100M
  tokens.** Per-step runtime: Muon **1.45×**, SOAP **1.72×** vs AdamW (in their distributed setup).
* **AlgoPerf 2024 (2502.15015).** Distributed Shampoo won external tuning at **+28%**;
  Schedule-Free AdamW won self-tuning at **+8%** and was the *only* self-tuning entry to beat
  the baseline. Validated directions: non-diagonal preconditioning and HP reduction.

**Synthesis:** expect **1.15×–1.35×** for Muon at our scale, not 2×. Note that our regime
(130M–1.2B; small batches; 3–9× Chinchilla) is *exactly* the band where the measured Muon advantage
is largest and where SOAP starts to compete.

### 2.5 Hyperparameter transfer and scaling laws

* **μP / μTransfer (2203.03466).** Width-wise HP transfer: parametrise so that per-layer LR scales
  as `1/fan_in` for hidden matrices, constant for embeddings and 1-D params, and output logits are
  scaled by `1/m_d`. Transfers LR, init scale, and (partly) β₂ from a narrow proxy.
* **Depth-μP (2310.02244 †)** and **CompleteP** extend transfer to depth (residual branch scaled
  `1/sqrt(L)` with block-output LR scaling); needed if we intend to change L between proxy and target.
* **u-μP (2407.17465 †, ICLR 2025)** combines μP with unit scaling: every tensor is unit-variance at
  init, HPs become orthogonal, and low-precision training is better behaved. Practical win: the LR
  optimum is near 1.0 and does not move.
* **Caveat: "Weight decay may matter more than μP for LR transfer in practice" (2510.19093).**
  Empirically, keeping the *effective* weight decay (`lr·wd` per step, or `wd` in AdamW's decoupled
  form which already multiplies by `lr`) fixed does most of the transfer work. Do not treat μP as
  magic; validate transfer with a 2-point width check.
* **Muon is partially self-μP.** With Moonshot's `0.2·sqrt(max(A,B))` scaling, the per-element update
  RMS is width-independent, so **the Muon LR needs no width scaling** — one fewer thing to transfer.
  Theory: Bernstein & Newhouse, "Old Optimizer, New Norm" (2409.20325 †); "Optimal Scaling Needs
  Optimal Norm" (2510.03871).
* **Over-training / inference-optimal (Sardana et al. 2401.00448).** When inference dominates
  lifetime cost, the compute-optimal model is *smaller and trained longer* than Chinchilla. For
  Prophet — which targets a 5090/Mac/iPhone — this argues strongly for **smaller N, more D**, which
  happily coincides with §1.3's arithmetic.
* **MiniCPM (2404.06395)** is the closest published analogue to Prophet: small model, WSD schedule,
  μP, and a deliberate over-train, with the scaling-law exponents refit for the WSD schedule.

### 2.6 LR schedules

* **WSD / trapezoidal (2404.06395; Hägele et al. 2405.18392 †).** Linear warmup → **constant** peak
  LR → short decay. Key measured properties:
  - Matches or beats cosine at the *same* token count.
  - **The total number of steps need not be known in advance** — this is the property that makes it
    correct for an interrupted, budget-uncertain project.
  - Decay fraction **~10–20% of total steps** is optimal; **`1-sqrt` decay shape slightly beats
    linear**, both beat exponential.
  - **The loss drops sharply and non-linearly during the decay phase** — a mid-run plateau
    checkpoint looks *much worse* than it is. Never benchmark a plateau checkpoint against a
    finished model.
* **Branching anneals — the key trick for Colab.** During the stable phase every checkpoint is in
  the same optimisation regime, so you can **fork N independent short decays from one plateau
  checkpoint**, each producing a properly-annealed, benchmark-able model, while the trunk run
  continues at peak LR. Uses: (i) mid-flight capability checks without contaminating the trunk;
  (ii) data-mixture ablations done *only* in the anneal (this is exactly how MiniCPM, Llama-3 and
  OLMo-2 pick their high-quality "anneal mix"); (iii) producing several release candidates
  (long-context anneal, code anneal, general anneal) from one pretraining run.
* **Cosine is actively harmful for us:** it bakes in a total-step count, and if the run is cut short
  or extended the schedule is wrong; and mid-run checkpoints are at an awkward LR for continuation.
* **Re-warming after a resume:** if you resume with the optimizer state intact and the LR unchanged
  (which WSD's plateau makes trivial), no re-warm is needed. If optimizer state was lost, re-warm
  over ~100–200 steps (cf. 2403.08763 † on continual pretraining re-warming).
* **Alternatives considered and rejected:** Schedule-Free AdamW (2405.15682 †) removes the schedule
  but gives up the WSD branching trick and adds optimizer state; WSM (2507.17634) replaces decay with
  checkpoint merging — elegant, but it needs many checkpoints held simultaneously, which our storage
  bandwidth forbids. Power Scheduler (2408.13359) is batch/token-agnostic and worth a look if the
  batch ramp misbehaves.

### 2.7 Stability

| Technique | Source | Verdict for Prophet |
|---|---|---|
| **QK-norm** (RMSNorm on q,k per head) | 2309.14322 (Wortsman et al.), OLMo-2 2501.00656, Gemma-3 | **ADOPT.** The attention-logit-growth instability appears in *small* models at high LR, and qk-layernorm fixes it there exactly as it does at scale. FlashAttention-2-compatible (applied before the kernel). Lets us run a higher LR safely. |
| **z-loss** `λ·log²(Z)` on the output softmax | Switch Transformer 2101.03961 †, OLMo-2 (λ=1e-5), PaLM | **ADOPT**, λ = 1e-4 during the bake-off, 1e-5 in the main run. Prevents output-logit divergence; costs ~0. |
| **Attention logit soft-capping** (Gemma-2, cap 50) | 2408.00118 † | **REJECT.** Incompatible with FlashAttention-2's fused softmax (you fall back to a slow path). Gemma-3 itself dropped it in favour of QK-norm. |
| **Final logit soft-capping** (cap 30) | Gemma-2 | Optional; z-loss achieves the same end more cheaply. Skip. |
| **Norm placement** | OLMo-2 "reordered norm" (norm on attn/MLP *outputs*); Peri-LN 2502.02732; Gemma-2 sandwich | **ADOPT sandwich/Peri-LN**: RMSNorm at both the input and the output of each sub-block. Measurably better gradient-norm behaviour than plain pre-LN at high LR; cost ≈ 0.5% FLOPs. |
| **Residual / output-projection init** | GPT-2, modded-nanogpt, GPT-NeoX | **Zero-init `o_proj` and expert `down_proj`.** Equivalent to (and more robust than) `1/sqrt(2L)` scaling; makes every block an identity at step 0. |
| **Init** | OLMo-2: truncated normal σ=0.02, truncated at ±3σ | **ADOPT** (with μP width correction `σ = σ_base·sqrt(d_base/d)` for the μP-scaled tensors). |
| **Gradient clipping** | universal | Global-norm clip at **1.0**, *plus* an adaptive skip rule (§4.6). |
| **QK-Clip** | Kimi K2 2507.20534 | **HOLD IN RESERVE.** Muon can inflate attention logits past 1000; Kimi found QK-norm insufficient for MLA specifically. If QK-norm + z-loss still spikes, add QK-Clip (rescale `W_q`,`W_k` when max logit exceeds a threshold τ≈100). |
| **Router stability (MoE)** | DeepSeek-V3 2412.19437 †, OLMoE 2409.02060 | Router logits in **fp32**; **aux-loss-free load balancing** (per-expert bias updated from load) + small router z-loss (1e-3); router on **AdamW, not Muon**. |

### 2.8 Memory and throughput engineering on one A100

| Technique | Effect | Notes for A100 |
|---|---|---|
| **FlashAttention-2** | 2–4× attention speed, O(s) attention memory | Only option (FA3 = Hopper). Use `head_dim ≤ 128` to stay on the fast path. |
| **torch.compile** | +10–30% throughput on small models (kernel fusion of norms, RoPE, SwiGLU, elementwise) | MoE dynamic shapes cause recompiles — pad expert buckets to fixed capacity and use `dynamic=False`, or `mark_dynamic` only the batch dim. |
| **Fused linear + cross-entropy** (Liger-Kernel, Cut-CE 2411.09009 †) | **The single biggest memory win for small models.** At 65,536 tokens × 64k vocab the logits alone are **7.8 GiB bf16 + 15.6 GiB fp32 softmax**; fused CE reduces this to <0.2 GiB. | Liger claims ~20% throughput / ~60% memory on standard LLM training; it also lets context scale to 16K where HF baselines OOM at 4K. |
| **Activation checkpointing** | Full per-layer: activations ≈ `2·L·s·b·d` bytes ⇒ **3.0 GiB** at L=24, d=2048, s=4096, micro-bs=8. Costs ~30% extra FLOPs (recompute fwd). | Prefer **selective/`SAC`**: checkpoint only the MLP/expert block (cheap to recompute, huge activations), keep attention outputs (FA2 already stores little). Typically ~10–15% FLOP cost instead of 30%. |
| **Sequence packing** | Removes padding waste entirely | Mandatory. Pack to exactly `s` tokens with a document-boundary attention mask (FA2 `varlen` API) so cross-document attention is blocked. |
| **8-bit optimizer (bitsandbytes)** | m,v at 1 byte each (blockwise quantization) ⇒ AdamW state 8→2 B/param | Useful only for the AdamW groups; Muon's state is already 2 B/param in bf16. |
| **Adafactor** | ~0.05 B/param | Costs a little loss. Only if desperate. |
| **CPU offload (ZeRO-Offload)** | Frees GPU state at PCIe cost | At ~20 GB/s, offloading 30 GB of state costs **~3 s/step** round-trip. With a 12.6 s/step budget that is a **24% throughput loss**. Use only as a last resort; prefer shrinking the model. |
| **FP8** | — | **Impossible on A100.** |
| **BF16 + fp32 master** | Standard | See §5.2 for whether it fits. |

**Realistic single-A100 MFU** (BF16 + FA2 + torch.compile + fused CE + packing):

| Model | Seq | Expected MFU | Notes |
|---|---|---|---|
| 300–600M dense | 2048–4096 | **42–50%** | small GEMMs; lm_head is 12–16% of FLOPs |
| 1.0–1.5B dense | 4096 | **45–52%** | best case on this GPU |
| MoE, top-k grouped GEMM | 4096 | **25–35%** | permute/unpermute + smaller per-expert GEMMs; 30% is the honest planning number |
| Naïve HF `Trainer`, no compile, no fused CE | 2048 | 15–25% | what you get if you don't do the work — **this alone is a 2× compute swing** |

---

## 3. What actually transfers to our scale

1. **Muon transfers, and our scale is its best scale.** The measured 1.4×@130M → 1.1×@1.2B decay
   (2509.02046) means Prophet's 300M–1.3B band should see **1.15–1.35×**. Two of our design choices
   *increase* the expected win: small batches (we are far below the critical batch size, where
   2607.20548 shows AdamW is *not* yet handicapped — this argues the win is at the lower end) and
   over-training (which 2509.02046 says favours SOAP/Kron over Muon — argues for testing SOAP on the
   *mini* model, where it fits).
2. **Muon's Newton–Schulz overhead is negligible for us.** Overhead ≈ `3.33·m/T` extra FLOPs
   (m = short matrix dimension, T = tokens per optimizer step). Computed:

   | m (short dim) | T=64K | T=128K | T=256K | T=512K |
   |---|---|---|---|---|
   | 768 | 3.9% | 2.0% | 1.0% | 0.5% |
   | 1024 | 5.2% | 2.6% | 1.3% | 0.7% |
   | 2048 | 10.4% | 5.2% | **2.6%** | **1.3%** |

   The 1.45× per-step figure in 2607.20548 reflects distributed all-gather overhead we do not have.
   **At batch ≥256K tokens on one GPU, Muon costs ≤3%.** Keep the batch large enough to amortise NS.
3. **Muon's memory advantage transfers and matters more than its speed.** 2 B/param (bf16 momentum)
   vs AdamW's 4–8 B/param, on a model whose total parameter count is the binding constraint (§5).
4. **SOAP/Shampoo/Kron do NOT transfer to the main model** — 3–6× AdamW state is unaffordable at
   4–10B total on 80 GB. They *do* transfer to the 300–600M dense mini model and should be tested
   there, especially since we over-train the mini (9× Chinchilla), the regime where they win.
5. **AdEMAMix does not transfer, for a non-obvious reason.** Its power comes from `β₃ = 0.9999`, an
   EMA with a **~10,000-step memory horizon**. In a preemptible setting every lost optimizer state
   costs ~10,000 steps of accumulated signal — and we only *have* 20–70k steps. Muon/AdamW momentum
   at β=0.95 has a ~20-step horizon, so losing it on a crash is essentially free (§4.7). This turns a
   nice-to-have into a structural argument.
6. **Sophia does not transfer.** The 2× claim did not survive a correctly-randomised dataloader;
   "no significant speedup for models under 0.5B."
7. **Adam-mini's headline gain does not transfer.** The 49.6% throughput came from memory and
   inter-GPU communication savings on a 2-GPU 7B run. On one GPU with a ≤4B model there is no
   comparable win; algorithmically it is ≈ AdamW.
8. **WSD transfers perfectly and is the highest-value non-optimizer decision in this report.**
   Unknown total step count + branchable anneals + resume-at-constant-LR is a precise match to
   "Colab kills you every 12–24h and you don't know your final budget."
9. **μP transfers for *width* only, cheaply.** Do not attempt depth transfer: fix L between proxy
   and target and vary only `d_model`. Combined with Muon (which is already width-invariant), the
   only things that need μP are the AdamW groups — embeddings, head, norms, router — which is a
   small, low-risk surface.
10. **The stability set transfers *because* it was validated at small scale.** 2309.14322's entire
    contribution is that these instabilities and their fixes reproduce at 100M–1B with high LR. We
    are that regime, and we *want* high LR to save compute.

---

## 4. Recommendation for Prophet

### 4.1 Headline recipe (one line)

> **Muon (Moonshot variant: decoupled WD + `0.2·sqrt(max(A,B))` RMS matching, NS-5, μ=0.95,
> Nesterov) on all ≥2-D hidden matrices + AdamW(0.9, 0.95) on embeddings / lm_head / all 1-D params
> / MoE router; WSD schedule (2% warmup, ~80% plateau, 18% `1-sqrt` decay) with branched anneals;
> BF16 autocast + FP32 master + FP32 router logits, no FP8; QK-norm + z-loss(1e-5) + sandwich
> RMSNorm + zero-init output projections + global-norm clip 1.0 + spike-skip; fused linear-CE and
> selective activation checkpointing; two-slot atomic checkpointing every ~20 min of wall-clock.**

### 4.2 Parameter groups — exactly which params get which optimizer and LR

μP base width `d_base = 256`; width multiplier `m_d = d_model / d_base` (e.g. `m_d = 8` at d=2048).
LRs below are the **values at `d_base`**; the "width scaling" column says how to get the value at
`d_model`.

| # | Parameter group | Optimizer | Base LR | Width scaling | Weight decay | Init | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `q_proj, k_proj, v_proj, o_proj` (2-D) | **Muon** | **0.02** | **none** (RMS-match is μP) | **0.1** | trunc-normal σ=0.02·sqrt(d_base/d); **`o_proj` = 0** | NS-5, μ=0.95, Nesterov |
| 2 | Dense MLP `gate/up/down` (2-D) | **Muon** | 0.02 | none | 0.1 | as above; **`down` = 0** | |
| 3 | **MoE expert** `gate/up/down` (3-D `[E,d,f]`) | **Muon**, orthogonalise **per-expert slice** | 0.02 | none | 0.1 | as above; `down` = 0 | See risk R3; bake-off arm uses AdamW here |
| 4 | Token embedding `embed_tokens` | **AdamW** | **4e-3** | **none** (μP: constant) | **0.0** | trunc-normal σ=0.02 | Never Muon (both reference impls exclude it) |
| 5 | Output head `lm_head` (untied) | **AdamW** | **2e-3** | **× 1/m_d** | 0.1 | **zeros** | logits also multiplied by `1/m_d` (μP output rule) |
| 6 | RMSNorm gains, QK-norm gains, biases (1-D) | **AdamW** | **4e-3** | none | **0.0** | 1.0 (gains), 0 (bias) | |
| 7 | **MoE router / gate** `[d, E]` | **AdamW** | **1e-3** (½ of head) | × 1/m_d | **0.0** | zeros or σ=0.006 | **fp32 master + fp32 logits**; router z-loss 1e-3; aux-loss-free load-balance bias updated *outside* the optimizer |
| 8 | Value/attention residual scalars, learned gates | **AdamW** | 4e-3 | none | 0.0 | as per arch | |

**AdamW settings for groups 4–8:** `β₁=0.9, β₂=0.95, ε=1e-8` (drop to `1e-12` if μP pushes any group
LR below ~1e-4, so ε does not dominate the denominator), decoupled WD as in PyTorch `AdamW`
(decay = `lr_t · wd`, so it automatically follows the LR schedule — this is the "effective WD stays
proportional to LR" behaviour that 2510.19093 finds matters).

**Sanity check on the LR relationship.** Muon at `lr=0.02` with the 0.2 factor produces a
per-element update RMS of `0.2 × 0.02 = 4e-3`, which matches the AdamW groups' `4e-3` update RMS.
That is the whole point of Moonshot's scaling: **one number (4e-3 update RMS) governs the entire
model**, and the bake-off only has to sweep that one number.

### 4.3 Schedule

| Phase | Fraction of tokens | LR | Batch (tokens) | Notes |
|---|---|---|---|---|
| Warmup | 0 → 2% | linear 0 → peak | 131,072 | ≥ 300 steps, ≤ 1000 steps |
| Ramp | 2% → 8% | peak (constant) | 131,072 → 262,144 (one doubling at 5%) | second doubling to 524,288 only if MFU improves |
| **Stable plateau** | 8% → 82% | **peak, constant** | 262,144 | *The resumable, branchable, budget-agnostic phase.* |
| Decay | 82% → 100% | peak → 0, **`1 - sqrt(t)` shape** | 262,144 | 18% of tokens |

* **Peak LR:** the value of `update-RMS` found in the bake-off; prior = **4e-3** (⇒ Muon `lr=0.02`).
* **Warmup applies to all groups simultaneously.** Weight decay is *not* warmed up.
* **Branched anneals.** At 30%, 55% and 80% of the token budget, fork the checkpoint and run a
  **1.5%-of-budget** `1-sqrt` decay to zero on a *separate* copy. Each fork yields a properly
  annealed model you can benchmark honestly. Cost: **3 × 1.5% = 4.5% of budget (~11 A100-h)**, and
  it is the only way to make go/no-go decisions mid-project without wrecking the trunk run.
  A plateau checkpoint evaluated directly will look 0.1–0.2 nats worse than it "is" — **never make a
  decision on an un-annealed checkpoint.**
* **Data-mixture ablations belong in the anneal, not the trunk** (Llama-3/OLMo-2/MiniCPM practice):
  fork three anneals with different high-quality mixes, pick the best, and only then run the real
  final decay.

### 4.4 Batch-size ramp

Two reasons to ramp: (i) early training has a small critical batch size, so large batches waste
compute; (ii) Muon's NS overhead is `3.33·m/T` and shrinks as `T` grows, so we want to *end* large.
Keep it to **two discrete doublings** — a continuous ramp complicates the dataloader-resume
determinism (§4.7) for little gain. **Do not change the LR when the batch changes** during the
plateau (LR-batch coupling is weak in the Muon spectral-norm parametrisation); log the grad-norm
before and after and revert if it jumps >2×.

### 4.5 Precision policy

| Item | Precision | Rationale |
|---|---|---|
| GEMM compute | **BF16** autocast | A100 tensor cores; **no FP8 available** |
| Master weights | **FP32** | needed because `lr·update ≈ 4e-3 · w` can be below bf16 ulp late in training |
| Gradients | BF16, **accumulated into an FP32 (or directly into the momentum) buffer** | see §5.3 |
| Attention | FA2 bf16, softmax fp32 internally | |
| RMSNorm / QK-norm | FP32 accumulate | |
| Loss / cross-entropy | **fused linear-CE, FP32 reduction** | avoids materialising [T,V] logits |
| Router logits + softmax | **FP32** | routing decisions are discrete; bf16 causes flapping |
| `torch.backends.cuda.matmul.allow_tf32` | **True** | free 2× on residual fp32 matmuls |
| Optimizer states | Muon momentum **BF16**; AdamW m,v **FP32** (or 8-bit if memory-bound) | Muon momentum is a short-horizon EMA and tolerates bf16 |

**If FP32 master weights must be dropped** (see §5.2), use **stochastic rounding** on the bf16
weight update. Deterministic bf16 rounding silently stalls training once `|Δw| < ulp(w)/2`.

### 4.6 Stability set (final)

1. **QK-norm**: RMSNorm on q and k, per head, before RoPE, learnable gain, applied outside the FA2 kernel.
2. **Sandwich RMSNorm**: `x + Attn(Norm(x))` → `Norm_out(...)` on both attention and MLP outputs (Peri-LN 2502.02732 / OLMo-2 reordering).
3. **z-loss**: `+ 1e-5 · mean(log²(Z))` on the output softmax (`1e-4` during the bake-off where LR is being pushed).
4. **Router z-loss**: `1e-3` on router logits; router bias-based aux-loss-free balancing.
5. **Zero-init** `o_proj`, expert `down_proj`, `lm_head`.
6. **Global gradient-norm clip 1.0**, computed over the *whole* model in fp32.
7. **Spike guard**: maintain an EMA and a rolling median of the grad norm. If `‖g‖ > 4 × median`,
   **skip the optimizer step** (but still zero the grads) and log. If >3 skips in 50 steps, **halt,
   roll back to the last checkpoint, and skip forward 1B tokens in the data stream** (the usual cause
   is a pathological document batch).
8. **Loss guard**: if `loss > EMA + 3σ` for 5 consecutive steps, same rollback procedure.
9. **Attention-logit monitor**: log `max|q·k/sqrt(d)|` every 100 steps. If it trends above ~100,
   enable **QK-Clip** (Kimi K2, 2507.20534) on the offending heads.
10. **No attention soft-capping** (breaks FA2). **No dropout** (we are data-limited, not overfitting).

### 4.7 Checkpoint / resume policy for a preemptible Colab

**The cadence is set by upload bandwidth, not by paranoia.** Computed sizes and upload times:

| Ckpt content (4B total, ~0.8B active) | Size | @50 MB/s | @200 MB/s |
|---|---|---|---|
| bf16 weights only ("light") | **7.5 GiB** | 2.7 min | 0.7 min |
| + fp32 master + bf16 Muon mom + fp32 Adam m,v ("full") | **31.4 GiB** | 10.7 min | 2.0 min |
| same for a 6B-total model ("full") | 47 GiB | 16 min | 4 min |

**Policy:**

* **Light checkpoint every ~20 min of wall-clock** (bf16 weights + RNG + dataloader position + step +
  LR state + config hash). ~7.5 GiB, uploaded **asynchronously from a background thread** while
  training continues. **Losing the optimizer state costs almost nothing**: Muon momentum (β=0.95)
  and AdamW `v` (β₂=0.95) both have a ~20-step horizon. On a light-checkpoint resume, re-warm the LR
  over 100 steps and continue. *(This is a strong extra argument against AdEMAMix, whose β₃=0.9999
  state has a 10,000-step horizon and cannot be cheaply discarded.)*
* **Full checkpoint every ~2 h** (all optimizer state), same async upload.
* **Two-slot atomic rotation.** Write to `slot_{step%2}`, upload, verify a SHA-256 manifest, and only
  then update a tiny `LATEST` pointer file. A preemption mid-upload must never be able to destroy the
  last good checkpoint. **This is the single most likely way to lose the project.**
* **Storage target, in preference order:** (1) **GCS same-region** with parallel composite uploads
  (200–400 MB/s from Colab); (2) **HF Hub with `hf_transfer`** (50–200 MB/s, free, versioned);
  (3) **Google Drive last** — 30–100 MB/s and highly variable; a 31 GiB full checkpoint takes
  ~10–17 min there, which is unworkable at any useful cadence. Local Colab disk is **not** durable
  (destroyed on VM teardown) — use it only as the staging buffer for the async uploader.
* **Deterministic dataloader resume.** Pre-tokenise into fixed-size binary shards with a global
  offset table. Make the sampler **stateless and index-addressable**: sample `i` is a pure function
  of `(seed, global_index)` via a counter-based PRNG (Philox / SplitMix64 hash), so resuming needs
  only the integer `global_token_index`. **Never** pickle a Python iterator or a `DataLoader` worker
  state. Save and restore `torch`, `torch.cuda`, `numpy` and Python RNG states so dropout-free but
  data-order-dependent behaviour is bit-reproducible.
* **Resume self-test.** On every resume, recompute the loss on a fixed held-out batch and compare to
  the value stored in the checkpoint; abort loudly if it differs by more than 1e-3. This catches
  silent config drift, which is the second most likely way to lose the project.
* **Keep a run ledger** (JSONL, appended and uploaded with each checkpoint): step, tokens, loss,
  grad-norm, LR, MFU, wall-clock, GPU model detected. Reconstructing "how many A100-hours have we
  actually spent" after 40 interrupted sessions is otherwise impossible.

### 4.8 What we are explicitly NOT doing

* Not using FP8 (A100 lacks it), FA3 (Hopper-only), or 2:4 sparsity.
* Not using SOAP/Shampoo on the main MoE (memory).
* Not using AdEMAMix (β₃ horizon vs preemption).
* Not using cosine LR (incompatible with an unknown/variable budget).
* Not using attention soft-capping (kills FA2).
* Not offloading optimizer state to CPU unless the memory table forces it (~24% throughput cost).

---

## 5. Compute and memory budget

Usable GPU memory: **77 GiB** (80 GB card = 79.2 GiB, minus ~1.2 GiB CUDA context/allocator slack;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to limit fragmentation).

### 5.1 Static memory vs total parameter count

Assumes ~93% of parameters are Muon-governed 2-D matrices, 7% are AdamW-governed
(embeddings + head + norms + router). Units are **GiB**.

**Config B — FP32 master + BF16 grads + BF16 Muon momentum + FP32 AdamW (m,v):**

| Total params | bf16 weights | bf16 grads | fp32 master | Muon mom (bf16) | AdamW m,v (fp32) | **Static total** | Left for activations (77 GiB) |
|---|---|---|---|---|---|---|---|
| **10 B** | 18.6 | 18.6 | 37.3 | 17.3 | 5.2 | **97.0** | **−20.0 ❌ does not fit** |
| **8 B** | 14.9 | 14.9 | 29.8 | 13.9 | 4.2 | **77.6** | **−0.6 ❌ does not fit** |
| **6 B** | 11.2 | 11.2 | 22.4 | 10.4 | 3.1 | **58.2** | 18.8 ⚠️ tight |
| **4 B** | 7.5 | 7.5 | 14.9 | 6.9 | 2.1 | **38.8** | **38.2 ✅** |
| **3 B** | 5.6 | 5.6 | 11.2 | 5.2 | 1.6 | **29.1** | 47.9 ✅ |
| **0.5 B** (mini, dense) | 0.9 | 0.9 | 1.9 | 0.8 | 0.4 | **4.9** | 72.1 ✅ |

For reference, the naïve **all-AdamW, fp32-states** variant of a 10B model is
`18.6 + 18.6 + 37.3 + 74.5 = 149 GiB` — nearly **2× the card**. Muon's single bf16 momentum buffer
is what brings 10B from "2× over" to "1.25× over", and it is still over.

**Verdict: a 10B-total MoE cannot be trained on one A100 80GB by any straightforward means, and an
8B-total model has literally zero memory left for activations.**

### 5.2 The three ways to make a larger total fit, costed

| Option | 8B total, static | Cost |
|---|---|---|
| **C — drop FP32 master; BF16 weights + stochastic rounding** | **47.8 GiB** (29 GiB free) | Fits. Risk: without stochastic rounding, late-training updates silently vanish; with it, an extra RNG op per weight. Used by modded-nanogpt-class runs but **not** by any published multi-T-token run. |
| **D — fold gradients into the momentum buffer during backward** (per-parameter `post_accumulate_grad_hook`; for Muon this is exactly valid: first micro-step `B.mul_(μ).add_(g/k)`, subsequent `B.add_(g/k)`, orthogonalise after the last) | **62.7 GiB** (14.3 GiB free) | Fits, keeps FP32 master. Requires ~100 lines of custom hook code and disables `torch.compile` on the optimizer step. **Recommended engineering trick if total params must exceed 6B.** |
| **CPU-offload master + AdamW states (ZeRO-Offload style)** | GPU 33 GiB, host 34 GiB | Fits, but ~30 GiB of PCIe traffic/step ⇒ **~3 s/step ≈ 24% throughput loss**, and Colab's ~83 GB host RAM leaves little headroom. |

### 5.3 Activation memory

With **full** per-layer checkpointing, stored activations ≈ `2 · L · s · b · d` bytes (bf16 layer
inputs only; the recompute peak of one block adds ~1–2 GiB):

| L | d | s | micro-bs | tokens/micro-batch | Activation memory |
|---|---|---|---|---|---|
| 24 | 2048 | 4096 | 8 | 32,768 | **3.0 GiB** |
| 24 | 2048 | 4096 | 16 | 65,536 | 6.0 GiB |
| 28 | 1536 | 4096 | 16 | 65,536 | 5.3 GiB |
| 30 | 2048 | 2048 | 16 | 32,768 | 3.8 GiB |

With **selective** checkpointing (MLP/expert only) multiply by roughly 2–3× but save ~15% of the
recompute FLOPs. Either way activations are **not** the binding constraint — optimizer state is.
Note that without fused linear-CE the logits for a 65,536-token micro-batch at V=64k alone are
**7.8 GiB (bf16) + 15.6 GiB (fp32 softmax)**, which *would* dominate everything. Use fused CE.

### 5.4 Throughput and step counts

`t_step = 6·N_active·T / (312e12 · MFU)`:

| Config | s/step | 250 A100-h ⇒ steps | ⇒ tokens | D/N |
|---|---|---|---|---|
| 1.3B-active MoE, T=512K, MFU 30% | 43.7 | 20,600 | **10.8 B** | 8× |
| 0.8B-active MoE, T=512K, MFU 32% | 25.2 | 35,700 | **18.7 B** | 23× |
| **0.8B-active MoE, T=256K, MFU 32%** | **12.6** | **71,400** | **18.7 B** | **23×** |
| 0.5B dense mini, T=256K, MFU 45% | 5.6 | 160,700 | 42.1 B | 84× |
| 0.35B dense mini, T=256K, MFU 45% | 3.9 | 229,500 | 60.2 B | 172× |

### 5.5 Recommended configurations (the actual ask to the architecture track)

| | Prophet-main | Prophet-mini |
|---|---|---|
| Total params | **≤ 4 B** (6 B only with the §5.2-D trick) | 0.35–0.5 B dense |
| Active params | **0.7–0.9 B** | = total |
| Static memory | **38.8 GiB** (4B) | 4.9 GiB |
| Budget | 170 A100-h | 80 A100-h |
| Tokens | **~13 B** | **~13–19 B** |
| Batch | 256K tokens (grad-accum 8 × micro-bs 8 × 4096) | 256K tokens |
| Steps | ~49,000 | ~52,000–73,000 |
| Optimizer | Muon + AdamW (§4.2) | Muon + AdamW; **SOAP is affordable here — test it** |

**The 10B-total / 1.3B-active target as stated does not fit and is not trainable to a useful loss
within this budget. R07 recommends ≤4B total / ≤0.9B active, and — if a larger MoE is required for
the product — reaching it by *sparse upcycling* a well-trained dense checkpoint late in the run
rather than training 10B-total from random init on 12B tokens.**

---

## 6. Risks and failure modes

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Colab serves A100-40GB, not 80GB** | High | Halves every memory budget; 4B total no longer fits | `nvidia-smi` gate at session start; maintain a 40GB config variant (≤2B total) and pick automatically |
| R2 | **Model too large for the token budget** (10B total on 12B tokens = 1.26 tok/param) | Certain if the stated target is kept | Model never leaves the "memorise nothing, generalise nothing" regime; loses to a 500M dense model | §5.5 — cap total at 4B, active at 0.9B; upcycle rather than train sparse from scratch |
| R3 | **Muon on MoE expert matrices misbehaves.** Each expert sees only ~k/E of tokens ⇒ its gradient is noisier and effectively lower-rank; orthogonalising a rank-deficient gradient amplifies noise directions | Medium | Silent quality loss, or expert collapse | Moonlight *did* apply Muon to MoE experts at 16B-A3B successfully — but with 5.7T tokens. **Make "Muon vs AdamW on experts" an explicit bake-off arm.** Monitor per-expert grad-norm and update-norm. |
| R4 | **Optimizer bake-off does not pay for itself.** Spending H hours to find a speedup `s` only helps if `(250−H)·s > 250` ⇒ for H=30, need **s ≥ 1.14×** | Medium | Wasted 30 A100-h | Trim the bake-off (§7.4): adopt Muon on the strength of published evidence and run a **confirmatory** 3-arm test, not a full 10-arm search |
| R5 | **Preemption corrupts the checkpoint mid-upload** | High over a multi-week project | Total loss of the run | Two-slot atomic rotation + SHA-256 manifest + `LATEST` pointer written last (§4.7). Test it by killing the process during an upload. |
| R6 | **Non-deterministic dataloader resume** ⇒ silent repeats or gaps in the data | High if a stateful sampler is used | Quality loss that is invisible until final evals | Stateless index-addressable sampler keyed on `(seed, global_index)`; assert token-index monotonicity in the ledger |
| R7 | **torch.compile ↔ MoE dynamic shapes** ⇒ recompilation storms, or graph breaks that silently drop the fusion | High | 20–40% throughput | Fixed expert capacity with padding; `dynamic=False`; assert compile-cache-hit count in the ledger; keep an eager fallback |
| R8 | **BF16 without FP32 master stalls late training** (option 5.2-C) | Medium | Loss plateaus for no visible reason | Stochastic rounding, plus a "fraction of weights whose update rounded to zero" metric logged every 500 steps |
| R9 | **Attention-logit explosion under Muon** (Kimi K2 observed >1000) | Medium | Loss spikes, divergence | QK-norm from step 0; monitor max logit; QK-Clip in reserve |
| R10 | **μP transfer fails** for the AdamW groups or the router | Medium | Peak LR is wrong ⇒ 10–30% of budget wasted | Two-point width check (d=256 → d=512 → predict d=1024) before committing; if the optimum moves, fall back to a direct 3-point LR sweep at the target width using 2% of the budget |
| R11 | **Benchmarking a plateau checkpoint** and concluding the run has failed | High (this fools everyone once) | Wrong go/no-go decision, possible project abandonment | Never evaluate without an anneal; §4.3 branched anneals exist precisely for this |
| R12 | **Over-tuning on a proxy that does not predict the target** (2509.02046's central warning: early-loss rankings flip late; small-scale rankings do not survive) | Medium | Pick the wrong optimizer | Decision rule requires the winner to lead at *both* proxy scales *and* at the final token count, not at an early step (§7.3) |
| R13 | **Grad-accumulation bug in the Muon-in-backward trick** (§5.2-D) — the momentum decay must be applied on the *first* micro-step only | Medium | Effective LR wrong by `k×` | Unit test: assert the momentum buffer after `k` micro-steps equals the buffer from a single full-batch step, to 1e-5 |
| R14 | Muon NS overhead balloons if the batch is small (10.4% at d=2048, T=64K) | Low | 10% throughput | Keep `T ≥ 256K` tokens/step; the formula `3.33·m/T` is in the ledger |

---

## 7. Ablation plan — small-scale optimizer bake-off

### 7.1 Proxy models (dense, μP-parametrised, fixed depth)

Depth is held fixed so only *width* transfer is exercised. Vocab 32k (a small vocab keeps the
lm_head from dominating the proxy's FLOPs and distorting the comparison). Seq 2048, packed.

| Proxy | d_model | L | heads | Non-emb params | Tokens (60 tok/param = **3× Chinchilla**) | MFU (est.) | **A100-h** |
|---|---|---|---|---|---|---|---|
| **P0** (LR sweep) | 256 | 8 | 4 | ~12 M | 0.72 B | 22% | **0.08** |
| **P1** | 384 | 8 | 6 | ~25 M | 1.5 B | 26% | **0.7** |
| **P2** | 512 | 12 | 8 | ~50 M | 3.0 B | 30% | **2.8** |
| **P3** (scale check) | 768 | 12 | 12 | ~110 M | 3.3 B (1.5× Chinchilla) | 33% | **3.7** |

Every arm stays **< 4 A100-h**, as required. 3× Chinchilla is chosen because it brackets the
regime our real run lands in (§1.3: 1.3×–9× depending on the final N), and it sits at the
Muon-vs-SOAP crossover identified by 2509.02046.

### 7.2 Arms

| Arm | Description | Cost (P0 sweep + P1 + P2) |
|---|---|---|
| **A** | **AdamW**, fully tuned (LR, β₂, WD, warmup swept) — *the baseline must be strong or the whole exercise is worthless* | 5×0.08 + 0.7 + 2.8 = **3.9 h** |
| **B** | **Muon** (RMS-match, WD 0.1) + AdamW on embed/head/1-D | 3.9 h |
| **C** | **Muon, no RMS-match** (Keller's `max(1,r/c)^0.5`), otherwise as B — tests whether Moonshot's scaling matters at our scale and whether it really removes the width-transfer burden | 3.9 h |
| **D** | **SOAP** (precond. freq. 10) | 3.9 h |
| **E** | **PSGD-Kron** (`memory_save_mode='smart_one_diag'`) | 3.9 h |
| **F** | **Muon + AdamW-on-experts** (MoE proxy only, P2 with 8 experts top-2) — tests risk R3 | 2.8 h |
| **G** | (contingency) **NAdamW / Cautious-AdamW** — cheapest possible upgrade if all matrix methods disappoint | 3.9 h |

**Stage-2 scale check:** top-2 arms only, at **P3** ⇒ 2 × 3.7 = **7.4 h**.
**Stage-0 systems sweep (no optimizer comparison):** torch.compile on/off, fused-CE on/off,
full vs selective checkpointing, micro-batch size, packing — measure tokens/s and MFU. **2 h.**

### 7.3 Decision rule

For each arm, at the **final** token count of each proxy (never at an intermediate step — 2509.02046
shows rankings flip late), record validation loss on a held-out 20M-token slice and the measured
wall-clock. Define

> **`speedup(arm) = (tokens AdamW needs to reach arm's final loss) / (tokens arm used)`,
> then corrected to wall-clock by `× (t_step_AdamW / t_step_arm)`.**

**Pick the arm that satisfies ALL of:**
1. Wall-clock speedup **≥ 1.10×** over tuned AdamW at **both P1 and P2**;
2. The speedup does **not decrease by more than 0.10** from P1→P2 (a decaying trend extrapolates to
   ≤1.0× at 800M — 2509.02046's central finding);
3. Optimizer state ≤ **4 bytes/param** (⇒ the main model still fits per §5.1);
4. ≤ 2 new hyperparameters beyond AdamW's.

If two arms tie within 0.5% loss, **pick the one with lower memory and fewer hyperparameters** —
i.e. Muon. If **no** arm clears 1.10×, **ship tuned AdamW-8bit** and bank the saved hours.

### 7.4 Budget and ROI — read this before running anything

Full protocol: `2 (systems) + 6×3.9 (arms A–E,G) + 2.8 (F) + 7.4 (P3) ≈ **35.6 A100-h**` ≈ **14%**
of a 250 h budget. The break-even speedup is `250/(250−35.6) = **1.166×**`.

**Recommended trimmed protocol (this is the actual proposal):** the published evidence for
Muon-over-AdamW at 130M–1.2B is strong and consistent across three independent groups, and Muon's
memory advantage is decisive regardless of speed. So **adopt Muon by prior** and spend the budget
only on what is genuinely uncertain for *us*:

| Step | Arms | Cost | Question it answers |
|---|---|---|---|
| Stage 0 — systems sweep | — | 2.0 h | Are we at 30%+ MFU? (biggest single lever) |
| Stage 1 — LR/update-RMS sweep at P0 | A, B, F | 5 LRs × 3 = 1.2 h | What is the peak LR? (needed regardless) |
| Stage 2 — confirm at P1 | A, B | 1.4 h | Does Muon beat tuned AdamW *here*? |
| Stage 3 — confirm at P2 | A, B, F | 8.4 h | Does the gap hold with scale? Muon-on-experts? |
| Stage 4 — μP 2-point width check | B only | 0.8 h | Does the LR transfer d=256→512→768? |
| **Total** | | **≈ 13.8 A100-h (5.5%)** | break-even speedup **1.058×** |

At 5.5% of budget the break-even is 1.058×, comfortably below every published estimate of Muon's
advantage. **Add SOAP and Kron arms only for the ≤600M dense mini model**, where they fit in memory
and where the over-trained (9× Chinchilla) regime is the one 2509.02046 says they win — budget
2 × 2.8 = 5.6 h there.

### 7.5 What to log in every arm

Validation loss vs (steps, tokens, wall-clock, FLOPs); grad-norm mean/median/max; per-group update
RMS (should equal `0.2·lr` for Muon and `≈lr` for AdamW — this is a direct correctness check on the
RMS matching); max attention logit; z-loss magnitude; fraction of weight updates rounding to zero;
MFU; per-expert token counts and grad norms; torch.compile cache hits; and the measured
`3.33·m/T` Newton–Schulz overhead as a fraction of step time.

---

## 8. References

Marked **(†)** = arXiv ID recalled from training data, **not re-verified in this session**
(arxiv.org is blocked by this environment's egress policy). All other IDs were surfaced by search in
this session.

**Optimizers**
- Liu et al., *Muon is Scalable for LLM Training* (Moonlight, 16B-A3B, 5.7T tokens) — **arXiv:2502.16982**
- Jordan et al., *Muon: MomentUm Orthogonalized by Newton-Schulz* — blog + `github.com/KellerJordan/Muon`, `github.com/KellerJordan/modded-nanogpt`
- Kimi Team, *Kimi K2: Open Agentic Intelligence* (MuonClip, QK-Clip, 15.5T tokens, zero loss spikes) — **arXiv:2507.20534**
- Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* — **arXiv:2409.11321**
- Gupta et al., *Scalable Second Order Optimization for Deep Learning* (Shampoo) — arXiv:2002.09018 **(†)**; impl. `github.com/facebookresearch/optimizers`
- Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer* — **arXiv:2305.14342**
- Zhang et al., *Adam-mini: Use Fewer Learning Rates To Gain More* — **arXiv:2406.16793**
- Chen et al., *Symbolic Discovery of Optimization Algorithms* (Lion) — arXiv:2302.06675 **(†)**
- Pagliardini et al., *The AdEMAMix Optimizer: Better, Faster, Older* — **arXiv:2409.03137**
- Yuan et al., *MARS: Unleashing the Power of Variance Reduction* — arXiv:2411.10438 **(†)**
- Pethick et al., *Training Deep Learning Models with Norm-Constrained LMOs* (Scion) — **arXiv:2502.07529**; `github.com/LIONS-EPFL/scion`
- Frans et al., *A Stable Whitening Optimizer for Efficient Neural Network Training* (SPlus) — **arXiv:2506.07254**
- Li, *Preconditioned Stochastic Gradient Descent* (PSGD-Kron) — `github.com/evanatyourservice/kron_torch`; `github.com/ClashLuke/HeavyBall`
- Defazio et al., *The Road Less Scheduled* (Schedule-Free) — arXiv:2405.15682 **(†)**
- Shazeer & Stern, *Adafactor* — arXiv:1804.04235 **(†)**
- Dettmers et al., *8-bit Optimizers via Block-wise Quantization* — arXiv:2110.02861 **(†)**
- Liang et al., *Cautious Optimizers* — arXiv:2411.16085 **(†)**
- Bernstein & Newhouse, *Old Optimizer, New Norm* — arXiv:2409.20325 **(†)**
- *Preconditioned Norms: A Unified Framework for Steepest Descent, Quasi-Newton and Adaptive Methods* — **arXiv:2510.10777**
- Zhang et al., *Why Transformers Need Adam: A Hessian Perspective* — **arXiv:2402.16788**

**Benchmarks / "what survives tuning"**
- Wen et al., *Fantastic Pretraining Optimizers and Where to Find Them* — **arXiv:2509.02046**
- Semenov et al., *Benchmarking Optimizers for Large Language Model Pretraining* — **arXiv:2509.01440**; `github.com/epfml/llm-optimizer-benchmark`
- *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales* (2026; up to 72B MoE, critical-batch-size analysis) — **arXiv:2607.20548**
- *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization* (2026) — **arXiv:2606.16899**
- Kasimbeg et al., *Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition* — **arXiv:2502.15015**; MLCommons AlgoPerf 2024 results
- *Towards Robust Scaling Laws for Optimizers* (2026) — **arXiv:2602.07712**
- *Muon is Not That Special: Random or Inverted Spectra Work Just as Well* (2026) — **arXiv:2605.11181**
- *Can Muon Fine-tune Adam-Pretrained Models?* (2026) — **arXiv:2605.10468**
- *How far away are truly hyperparameter-free learning algorithms?* — **arXiv:2505.24005**

**Parametrisation and scaling laws**
- Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot HP Transfer* (μP/μTransfer) — **arXiv:2203.03466**
- Bordelon et al., *Depthwise Hyperparameter Transfer in Residual Networks* (Depth-μP) — arXiv:2310.02244 **(†)**
- Blake et al., *u-μP: The Unit-Scaled Maximal Update Parametrization* (ICLR 2025) — arXiv:2407.17465 **(†)**
- *Weight Decay may matter more than μP for Learning Rate Transfer in Practice* — **arXiv:2510.19093**
- *Optimal Scaling Needs Optimal Norm* — **arXiv:2510.03871**
- *How to Set the Learning Rate for Large-Scale Pre-training?* (2026) — **arXiv:2601.05049**
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla) — arXiv:2203.15556 **(†)**
- Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* — **arXiv:2401.00448**

**Schedules**
- Hu et al., *MiniCPM: Unveiling the Potential of Small LMs with Scalable Training Strategies* (WSD) — **arXiv:2404.06395**
- Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* — arXiv:2405.18392 **(†)**
- Shen et al., *Power Scheduler: A Batch Size and Token Number Agnostic LR Scheduler* — **arXiv:2408.13359**
- *WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training* — **arXiv:2507.17634**
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train LLMs* (re-warming) — arXiv:2403.08763 **(†)**

**Stability and architecture recipes**
- Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* — **arXiv:2309.14322**
- OLMo Team, *2 OLMo 2 Furious* (reordered norm, QK-norm, z-loss 1e-5, trunc-normal σ=0.02) — **arXiv:2501.00656**
- Kim et al., *Peri-LN: Revisiting Normalization Layer in the Transformer Architecture* — **arXiv:2502.02732**
- Gemma Team, *Gemma 2* (soft-capping, sandwich norm) — arXiv:2408.00118 **(†)**
- Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models* — **arXiv:2409.02060**
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (aux-loss-free load balancing) — arXiv:2412.19437 **(†)**
- Fedus et al., *Switch Transformers* (z-loss) — arXiv:2101.03961 **(†)**
- Hu et al., *YuLan-Mini: An Open Data-efficient Language Model* — **arXiv:2412.17743**

**Systems**
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — **arXiv:2307.08691**; hardware support confirmed from `github.com/Dao-AILab/flash-attention` (FA2 = Ampere/Ada/Hopper; **FA3 requires H100/H800**; FA4 = Hopper/Blackwell)
- Liger-Kernel (fused RMSNorm/RoPE/SwiGLU/linear-CE; ~20% throughput, ~60% memory) — `github.com/linkedin/Liger-Kernel`
- Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (Cut Cross-Entropy) — arXiv:2411.09009 **(†)**
- *Computational Bottlenecks of Training Small-scale Large Language Models* — **arXiv:2410.19456**
- Unsloth (Triton kernels, padding-free packing) — `github.com/unslothai/unsloth`
- bitsandbytes (blockwise 8-bit optimizers, paged optimizers) — `github.com/bitsandbytes-foundation/bitsandbytes`

---

### Appendix A — formulas used, so every number here is reproducible

```
C            = 312e12 * MFU * 3600 * A100_hours          # useful FLOPs
D            = C / (6 * N_active)                        # tokens (ignores attention term)
attn_extra   = (6 * L * d_model * s) / (6 * N_active)    # +15% at L=24,d=2048,s=4096
t_step       = 6 * N_active * T / (312e12 * MFU)         # T = tokens per optimizer step
muon_ns_cost = 3.33 * m / T                              # m = short matrix dim; NS-5 quintic
act_mem      = 2 * L * s * b * d_model                   # bytes, full per-layer checkpointing
logit_mem    = T_micro * V * (2 + 4)                     # bytes, WITHOUT fused linear-CE
static_mem   = 2*N + 2*N + 4*N + 2*N_muon + 8*N_adam     # bf16 w + bf16 g + fp32 master + moms
N_chinchilla = sqrt(C / 120)                             # compute-optimal N at D = 20N
breakeven_s  = H_total / (H_total - H_ablation)          # min speedup that pays for the bake-off
```
