# A1 — Independent codebase review

> **Snapshot reviewed.** Working tree at 2026-09-04 23:52 → 2026-09-05 00:20 UTC, i.e. *after* the
> uncommitted edits with mtimes 23:49–23:51 on `CLAUDE.md`, `prophet/budget.py`, `prophet/config.py`,
> `prophet/modeling/layers.py`, `prophet/modeling/model.py`, `prophet/train/loop.py`,
> `tests/test_modeling.py`, and after `README.md` / `docs/01_ARCHITECTURE.md` were rewritten during the
> review with "Révision (revue A1)" notes. Four findings from my first pass were fixed by those edits
> before this file was written and are listed as **fixed mid-review** rather than as open bugs:
> `nope_layers` honoured (`config.py:399`, `layers.py:272,312`), `count_parameters` walks
> `section_layout()` (`budget.py:251`), confidence/halt heads counted (`budget.py:270-273`),
> `mtp_loss_weight`/`confidence_loss_weight` sourced from the model config (`loop.py:140-145`).
> Everything below is against the state *after* those edits. All line numbers are from that snapshot.
> Every number marked *(measured)* comes from code I ran; the script is reproduced in §2.1.

## 1. Verdict in one paragraph

The repository is a well-written **design document with executable illustrations**, not a training
system. The pure-Python pieces that only have to be *consistent* (config schema, budget arithmetic,
allocator, mixture validation, tokenizer, checkpoint rotation) are careful and now agree with each
other to 1e-4. The pieces that have to be *correct under load* are not: the model that every shipped
config builds silently zeroes the gated-delta forget-gate bias at init (α = 0.5 instead of ≈0.95),
multiplies every residual branch by 0.11 at forward time (which also destroys the "direct copy" of a
converted donor), and trains its halting head with a loss whose gradient is identical at every
position, so the input-dependent depth that W1 turned from option into requirement cannot be learned
by construction. Attention with a cache is only correct for the two cases the tests exercise
(full prefill, then one token at a time); a second prompt chunk or any prompt longer than the sliding
window produces wrong output with no error. The trainer runs fp32 with no autocast, no activation
checkpointing, no CUDA RNG in the checkpoint and no fused delta-rule kernel; the budget that gates
`scripts/train.py` assumes bf16, checkpointing and an 8-bit optimiser that do not exist, and
`train.py` exits with code 2 before building a model unless `--smoke` is passed because nothing
turns a dataset into tokens. Roughly 20 configuration fields are accepted, serialised and never read.
The documentation is unusually honest about *scientific* limits and unusually optimistic about
*engineering* state: "235 tests passent" (there are 295), "0.97B / 89 %" (now 1.02B / 85 %),
"session state under a megabyte" (11 MB, measured), "BF16 with FP32 master weights" (fp32 only).
None of this is unusual for a fast-moving research repo; what is unusual is that the docs present it
as an infrastructure that "tourne de bout en bout".

## 2. Bugs

Severity: **S1** silent — trains normally, model is wrong; **S2** wrong output/number in a path the
project relies on; **S3** loud or blocking; **S4** minor.

| # | Where | What is wrong | How it manifests | Sev | Suggested fix |
|---|---|---|---|---|---|
| B1 | `prophet/modeling/model.py:319-322`, `prophet/modeling/layers.py:438-439` | `ProphetModel._init_weights` zeroes every `nn.Linear` bias unless `_prophet_keep_bias` is set — and nothing in the repo ever sets it. `GatedDeltaNet.__init__` sets `a_proj.bias = 3.0` (forget gate α = σ(3) ≈ 0.95, "initialise near 1 … otherwise the layer is untrainable") and the model init immediately overwrites it. *(measured: `sections["core"][0].mixer.a_proj.bias.unique() == [0.0]`, α = 0.5 on `prophet_500m_probe.json`; a standalone `GatedDeltaNet` keeps 3.0)* | Every recurrent layer starts with a state half-life of one token. The loss still falls (the smoke corpus is a 256-token cycle), so nothing flags it. All four shipped configs and the converted donor are affected. | **S1** | Set `self.a_proj._prophet_keep_bias = True` (and `b_proj`) in `GatedDeltaNet.__init__`, or exclude `a_proj`/`b_proj` in `_init_weights`; add a test asserting `sigmoid(bias) > 0.9` on a built `ProphetModel`. |
| B2 | `prophet/train/loss.py:136-141`; `prophet/modeling/model.py:437-438`, `loss.py:139` | (a) The "expected LM loss over stopping times" is `Σ_i mean(p[...,i]) · mean(loss_i)` — a product of two batch means. Its gradient w.r.t. `p[b,t,i]` is `mean(loss_i)/(B·T)`, the *same value at every position*. *(measured: 1 unique gradient value per step across 10 positions; a per-token expectation gives 10)*. The KL term is per-position but pulls toward a fixed prior. The optimum of this objective is one constant distribution `p_i ∝ prior_i·exp(−mean loss_i)` for every token. (b) `hidden_per_step` is the coda output **before** `norm_out`; `project(hidden)` applies `lm_head` to it directly. *(measured: rms 0.100 vs 0.999 after `norm_out`)* — the per-step losses that weight the halting distribution are computed on near-uniform logits. | The halting head, which D4b/W1 make "Requis", can only learn a constant. `ponder/expected_depth` will move during training and look input-dependent while being a function of the batch mean only. Every ablation of "learned depth vs fixed dial" built on this loss is confounded. | **S1** | `F.cross_entropy(..., reduction="none")`, weight per token: `expected += (p[:, :-1, i] * ce_i).mean()`; apply `norm_out` before `project` (or store the normed tensor in `hidden_per_step`). Add a synthetic test where optimal depth differs by token and assert the head separates them. |
| B3 | `prophet/modeling/model.py:238-239, 156-157`; `prophet/config.py:337` | `residual_scaling` is documented as "scale residual branches by 1/√(2·depth) **at init**"; the implementation multiplies *both* branch outputs by the constant at **forward time, forever**. With `effective_depth(train_loop_max)`: 0.112 (main), 0.151 (mini), **0.102 for the Qwen3-1.7B conversion** *(measured)*. | For conversion this is fatal to the premise: the "direct copy" prelude/coda blocks compute `x + 0.10·f(x)` where the donor computed `x + f(x)`; the converted model does not reproduce the donor's function in a single layer, and "85 % coverage" is a tensor-count, not a function-count. For from-scratch training it is an un-ablated 9× attenuation of every branch that Muon's RMS-matched update cannot undo quickly. | **S1** (conversion) / S2 | Implement the documented behaviour (scale `o_proj`/`down_proj` init std), or expose the forward-time multiplier as a separate, ablated option; `prophet_config_for_donor` must set `residual_scaling=False`; add a test that a converted prelude block equals the donor block on the same input. |
| B4 | `prophet/modeling/layers.py:330-335` | With a cache and `s > 1`, full attention takes the `self.window is None` branch → `is_causal=True`. PyTorch's `is_causal` mask is **top-left** aligned (`tril` of an `L×S` matrix), so query `i` of the chunk attends to keys `0..i`, not `0..offset+i`. *(measured: max |full − chunked| = 1.30 for a 4-token chunk after 8 cached tokens)* | Any multi-token continuation with a cache (chunked prefill, feeding the next user turn as a block, speculative-decoding verification) is silently wrong in every full-attention layer. The suite only tests prefill-once + single-token steps. | **S2** | When `cache is not None and s > 1`, build an explicit mask with `offset = kv_len − q_len` exactly as `_windowed_mask` does. |
| B5 | `prophet/modeling/layers.py:321, 195-219, 342-350` | For SWA, `cache.append` evicts to `window + sinks` **before** attention and returns the truncated K/V; `_windowed_mask(s, kv_len)` then gets `offset = kv_len − q_len < 0` and the mask is meaningless. Independently, `_windowed_mask` uses buffer indices, not absolute positions, so after any eviction a chunk's window is misaligned. *(measured: max |no-cache − with-cache| = 0.80 for a 24-token prefill, window 8, 2 sinks)* | Every prompt longer than `sliding_window + 1` (2049 tokens on the shipped configs) prefilled with a cache is wrong in every SWA layer — i.e. every real 4k–32k prompt. | **S2** | Attend over the un-evicted K/V for the current chunk and evict afterwards; keep absolute positions in `AttentionCache` and mask on them. Test: prefill of `3×window` tokens with cache == without cache. |
| B6 | `prophet/modeling/model.py:431-446, 482` | Halting at inference: (a) probe coda passes are cache-free, so at decode time (`s = 1`) the halting decision is computed by a coda that sees **only the current token** — attention layers with an empty cache, GDN with a fresh state. *(measured: halt_probs differ between full forward and incremental decode on an untrained model; logits agree to 4e-7 because the real coda is cached)*. (b) Early exit leaves the core slots of skipped iterations stale: *(measured: after a threshold exit at iteration 1, slots `(core,0,1..3)` show `seen=5` while `cache.position=6`)* — the next step that runs deeper reads a state that never saw the previous token. (c) `loop_k=k` is reported even after `break`. (d) The threshold is on `(1−survived).mean()` over batch **and** sequence — one number decides for every sequence in the batch. | `halt_threshold` at inference produces fluent, plausible, wrong output. The "vérifié par test" in `07_WALLS.md` §A.3 covers only the no-threshold case. | **S2** | Decide halting on the prefill only, or give probe passes their own per-iteration coda caches; when exiting early with a cache either advance all slots or forbid variable `k`; return the iterations actually run; threshold per sequence. Add an equivalence test with `halt_threshold` set. |
| B7 | `prophet/config.py:509-513`, `prophet/budget.py:256`, `prophet/modeling/model.py:391-394` | `memory.layers` is validated against the **global** parameterised depth and budgeted at the global index, but read with the **section-local** index and only in `trunk`/`coda`. On a 2/2/2 recurrent model `layers=(5,)` validates, allocates a ledger and is never read. *(measured: writing 50× noise into `ledgers["5"]` leaves the logits unchanged)*. `kind="fast_weight"` validates and is budgeted (`budget.py:181-183`) but builds nothing. | A headline feature can be "enabled" and be a no-op on every recurrent config; the budget counts parameters that do no work. | **S2** | Key ledgers by `(section, idx)`; validate against `section_layout()`; remove `fast_weight` or implement it. |
| B8 | `prophet/memory/consolidate.py:84, 134, 391`; `prophet/modeling/model.py:272-273, 392-394`; `prophet/memory/ledger.py:90, 112` | Three incompatible views of one ledger: the in-model ledger reads the **pre-norm residual** at a block output; `consolidate`/`recall_error`/`depth_agreement` write and read targets in `out.hidden` (**post-`norm_out`**) space; the ledger the model builds from `MemoryConfig` gets `write_lr=0.01, decay=0.999` while every number in `06_MEMORY.md` was measured with `LedgerConfig` defaults (`write_lr=1.0`, `decay=1.0`). `ledger.query` is a trainable `nn.Linear` routed to Muon while the class docstring says keys are frozen. | A ledger consolidated offline cannot be mounted via `cfg.memory.layers`; the integrated ledger takes 1 % of the "exact" step; training moves the addressing under stored associations — the failure the docstring warns about. | **S2** | Pick one read point (post-norm is easier), make `consolidate` use the mounted ledger's read path, align `MemoryConfig` defaults with `LedgerConfig`, freeze `query` (buffer) or document that it must be trained before any write. |
| B9 | `prophet/data/decontaminate.py:131-134, 94, 145-149` | (a) `found[key] += 1` per n-gram **occurrence**, not per distinct n-gram: one repeated 13-gram can cross the threshold and containment can exceed 1.0. *(measured: a document repeating a single trigram 5× scores containment 0.625 against an 8-gram item)*. (b) With `n=13`, `min_example_ngrams=3`, every item under 15 words is matched by `short in text` substring against every document. *(measured: "Yes", "The answer is 4", "What is the capital of France?" reject three innocuous sentences)*. | Rule 5 routes every source through this; at scale it silently discards clean data and does so most on the short-item benchmarks (MC questions, GSM answers). | **S2** | Count `set(matched)`; require a minimum word count and word-boundary matching for the exact path; report containment ≤ 1. |
| B10 | `scripts/train.py:89-91`; `prophet/train/loop.py:63, 194-237`; `prophet/train/optim.py:131`; `prophet/budget.py:341-349` | The trainer is fp32 end to end: no autocast (`TrainConfig.dtype` is never read), fp32 grads, Muon momentum `zeros_like(p)` = fp32, no activation checkpointing, no TF32. `train.py` gates the run with `training_memory(..., optimizer_bytes_per_param=2.0)` — bf16 weights + fp32 master + bf16 grads + **8-bit optimiser** + checkpointed activations — none of which exist. *(measured: printed 37.1 GB "fits"; real static state 3.83B × 12 B = 42.8 GB before activations; activations 2× the estimate per element and 7× without checkpointing)*. The budget also ignores the `k` extra coda passes halting adds (`budget.py:341`). | The gate says "fits" for a run that does not; the MFU planning assumes 312 TFLOPS bf16 while fp32 without TF32 peaks at 19.5. | **S2** | Autocast bf16 + fp32 master, `allow_tf32`, checkpointing; make the gate compute with the trainer's real dtypes; count probe passes. |
| B11 | `prophet/train/loop.py:170, 181-182`; `prophet/modeling/model.py:413, 403` | Only the CPU RNG is checkpointed. `state_init="randn"` draws `torch.randn_like(x)` on the model's device — the CUDA generator on an A100 — and `sample_loop_k` never receives a generator. | Resume is bit-identical on CPU (the test) and not on the target hardware; CLAUDE.md's "reprenable de façon déterministe" does not hold where it matters. | **S2** | Save/restore `torch.cuda.get_rng_state_all()`; better, draw init noise from a per-step seeded generator so resume does not depend on device RNG at all. |
| B12 | `prophet/budget.py:513, 517-521`; `scripts/build_configs.py:88`; `prophet/modeling/model.py:437` | Token budget: `avg_loop_k = (min+max)/2 = 4.5` but the sampler is log-uniform, *(measured E[k] = 3.38)*; and halting (`"ponder"` on **all** shipped configs) runs the coda `k` extra times with grad, which `tokens_affordable` and `training_memory` do not count. On main (4/4/4 blocks) block-passes per token go from `8+4k` to `8+8k`: ×1.63 at k = 3.4, minus the 4.5→3.4 correction → the "22.4B tokens" is ≈35–40 % too high. MTP head cost is also excluded. | The numbers that set the whole plan (tokens/param, "0.5× Chinchilla") are wrong in the flattering direction. | **S2** | Compute E[k] from the configured distribution; add `coda_layers·k` block passes when `halting == "ponder"` plus the MTP block; do the same in `training_memory`. |
| B13 | `prophet/convert/weights.py:165-182`; `prophet/convert/plan.py:158-191` | GDN seed head-scrambling: q for GDN head `h` comes from donor q-head `h`, k from kv-head `h // n_rep`, but the value rows (`expanded.repeat(factor,1)/factor`) and output columns (`widened[:, :o.shape[1]]`) for GDN head `h` are donor heads `2h, 2h+1` (whole-matrix tiling, head_v = 2·head_dim). The generated config also keeps `rope_theta=500k` (Qwen3: 1e6), puts `swa(2048)` + 1 sink on layers the donor ran as full attention, and `residual_scaling=True` (B3). | The claim "the layer's initial function is as close to the donor's attention as a bounded-state mixer can be" is false; the seed pairs head h's queries with other heads' values. | **S3** | Tile per head (`view(h, hd, d).repeat_interleave` along the head_v axis); copy `rope_theta` from the donor; match the donor's attention kind per copied layer; set `residual_scaling=False`. |
| B14 | `scripts/train.py:37-48, 143-144` | The SIGINT/SIGTERM handler sets `_STOP`, which **nothing reads**. Installing it also removes Python's default `KeyboardInterrupt`, so Ctrl-C no longer stops training at all and the `finally:` checkpoint never runs on a polite stop. *(observed: `timeout 500 python3 scripts/train.py --smoke --max-steps-this-session 8` — SIGTERM at 500 s was printed and ignored; the process ran its remaining steps and wrote `ckpt_slot0.pt` ~40 s later.)* | Exactly the opposite of the docstring: the only way out is SIGKILL. | **S3** | Check `_STOP` in `Trainer.train` (or raise from the handler). |
| B15 | `prophet/modeling/layers.py:478-484, 512` | The only path that can train (`flash-linear-attention`) is `pragma: no cover` and never executed here (`HAS_FLA=False`). It passes `head_first=False` (removed in recent `fla` releases), fp32 `g`/`beta` next to bf16 `q/k/v`, and expects `new_state` in the scan's `(b,h,dv,dk)` layout while FLA's `initial_state`/final state are `[B,H,K,V]`. | Either a `TypeError` on the first A100 step, or a silently transposed state whenever a session crosses devices (A100 → Mac/iPhone is the stated use of `session.py`). | **S3** (unverified) | Pin an `fla` version; add a GPU test `chunk_gated_delta_rule ≡ _scan` incl. final state layout; transpose explicitly. |
| B16 | `prophet/memory/session.py:111` | `flat[::step][:n_sampled].numpy()` on a tensor still on the model's device. | `TypeError` for any CUDA model — `model_fingerprint`, hence `extract_session`, is CPU-only. | **S3** | `.cpu()` first. |
| B17 | `prophet/data/streaming.py:66-81`; `prophet/data/mixture.py` | The loader has one weight vector and no phases; `Mixture` phases A/B/C, `context_len` and `lr_schedule` strings are documentation. `TokenSource` wraps to epoch 2+ silently, so the 4-epoch rule `Mixture.validate` enforces is not enforced where tokens are drawn. Nothing converts a Hub dataset or a file into a `TokenSource`; no script trains the production tokenizer; decontamination is not in any path. | `scripts/train.py:116-123` returns 2 for any non-`--smoke` run. | **S3** | Build the missing pipeline (see §6). |
| B18 | `prophet/memory/session.py:119-135, 166` | `extract_session` drops every attention cache; `restore_session` sets `cache.position = tokens_seen`. After restore, prelude/coda attention layers (and the SWA sinks) see an empty cache at position N. | "Resume tomorrow where it stopped today" is not what happens: the prelude that feeds the core has lost its context. | **S3** | Document what is restored, or persist the (bounded) SWA windows and the sinks at least. |
| B19 | `prophet/modeling/model.py:388-390` (inside probe passes) | With halting, the coda's MoE layers run `k+1` times per step: router `aux_loss` and `expert_bias` nudges are applied `k+1×` for coda routers and `1×` for prelude routers; z-loss is `(k+1)×` weighted. | On `prophet_main.json` (all-MoE, halting on) the four coda routers balance ~4× faster and dominate the aux loss. | **S4** | Skip stats/aux/bias updates when `use_cache=False`, or scale by `1/(k+1)`. |
| B20 | `prophet/train/optim.py:187` | `Linear(d,1)` halt/confidence heads are 2-D and not in `ADAMW_ONLY_PATTERNS` → Muon. Newton-Schulz on a `1×d` matrix is normalisation with a `0.2·√d` scale. | Harmless, unintended; the docstring says routers/heads are AdamW's. | **S4** | Route `min(shape)==1` to AdamW. |
| B21 | `prophet/modeling/layers.py:107-108` | `rope_scaling="linear"` multiplies `theta` — that is NTK/base scaling, not linear position interpolation. | Wrong extrapolation recipe if ever used. | **S4** | Divide positions by the factor. |
| B22 | `prophet/modeling/model.py:343, 403` | `"poisson"` ignores the generator; `forward` never passes one. | Depth sampling depends on the global RNG (see B11). | **S4** | Thread a generator. |
| B23 | `prophet/modeling/moe.py:165-171`; `prophet/data/streaming.py:161-164`; `prophet/data/tokenizer.py:255` | 128 `nonzero` device syncs per MoE layer per micro-batch; `SequencePacker.pop` copies the whole buffer per sequence; `_encode_unit` cache is unbounded. | Throughput only. | **S4** | Sort-and-split dispatch; deque; LRU. |

### 2.1 Minimal reproductions

All of the following ran on the snapshot with `python3` (CPU, torch 2.14, no `fla`).

```python
# B1 — forget gate zeroed by model init
from prophet.config import ProphetConfig; from prophet.modeling.model import ProphetModel
m = ProphetModel(ProphetConfig.from_json("configs/prophet_500m_probe.json"))
print(m.sections["core"][0].mixer.a_proj.bias.unique())            # tensor([0.])  (layers.py:438 set 3.0)

# B4 — chunked prefill, full attention
import torch; from prophet.modeling.layers import CausalSelfAttention, RotaryEmbedding, AttentionCache
a = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16).eval(); r = RotaryEmbedding(16, theta=1e4)
x = torch.randn(1, 12, 64); cos, sin = r(torch.arange(12)[None])
with torch.no_grad():
    full = a(x, cos=cos, sin=sin); c = AttentionCache()
    a(x[:, :8], cos=cos[:, :8], sin=sin[:, :8], cache=c)
    chunk = a(x[:, 8:], cos=cos[:, 8:], sin=sin[:, 8:], cache=c)
print((full[:, 8:] - chunk).abs().max())                            # 1.297

# B5 — SWA prefill longer than the window, with a cache
s = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16, window=8, sink_tokens=2).eval()
x = torch.randn(1, 24, 64); cos, sin = r(torch.arange(24)[None])
with torch.no_grad(): print((s(x, cos=cos, sin=sin) - s(x, cos=cos, sin=sin, cache=AttentionCache())).abs().max())  # 0.801

# B2 — the ponder gradient is the same at every position
B, T, K = 2, 5, 3; p = torch.rand(B, T, K).softmax(-1).requires_grad_(); L = [torch.rand(B, T) for _ in range(K)]
g = torch.autograd.grad(sum(p[..., i].mean() * L[i].mean() for i in range(K)), p)[0]   # as loss.py:136-141
print([g[..., i].unique().numel() for i in range(K)])                # [1, 1, 1]

# B7 — a validated, budgeted, never-read ledger
from prophet.config import *; 
cfg = ProphetConfig(d_model=64, frontend=FrontendConfig(vocab_size=128),
    mixer=MixerConfig(pattern=["swa","full_attn"], n_heads=4, n_kv_heads=2, head_dim=16, sliding_window=8, linear_heads=2, linear_head_dim=16, nope_layers=(1,)),
    recurrent=RecurrentCoreConfig(enabled=True, prelude_layers=2, core_layers=2, coda_layers=2, core_pattern=["gdn"]),
    ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    memory=MemoryConfig(enabled=True, kind="product_key", layers=(5,), memory_dim=32, n_slots=1024))
cfg.validate(); m = ProphetModel(cfg).eval(); ids = torch.randint(0, 128, (1, 6))
m.ledgers["5"].write(torch.randn(1, 3, 64), 50 * torch.randn(1, 3, 64))
with torch.no_grad(): a = m(ids).logits; m.ledgers["5"].values.zero_(); b = m(ids).logits
print(torch.allclose(a, b))                                          # True — the ledger is never read

# B9 — decontaminator double counting
from prophet.data.decontaminate import Decontaminator
d = Decontaminator(n=3, threshold=0.5); d.add_benchmark("b", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"])
print(d.check("alpha beta gamma. " * 5 + "unrelated weather"))       # containment 0.625 from one repeated trigram
```

## 3. Claims the code does not honour

| # | Claim | Where it is made | What the code actually does |
|---|---|---|---|
| C1 | "masque d'attention comme objet de première classe supportant les segments bidirectionnels, chemin `inputs_embeds`" | `docs/01_ARCHITECTURE.md:213-216`, `docs/00_PROBLEM_LANDSCAPE.md:317` | No attention-mask object exists; `ProphetModel.forward` (`model.py:348-358`) has no `inputs_embeds`; `ModalityConfig.bidirectional_spans` / `adapter_mount_points` (`config.py:312,315`) are never read. |
| C2 | "Scale residual branches by 1/sqrt(2·depth) **at init**" | `config.py:337` | Forward-time multiplier on every branch, every step (`model.py:156-157`); B3. |
| C3 | "Initialise the forget gate near 1 … starting with an aggressively forgetting state makes the layer untrainable" | `layers.py:435-439` | Bias zeroed by `ProphetModel._init_weights` (`model.py:321`); B1. |
| C4 | "Each candidate stopping point is scored on its own read-out, so the halting head learns which iterations were actually good enough to stop at" / "profondeur dépendant de l'entrée" | `loss.py:132-135`; `docs/07_WALLS.md` §A.3, §B; `01_ARCHITECTURE` D4b | Gradient identical at every position (B2); read-outs un-normalised. |
| C5 | "For Prophet-mini the whole state is under a megabyte" / "état de session ~0.6 Mo" | `session.py:9`; `06_MEMORY.md` §2 table | 10.98 MB at k = 2 on `prophet_mini.json` *(measured)*: 4 core blocks × k iterations × (10·128·256 fp32). Scales with k. |
| C6 | "registre ~33 Mo (mini)" | `06_MEMORY.md` §2 | 33.5 MB is `LedgerConfig()` defaults with `dim=256`; the model builds values of shape `(n_slots, d_model)` (`ledger.py:121`): mini's `MemoryConfig` gives 10.5 MB nominal, **21 MB** actual (fp32 buffer; `n_bytes(dtype_bytes=2)` reports half). |
| C7 | "un token isolé atteint sa cible à 1e-7 près en une seule écriture" | `06_MEMORY.md` §3 | With `write_lr=1.0`; the ledger the model builds uses `write_lr=0.01` (`model.py:272`) → 1 % of the step. |
| C8 | "Keys are **frozen after initialisation**" | `ledger.py:90` | `self.query` (`ledger.py:112`) is a trainable Linear routed to Muon. |
| C9 | `t = m(h⁻) + λ(h⁺ − h⁻)` | `consolidate.py:12-16` module docstring | Code uses the absolute target (`consolidate.py:134`); `06_MEMORY.md` §4 says the docstring's formula is the wrong one. |
| C10 | "Muon: **2 bytes/param of state** versus AdamW's 4-8" / "2 octets/paramètre" | `optim.py:15`; `docs/03_TRAINING.md:30` | `momentum_buffer = zeros_like(p)` → fp32, 4 bytes *(measured)*. |
| C11 | "Entraînement : **BF16 avec poids maîtres FP32**" | `docs/03_TRAINING.md:127` | No autocast, no bf16, no master copy anywhere in `prophet/train/`. |
| C12 | "training memory 37.1 GB … fits" | `scripts/train.py:89-98`, `python -m prophet.budget configs/prophet_main.json` | Assumes an 8-bit optimiser and checkpointing that do not exist; B10. |
| C13 | "1.72B → **0.97B**, **89 %** des paramètres hérités" | `README.md:127,132`; `01_ARCHITECTURE.md:248-250`; `05_ROADMAP.md:15` | `convert_donor.py --plan-only` now prints **1.02B total, 85 %** (865M/1016M) after the estimator fix. |
| C14 | "Configurations … produites par `scripts/design_search.py`" | `README.md:59`, `01_ARCHITECTURE.md:123` | Shipped JSONs come from `build_configs.py` with `n_kv_heads=2` and `halting="ponder"`; the search evaluates `n_kv_heads=max(n_heads//8,1)=1` and no halting (`design_search.py:106-121,175`) — 405.3M vs 408.4M active, and the search never priced the probe passes. |
| C15 | "**235 tests passent**" (×2) | `README.md:76,111` | 295 collected; CLAUDE.md now says "~300". |
| C16 | "a resumed run is asserted … to produce identical weights" / "Exactness … byte-identical" | `scripts/train.py:5-7`; `streaming.py:14-16`; CLAUDE.md non-negotiable 3 | Asserted on CPU only; CUDA RNG not checkpointed (B11). |
| C17 | "harnais d'évaluation à trois niveaux" | `README.md:74-76`; `docs/04_EVAL.md:39-41`; CLAUDE.md | `run_suite` runs whatever lambdas the caller passes (`harness.py:188`); no runner for any named task exists; only `evaluate_bpb` executes a model. |
| C18 | "**One code path.** Training, prefill and single-token decoding share the same forward; passing a cache switches behaviour" | `layers.py:13-14` | Chunked prefill and prefill-beyond-window are wrong with a cache (B4, B5). |
| C19 | "Checkpoint and exit cleanly if the platform gives us any warning at all" | `scripts/train.py:41-44` | Handler sets a flag nothing reads; Ctrl-C disabled (B14). |
| C20 | "Registre à 65 536 emplacements : 201 Mo, contre **158 Mo de poids**" | `docs/07_WALLS.md` §D.4 | No shipped config has 158 MB of weights (mini ≈ 130 MB int4 / 510 MB bf16; main 1.92 GB int4). Unsourced. |
| C21 | "The default of 40B is what `prophet.scaling` says 300 A100-hours buys at ~500M active" | `recipes.py:172-177`; `docs/02_DATA.md` generated at 40B | `prophet.budget` says 22.2B (main) / 52.1B (mini); nothing in the repo yields 40B. |
| C22 | "configs/ générées par `scripts/build_configs.py` (**jamais à la main**)" | `CLAUDE.md:54` | `configs/prophet_tiny_smoke.json` is not in `CONFIGS` (`build_configs.py:97-112`) — hand-written (`n_kv_heads=2`, 4 sinks, `halting="none"`). |
| C23 | "`n_kv_heads` = `n_heads / 8` (GQA)" | `01_ARCHITECTURE.md` §4 table | Shipped: 12/2 (main), 10/2 (mini), 12/3 (probe). |
| C24 | `kv_compression="mla"` shrinks the KV cache (`test_mla_shrinks_the_kv_cache`) | `config.py:111-113`, `budget.py:135-141, 411-419` | Only the estimator knows MLA; the model builds plain GQA. The test tests an estimate of a mixer that does not exist. |
| C25 | `activation`, `norm_kind`, `dropout`, `router_dtype`, `halting="entropy"` are "switches" | `config.py:217,335,343,240,202`; module docstring "Nothing is hard-wired" | All accepted and serialised, none honoured (SwiGLU/RMSNorm/no dropout/fp32 router/no entropy halting are hard-wired). |
| C26 | "Un champ de configuration que rien ne lit est un bug" | `CLAUDE.md:73-75` (added mid-review) | Twenty such fields remain (§4). |

**Honoured, for the record (checked):** budget ≡ model parameter count on all four shipped configs and the
converted donor to ≤1.2e-4 *(measured, after the mid-review fix)*; `linear_beta_max` reaches
`GatedDeltaNet` (`layers.py:476`); the delta-rule scan matches its docstring formula; NoPE now reaches
`CausalSelfAttention`; the probe-pass cache fix (10 positions for 10 tokens) holds; checkpoint
rotation, mixture sums, tokenizer invariants and the allocator behave as documented.

## 4. Dead and unread code

**Config fields that no code outside `prophet/config.py` reads** (grep over `prophet/` and `scripts/`,
excluding each field's own definition line):

| Field | Line | Status |
|---|---|---|
| `FrontendConfig.patch_target_bytes` | `config.py:56` | unread; byte frontend is `NotImplementedError` (`model.py:211-215`) yet fully budgeted (`budget.py:199-224`) |
| `FrontendConfig.patch_entropy_threshold` | `config.py:59` | unread |
| `FrontendConfig.patch_max_bytes` | `config.py:58` | read only by `validate()` (`config.py:517`) |
| `FrontendConfig.local_window` | `config.py:65` | unread |
| `MixerConfig.kv_compression`, `kv_lora_rank` | `config.py:111-113` | budget only; model has no MLA (C24) |
| `MixerConfig.rope_scaling="linear"` | `config.py:133` | mis-implemented (B21) |
| `RecurrentCoreConfig.halting="entropy"` | `config.py:202` | accepted; only `"ponder"` builds a head (`model.py:295-299`) |
| `FeedForwardConfig.activation` | `config.py:217` | unread; SwiGLU hard-wired |
| `FeedForwardConfig.router_dtype` | `config.py:240` | never passed to `MoERouter` (`model.py:126-138`); `moe.py:64` has its own default |
| `MemoryConfig.kind="fast_weight"` | `config.py:259` | validates, budgeted (`budget.py:181-183`), builds nothing |
| `MemoryConfig.update_rule` | `config.py:265` | unread |
| `MemoryConfig.surprise_threshold` | `config.py:269` | unread |
| `MemoryConfig.persist_across_sessions` | `config.py:273` | unread |
| `MemoryConfig.max_persisted_writes` | `config.py:274` | unread |
| `ModalityConfig.bidirectional_spans` | `config.py:312` | unread (C1) |
| `ModalityConfig.adapter_mount_points` | `config.py:315` | unread (C1) |
| `ProphetConfig.norm_kind` | `config.py:335` | unread; RMSNorm hard-wired |
| `ProphetConfig.max_seq_len` | `config.py:342` | unread; no length check anywhere |
| `ProphetConfig.dropout` | `config.py:343` | unread; no dropout anywhere |
| `TrainConfig.dtype` | `loop.py:63` | unread (B10) |
| `TrainConfig.seed` | `loop.py:61` | unread (`train.py` seeds torch itself) |
| `TrainConfig.confidence_weight` | `loop.py:48, 142-145` | resolved from the model config, **never passed** to `compute_loss` (`loop.py:217-225`); no confidence targets exist anywhere → the confidence head (on every shipped config) is never trained |
| `MixerConfig.nope_layers` | `config.py:135` | **fixed mid-review** (`config.py:399-419`, `layers.py:554`) |

**Dead functions, parameters and objects:**

- `count_parameters(cfg, loop_k=...)` — `loop_k` is accepted and unused (`budget.py:227`).
- `build_schedule` (`schedule.py:127`) — never called; `CosineSchedule` — tests only.
- `SparseMoE.aux_loss()` (`moe.py:178`) — never called.
- `Mixture.license_warnings` (`mixture.py:202`) — never called by any script (`verify_datasets.py` reimplements the check).
- `ProphetTokenizer.fertility` (`tokenizer.py:282`), `Decontaminator.index_bytes` (`decontaminate.py:199`), `MixtureSampler.empirical_shares` (`streaming.py:109`) — tests or nothing.
- `CheckpointManager.verify` (`checkpoint.py:195`, "Cheap insurance to run at startup") — `train.py` never calls it.
- `consolidate.py:123, 141` — `slots` set is filled and discarded.
- `scripts/train.py:37-48` — `_STOP` written, never read.
- `GatedDeltaNet.allow_fused` — never set to `False` by anything; `HAS_FLA` exported and never reported.
- `prophet/kernels/` — an empty directory with no `__init__.py`, listed in `CLAUDE.md:53` as a package.
- `ProphetOutput.router_stats` from probe passes — appended `k` extra times (B19).
- `RecurrentState.seen` — set by `GatedDeltaNet` and `restore_session` (to the global count), read by nothing.
- `AttentionCache.seen` — read by one test.

## 5. Doc drift

| Where | Says | Is |
|---|---|---|
| `README.md:76,111` | 235 tests | 295 collected (`pytest --collect-only`); `CLAUDE.md:56` "~300" (was "173" before the mid-review edit) |
| `README.md:127,132`; `01_ARCHITECTURE.md:248-250`; `05_ROADMAP.md:15` | 0.97B, 89 % | 1.02B, 85 % (`convert_donor.py --plan-only`, this snapshot) |
| `README.md:65-69`, `01_ARCHITECTURE.md:129-133` | main 3.83B / 408M, mini 253M / 236M, "produites par design_search.py" | `design_search.py` prints 405.30M / 235.81M for configs that differ from the shipped ones (C14); `prophet.budget` on the shipped JSON: 408.44M / 237.8M |
| `01_ARCHITECTURE.md:135-141` "colle … à 10⁻⁴ … un test le garde" | | True *(measured ≤1.2e-4)*; the test (`test_modeling.py::test_budget_estimate_matches_every_shipped_config`) was added in the same uncommitted edit |
| `01_ARCHITECTURE.md` §4 | GQA `n_heads/8`, window 2048 "identical on every attention layer" | shipped 12/2, 10/2, 12/3; `MixerConfig` default is 4096 |
| `01_ARCHITECTURE.md` §5 | "Rétropropagation tronquée aux 3 dernières itérations" | shipped configs 3, schema default 4 (`config.py:198`) |
| `01_ARCHITECTURE.md` §6 | mask object, `inputs_embeds` | absent (C1) |
| `06_MEMORY.md` §2 | 0.6 MB session, 33 MB ledger | 11 MB, 10.5/21 MB (C5, C6) |
| `06_MEMORY.md` §3, §5 | write exact to 1e-7 | with `write_lr=1.0`; model uses 0.01 (C7) |
| `docs/03_TRAINING.md:30,127` | Muon 2 B/param; BF16 + FP32 master | 4 B/param; fp32 only (C10, C11) |
| `docs/02_DATA.md` (generated), `recipes.py:172-177` | 40B tokens | budget: 22.2B / 52.1B (C21) |
| `docs/07_WALLS.md` §D.4 | 158 MB of weights | matches no config (C20) |
| `docs/07_WALLS.md` §A.3 | "la halte est … implémentée" and gradient reaches the head | head receives a position-independent gradient (B2) |
| `CLAUDE.md:54` | configs never hand-written | `prophet_tiny_smoke.json` is (C22) |
| `CLAUDE.md:44` | "attention GQA/SWA/NoPE" | NoPE only since the mid-review edit |
| `prophet/train/optim.py:18-20` | "routers … keep AdamW" | halt/confidence heads go to Muon (B20) |
| `prophet/data/tokenizer.py:23-26` | "At d_model=1024 a 128k vocabulary spends 197M of a 268M model" | fine, but no shipped config is d=1024 and no trained vocabulary exists in the repo |

## 6. What breaks on a real run first

Walking `python scripts/train.py --config configs/prophet_main.json --steps 40000 --checkpoint-dir /content/drive/…` on an A100 tomorrow:

1. **Exit code 2 before any model is built.** `training_memory(...)` prints "37.1 GB estimated, fits" (wrong, B10), then `scripts/train.py:116-123` prints "Real-corpus streaming is not wired up yet" and returns 2. There is no code that turns a Hub dataset, a file, or a `Mixture` into a `TokenSource`; no trained Prophet-Tok vocabulary artifact and no script to train one (`BPETrainer` is "far too slow for a full corpus"); no phase schedule; decontamination is called by nothing.
2. **If you wire a source:** `ProphetModel(cfg)` allocates 3.83B fp32 parameters on the **host** (15.3 GB; a standard Colab VM has 12.7 GB → host OOM before the GPU is touched), then `.to("cuda")`; Muon momentum adds 15.3 GB; ≈43 GB static on the GPU.
3. **First forward, first core iteration:** `HAS_FLA=False`, so `GatedDeltaNet._scan` (`layers.py:495-529`) runs a Python loop of `seq_len × 4 core layers × k` sequential steps (≈14 000 for seq 1024, k ≈ 3.4), and autograd retains every `S_t`: **144 GB for 8 192 tokens** *(computed)* → CUDA OOM. With `fla` installed instead: `head_first` TypeError on recent versions, or an untested kernel with fp32 gates and a transposed final state (B15).
4. **If that is fixed:** fp32 matmuls without TF32 (19.5 TFLOPS, 6 % of the bf16 peak the 35 % MFU plan assumes); 128-expert Python dispatch with a device sync per expert per layer (B23); halting runs the coda — half of which is attention over the full sequence — `k` extra times per step with grad (B12). Expect single-digit percent of the planned tokens/hour.
5. **Training proceeds and looks healthy** while: every GDN forget gate started at α = 0.5 (B1); every residual branch is scaled ×0.112 (B3); the halting head is fitting a constant (B2); coda routers are balanced 4× faster than prelude routers (B19); the confidence head receives no gradient (§4).
6. **Step 200, first checkpoint:** `torch.save` of ≈31 GB (fp32 weights + Muon state) followed by a SHA-256 over it, alternating two slots on Drive — many minutes per save, 60+ GB of Drive.
7. **Session dies, rerun:** resume works, but `randn_like` state init drew from the CUDA generator → not bit-identical to the uninterrupted run (B11). Ctrl-C to stop early does nothing (B14).
8. **First eval:** nothing to run except `evaluate_bpb` on hand-built `(tokens, n_bytes)` batches; byte counts need the tokenizer that does not exist.
9. **First inference on the 5090 / Mac:** any prompt over 2 049 tokens is wrong in every SWA layer (B5); any multi-turn continuation fed as a block is wrong in every full-attention layer (B4); `halt_threshold` yields confident nonsense (B6); `model_fingerprint` crashes on CUDA (B16).

For calibration, the path that *does* run today: `scripts/train.py --smoke` on `prophet_tiny_smoke.json`
(6.4M parameters, 4×128 tokens per step) took **11.7 s, 8.1 s and 2.8 s** for its first three steps
single-threaded on this CPU *(measured; the spread is `k` drawn from log-uniform 1..4)* — about
50 tokens/s, i.e. the reference delta-rule scan makes even the toy path two to three orders of
magnitude slower than a dense model of that size. Loss fell 7.67 → 7.48 over those steps, so the
loop is wired correctly; it is the model and the scale, not the loop, that are the problem.

## 7. Test gaps (ranked)

1. **No GPU, bf16, autocast or `fla` test at all** (`grep -rl cuda|autocast|bfloat16|HAS_FLA tests/` → nothing). The only path that can train is `pragma: no cover`. Needed: `chunk_gated_delta_rule ≡ _scan` (outputs and final-state layout), skipped when unavailable.
2. **Cache equivalence covers two shapes only.** `test_prefill_then_decode_matches_full_forward` uses `n_kv_heads=4` (no GQA), no window, and one prefill. Needed: chunked prefill (q_len > 1 with a non-empty cache) for full attention; prefill of `3×window` tokens with a cache for SWA; both at the model level. Would have caught B4/B5.
3. **Halting is tested for shapes, not for learning.** `test_a_higher_threshold_buys_more_iterations` checks monotonicity of an untrained head — it passes for any head. Needed: a synthetic task whose optimal depth differs by token; assert the trained head separates them (fails today, B2); halting + cache + threshold equivalence (B6).
4. **No test that model init preserves layer-local inits** (B1). `test_training_reduces_loss_on_a_learnable_task` passes with α = 0.5 because the corpus is a 32-token cycle.
5. **Resume determinism on CPU only** (B11). A test can assert that `state_dict()` carries CUDA RNG when `torch.cuda.is_available()`, and the noise path can be made device-independent and tested on CPU.
6. **MoE + recurrence + halting are never combined** in a test; `test_moe_model_trains_and_reports_router_stats` uses `tiny_config()` without halting. B19 is invisible.
7. **Conversion is tested tensor-wise, never function-wise.** `test_attention_weights_are_copied_verbatim` compares matrices; no test runs a donor block and the converted prelude block on the same input (would catch B3, `rope_theta`, swa/sinks, B13).
8. **`training_memory` is never compared to a real allocation.** A CPU test can compare `sum(t.numel()·t.element_size())` of params + grads + optimiser state on the tiny config to the estimate computed with the trainer's real dtypes (B10).
9. **Memory integration is tested on a non-recurrent trunk only** (`_model_with_memory`, `layers=(2,)`). A recurrent model with `memory.layers` would expose B7; no test reads a `consolidate()`-written ledger through `cfg.memory.layers` (B8); no test that the integrated ledger's `write_lr=0.01` still converges.
10. **Decontaminator false positives.** No test for repeated n-grams; `test_short_examples_are_matched_exactly` asserts the substring behaviour as desired (B9).
11. **Tokenizer at scale.** Trained on four sentences; no fertility test on real text; no test that a saved vocabulary's `n_tokens ≤ frontend.vocab_size`; nothing trains or ships the production vocabulary.
12. **Streaming.** No test of documents longer than `seq_len` spanning several batches across a resume, no multi-phase mixture, no test that the 4-epoch cap is enforced at draw time (it is not).
13. **Tests that test the mock:** every harness test passes lambdas; `test_evaluate_bpb_runs_against_the_real_model` scores random tokens; the `*_renders` tests assert substrings.
14. **The 15 % tolerance that let a 31M miscount live** was tightened mid-review; the same looseness remains in `test_donor_specs_reproduce_their_advertised_sizes` (0.85–1.15 on unverified specs).
15. **Signal handling, `verify()` at startup, and `train.py` itself** have no test; the smoke path is exercised only by hand.

## 8. The ten things to fix first

1. **B1** — stop `_init_weights` from zeroing `a_proj`/`b_proj` biases; test `sigmoid(bias) > 0.9` after `ProphetModel(...)`.
2. **B2** — per-token ponder expectation, `norm_out` before `project`; a learnability test on a synthetic depth task.
3. **B3** — make `residual_scaling` do what its docstring says (init-time), or expose the forward multiplier as a separate ablated switch; `prophet_config_for_donor` sets it off; add a donor-block ≡ converted-block equivalence test (also catches B13 and `rope_theta`).
4. **B4/B5** — explicit offset mask whenever `cache is not None and s > 1`; attend before evicting; model-level equivalence tests for chunked prefill and prefill > window.
5. **B10/B15** — bf16 autocast with fp32 master, TF32, activation checkpointing; pin `fla`, transpose its state, add the GPU equivalence test; make `train.py` refuse a non-smoke run when `HAS_FLA` is `False`; have the memory gate use the trainer's real dtypes and count the probe passes.
6. **B17** — the data path: a tokenizer-training script that ships an artifact, Hub/file → `TokenSource`, `Mixture` phases → loader schedule, decontamination in the path (with B9 fixed), epoch cap enforced at draw time. Until this exists the README should not say the loop "tourne de bout en bout".
7. **B6** — define halting semantics with a cache (decide on prefill, or per-iteration coda caches), report the iterations actually run, threshold per sequence, add the equivalence test with a threshold.
8. **B7/B8** — one ledger read point shared by the model and `consolidate`; key ledgers by `(section, idx)`; align `MemoryConfig` with `LedgerConfig`; freeze `query`; delete `fast_weight` or implement it; correct the sizes in `06_MEMORY.md`.
9. **B11/B14/B16** — checkpoint CUDA RNG (or make the noise path device-independent); read `_STOP`; `.cpu()` in `model_fingerprint`; call `verify()` at startup.
10. **§4 + B12 + §5** — delete or implement the twenty unread fields (per CLAUDE.md's own new rule); compute E[k] from the configured distribution and add halting/MTP passes to `tokens_affordable`/`training_memory`; refresh README/01_ARCHITECTURE/05_ROADMAP numbers (295 tests, 1.02B / 85 %, session 11 MB, "produced by build_configs.py").
