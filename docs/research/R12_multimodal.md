# R12 — Multimodality and On-Device Perception: Verdict, Cheapest Path, and Day-One Architectural Hooks

**Track:** R12 | **Status:** research complete, recommendation ready for architecture freeze
**Author:** research agent, Prophet project
**Date:** 2026-09-03

> **Sourcing note.** arxiv.org, huggingface.co and semanticscholar were blocked by this session's
> egress policy, and the shared WebSearch budget was exhausted. Every number below marked
> **[V]** was verified in this session by fetching primary sources over the one reachable
> channel (`raw.githubusercontent.com`): official repo READMEs, HF blog markdown sources, and
> `transformers` model configs. Numbers marked **[M]** are from model knowledge (cutoff May 2026)
> and could not be re-verified here — treat them as approximately right and re-check before
> anyone bets compute on them. arXiv IDs are similarly tagged in §8.

---

## 1. Problem statement + strategic verdict

### 1.1 The question

Prophet's win condition is **text**: beat Qwen3-1.7B/4B, Gemma-3-4B, Phi-4-multimodal on
reasoning, math, code and knowledge, at ~1–1.5B active parameters, trained on a few hundred
A100-hours on a single Colab A100-80GB. Multimodality would get at most **10–15% of that
budget, i.e. ~30–75 A100-hours**.

Three sub-questions:
1. Does adding vision help us win, or does it cost us the text benchmarks that *are* the win?
2. If we do it, what is the cheapest recipe that leaves the text model provably untouched?
3. What must the text architecture contain from day one so that (2) is possible later?

### 1.2 Verdict: **LATER — but the hooks are NOW, and they are non-negotiable.**

Concretely:

| Phase | What | GPU cost | Text impact |
|---|---|---|---|
| **v1 (now)** | Implement the 10 architectural hooks in §4.3. Ship a **text-only** model. | **0 A100-h** | Zero (all hooks are provable no-ops) |
| **v1.5 (opt.)** | nanoVLM-style connector-only proof-of-life on the **mini** (300–600M dense). Validates hooks end-to-end. | **6–12 A100-h** | Zero (LLM frozen) |
| **v2 (later)** | Real VLM: frozen SigLIP2 + pixel-shuffle connector + modality-LoRA on the main MoE. | **~26 A100-h** (hard cap 50) | Zero with adapter off (bit-identical weights) |
| **v3+ (not now)** | Audio-native, video, early fusion, image generation. | ≫ budget | — |

### 1.3 Why LATER and not NOW

**Vision adds zero points to our primary benchmarks.** MMLU, GSM8K, MATH, HumanEval, ARC do not
move because the model can see. Every hour of the text budget diverted is a direct subtraction
from the only scoreboard we have said we intend to win.

**We cannot beat the dedicated VLMs on their own turf, and we should stop pretending otherwise.**
The comparison set — InternVL3-2B, Qwen2.5-VL-3B, SmolVLM2, Moondream — were trained on
multimodal corpora 2–3 orders of magnitude larger than ours. SmolVLM alone consumed The Cauldron
(50 datasets) plus Docmatix (2.4M images / 9.5M QA pairs **[V]**) on a multi-node cluster, and
*still* scores DocVQA 81.6 vs Qwen2-VL-2B's 90.1 **[V]**. A 26-A100-hour run will not close a gap
that a cluster-scale run did not close. Any plan whose success criterion is "beat Qwen2.5-VL-3B on
OCRBench" is not credible and should be rejected on sight.

**The binding constraint is not FLOPs, it is data I/O on Colab.** This is the single most
underrated risk in this track. The Cauldron and FineVision are multi-TB with per-sample JPEG
decode; Colab A100 instances have limited local disk, weak CPU, and rate-limited network. A
vision run that is nominally 6 GPU-hours becomes a 30-hour GPU-idle disaster if the loader
starves. §6-R1 gives the mitigation (precompute vision features once, ship packed shards).

### 1.4 Why not NEVER

**The marginal cost is genuinely small — *if and only if* the hooks exist.** ~26 A100-hours is
5–9% of a 300–500 hour budget. That is affordable. What is *not* affordable is discovering at
v2 that the vocabulary has no reserved IDs, that RoPE is hard-coded 1-D, that the attention call
site is `is_causal=True` with no mask path, and that adding any of it requires a re-pretrain.
**Retrofitting the hooks costs a full training run; installing them now costs a few hundred
thousand parameters and a week of engineering discipline.** That asymmetry is the entire
argument of this report.

**Half our comparison set is multimodal.** Gemma-3-4B and Phi-4-multimodal see. A text-only
Prophet loses every multimodal comparison by forfeit, and "it beats Qwen3-1.7B on text *and*
it sees" is a materially stronger claim than the first half alone.

**Our deployment target is a phone with a camera.** On-device vision is the archetypal
justification for a small local model. It is the story, even if it is not the scoreboard.

### 1.5 Audio: **NO** for v1 and v2.

Use a **pipelined** ASR front-end (Moonshine or `distil-whisper/distil-small.en`, 166M params,
short-form WER 12.1, within 4% WER of Whisper-large-v3 **[V]**) that emits text into Prophet.
Audio-native fusion (Qwen2.5-Omni's Thinker-Talker + TMRoPE, Phi-4-MM's speech-LoRA) is a
different order of engineering and needs speech data we do not have. **Reserve the token IDs and
the modality type now (§4.3 H1/H2); write no audio code.** The hooks are shared with vision, so
this reservation is free.

---

## 2. State of the art

### 2.1 Cheap VLM recipes — verified training cost

| Recipe | Vision enc. | LLM | Data | **Measured cost** | Result |
|---|---|---|---|---|---|
| **nanoVLM-222M** **[V]** | SigLIP-B/16-224 (85M) | SmolLM2-135M | 1.7M samples of `the_cauldron` | **~6 h on 1×H100** (≈10–14 A100-h) | **MMStar 35.3** |
| **LLaVA-1.5-7B stage 1** (projector only) **[V]** | CLIP-L/14-336 | Vicuna-7B | `liuhaotian/LLaVA-Pretrain` 558K | 3.5 h × 8×A100-80G = **28 A100-h** | connector aligned |
| **LLaVA-1.5-7B stage 2** (full SFT) **[V]** | " | " | `llava_v1_5_mix665k` 665K | 10 h × 8×A100-40G = **80 A100-h** | VQAv2 78.5 **[M]**, TextVQA 58.2 **[M]** |
| **LLaVA-1.5-13B** total **[V]** | " | Vicuna-13B | 558K + 665K | (5.5 + 20) h × 8 = **204 A100-h** | SoTA-on-11-benchmarks (2023) |
| **LLaVA-Lightning** **[V]** | " | " | reduced | **$40 / 3 h** | "lite multimodal GPT-4" |
| **SmolVLM-2.2B** **[V]** | SigLIP-SO400M-p14-384 (shape-opt.) | SmolLM2-1.7B | Cauldron + Docmatix | multi-node, not published | see §2.3 |
| **TinyLLaVA-Factory** **[V]** | SigLIP-SO400M-384 | swappable | 558K + 665K | ~1 node-day class | see §2.2 |
| **MolmoE-1B** **[V]** | CLIP | **OLMoE-1B-7B (MoE, 1B active / 7B total)** | PixMo | cluster | — |

**The single most important datapoint for us is nanoVLM: ~6 H100-hours produced a usable VLM
(MMStar 35.3) from frozen-ish 85M + 135M backbones on 1.7M samples.** That is the floor. Scaled
to a 1.5B-active decoder with a 4–7× smaller token budget per image, the arithmetic in §5 lands
at ~26 A100-hours for a much better model.

Also note the *shape* of the LLaVA numbers: **stage 1 (connector alignment) is 26% of the cost of
stage 2 (instruction tuning)**. The expensive part is not teaching the projector to speak; it is
the SFT. That is where LoRA saves us.

### 2.2 TinyLLaVA-Factory backbone sweep — all SigLIP-SO400M-384, identical 558K+665K recipe **[V]**

This is the cleanest available controlled study of "how good does a small LLM + a good frozen
encoder get on the standard cheap recipe":

| LLM backbone | VQA-v2 | GQA | SQA-I | TextVQA | MM-Vet | POPE | MME | MMMU-val |
|---|---|---|---|---|---|---|---|---|
| OpenELM-450M-Instruct | 71.7 | 53.9 | 54.1 | 44.0 | 20.0 | 85.4 | 1118.8 | 24.0 |
| Qwen2-0.5B | 72.3 | 55.8 | 60.1 | 45.2 | 19.5 | 86.6 | 1153.0 | 29.7 |
| **Qwen2.5-0.5B** | 75.3 | 59.5 | 60.3 | 48.3 | 23.9 | 86.1 | 1253.0 | 33.3 |
| TinyLlama-1.1B | 75.5 | 58.6 | 64.0 | 49.6 | 23.5 | 86.3 | 1256.5 | 28.3 |
| StableLM-2-1.6B | 78.2 | 60.7 | 66.7 | 56.0 | 29.4 | 86.3 | 1319.3 | 32.6 |
| Gemma-2B-it | 78.4 | 61.6 | 64.4 | 53.6 | 26.9 | 86.4 | 1339.0 | 31.7 |
| Phi-2 (2.7B) | 79.2 | 61.6 | 71.9 | 57.4 | 35.0 | 87.2 | 1462.4 | 38.2 |
| **Qwen2.5-3B** | 79.4 | 62.5 | 74.1 | 58.3 | 34.8 | 87.4 | 1438.7 | 39.9 |
| Phi-2 + ShareGPT4V ("share" recipe) | **80.1** | 62.1 | 73.0 | **60.3** | **37.5** | 87.2 | **1466.4** | 38.4 |
| Phi-2, LoRA instead of full FT | 77.6 | 59.7 | 71.6 | 53.8 | 33.3 | 87.9 | 1413.2 | 35.6 |

**Two findings that directly shape our plan:**
1. **The LLM backbone quality dominates.** Same encoder, same data, same recipe: MMMU goes
   24.0 → 39.9 purely by swapping the decoder. *This is the strongest possible argument for
   spending our budget on text first.* A better Prophet makes a better Prophet-VL for free.
2. **LoRA instead of full fine-tuning costs ~1.5–3 points** (Phi-2: 79.2→77.6 VQAv2,
   57.4→53.8 TextVQA, 38.2→35.6 MMMU). That is the *measured price* of the bit-identical-text
   guarantee. It is a price worth paying, and it should be stated honestly in the plan rather
   than hand-waved.

### 2.3 Small-VLM benchmark landscape, with memory **[V]** (from the SmolVLM release)

| Model | MMMU val | MathVista | MMStar | DocVQA | TextVQA | **Min GPU RAM (GB)** |
|---|---|---|---|---|---|---|
| SmolVLM 2.2B | 38.8 | 44.6 | 42.1 | 81.6 | 72.7 | **5.02** |
| Qwen2-VL 2B | **41.1** | **47.8** | 47.5 | **90.1** | **79.7** | 13.70 |
| InternVL2 2B | 34.3 | 46.3 | **49.8** | 86.9 | 73.4 | 10.52 |
| PaliGemma 3B 448px | 34.9 | 28.7 | 48.3 | 32.2 | 56.0 | 6.72 |
| moondream2 | 32.4 | 24.3 | 40.3 | 70.5 | 65.2 | **3.87** |
| MiniCPM-V-2 | 38.2 | 39.8 | 39.1 | 71.9 | 74.1 | 7.88 |
| MM1.5 1B | 35.8 | 37.2 | — | 81.0 | 72.5 | — |

**Read the last column.** SmolVLM matches or beats Qwen2-VL-2B on nothing, yet uses **2.7× less
memory**. The entire SmolVLM thesis is the token budget, and it is the right thesis for us:
> "SmolVLM encodes each 384×384 image patch to **81 tokens**. This results in SmolVLM encoding
> our test prompt and a single image in **1.2k tokens, whereas Qwen2-VL uses 16k tokens**."
> — and therefore "prefill throughput is **3.3 to 4.5× faster**, generation throughput
> **7.5 to 16× faster**" **[V]**

That is a **13× token reduction** buying a 2.7× memory reduction and a ~4× prefill speedup, at a
cost of a few benchmark points. On a phone, that trade is not close.

### 2.4 The token budget problem — the core on-device constraint

Raw patch counts (image side / patch size, squared):

| Config | Raw tokens | With compression | Used by |
|---|---|---|---|
| 224px / p16 | 196 | — | SigLIP-B baseline |
| 336px / p14 | **576** | — | **LLaVA-1.5** |
| 364px / p14 | 676 | **169** (pixel-shuffle ×2, 4× reduction) | **Idefics3** **[V]** |
| 384px / p14 | **729** | **81** (pixel-shuffle ×3, 9× reduction) | **SmolVLM** **[V]** |
| 448px / p14 | 1024 | — | PaliGemma-448 **[M]**, Phi-4-MM (`image_size: 448, patch_size: 14`) **[V]** |
| 512px / p16 | 1024 | **64** (pixel-shuffle factor 4) | **nanoVLM current default** **[V]** |
| 896px / p14 | **4096** | — | PaliGemma-896 **[M]** |
| 4×4 tiling + global @81/tile | — | **1377** | SmolVLM high-res |

nanoVLM's config is the cleanest statement of the target **[V]**:
```python
vit_model_type = 'google/siglip2-base-patch16-512'   # 32×32 = 1024 patches
mp_pixel_shuffle_factor = 4                          # 4×4 = 16× reduction
mp_image_token_length  = 64                          # → 64 tokens per view
```

**Compression methods, ranked for us:**

| Method | Params | Cost | Reduction | Verdict for Prophet |
|---|---|---|---|---|
| **Pixel-shuffle / space-to-depth** | 0 (+MLP) | free reshape+permute | r² (4×, 9×, 16×) | **✅ Use this.** ANE-friendly (pure reshape). Idefics2→Idefics3 *moved off* Perceiver *onto* pixel shuffle — a strong signal. |
| Perceiver resampler (Flamingo, Idefics2, Qwen-VL) | ~10–100M | trained cross-attn | fixed 64–256 out | ⚠️ Fixed output length is attractive, but it's a from-scratch trained module and cross-attention exports badly to ANE. |
| Q-Former (BLIP-2) | ~100M | multi-stage pretraining | 32 queries | ❌ Effectively abandoned by 2024–25. Needs its own curriculum. |
| **ToMe** (token merging in ViT) | 0 | training-free | ~2× throughput **[M]** | ⚪ Optional free extra. |
| **FastV** | 0 | training-free, inference | prune 50% after layer 2 → **45% FLOPs cut**, ~no loss **[V]** | ⚪ Free at inference. Gains are small when tokens are already 64. |
| **VisionZip** | 0 | training-free, inference | keep **54 dominant + 10 contextual = 64** of 576; **10% of tokens → ~95% of performance** **[V]** | ✅ Best inference-time knob. Text-agnostic, so it composes with KV-cache and multi-turn. |
| **FastVLM / FastViTHD** (encoder redesign) | new encoder | full retrain | see §2.6 | ⚪ Right idea, wrong licence & no HF drop-in. Watch for v3. |

**Key architectural consequence:** pixel-shuffle wins because it is a *reshape*. On the ANE, a
reshape+permute costs nothing and compiles cleanly; a Perceiver's cross-attention does not. The
cheapest compression is also the most exportable one. Take the free win.

### 2.5 Vision encoders

| Encoder | Params | Res | Notes |
|---|---|---|---|
| SigLIP-B/16-224 | 85–86M **[V]** | 224 | nanoVLM's original choice; 196 tokens |
| SigLIP-SO400M-p14-384 | ~400M **[M]** | 384 | 729 tokens; SmolVLM, TinyLLaVA, Kimi-VL (as MoonViT) **[V]** |
| **SigLIP2-base-p16-512** | ~86M **[M]** | 512 | **nanoVLM's current default**; 1024→64 tokens **[V]** |
| **SigLIP2 NaFlex** | base | **variable** | Native aspect ratio; `max_num_patches` default **256**, tunable **[V]**. A token budget knob *inside the encoder* — very relevant for documents. |
| SigLIP2-so400m-p14-384 | ~400M **[M]** | 384 | Multilingual, + decoder-pretraining/self-distillation/masked prediction **[V]** |
| DINOv2 ViT-L/14 | 300M **[M]** | 224/518 | Dense/spatial features, no language alignment |
| DINOv3 | ViT-S/B/L distilled from 7B **[M]** | — | Strongest dense features (2025) |
| **AIMv2-L distilled** | ~300M **[M]** | 224/336/448/native | Apple explicitly labels it **"recommended for multimodal applications"** **[V]**; AIMv2 "outperforms OAI CLIP and SigLIP on the majority of multimodal understanding benchmarks", AIMv2-3B hits **89.5% IN-1k frozen trunk** **[V]** |
| FastViTHD | 3.4× smaller than LLaVA-OV-0.5B's **[V]** | high-res | Hybrid conv-transformer, few tokens by construction |

**Recommendation: `google/siglip2-base-patch16-512`** — 86M, 512px (good for documents), and
1024 patches divides cleanly by 16 to give exactly 64 tokens. It is nanoVLM's own current choice,
so the recipe is pre-debugged. AIMv2-L-distilled is the upgrade path if we have spare hours;
SigLIP2-NaFlex is the upgrade path specifically for OCR.

### 2.6 FastVLM (Apple, 2412.13303, CVPR 2025) — designed for exactly our target **[V]**

Verified claims from the official repo:
- **FastViTHD**, a hybrid vision encoder that "output[s] fewer tokens and significantly reduce[s]
  encoding time for high-resolution images."
- "Our smallest variant outperforms **LLaVA-OneVision-0.5B with 85× faster Time-to-First-Token
  (TTFT)** and a **3.4× smaller vision encoder**."
- "Our larger variants using Qwen2-7B outperform Cambrian-1-8B with a **7.9× faster TTFT**."
- Sizes 0.5B / 1.5B / 7B; three training stages; LLaVA codebase.
- Ships an **iOS 18.2+ / macOS 15.2+ app that displays TTFT per inference**.
- **Deployment split (this is the important part):** the vision encoder is exported with
  **coremltools** (`export_vision_encoder.py`), while the **LLM runs on MLX** with 4-bit or 8-bit
  quantization (`mlx_vlm.convert --only-llm -q --q-bits 8`).

**The lesson for Prophet is the deployment split, not the encoder.** Apple sends the ViT to
CoreML (→ ANE, compute-bound, loves fixed shapes and convs) and the decoder to MLX (→ GPU/Metal,
memory-bandwidth-bound). This directly implies hook **H9**: the vision encoder must remain a
standalone module with its own export target, never fused into the LM graph.

The secondary lesson: at high resolution, **vision encoding latency dominates TTFT**, not LLM
prefill — which is why an 85× TTFT speedup is even possible. Our defence against this is the same
one: few tokens, small encoder, and no tiling on-device.

### 2.7 Early/native fusion vs. late fusion

| Model | Approach | Image representation | Verdict for us |
|---|---|---|---|
| **Chameleon** (Meta) | Early fusion, discrete VQ tokens, single transformer **[V]** | VQ tokens, 1024+/image **[M]** | ❌ From-scratch mixed-modal pretraining |
| **Fuyu-8B** (Adept) | No encoder at all; linear projection of raw patches | raw patches | ❌ Vision must be *learned in the decoder* — the expensive thing. No paper, blog only. |
| **Transfusion** | AR text + diffusion image in one transformer **[M]** | continuous | ❌ Way out of budget |
| **Emu3** (BAAI) | "trained solely with next-token prediction... a single transformer **from scratch**" **[V]** | VQ (`BAAI/Emu3-VisionTokenizer`) | ❌ "from scratch" is the disqualifier |
| **Janus / Janus-Pro** (DeepSeek) | **Decoupled** visual encoding: separate paths for understanding vs generation, one transformer **[V]** | 576 tokens/image **[V]** | ⚪ Elegant; but generation is out of scope |
| **Scaling Laws for Native Multimodal Models** (2504.07951) | early fusion ≈ compute-optimal-competitive with late fusion; **sparse MoE substantially helps NMMs by letting experts specialize per modality** **[M]** | — | ⚠️ See below |

**Critical reading of 2504.07951 for our situation.** The paper's headline — that native
early-fusion multimodal models are not worse per FLOP than late-fusion ones, and that MoE helps
them by learning modality-specialized weights — is a statement about **from-scratch training runs
at a given compute budget**. It emphatically does *not* say early fusion can be obtained cheaply
from an existing text model. For us:

- We can afford **exactly one** from-scratch pretraining run. Early fusion would mean mixing
  image data into that single run, which trades directly against the text benchmarks that are our
  win condition. **The scaling law says the trade is fair; our objective function says the trade
  is a loss**, because we are not optimizing joint multimodal loss — we are optimizing text
  benchmarks with vision as a side quest.
- Discrete-token early fusion additionally pushes image sequences to 1024+ tokens, the exact
  opposite of our on-device requirement.

**But steal one idea from it:** MoE experts naturally specialize by modality. That is a direct
argument for hooks **H5(c)** and **H5(d)** — design the MoE so modality-specialist capacity can be
*added* later, and keep always-on shared experts as the natural cross-modal home.

**Decision: late fusion (LLaVA-style), frozen encoder, trained connector, modality-LoRA.**

### 2.8 Audio

| System | Params | Numbers | Role |
|---|---|---|---|
| `distil-whisper/distil-small.en` | **166M** **[V]** | Short-form WER **12.1**, long-form **12.8**, rel. latency 5.6×, "within 4% WER of Whisper large-v3" **[V]** | ✅ **Our ASR front-end** |
| `distil-whisper/distil-large-v3` | 756M **[V]** | WER 9.7 / 10.8, 6.3× faster, 49% smaller than Whisper **[V]** | Desktop tier |
| **Moonshine Voice** | tiny → large | Runs on-device across Python/JS-WASM/iOS/Android/Mac/Linux/Win/RPi; claims "higher accuracy than Whisper Large V3" at the top end **[V]**; streaming-first (works while the user is still talking) **[V]** | ✅ **Best streaming/edge option**; MIT |
| **Moonshine Micro** | ~1.3 MiB STT | Runs in **~470 KB RAM** on an RP2350 (80¢ MCU); VAD 89 KiB flash / 36 KiB SRAM; full VAD+STT+TTS pipeline ~3.6 MiB flash **[V]** | Reference for how small this *can* go |
| **Phi-4-multimodal** | 5.6B + LoRAs | `input_size: 80` mel, `time_reduction: 8`, `num_blocks: 24`, `hidden_size: 1024` **[V]** → **12.5 audio tokens/s** | ⭐ The LoRA-per-modality trick — see §2.9 |
| **Qwen2.5-Omni** | 7B | Thinker-Talker; **TMRoPE** (Time-aligned Multimodal RoPE) interleaves audio+video and aligns timestamps; block-wise streaming encoders; sliding-window DiT for low first-packet latency **[V]** | ❌ Out of budget; ⭐ but TMRoPE informs hook H3 |

**The audio token-rate number matters.** Phi-4-MM's audio encoder does 80-mel at the standard
100 fps with `time_reduction: 8` → **12.5 tokens/second**, so one minute of speech is ~750 tokens.
Whisper's encoder is ~50 tokens/s **[M]** — 4× more expensive. If audio is ever added, copy
Phi-4-MM's rate, not Whisper's.

### 2.9 ⭐ Phi-4-multimodal's LoRA-per-modality trick — the single most relevant prior art

Verified from the `transformers` implementation **[V]**:

> "Phi4 Multimodal is a multimodal model capable of text, image, and speech and audio inputs or
> any combination of these. It features a **mixture of LoRA adapters** for handling different
> inputs, and each input is **routed to the appropriate encoder**."

```python
model.load_adapter(model_path, adapter_name="vision", adapter_kwargs={"subfolder": "vision-lora"})
model.set_adapter("vision")     # and separately, "speech-lora"
```
Config constants **[V]**: `image_token_id = 200010`, `audio_token_id = 200011`;
vision tower `hidden_size 1152, layers 27, image_size 448, patch_size 14` (a SigLIP-shaped ViT);
audio tower `hidden_size 1024, num_blocks 24, input_size 80, time_reduction 8`.

**Why this is exactly our answer:** the base text weights are shipped **unmodified**. Vision and
speech are *additive, swappable, per-modality* LoRA deltas plus an encoder plus a projector. With
the adapter disabled, the model is byte-for-byte the text model. That converts "will vision hurt
our text scores?" from an empirical question requiring an expensive ablation into a **structural
guarantee**. This is the pattern Prophet should adopt, and hook **H5** is its architectural
prerequisite.

### 2.10 Training data — HF IDs and sizes

| HF ID | Size | Use |
|---|---|---|
| `liuhaotian/LLaVA-Pretrain` | **558K** image-caption (LAION-CC-SBU + BLIP captions) **[V]** | ✅ **Stage A connector alignment.** Small on disk, most-reproduced alignment set in existence. |
| `liuhaotian/LLaVA-Instruct-150K` (→ `llava_v1_5_mix665k.json`) | **665K** (150K GPT-generated + ~515K academic VQA) **[V]** | ✅ **Stage B core.** |
| `HuggingFaceM4/the_cauldron` | **50 datasets** **[V]**, ~30M samples **[M]** | ✅ Subsample. nanoVLM used **1.7M samples** of it **[V]**. |
| `HuggingFaceM4/Docmatix` | **2.4M images / 9.5M Q-A pairs** from 1.3M PDFs; **240× larger** than prior DocVQA sets; fine-tuning Florence-2 on it gave **+20% DocVQA** **[V]** | ✅ **Subsample ~300K.** Highest-leverage data we can add — document/OCR is where small VLMs are weakest and where a win is most visible. |
| `pixparse/pdfa-eng-wds` | 2.1M PDFs (Docmatix's source) **[V]** | ⚪ Raw OCR |
| `HuggingFaceM4/FineVision` | ~24M samples / 200 datasets **[M]**; nanoVLM's current default is `HuggingFaceM4/FineVision_concat_shuffled_2` **[V]**, with per-sample quality ratings (`relevance`, `image_correspondence`, `visual_dependency`, `formatting`) **[V]** | ⚠️ Best modern mix, but **multi-TB**. Only if I/O is solved. The quality-rating filters are genuinely useful. |
| `lmms-lab/LLaVA-OneVision-Data` | ~3.5–4M **[M]** | ⚪ Too big for v2 |
| `nyu-visionx/Cambrian-10M` (Cambrian-7M curated subset) | 10M / 7M **[M]** | ⚪ Too big for v2 |
| **PixMo** (all `allenai/`, VLM-free construction **[V]**): `pixmo-cap` (dense captions, ~200 words avg), `pixmo-ask-model-anything`, `pixmo-cap-qa`, `pixmo-points` (grounding/counting), `pixmo-point-explanations`, `pixmo-docs` (synthetic charts/tables/diagrams), `pixmo-clocks`, `pixmo-count` **[V]** | varies | ✅ `pixmo-docs` + `pixmo-cap` subsamples. ⚠️ Images are **URLs** — link rot is real; budget for a lower yield. |
| `HuggingFaceH4/rlaif-v_formatted` | — | ⚪ DPO for hallucination reduction **[V]** |
| ShareGPT4V | — | ⚪ TinyLLaVA's "share" recipe gained **+0.9 VQAv2 / +2.9 TextVQA / +2.5 MM-Vet** over "base" **[V]** — cheap upgrade |

---

## 3. What transfers to our scale

**Transfers cleanly:**

1. **The frozen-encoder + trained-connector recipe.** nanoVLM proves ~6 H100-hours suffices for a
   working VLM **[V]**. This is the only part of the VLM literature that is genuinely cheap.
2. **Pixel-shuffle token compression.** Zero parameters, pure reshape, 4–16× reduction, ANE-clean.
   SmolVLM's 81 tokens and nanoVLM's 64 tokens are both reachable **[V]**.
3. **Phi-4-MM's modality-LoRA.** Gives a *structural* rather than empirical guarantee of text
   preservation **[V]**. This is the key unlock for a text-first project.
4. **MoE decoders work fine for VLMs.** MolmoE-1B (OLMoE 1B-active/7B-total) **[V]**, Kimi-VL-A3B
   (16B total / 2.8B active, MoonViT encoder) **[V]**, DeepSeek-VL2, Llama-4 **[V]**. Our exact
   sparsity shape is a well-trodden path — this is reassuring and worth stating plainly.
5. **`lmms-eval`** as the harness. nanoVLM's default task string is a ready-made eval suite **[V]**:
   `mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa`.
6. **The encoder/decoder deployment split** (CoreML-ANE for the ViT, MLX/GGUF for the LM) **[V]**.
7. **Docmatix for documents.** +20% DocVQA on Florence-2 from one dataset **[V]** is the best
   points-per-GPU-hour available anywhere in this literature.

**Does NOT transfer:**

1. **Any from-scratch native/early-fusion training** (Chameleon, Emu3, Transfusion). Requires the
   one pretraining run we cannot spend twice.
2. **High-resolution tiling.** SmolVLM's 4×4+global = 1377 tokens **[V]** is fine on a GPU and
   fatal on a phone. Desktop-only feature.
3. **Cluster-scale data mixes.** FineVision at 24M samples, Cambrian-7M, LLaVA-OneVision-Data. Not
   because of FLOPs — because of Colab disk and I/O (§6-R1).
4. **Full-finetune SFT.** LLaVA-1.5's 80–160 A100-hours **[V]** for stage 2 *and* it destroys the
   text guarantee. Replaced by LoRA at a measured cost of ~1.5–3 benchmark points **[V]**.
5. **Audio-native fusion.** Qwen2.5-Omni's Thinker-Talker + TMRoPE + streaming DiT **[V]** is a
   multi-team effort.
6. **Beating Qwen2.5-VL-3B / InternVL3-2B on OCR/document benchmarks.** State this in the plan so
   nobody promises it later.

**The most important transferable insight, stated bluntly:** in TinyLLaVA's controlled sweep,
holding encoder and data fixed and swapping only the decoder moved MMMU from 24.0 to 39.9 **[V]**.
**The best thing we can do for Prophet-VL is to make Prophet better at text.** Vision is downstream
of the decoder's quality. That is a research finding, and it happens to also be our strategy.

---

## 4. Recommendation for Prophet

### 4.1 Phased plan

**v1 — Text-only, hooks installed. 0 A100-hours of vision compute.**
Build every hook in §4.3. Prove each is a no-op (§7 V0). Ship text.

**v1.5 — Proof of life on the mini. 6–12 A100-hours. Optional but strongly recommended.**
nanoVLM recipe on the 300–600M dense mini: `google/siglip-base-patch16-224` (frozen) + connector +
frozen mini. Success = **MMStar ≥ 35** (nanoVLM parity **[V]**). Purpose is *not* a good VLM; it is
to prove the hooks work end-to-end **before** the main model's weights are frozen and shipped. If a
hook is broken, this is where we find out, for 6 hours instead of a re-pretrain.

**v2 — The real VLM. ~26 A100-hours (hard cap 50).** Recipe in §4.2.

**v3+ — Not now.** Audio-native, video, high-res tiling on device, image generation, FastViTHD-class
encoder redesign, early fusion.

### 4.2 The cheapest credible VLM recipe (v2)

**Components**
- **Vision encoder:** `google/siglip2-base-patch16-512` (~86M, frozen throughout).
  512px / p16 → 32×32 = **1024 patches**.
- **Connector:** pixel-shuffle factor 4 (16× reduction) → **64 tokens**, then a 2-layer MLP
  1024·(768→d_model). ~10–25M params. Trained in both stages.
  *Rationale: exactly nanoVLM's current, debugged configuration* **[V]**.
- **Decoder:** Prophet main (1–1.5B active / 8–12B total MoE), **frozen base weights**, plus a
  vision LoRA (r=32, α=64) on attention q/k/v/o **and the shared/always-on expert MLP only** —
  never on routed experts (§4.3 H5c).
- **Attention over image spans:** PaliGemma-style prefix-LM — bidirectional within the image span
  and the prefix, causal over the answer. Enabled by hook H4.
- **Optional inference-time:** VisionZip on top (54 dominant + 10 contextual **[V]**) for a further
  free reduction on desktop when tiling is used.

**Stage A — connector alignment (~6 A100-hours)**
- Data: `liuhaotian/LLaVA-Pretrain` (558K) **[V]**.
- Trainable: **projector only**. Encoder frozen, decoder frozen, no LoRA.
- **Precompute all vision features to disk first** (one pass, 558K × 64 × 768 × 2 bytes ≈ **55 GB**),
  then stage A never touches the encoder again. This is the difference between a 6-hour run and a
  30-hour one on Colab.
- LR: high on the projector (nanoVLM uses **lr_mp = 5.12e-3** vs 5e-5 on backbones **[V]**).
- Text impact: **bit-identical** (no decoder weight is touched).

**Stage B — visual instruction tuning with modality LoRA (~20 A100-hours)**
- Data mix, ~1.2M samples:
  - `llava_v1_5_mix665k` — 665K **[V]** (general VQA + instruction following)
  - `HuggingFaceM4/Docmatix` — subsample **300K** **[V]** (documents/OCR; highest leverage)
  - `allenai/pixmo-docs` + `allenai/pixmo-cap` — subsample **~150K** **[V]** (charts/diagrams, dense captions)
  - `HuggingFaceM4/the_cauldron` OCR/chart subsets — **~100K** **[V]**
  - ShareGPT4V if available (worth ~+3 TextVQA **[V]**)
- Trainable: projector + vision LoRA. Base decoder weights **frozen**.
- Text impact with adapter off: **bit-identical, provably**.

**Total: ~26 A100-hours ≈ 5–9% of a 300–500 A100-hour budget.** Hard cap 50; if stage B exceeds it,
cut data, not the freeze.

**Honest expected result:** MMStar 40–45, MMMU 35–38, TextVQA 60–70, DocVQA 65–75, AI2D 60–65,
OCRBench 450–550. That is **SmolVLM-2.2B-class at ~64 tokens/image** — competitive with SmolVLM2
and Moondream, **below Qwen2.5-VL-3B and InternVL3-2B on OCR/document tasks**. Our claim should be
*efficiency parity at lower token cost, with zero text regression*, not benchmark supremacy.

### 4.3 ⭐ THE ARCHITECTURAL HOOKS SPEC — implementable requirements for v1

These are the requirements the **text** model must satisfy from day one. Each is specified with
(a) what to build, (b) the v1 cost, (c) why retrofitting is expensive.

---

#### **H1 — Reserved vocabulary block (256 IDs)**

**Build:** `vocab_size = text_vocab_size + 256`. The tokenizer must never emit these IDs. The LM
head must mask them to `-inf` at inference in v1.

Fixed allocation (freeze this table now; changing it later invalidates every checkpoint):
```
+0    <|img|>            image feature placeholder (projector writes here)
+1    <|img_start|>
+2    <|img_end|>
+3    <|global_img|>     the downscaled whole-image view in a tiled layout
+4..+67   <row_i_col_j>  i,j in 1..8  -> supports up to an 8x8 tile grid (64 IDs)
+68   <|audio|>
+69   <|audio_start|>
+70   <|audio_end|>
+71   <|video_frame|>
+72..+87  <|ts_0|>..<|ts_15|>   coarse timestamp buckets (TMRoPE-style, unused in v1/v2)
+88..+255  RESERVED — do not allocate
```
**Precedent [V]:** nanoVLM reserves `extra_token_amount = 66` (`<|image|>`, `<|global_image|>`, and
exactly the 64 `<row_i_col_j>` tokens for an 8×8 grid). SmolVLM: `image_token_id = 128257`.
Phi-4-MM: `image_token_id = 200010`, `audio_token_id = 200011`.

**v1 cost:** `256 × d_model` parameters. At d_model=2048 that is **0.52M params — 0.006% of an 8B
model.** Effectively free.

**Why now:** growing the vocabulary later means resizing the embedding matrix *and* (with tied
weights) the LM head, which changes the softmax normalizer over the whole vocabulary, invalidates
every quantized/exported artifact, and forces re-calibration of the sampler. Reserved rows also get
trained-toward-unused during pretraining, which is a *better* initialization than random.

---

#### **H2 — Modality-typed embeddings and a `modality_ids` tensor threaded through the forward pass**

**Build:**
```python
MODALITY_TEXT, MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO = 0, 1, 2, 3
self.modality_emb = nn.Embedding(8, d_model)   # 8 reserved types
nn.init.zeros_(self.modality_emb.weight)       # v1: exact no-op

def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
            modality_ids=None, attn_spec=None, adapter=None, ...):
    ...
    h = tok_emb + self.modality_emb(modality_ids)   # v1: adds exactly 0
```
`modality_ids` must be `[B, L]`, default all-zeros, and **must be threaded through every layer,
the KV-cache, the packing/collator, the export path, and the inference server**, even though it is
constant in v1.

**v1 cost:** `8 × d_model` params (~16K). Zero FLOPs meaningfully. The real cost is *plumbing
discipline*.

**Why now:** this is the hook whose retrofit cost is almost entirely non-parametric. Adding a
per-token side-channel later touches every layer signature, every fused kernel call, the KV-cache
layout, the packing code, the CoreML/GGUF export graphs, and the serving API. Threading a
zeros tensor through on day one costs nothing; threading it through a shipped, quantized,
exported, served stack costs weeks.

---

#### **H3 — M-RoPE-compatible position encoding + a long-context-ready RoPE base**

**Build:**

(a) **Partition head_dim into `[t, h, w]` sections** (e.g. head_dim 128 → t=64, h=32, w=32). In v1,
set `h_idx = w_idx = t_idx = position` for every token, which makes M-RoPE **numerically identical
to standard 1-D RoPE**. This is the Qwen2-VL trick and it costs nothing.

(b) **`position_ids` must have shape `[3, B, L]`** and be *computed and passed by the model*, never
implicitly `torch.arange` inside the attention kernel. In v1 it is `arange` broadcast to 3 sections.

(c) **RoPE base must be long-context-ready.** Pretrain at **≥8k context with theta ≥ 500,000**, or
explicitly train the final 5–10% of tokens at long context.

> **Precedent — this is a mistake SmolVLM had to pay for [V]:** *"SmolLM2's pre-training context
> window is insufficient for VLMs. Images are encoded into many tokens... we extended it to 16k
> tokens by increasing the RoPE base value from 10k to 273k"* — requiring a whole separate
> context-extension run (EasyContext) on a long/short data mixture, with an upsampled math set to
> repair a GSM8k regression it caused. **We can avoid that entire run by picking theta correctly
> once.**

(d) **Nothing may assume position ids are contiguous or increment by 1.** Image blocks consume a
*range*. Audit varlen FlashAttention packing paths, KV-cache index arithmetic, and any
sliding-window mask that derives from position deltas.

**v1 cost:** zero parameters, zero FLOPs (identical arithmetic).
**Why now:** (c) is the expensive one — it is a **pretraining decision**, and getting it wrong
costs a context-extension run plus the text regression that run introduces.
Reference: TMRoPE (Qwen2.5-Omni) extends exactly this scheme with a time axis for audio/video
alignment **[V]** — the `[t,h,w]` partition is forward-compatible with it.

---

#### **H4 — Attention masking as a first-class object, with a bidirectional-span option**

**Build:** the attention call site must accept an `attn_spec` describing (i) causality and (ii) a
list of `(start, end)` spans that attend **bidirectionally** within themselves and to everything
before them. v1 passes `spans=[]`, giving an exactly causal mask.

```python
# FORBIDDEN at the call site:
F.scaled_dot_product_attention(q, k, v, is_causal=True)
# REQUIRED:
F.scaled_dot_product_attention(q, k, v, attn_mask=self.build_mask(attn_spec))
# with a fast path: if attn_spec.is_pure_causal: use is_causal=True
```
Recommended implementation: PyTorch **FlexAttention** `BlockMask` with a `mask_mod`, which keeps
the fast path fast while making arbitrary span masks expressible.

**Why:** PaliGemma's prefix-LM mask (bidirectional over image + prefix, causal over the answer) is
a measurable win for VQA/OCR **[M]** and costs nothing to support if planned. Building the whole
model around a causal-only fused kernel makes it impossible.

**⚠️ Cross-track constraint — escalate this to the architecture owner.** If another track (attention
/ efficiency) proposes **Mamba/SSM layers, linear attention, or aggressive sliding-window
attention**, then: (i) bidirectional image spans become hard or impossible, (ii) long tiled-image
contexts interact badly with sliding windows, and (iii) the "image span" abstraction has no
meaning in a recurrent state. **A hybrid-attention decision and the vision roadmap are coupled and
must be decided together, not sequentially.** This is the highest-value thing R12 has to say to
the other tracks.

**v1 cost:** zero, given a fast path.

---

#### **H5 — LoRA / adapter mount points, declared and named**

**Build:**

**(a)** Every `nn.Linear` in attention (q, k, v, o) and in MLP/expert blocks is a wrapper class
supporting an optional additive low-rank delta:
```python
class AdaptableLinear(nn.Module):
    def forward(self, x, adapter=None):
        y = self.base(x)
        if adapter is not None:
            y = y + adapter.scale * adapter.B(adapter.A(x))
        return y
```
When `adapter is None`, this must be **bitwise identical** to `self.base(x)` — no reordering, no
extra cast. Assert this in CI.

**(b) Activation granularity: per-request, with per-token as an experiment.** The safe, proven
pattern (Phi-4-MM) is: *does this sequence contain modality m? → activate adapter m for the whole
sequence.* This is correct because the **text answer tokens must themselves be adapted** in order
to reason over image features. The guarantee we sell is therefore precisely:
> **text-only input → no adapter active → base weights → bit-identical output.**
Per-token gating on `modality_ids` (gather → apply → scatter-add) is a possible refinement; do not
make the v2 plan depend on it.

**(c) MoE placement rule: attach modality LoRA to attention projections and the shared/always-on
expert only. Never to routed experts.** A routed expert sees a small, router-dependent fraction of
tokens, so a LoRA on it trains on too little data and interacts unpredictably with router drift.
→ **Requirement on the MoE design: reserve at least 1–2 always-on shared experts per layer.**
This is where cross-modal capacity lives later. (2504.07951's finding that MoE experts specialize
by modality is the theoretical backing **[M]**; MolmoE-1B and Kimi-VL-A3B are the empirical
precedents **[V]**.)

**(d) Router forward-compatibility — choose one, now:**
- *Option 1 (cheap, recommended):* accept that adding modality-specialist experts later requires
  retraining the router, and don't add any. Modality capacity comes from LoRA + shared expert.
- *Option 2 (costs total-param budget):* reserve k experts per layer, initialized but masked out of
  the router softmax in v1 (`router_logits[:, reserved] = -inf`); unmask at v2 and train only those
  experts plus a router bias. Zero v1 FLOPs, but the reserved experts count against the 8–12B
  total-parameter ceiling.
**Recommendation: Option 1**, with the shared-expert reservation from (c) as the hedge.

**v1 cost:** zero params, one predictable branch per linear.
**Why now:** retrofitting `AdaptableLinear` into a shipped, exported, quantized model means
regenerating every artifact. And this hook is what makes the entire text-preservation guarantee
*structural* rather than empirical.

---

#### **H6 — `inputs_embeds` path and a tested splice API**

**Build:**
- `forward()` accepts `inputs_embeds` in place of `input_ids`.
- **No hidden coupling** that recovers token ids from embeddings — e.g. no loss-masking or
  attention logic that secretly re-reads `input_ids`. nanoVLM makes this an explicit config flag
  **[V]**: `lm_use_tokens: bool = False  # if using as a backbone for the VLM, set to False`.
- Ship and unit-test a splice utility:
  `replace_placeholders(embeds, input_ids, placeholder_id=IMG_ID, features) -> embeds`
  asserting `count(input_ids == IMG_ID) == features.shape[1]`.
- **The loss must support masking positions out of the LM objective** (image tokens have no target).
  The `labels = -100` convention is sufficient; just make sure it exists and is tested in v1.

**v1 cost:** zero.

---

#### **H7 — Published embedding/normalization contract for the connector**

**Build:** document and *freeze* (i) `d_model`, (ii) the exact norm applied to layer 0's input
(RMSNorm vs LayerNorm, eps, whether it is pre- or post-embedding), and (iii) any embedding scaling.
Add an explicit config field `embed_scale: float = 1.0` rather than an implicit
`* sqrt(d_model)` hidden in the embedding call.

**Why:** a statistical mismatch between projector outputs and the decoder's expected input
distribution is the number-one cause of connector-training divergence, and it is *silent* — the
loss just plateaus. A written contract plus an explicit scale field turns a multi-day debugging
session into a one-line fix.

**v1 cost:** zero (documentation + one config field).

---

#### **H8 — KV-cache and packing must tolerate long, image-heavy prefills**

**Build:**
- KV cache sized by a **runtime max**, not a compile-time text-only constant. Paged if possible.
- The packing/collator must express **"this image block belongs to document i"** so that
  document-boundary masking does not split an image across a boundary. nanoVLM's training config
  already carries the shape of this **[V]**: `max_images_per_example: 4`,
  `max_images_per_knapsack: 18`, `max_sample_length: 4096`.
- **Prefix caching should key on an image content hash**, so a repeated image in a multi-turn
  conversation does not re-prefill. (VisionZip is text-agnostic, so it composes with this **[V]**.)

**v1 cost:** near zero; mostly avoiding a hard-coded constant.

---

#### **H9 — Export-path separation (the on-device hook)**

**Build:**
- The vision encoder is a **separate, standalone module** with its own `forward` and its own export
  target. It is **never** fused into the LM graph. The interface between them is a plain
  `[B, N_tokens, d_model]` tensor.
- Rationale, verified from Apple's own FastVLM shipping code **[V]**: the ViT is exported with
  **coremltools** (→ ANE; compute-bound, wants fixed shapes and convs) while the LLM is converted
  with **MLX** at 4/8-bit (→ GPU/Metal; memory-bandwidth-bound). Two runtimes, two artifacts.
- **Connector op restrictions (ANE-friendliness):** use only Conv2d / Linear / GELU / LayerNorm /
  reshape / permute. Avoid dynamic shapes, avoid `einsum` with >4 dims, avoid odd channel counts.
  **Pixel-shuffle is a reshape+permute and is therefore free on ANE — this is an additional,
  independent reason to prefer it over a Perceiver resampler.**
- Note for the roadmap: a stock HF SigLIP will not achieve good ANE residency without attention
  reshaping to Apple's 4-D `(B, C, 1, S)` convention **[M]**. Budget real engineering, or accept
  Metal execution for the encoder in v2.

**v1 cost:** zero — it is a rule about module boundaries, not code.

---

#### **H10 — Config and checkpoint forward-compatibility**

**Build:** from v1, the config schema contains optional, default-`None` fields:
`vision_config`, `audio_config`, `modality_adapters`, `connector_config`; the config is
**versioned**; and the checkpoint loader **tolerates unknown keys** rather than raising.

**Precedent [V]:** `SmolVLMConfig` carries `sub_configs = {"text_config": ..., "vision_config": ...}`
with `vision_config: ... | None = None`; Phi-4-MM carries both `vision_config` and `audio_config`
as optional sub-configs. Both are exactly this pattern.

**v1 cost:** zero. **Why now:** without it, every v2 checkpoint breaks every v1 tool.

---

#### Hooks summary

| # | Hook | v1 param cost | v1 FLOP cost | Retrofit cost if skipped |
|---|---|---|---|---|
| H1 | 256 reserved vocab IDs | ~0.5M (0.006%) | 0 | Embedding+head resize; all exports invalid |
| H2 | Modality-typed embeddings + `modality_ids` plumbing | ~16K | 0 | Every layer signature, cache, export, server |
| H3 | M-RoPE `[t,h,w]` sections + theta ≥ 500k @ ≥8k ctx | 0 | 0 | **A full context-extension run** (SmolVLM paid this) |
| H4 | Mask as first-class object, bidirectional spans | 0 | 0 (fast path) | Cannot do prefix-LM; kernel rewrite |
| H5 | `AdaptableLinear` + shared-expert reservation | 0 | ~0 | Regenerate all artifacts; lose the text guarantee |
| H6 | `inputs_embeds` + splice API + loss masking | 0 | 0 | Deep refactor of the forward/loss coupling |
| H7 | Embedding/norm contract + `embed_scale` | 0 | 0 | Days of silent connector-divergence debugging |
| H8 | Runtime-sized KV cache, image-aware packing | 0 | 0 | Serving rewrite |
| H9 | Encoder/decoder export separation | 0 | 0 | On-device path blocked |
| H10 | Optional sub-configs, versioned, tolerant loader | 0 | 0 | v1 tooling breaks on every v2 checkpoint |
| | **TOTAL** | **~0.54M params (≈0.006%)** | **0** | **≥ one full pretraining run** |

---

## 5. Compute & memory budget

### 5.1 Training compute

**Stage A (connector alignment), FLOP estimate:**
558K samples × ~250 tokens ≈ **140M tokens**. Backward must still traverse the frozen decoder to
reach the projector (only the *optimizer state* is saved, not the backward pass), so use the full
6ND: `6 × 1.5e9 × 1.4e8 = 1.26e18 FLOPs`. At A100 bf16 ~312 TFLOP/s peak, 40% MFU ≈ 125 TFLOP/s →
**~2.8 GPU-hours of pure compute**. Vision encoder forward is precomputed once
(86M × 1024 tokens × 2 × 558K ≈ 1.0e17 → ~0.2 h). **Budget 6 A100-hours wall-clock** for loader
inefficiency and checkpointing.

**Stage B (visual instruction tuning with LoRA):**
1.2M samples × (64 image + ~200 text) ≈ **320M tokens**. `6 × 1.5e9 × 3.2e8 = 2.9e18 FLOPs` →
**~6.4 GPU-hours pure compute**. LoRA removes optimizer state but not the backward pass.
**Budget 20 A100-hours wall-clock.**

**v1.5 proof-of-life on the mini:** nanoVLM measured ~6 h on 1×H100 for 1.7M samples at 222M total
**[V]**. At ~0.5× A100/H100 throughput for this workload → **~12 A100-hours**; a 600M mini with 64
(not 196) tokens/image lands nearer **6–10**.

| Item | A100-hours |
|---|---|
| Hooks (v1) | **0** |
| v1.5 proof-of-life (mini) | 6–12 |
| Stage A (connector) | ~6 |
| Stage B (LoRA SFT) | ~20 |
| Ablation ladder on mini (§7 V-abl) | 12–18 |
| **Total** | **44–56** |
| **Hard cap** | **75** (= 15% of a 500 h budget) |

This fits the stated 10–15% envelope, with the ablations as the discretionary item to cut first.

### 5.2 Training memory (A100-80GB)

nanoVLM's measured VRAM curve for a 222M VLM on an 80GB card **[V]**:
`model load 871 MB; bs=1 → 4.45 GB; bs=8 → 5.37; bs=16 → 7.60; bs=32 → 12.07; bs=64 → 21.0;
bs=128 → 38.8; bs=256 → 74.6; bs=512 → OOM`.
For Prophet at 1.5B active with a frozen base + LoRA and gradient checkpointing, expect
**bs ≈ 8–16 at 2k sequence length**. Note MoE total parameters (8–12B) must be resident even though
only 1–1.5B are active: at bf16 that is **16–24 GB of weights alone**, so use 8-bit or 4-bit base
weights during the frozen stages (the base is frozen — there is no reason to hold it in bf16).
This is a genuine advantage of the freeze-the-base plan.

### 5.3 Inference memory / on-device

**Per-image KV-cache cost** — the whole argument, in one calculation. For a decoder with 24 layers,
4 KV heads, head_dim 128, bf16: `2 × 24 × 4 × 128 × 2 B = 12,288 B/token`.

| Config | Tokens/image | KV cache for 1 image |
|---|---|---|
| **Prophet target (512px, PS×4)** | **64** | **0.8 MB** |
| SmolVLM (384px) | 81 | 1.0 MB |
| LLaVA-1.5 (336px) | 576 | 7.1 MB |
| PaliGemma-896 | 4096 | 50 MB |
| SmolVLM 4×4 tiled + global | 1377 | 17 MB |
| Qwen2-VL (per SmolVLM's measurement, ~16k tokens) **[V]** | ~16,000 | **~197 MB** |

**Device tiering — this needs to be an explicit decision, and it is not the obvious one:**

| Target | Model | Weights | Encoder | Verdict |
|---|---|---|---|---|
| **iPhone 17 Pro (~8 GB, ANE)** | **mini 600M dense @ 4-bit** | ~340 MB | SigLIP2-base int8, ~86 MB | ✅ well under 1 GB total |
| RTX 5090 32 GB | main 8–12B MoE @ 4-bit | 4.5–6.5 GB | fp16, 172 MB | ✅ comfortable |
| Mac Studio | main @ 4–8 bit | 4.5–13 GB | fp16 | ✅ comfortable |

**The main 8–12B MoE does not belong on the iPhone.** Even at 4-bit it is 4.5–6.5 GB, and iOS
realistically grants an app well under the nominal 8 GB. **Therefore the *mini* is the on-device
vision target, and the hooks in §4.3 must be implemented in the mini as well as the main model** —
a point that is easy to miss and expensive to discover late.

**Prefill cost on-device:** for the 600M mini, 64 image tokens ≈ `2 × 0.6e9 × 64 = 7.7e10 FLOPs` —
trivial. A 4×4 tiled image at 1377 tokens ≈ `1.7e12 FLOPs` ≈ **~1 s** on a phone-class GPU at
~2 TFLOP/s effective. **Conclusion: tiling, not the single image, is what breaks the phone.
On-device default = one global 512px view = 64 tokens. Tiling is a desktop-only feature.**
This matches Apple's finding that at high resolution vision encoding dominates TTFT **[V]**.

---

## 6. Risks

**R1 — Data I/O on Colab is the binding constraint, not FLOPs. [HIGHEST]**
The Cauldron and FineVision are multi-TB with per-sample JPEG decode. Colab A100 instances have
limited local disk, weak CPUs, and rate-limited network. A nominally 6-GPU-hour run becomes 30
GPU-idle hours if the loader starves.
*Mitigations:* (i) **precompute vision features once** into packed shards (55 GB for stage A) and
never run the encoder during training; (ii) prefer `liuhaotian/LLaVA-Pretrain` (558K, small on
disk) over FineVision; (iii) use WebDataset/tar shards, not per-file reads; (iv) measure loader
throughput *before* committing GPU hours, and treat GPU utilization < 70% as a stop-the-line defect.

**R2 — Colab session limits and preemption.**
Sessions cap at 12–24 h and can be preempted. Stage B at ~20 h needs 2–3 sessions.
*Mitigation:* checkpoint every ~15 minutes (nanoVLM checkpoints every 250 steps **[V]**; SmolVLM
saved every **25** optimization steps **[V]**); every stage must be resumable from an arbitrary
step including optimizer and dataloader state.

**R3 — The multimodal tax on text.**
Full-finetune multimodal SFT measurably degrades text.
*Mitigation:* the frozen-base + LoRA design makes adapter-off text **bit-identical by
construction**. But **verify, do not assume** — §7 V2 gates the release on it. Note also the
*measured* price of LoRA over full FT: ~1.5–3 points on vision benchmarks **[V]**. Accept it
explicitly.

**R4 — Context length insufficiency. [SPECIFIC AND AVOIDABLE]**
If v1 pretrains at 4k with theta = 10k, then a tiled document (8×8 tiles ≈ 5184 tokens) does not
fit and a context-extension run becomes mandatory. **SmolVLM paid exactly this cost** (10k → 273k
RoPE base, a separate EasyContext run, plus an upsampled math set to repair the GSM8k regression
it introduced) **[V]**.
*Mitigation:* hook H3(c) — pretrain at ≥8k with theta ≥ 500k. This is a **decision to make before
the pretraining run starts**, and it is the single most time-sensitive item in this report.

**R5 — Hybrid attention forecloses vision. [CROSS-TRACK]**
If another track selects Mamba/SSM, linear attention, or aggressive sliding-window attention, then
bidirectional image spans (H4) become hard or impossible, and long tiled contexts interact badly
with sliding windows.
*Mitigation:* escalate now; the attention decision and the vision roadmap must be made jointly.

**R6 — MoE × LoRA interaction.**
LoRA on routed experts trains on too few tokens; and the router, having never seen image tokens,
may route them arbitrarily.
*Mitigation:* H5(c) — LoRA on attention + shared expert only; reserve 1–2 always-on shared experts.
Optionally train a small router bias at v2. Measure router entropy on image tokens as a diagnostic.

**R7 — ANE support for ViT ops.**
A stock HF SigLIP will not achieve good ANE residency without attention reshaping to Apple's 4-D
convention **[M]**. FastVLM ships coremltools export precisely because this needs bespoke work **[V]**.
*Mitigation:* H9 op restrictions; measure ANE residency % explicitly; accept Metal for v2 and treat
ANE as a v3 optimization rather than a v2 requirement.

**R8 — Benchmark expectation management.**
We will lose to Qwen2.5-VL-3B and InternVL3-2B on OCR/document benchmarks.
*Mitigation:* pre-commit the success criterion now — *parity with SmolVLM-2.2B / Moondream at
lower token cost, with provably zero text regression* — and write it into the plan before anyone
promises otherwise.

**R9 — Data licensing and link rot.**
The Cauldron aggregates 50 datasets with heterogeneous licences **[V]**. PixMo distributes **image
URLs**, not images **[V]** — a growing fraction are dead. FastVLM's models carry a separate,
restrictive `LICENSE_MODEL` **[V]**.
*Mitigation:* licence audit before any release; assume a materially reduced PixMo yield; do not
build on FastVLM weights without a legal read.

**R10 — Scope creep into audio/video/generation.**
Every one of these is individually seductive and collectively fatal to the budget.
*Mitigation:* the hooks reserve the IDs so the *option* stays open at zero cost. Ship nothing.

---

## 7. Validation plan

### V0 — Hooks are provable no-ops (v1, **0 GPU-hours**, blocking merge)
Automated tests, run in CI on every commit:
1. **Bit-identity:** hooked model vs. reference implementation, logits identical (`torch.equal`) on
   ≥10k text tokens across ≥8 batch shapes.
2. **M-RoPE degeneracy:** with `t_idx = h_idx = w_idx`, M-RoPE output matches 1-D RoPE to < 1e-6.
3. **Mask degeneracy:** `attn_spec(spans=[])` produces a mask bitwise equal to the causal mask; the
   fast path is taken.
4. **Adapter degeneracy:** `AdaptableLinear(x, adapter=None)` is bitwise equal to `base(x)`.
5. **Vocab safety:** reserved IDs are unreachable by the tokenizer and masked to `-inf` by the head.
6. **Plumbing:** `modality_ids=None` and `modality_ids=zeros` produce identical output; a non-zero
   `modality_ids` changes it (proving the wire is actually connected, not silently dropped).
7. **`inputs_embeds` path:** `forward(input_ids=x)` == `forward(inputs_embeds=embed(x))`.
8. **Config round-trip:** a config carrying `vision_config`/`audio_config` loads in v1 tooling.
9. **Position discontinuity:** a sequence with a deliberate position gap runs without error and
   matches a hand-computed reference.

*If any of these fails, vision is not deferred — it is foreclosed. This is the highest-value,
lowest-cost work in the whole track.*

### V-abl — Ablation ladder on the mini (12–18 A100-h, before committing main-model compute)
Each ~3–6 h, on the 300–600M mini, scored on MMStar + DocVQA-val:
- **Pixel-shuffle factor** {2, 4, 9} → {256, 64, ~28} tokens. *Find the knee.*
- **Encoder** {SigLIP2-B/16-512, SigLIP2-SO400M-384, SigLIP2-NaFlex@256, AIMv2-L-distilled}.
- **Connector** {2-layer MLP, Perceiver-64}. *Expect MLP to win on cost; confirm.*
- **Mask** {pure causal, prefix-LM over the image span}. *Isolates the value of H4.*
- **Adapter** {projector-only, +LoRA r=16, +LoRA r=32, full FT}. *Prices the text guarantee
  directly — TinyLLaVA's Phi-2 numbers predict ~1.5–3 points **[V]**; confirm on our stack.*

### V1 — Proof of life (v1.5, 6–12 A100-h)
**Gate: MMStar ≥ 35** on the mini, matching nanoVLM-222M's 35.3 **[V]**. Purpose: prove the hooks
work end-to-end before the main model ships.

### V2 — Text non-regression (**hard release gate**)
- **Adapter OFF:** full text suite (MMLU, GSM8K, MATH, HumanEval, HellaSwag, ARC, WinoGrande) must
  be **bit-identical** to the v1 text model — not "within noise," *identical*, because the weights
  are frozen. Any deviation is a bug in H5, not a training outcome.
- **Adapter ON, text-only input:** within **0.5 points** on every benchmark.
- **Long-context:** verify no regression at 8k after any RoPE work (SmolVLM's GSM8k regression
  during context extension **[V]** is the specific failure mode to watch for).

### V3 — Vision quality (v2)
Via `lmms-eval`, using nanoVLM's own task string **[V]**:
`mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa`
**Targets (stated as parity, not supremacy):** MMStar ≥ 42 (SmolVLM 42.1 **[V]**), MMMU ≥ 36,
DocVQA ≥ 70, TextVQA ≥ 65, at **≤ 96 tokens/image** — i.e. SmolVLM-class quality at ~1.25× fewer
tokens. Explicitly **not** targeting Qwen2.5-VL-3B on OCRBench.

### V4 — Efficiency (v2)
- **Tokens per image ≤ 96** for a single 512px view (hard constraint).
- Prefill TTFT and generation throughput on RTX 5090 and Mac Studio, vs. SmolVLM-500M as the
  reference efficiency baseline. SmolVLM's own gain over Qwen2-VL was 3.3–4.5× prefill /
  7.5–16× generation **[V]**; we should be at least at SmolVLM's level.
- Peak RSS with 1 and 2 images (the SmolVLM memory table **[V]** is the comparison; the 2-image
  delta is the discriminating measurement).

### V5 — On-device (v2/v3)
- CoreML export of the **encoder alone** (H9); report **ANE residency %** and encoder latency.
- MLX and GGUF export of the decoder at 4-bit and 8-bit.
- **End-to-end TTFT < 1.5 s** for a 512px single-view image on the mini on the target iPhone.
- Confirm the FastVLM finding on our stack: measure encoder latency vs. LLM prefill separately, so
  we know which half to optimize.

---

## 8. References

> **Verification status.** **[V]** = primary source fetched and read in this session (repo README,
> HF blog markdown, or `transformers` source/config). **[M]** = from model knowledge; arXiv was
> unreachable from this session, so the ID and the figures should be re-checked before use.

**Small / cheap VLMs**
- SmolVLM: *Redefining small and efficient multimodal models*, **arXiv 2504.05299** **[V id]**;
  Marafioti et al. 2025. Architecture, 81 tokens/image, benchmark + memory table, RoPE 10k→273k
  context extension: all **[V]** from the HF release post and `huggingface/smollm`.
- nanoVLM — `github.com/huggingface/nanoVLM` **[V]**. 222M = SigLIP-B/16-224 (85M) + SmolLM2-135M;
  ~6 h on 1×H100 on 1.7M Cauldron samples → MMStar 35.3; VRAM table; current config
  (SigLIP2-B/16-512, pixel-shuffle 4, 64 tokens, 66 reserved tokens, FineVision).
- LLaVA / LLaVA-1.5 — **arXiv 2304.08485** **[M]** / **2310.03744** **[V id]**;
  `github.com/haotian-liu/LLaVA` **[V]** for all training-time numbers (3.5 h / 5.5 h / 10 h / 20 h
  on 8×A100; 558K + 665K; LLaVA-Lightning $40/3 h).
- TinyLLaVA Factory — `github.com/TinyLLaVA/TinyLLaVA_Factory` **[V]**. Full backbone sweep table
  including the base-vs-LoRA-vs-share comparison.
- Moondream — `github.com/vikhyat/moondream` **[V]** (2B and 0.5B distillation target).
- Idefics3 — **arXiv 2408.12637** **[M]**. Pixel-shuffle 4×, 364px patches: **[V]** via SmolVLM.
- PaliGemma — **arXiv 2407.07726** **[M]**. Prefix-LM mask; 224/448/896 → 256/1024/4096 tokens **[M]**.
- Florence-2 — **arXiv 2311.06242** **[M]**. Docmatix fine-tuning gave +20% DocVQA **[V]**.
- MM1.5 — **arXiv 2409.20566** **[M]**.
- Molmo / PixMo — **arXiv 2409.17146** **[M]**; `github.com/allenai/molmo` **[V]** for MolmoE-1B
  (OLMoE-1B-7B MoE backbone) and the full PixMo dataset list.
- Kimi-VL-A3B — **arXiv 2504.07491** **[M]**; MoonViT + 16B-total/2.8B-active MoE **[V]**.
- InternVL3 — **arXiv 2504.10479** **[M]**. Qwen2.5-VL — **arXiv 2502.13923** **[M]**.

**Vision encoders**
- SigLIP — **arXiv 2303.15343** **[M]**. SigLIP2 — **arXiv 2502.14786** **[V id]**; NaFlex variant,
  `max_num_patches` default 256 **[V]**.
- AIMv2 — **arXiv 2411.14402** **[V id]**; `github.com/apple/ml-aim` **[V]**: beats CLIP and SigLIP
  on most multimodal benchmarks; AIMv2-3B 89.5% IN-1k frozen trunk; distilled ViT-L "recommended
  for multimodal applications". AIMv1 — **arXiv 2401.08541** **[V id]**.
- DINOv2 — **arXiv 2304.07193** **[M]**. DINOv3 — **arXiv 2508.10104** **[M]**;
  `github.com/facebookresearch/dinov3` **[V]**.

**Token compression**
- Pixel shuffle (ESPCN, sub-pixel conv) — **arXiv 1609.05158** **[V id]** (cited by nanoVLM).
- Perceiver resampler / Flamingo — **arXiv 2204.14198** **[M]**. Q-Former / BLIP-2 — **2301.12597** **[M]**.
- ToMe — **arXiv 2210.09461** **[M]**.
- FastV, *An Image is Worth 1/2 Tokens After Layer 2* — **arXiv 2403.06764** **[M id]**;
  `github.com/pkunlp-icler/FastV` **[V]**: 45% FLOPs reduction; K/R latency-memory table.
- VisionZip — **arXiv 2412.04467** **[V id]**; `github.com/dvlab-research/VisionZip` **[V]**:
  10% of tokens → ~95% of performance training-free; default 54 dominant + 10 contextual.
- FastVLM — **arXiv 2412.13303** **[V id]**, CVPR 2025; `github.com/apple/ml-fastvlm` **[V]**:
  FastViTHD; 85× TTFT vs LLaVA-OneVision-0.5B with a 3.4× smaller encoder; 7.9× vs Cambrian-1-8B;
  CoreML encoder export + MLX 4/8-bit LLM; iOS 18.2+/macOS 15.2+ app reporting TTFT.

**Early / native fusion**
- Chameleon — **arXiv 2405.09818** **[M]**; `github.com/facebookresearch/chameleon` **[V]**.
- Fuyu-8B — Adept blog, **no arXiv paper** **[M]**.
- Transfusion — **arXiv 2408.11039** **[M]**.
- Emu3 — **arXiv 2409.18869** **[V id]**; `github.com/baaivision/Emu3` **[V]** ("single transformer
  from scratch", `BAAI/Emu3-VisionTokenizer`).
- Janus — **arXiv 2410.13848** **[V id]**; JanusFlow — **2411.07975** **[V id]**;
  `github.com/deepseek-ai/Janus` **[V]** (576 image tokens).
- **Scaling Laws for Native Multimodal Models — arXiv 2504.07951** **[M]**. Early fusion is
  compute-competitive with late fusion; sparse MoE lets NMMs learn modality-specialized weights.
  *Read as a statement about from-scratch runs — see §2.7.*
- MoE-LLaVA — **arXiv 2401.15947** **[V id]**.

**Audio**
- Whisper — **arXiv 2212.04356** **[M]**. Distil-Whisper — **arXiv 2311.00430** **[M]**;
  `github.com/huggingface/distil-whisper` **[V]** for the full params/WER/latency table.
- Moonshine — **arXiv 2410.15608** **[M]**; `github.com/moonshine-ai/moonshine` **[V]**;
  Moonshine Micro (470 KB RAM on an RP2350) **[V]**.
- **Phi-4-multimodal — arXiv 2503.01743** **[V id]**; `transformers` docs + config **[V]**:
  mixture of LoRA adapters, `vision-lora`/`speech-lora`, `image_token_id 200010`,
  `audio_token_id 200011`, audio `input_size 80` / `time_reduction 8` → 12.5 tokens/s.
- **Qwen2.5-Omni — arXiv 2503.20215** **[M id]**; `transformers` docs **[V]** for the abstract:
  Thinker-Talker, **TMRoPE**, block-wise streaming encoders, sliding-window DiT.
- Qwen2-VL (M-RoPE) — **arXiv 2409.12191** **[M]**.
- Kyutai Moshi — `github.com/kyutai-labs/moshi` **[V]**.

**Data**
- The Cauldron — `HuggingFaceM4/the_cauldron` **[V]** (50 datasets).
- Docmatix — `HuggingFaceM4/Docmatix` **[V]** (2.4M images / 9.5M QA / 1.3M PDFs; from
  `pixparse/pdfa-eng-wds`; +20% DocVQA on Florence-2).
- FineVision — `HuggingFaceM4/FineVision` **[M]**; nanoVLM default
  `HuggingFaceM4/FineVision_concat_shuffled_2` with quality-rating filters **[V]**.
- LLaVA-Pretrain (558K) / LLaVA-Instruct-150K → `llava_v1_5_mix665k` — `liuhaotian/*` **[V]**.
- PixMo family — `allenai/pixmo-{cap,ask-model-anything,cap-qa,points,point-explanations,docs,clocks,count}` **[V]**.
- LLaVA-OneVision-Data — `lmms-lab/LLaVA-OneVision-Data`; **arXiv 2408.03326** **[M]**.
- Cambrian-1 / Cambrian-10M — `nyu-visionx/Cambrian-10M`; **arXiv 2406.16860** **[M]**.
- RLAIF-V (DPO) — `HuggingFaceH4/rlaif-v_formatted` **[V]**.

**Other**
- Scaling Laws of RoPE-based Extrapolation — **arXiv 2310.05209** **[V id]** (the basis for
  SmolVLM's 10k→273k RoPE base change).
- Apollo (video data mixtures, informed SmolVLM2's image/video balance) — **arXiv 2412.10360** **[M]**.
- `lmms-eval` — `github.com/EvolvingLMMs-Lab/lmms-eval` **[V]**. VLMEvalKit —
  `github.com/open-compass/VLMEvalKit` **[V]**.
- *Vision Language Models (2025 update)* — `huggingface/blog/vlms-2025.md` **[V]** (MoE-decoder VLM
  landscape, any-to-any models, smol-model survey).
