# Study Plan & Daily Requirements

*What to read, understand, calculate, and ship each day. Full links for each
day. Days 6-12 appended as we reach them.*

---

## Day 1 — Product + Data Contract (done 2026-08-21, tag v0.0.2)

**Study (~2h):**
- ML systems design — Huyen, ch.1-2 (ML systems like software engineering)
- YC Requests for Startups (Fall 2026): https://www.ycombinator.com/rfs
- Pydantic v2 basics: https://docs.pydantic.dev/latest/
- ADR format (architecture decision records)

**Build:**
- Product contract (≤25M params, ≤8MB 4-bit, <100ms CPU) + acceptance criteria
- Pydantic `DatasetExample` schema + `forger-data-validate` CLI
- TinyStories sample with sha256 checksum
- ruff + pytest + GitHub Actions CI

**Ship:** tests (hypothesis round-trip), `benchmarks/data_contract.md`, ADR-01, tag v0.0.2.

## Day 2 — BPE Tokenizer (done 2026-08-22, tag v0.0.3)

**Read:**
- Sennrich et al. 2016, sections 2–3: https://arxiv.org/abs/1508.07909
- Karpathy `minbpe` source: https://github.com/karpathy/minbpe
- OpenAI GPT-2 tokenizer: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- HF BPE docs: https://huggingface.co/docs/tokenizers/en/quicktour
- `regex` package (GPT-2 Unicode patterns): https://pypi.org/project/regex/

**Understand:** token/symbol/pair · pair frequency counting · most-frequent-pair
merging · deterministic merge order · why byte-level avoids unknown characters ·
UTF-8 · pretokenization vs BPE merging · train-frequency vs encode-rank ·
special tokens · decode = concatenate bytes THEN decode once · vocab =
bytes + merges · save/reload vocab + merges.

**Explain:** `"low low lower"` → pretokenized pieces → UTF-8 bytes → byte IDs →
frequent pair merges → final token IDs.

**Ship:** tests (17), parity 10/10 vs HF, `benchmarks/tokenizer_parity.md`, ADR-02, tag v0.0.3.

## Day 3 — GPT Model Core (done 2026-08-22, tag v0.0.4)

**Read:**
- Karpathy nanoGPT model.py: https://github.com/karpathy/nanoGPT/blob/master/model.py
- Vaswani 2017 §3: https://arxiv.org/abs/1706.03762
- nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- sdpa: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- RMSNorm: https://arxiv.org/abs/1910.07467
- RoPE: https://arxiv.org/abs/2104.09864
- GLU/SwiGLU: https://arxiv.org/abs/2002.05202
- KV-cache: https://huggingface.co/docs/transformers/cache_explanation

**Understand:** embedding lookup · QKV projections · scaled dot-product
attention · causal masking · multi-head shapes · d_model % n_heads == 0 ·
LayerNorm vs RMSNorm · absolute vs RoPE · RoPE rotation · SwiGLU gate/up/down ·
residuals · pre-norm blocks · logits projection · cross-entropy shapes · tied
embedding/head · KV-cache speedup · training forward vs one-token decode.

**Calculate:** d=256, heads=4 → head_dim=64; [B,T] → [B,T,256] → [B,T,4096].

**Hand-trace:** Q@Kᵀ → /√head_dim → causal mask → softmax → ×V → concat heads → output proj.

**Do NOT spend time on:** distributed training, FlashAttention internals, full
KV-cache, MoE, long-context scaling, GPU kernels.

**Ship:** 10 tests, `benchmarks/model_forward.md`, ADR-03, tag v0.0.4.

## Day 4 — Training Pipeline (done 2026-08-22, tag v0.0.5)

**Read:**
- AdamW: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- AMP: https://pytorch.org/docs/stable/amp.html
- Cosine schedule + warmup (HF): https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules
- Gradient accumulation: https://huggingface.co/docs/accelerate/concept_guides/gradient_accumulation
- Hydra: https://hydra.cc/docs/1.3/intro/
- HF datasets streaming: https://huggingface.co/docs/datasets/stream

**Understand:** cross-entropy [B·T,V] vs [B·T] · contiguous windows with
shift-by-1 labels · seeds/shuffle = reproducibility · warmup + cosine decay ·
bs_eff = bs × accum · fp16 needs GradScaler, CPU uses fp32 (bf16 measured 7×
slower here!) · checkpoint = model+optimizer+scheduler+step · eval loop ·
streaming avoids RAM blowup.

**Calculate:** 200K chars ≈ 50K tokens · bs 8 × 512 = 4096 tok/step ·
ln(4096) ≈ 8.32 initial loss · CPU step ≈ 3× forward.

**Ship:** 6 tests (incl. bit-exact resume), `benchmarks/train_smoke.md`, ADR-04, tag v0.0.5.

## Day 5 — Baseline Training (done 2026-08-22, tag v0.1.0)

**Read:**
- Kaggle notebooks + GPU (T4/P100): https://www.kaggle.com/docs/notebooks
- HF Hub upload: https://huggingface.co/docs/hub/en/upload
- TinyStories card: https://huggingface.co/datasets/roneneldan/TinyStories

**Understand:** CPU→GPU derivability (device flag only) · T4 16GB fits bs 64
@ ctx 128 for 5.25M params · checkpoint push/pull via HF Hub · evidence
capture (wall-time, curves) · **memorization**: unique data = stories × tokens,
not windows × tokens; more stories beats more steps.

**Gotchas learned (all in development-log.md):** `sys.path` for scripts/ ·
stale clones on Kaggle · `model.to(device)` · `get_secret()` doesn't export env
· deleting the kernel's cwd · window-cap `min(windows, len//ctx)` · ctx 256
discards 91% of TinyStories · HF repo must be under YOUR username.

**Ship:** `benchmarks/baseline_train.md` (eval 2.0247), ADR-05, scripts
(kaggle_baseline, smoke_infer, download_from_hub), tag v0.1.0.

## Day 6 — Evaluation Harness (done 2026-08-22, tag v0.2.0)

**Read:**
- Eleuther lm-evaluation-harness structure: https://github.com/EleutherAI/lm-evaluation-harness
- Perplexity math: https://huggingface.co/docs/transformers/perplexity
- Distinct-n / diversity metrics: https://aclanthology.org/P16-1162/ (Li et al. 2016)

**Understand:** sliding-window ppl (stride < ctx, count fresh tokens only) ·
bits-per-byte (NLL / ln2 × bytes) · why eval data must be PINNED (checksummed),
not re-streamed · reproducibility block in reports · top-k/temperature sampling
semantics · distinct-n as diversity, repetition rate as degeneracy.

**Ship:** ppl+bpb on pinned 500-story corpus (8.449 / 1.0366), 5-prompt
generation table + metrics, ADR-06, tag v0.2.0.

## Day 7 — Inference Engine (done 2026-08-22, tag v0.2.1)

**Read:**
- FlashAttention recap (Milakov/Gimelshein): https://arxiv.org/abs/2205.14135
- torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
- KV-cache: https://pharath.github.io/posts/nanoGPT/
- Dynamic int8: https://pytorch.org/docs/stable/quantization.html

**Understand:** prefill vs decode phases · why cache makes decode O(1)-ish per
token · torch.compile needs a C++ compiler on Windows (Inductor) · dynamic int8
shrinks weights 4x but kernel overhead can beat the gain on small matrices ·
measure protocol: fixed prompt, first-token vs per-token vs prefill tok/s.

**Ship:** Engine (≡ reference greedy, context-limit safe), int8 4MB variant,
`benchmarks/serve_speedup.md` (fp32 123 tok/s, first-token 10ms), ADR-07,
tag v0.2.1.

---

## Upcoming (append as we go)
- Day 6 — Evaluation harness: lm-evaluation-harness structure, perplexity math, distinct-n
- Day 7 — Inference engine: FlashAttention recap, torch.compile, KV-cache, dynamic INT8
- Day 8 — LoRA SFT: Hu et al. 2021 §3-4, peft LoraConfig
- Day 9 — QLoRA: Dettmers et al. 2023 §2-3 (NF4, double quant)
- Day 10 — Optimized serving: ONNX export + ORT quantization
- Day 11 — API + deploy: FastAPI, Docker multi-stage, OWASP API top-10
- Day 12 — Security/release: HF ModelCard, SemVer, SRE/SLOs