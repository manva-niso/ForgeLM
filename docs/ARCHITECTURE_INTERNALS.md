# Architecture Internals & Hyperparameter Ledger

*Everything inside the model, and every knob we changed, with the exact values
that shipped. Companion to docs/JOURNEY.md (the story) and
docs/architecture.md (decisions).*

---

## 1. Tokenizer Internals (`forger/tokenizer/bpe.py`)

| Component | Value |
|---|---|
| Algorithm | byte-level BPE (GPT-2 style) |
| Base byte tokens | 256 (IDs 1..256 = bytes 0..255) |
| Special token | `<|endoftext|>` = ID 0 |
| Target vocab | 4096 → merge tokens = 4096 - 256 - 1 = **3839** |
| Pre-tokenization | GPT-2 regex (`regex` pkg): `'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` |
| Byte mapping | GPT-2 `bytes_to_unicode` (bytes → displayable chars, for vocab.json + HF parity) |
| Tie-breaking | highest frequency, then lexicographically smallest pair |
| Training corpus cap | default `max_chars=200,000` (speed); CLI override |
| Encoding | single-pass rank scan + piece cache (100,000 entries) |
| Serialization | `vocab.json`, `merges.txt` (rank order), `config.json` |

## 2. Model Internals (`forger/model/`)

### GPTConfig (defaults)
| Field | Value |
|---|---|
| vocab_size | 4096 |
| d_model | 256 |
| n_heads | 4 (head_dim = 64) |
| n_layers | 4 |
| context_length | 512 (training used 128-256) |
| ffn_mult | 4 (MLP hidden = 1024) |
| Total params | **5,249,280** (~5.25M) |

Param budget: embedding/tied head 4096×256 = 1,048,576 + per layer
(c_attn 3×d²+3d, c_proj d²+d, gate+up 2×d×(4d), down 4d×d, 2×RMSNorm d) ≈
1,050,112 × 4 layers + ln_f 256.

### Blocks
- **RMSNorm**: weight only (no bias), eps 1e-5; `x·rms⁻¹` normalization.
- **RoPE (RotaryEmbedding)**: base 10,000, `dim/2` frequencies, duplicated
  across the pair dimension; precomputed cos/sin tables for `context_length`;
  `offset` parameter for KV-cache decoding (global position).
- **CausalSelfAttention**: `c_attn` Linear(d → 3d, bias) → split Q/K/V →
  reshape (B, H, T, head_dim) → RoPE → sdpa (causal) → `c_proj` Linear(d → d,
  bias). Cache: concatenated (k, v); explicit float(-inf) triu mask when
  T_q ≠ T_k (bool masks in sdpa mean True = attend — trap documented in ADR-03).
- **MLP (SwiGLU)**: `gate`, `up` (d → 4d, no bias), SiLU(gate) ⊙ up, `down`
  (4d → d, no bias).
- **Block**: pre-norm — `x + attn(ln_1(x))`, `x + mlp(ln_2(x))`.
- **GPT**: token_embedding **tied** to lm_head (same weight tensor, bias=False),
  `ln_f` final RMSNorm.
- **init_weights**: Linear/Embedding N(0, 0.02); residual projections
  (c_proj, down) scaled to `0.02/sqrt(2·n)` (GPT-2 style).

## 3. Training Pipeline Internals (`forger/train/`)

| Component | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight_decay 0.1 |
| LR schedule | linear warmup → cosine decay to 10% of peak |
| Loss | cross-entropy, logits [B·T, V] vs targets [B·T] |
| AMP | fp16 + GradScaler on CUDA; **plain fp32 on CPU** (bf16 measured 7× slower) |
| Batching | WindowDataset: contiguous 128-token windows, labels shifted by 1 |
| Resume | exact continuation (deterministic get_batch by step) |
| Checkpoints | `checkpoint.pt` (model+optimizer+step), `best_model.pt` (best eval), `model.pt`+`config.json` (for GPT.load), JSON logs |
| Extras | TensorBoard (`runs/`), eval-rise warning (3 rising evals), best-eval tracking |

## 4. Hyperparameter Change Ledger (chronological)

| # | When | Knob | From → To | Why |
|---|---|---|---|---|
| 1 | Day 2 | tokenizer `max_chars` | none → 200,000 | naive training loop timed out (15 min); 200K chars → 89s |
| 2 | Day 2 | `_bpe` algorithm | dict-rebuild loop → single-pass rank scan + piece cache | profiled 22.6M dict gets; **20× faster**, parity 10/10 unchanged |
| 3 | Day 3 | GPTConfig ctx | 512 (fixed default) | plan target; later reduced for data retention |
| 4 | Day 4 | AMP on CPU | bf16 → fp32 | measured **7× slower** (1.35s vs 0.19s forward) |
| 5 | Day 4 | smoke.yaml | bs 4, ctx 256, steps 10, warmup 2 | 8.1s smoke, loss 8.11→7.36 |
| 6 | Day 4 | baseline.yaml | bs 8, ctx 512, steps 500, warmup 20 | local CPU reference config |
| 7 | Day 5 | CPU baseline run | bs 8, ctx 256, steps 300 | 358s, loss 8.11→3.34 |
| 8 | Day 5 | Kaggle v1 | bs 64, ctx 256, steps 4000, warmup 100 | first T4 attempt → memorized (train→0.01, eval→7.7) |
| 9 | Day 5 | windows_per_story | 1 → 16/4 (train/eval) | memorization fix attempt 1 (partially wrong — see #11) |
| 10 | Day 5 | max-stories | 2000 → 20000 | encoder speedup made large corpora affordable (~8 min encode) |
| 11 | Day 5 | window count formula | `min(windows, len//ctx)` → unconditional | **cap bug**: crops silently = 1 for stories < 2×ctx |
| 12 | Day 5 | Kaggle ctx | 256 → **128** | **91% of TinyStories discarded** at 257+ tokens (avg story ~150) |
| 13 | Day 5 | best-eval safeguard | none → `best_model.pt` + `best_eval.json` | late-training overfit must never destroy the best model |
| 14 | Day 5 | CPU fallback intervals | fixed 250 → `steps//16` (eval), `steps//40` (log, warmup) | adaptive for short runs |
| 15 | Day 5 (final) | Kaggle final config | bs 64, **ctx 128**, steps 4000, warmup 100, eval_every 250, windows 16/4, seed 0 | **result: eval 2.0247, no memorization** |

### Final shipped hyperparameters (Kaggle baseline, `configs/train/kaggle.yaml`)
```
steps: 4000    batch_size: 64    context_length: 128    lr: 3e-4
weight_decay: 0.1    warmup_steps: 100    grad_accum: 1
eval_every: 250    eval_windows: 20    log_every: 50
device: cuda    seed: 0
windows_per_story_train: 16    windows_per_story_eval: 4
```
Result: **best eval 2.0247** (step 3250), final train ~2.0, coherent generations.

## 5. What Did NOT Change (deliberately kept)
- lr 3e-4, AdamW betas, weight decay 0.1, model dims, SwiGLU, RoPE, RMSNorm,
  tied head, vocab 4096 — stable across all runs (only data/ctx/windows/AMP moved).