# ADR-03: Model Core

Status: accepted
Date: 2026-08-29

## Context
Day 3 requires the GPT core: a small causal LM to be trained from Day 4 onward.

## Decision
- Architecture: decoder-only GPT, pre-norm blocks, tied embedding/LM head.
- Config: vocab 4096 (tokenizer artifact), d_model 256, n_heads 4 (head_dim 64),
  n_layers 4, context 512, ffn_mult 4. Actual params: **5,249,280** (~5.25M;
  plan estimated ~3.6M - SwiGLU's third projection + full embedding account for
  the delta; still well under the 25M product cap).
- Normalization: RMSNorm (cheaper than LayerNorm; standard in modern LMs).
- Positional encoding: RoPE (rotary), applied inside attention to q/k.
- Activation: SwiGLU (gate*up, SiLU) in the MLP; 3 linear projections.
- Attention: `F.scaled_dot_product_attention` (flash/mem-efficient kernels on
  supported backends, causal masking built-in).
- KV-cache stub: attention accepts optional (k,v) cache; chunked decode proven
  identical to full forward. Two subtle bugs fixed during validation:
  1. RoPE must use global position offsets when a cache is present.
  2. With T_q != T_k, is_causal cannot be used; explicit float(-inf) mask is
     required (bool masks in sdpa mean "True = attend", an easy trap).
- Weight init: N(0, 0.02), c_proj/down scaled by 1/sqrt(2*n) (GPT-2 style).

## Consequences
Correct, deterministic, cache-consistent model; 10 model tests green.
Forward: ~211ms for B=2/T=512 on CPU (4,855 tok/s), evidence in
`benchmarks/model_forward.md`. Model is CPU-trainable at reduced scale (Day 4).