# ADR-08: LoRA SFT

Status: accepted
Date: 2026-08-28

## Context
Day 8: specialize the baseline with supervised fine-tuning while keeping the
"small, portable" constraint (full fine-tuning is wasteful at any scale).

## Decision
- Hand-rolled LoRA (`forger/ft/lora.py`): `LoRALinear` = frozen base +
  low-rank A (r x in, kaiming init) and B (out x r, zeros) with alpha/r
  scaling. `apply_lora` targets c_attn/c_proj/gate/up/down (NOT the tied
  embedding/lm_head). After LoRA, ALL other params freeze (incl. RMSNorms) -
  the classic LoRA freeze contract; Trainer optimizes only requires_grad
  params.
- `merge_lora` folds W + (alpha/r)*B@A into the base weight; `convert_merged`
  swaps LoRALinear back to plain Linear so the checkpoint loads as a normal GPT.
- SFT data: **story-domain** (TinyStories formatted
  "### Instruction: Write a story about {topic}\n### Response: {story}").
  Initial choice (databricks-dolly-15k, Apache-2.0) was REJECTED after
  evidence: see "Domain lesson" below. `forger/ft/story_sft_data.py` +
  `scripts/kaggle_sft.py` (streaming mode for GPU runs).

## Domain lesson (the important part)
Dolly SFT (15K rows, 3000 T4 steps) produced a model with ppl 38 on stories
and ppl 68 on held-out dolly, generating confident garbage. Root cause was
NOT the pipeline (merge verified exact, zero-LoRA == baseline, deltas small):
a 5.25M model trained only on TinyStories has no general knowledge, and dolly
asks general-knowledge questions. SFT teaches response FORM, not FACTS.
Fixing the data domain (story instructions over TinyStories content) made the
same pipeline produce coherent instruction-following (loss 1.96).
- `forger/ft/train_sft.py`: loads baseline checkpoint, applies LoRA, trains
  (device flag - CPU local, CUDA Kaggle), saves Trainer checkpoint + merged
  model.

## Results
Local evidence run: 120 steps, 2K dolly rows, loss 5.08, 3.3% trainable
params, 75s. Quality at this step count is proof-of-life; the Kaggle run
(15K rows, thousands of steps) is the real deliverable.

## Consequences
Adapter is mergeable (model stays a plain GPT); LoRA trainable fraction
~3.3% of params; Dolly Apache-2.0 keeps licensing clean.