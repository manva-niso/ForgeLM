# Benchmark: LoRA SFT — Day 8 (2026-08-22)

## Local CPU evidence run
| Item | Value |
|---|---|
| Base | `models/forgelm-baseline` (best eval 2.0247) |
| Data | 2,000 databricks-dolly-15k rows (Apache-2.0), instruction format |
| LoRA | r=8, alpha=16, on c_attn/c_proj/gate/up/down (20 modules) |
| Trainable params | ~172,032 (3.3% of 5.25M) |
| Steps | 120, bs 8, ctx 128, lr 1e-3, CPU |
| Final loss | 5.0758 (best eval 5.2064) |
| Wall time | 75s |
| Merged model | `checkpoints/sft/merged/` (folded W + BA*scale, converted to plain Linear) |

## Qualitative (local 120-step run — proof-of-life, NOT final)
Prompt: `### Instruction: Write a story about a cat\n### Response:`
- Baseline (zero-shot): still writes story-style text but ignores instruction
  format; e.g. "Once upon a time there was a little cat named Tom..."
- After 120-step SFT: follows the ### Response: marker but garbled text
  ("That is a perfect with the danger quike dogies...") — expected at 120 steps.

## Kaggle upgrade (pending, ~30-45 min)
- `scripts/kaggle_sft.py` — 15K dolly rows, ~2000-4000 steps T4, saves adapter
  + merged model, pushes to HF `Manvaniso/forgelm-sft`. Run via the Kaggle
  notebook (kaggle branch). Quality table updated after the run.