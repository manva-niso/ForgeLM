# ForgeLM Baseline Weights

| Item | Value |
|---|---|
| File | `model.pt` (state dict, 24.3 MB) |
| Eval loss (best) | 2.0247 (step 3250 of 4000) |
| Config | vocab 4096, d 256, 4 heads, 4 layers, ctx 128, ffn 4 |
| Params | 5,249,280 |
| Trained on | 20,000 TinyStories stories (293,200 windows), Kaggle T4 |
| Tokenizer | `artifacts/tokenizer/` (in repo) |
| Source | Kaggle baseline run, best-eval checkpoint |

Load + generate:
```powershell
uv run python scripts/smoke_infer.py --ckpt models/forgelm-baseline --prompt "Once upon a time"
```

## Known limitations (baseline is a proof-of-life, not a final model)
- Eval loss 2.02 (~ppl 7.6); repetitive phrasing, short-range coherence only.
- Trained on only 20,000 of TinyStories' ~2.1M stories.

## Improvement path (planned after the project's remaining days)
- Train on full TinyStories (~2.1M stories) — encode now ~0.024s/story (20x speedup), so ~14h encode or a batched/parallel pass on GPU.
- More steps / longer cosine decay; optionally bigger d_model or more layers (product cap 25M params).
- Then fine-tune path already in the plan: LoRA SFT (Day 8) + QLoRA int4 (Day 9).