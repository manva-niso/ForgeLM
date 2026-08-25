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