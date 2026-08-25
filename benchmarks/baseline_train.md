# Benchmark: Baseline Training - Day 5 (2026-08-22)

## Run 1 - local CPU (evidence, checkpoint produced)
| Item | Value |
|---|---|
| Model | GPTConfig(vocab 4096, d 256, 4 heads, 4 layers, ctx 256) - 5,249,280 params |
| Data | TinyStories sample 1000 stories (95/5 split), ~50K chars, tokenizer artifacts/tokenizer |
| Config | bs 8, ctx 256, lr 3e-4, AdamW, warmup 8, cosine, seed 0 |
| Device | CPU fp32 (torch 2.13.0, Windows) |
| Steps | 300 |
| Wall time | 358s (0.84 step/s) |
| Train loss | 8.11 -> 3.34 |
| Eval loss | 6.19 (step 25) -> 3.95 (step 300) |
| Checkpoint | checkpoints/baseline/ (checkpoint.pt + train_log.json + eval_log.json) |

Loss curves: `runs/baseline/` (TensorBoard) - monotonic decrease, eval tracks train,
still improving at step 300 -> more steps would help (Kaggle run).

## Run 2 - Kaggle T4 (attempt 1 FAILED: memorization)
- Symptoms: train loss -> 0.01 by step 800, eval loss rising 5.1 -> 7.7.
- Root cause: 2000 stories x 1 window = 2K unique windows cycled ~128x in 4000 steps.
- Fix: windows_per_story_train 16 (32K unique windows / ~8M tokens), eval 4.

## A/B evidence (CPU, ctx 128, bs 8) - memorization isolation
| Setup | final train loss | final eval loss | verdict |
|---|---|---|---|
| 50 stories, 1 win, 400 steps | 0.14 | 7.46 | memorized |
| 50 stories, 16 win, 400 steps | 0.15 | 7.84 | memorized (data too small either way) |
| 400 stories, 1 win, 600 steps | 2.71 | 4.92 | healthy |
| 400 stories, 16 win, 600 steps | 2.80 | 4.89 | healthy |

Conclusion: unique-data volume is the lever; 16 windows = 16x unique data at
zero extra encode cost. Kaggle re-run (Run 2b) pending with the fix.

## Run 2b - Kaggle T4 (pending upgrade, ~30-45 min)
- Notebook: `scripts/kaggle_baseline.ipynb` (import into Kaggle, GPU T4 x2)
- Script: `scripts/kaggle_baseline.py --device cuda --steps 4000 --batch-size 64 --context-length 256 --data stream --max-stories 50000 --hf-repo forge-lm/baseline`
- Expected: 4k steps on full TinyStories stream; checkpoint pushed to HF Hub
  `forge-lm/baseline`; pull back via `scripts/download_from_hub.py --repo forge-lm/baseline`
- This table will be updated with GPU numbers after the run.

## Reproducibility
- Same code path CPU/CUDA (device flag only); seed 0; deterministic windows;
  resume exact. uv.lock pins environment.