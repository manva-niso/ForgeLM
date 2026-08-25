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
| 50 stories, 16 win, 400 steps | 0.15 | 7.84 | memorized |
| 400 stories, 1 win, 600 steps | 2.71 | 4.92 | not yet collapsed |
| 400 stories, 16 win, 600 steps | 2.80 | 4.89 | not yet collapsed |
| 400 stories, 16 win, 1200 steps | ~2.5 | 5.09 -> 5.66 | memorization creeping back |

CORRECTION: windows from the same story overlap ~completely; unique data =
stories x tokens (400 stories = ~52K unique tokens). The window multiplier
adds redundant views only. Real lever = story count, gated by encoder speed.
Encoder optimized 20x (37.3s -> 1.8s per 20 stories; single-pass rank scan +
piece cache, parity still 10/10). Kaggle re-run now uses 20000 stories.

CORRECTION 2 (the real root cause): `min(windows_per_story, len//ctx)` capped
crops at 1 for stories < 2x ctx, and ctx 256 discarded 91% of TinyStories
(avg story ~150 tokens). Fixed: unconditional windows_per_story + kaggle
context 128 (97% qualify; 6176 vs 386 windows for 400 stories) + best-eval
checkpoint. Validation: eval DECREASING 4.63 -> 4.58, train 3.22.

## Run 2b - Kaggle T4 (COMPLETE - 2026-08-22)
| Item | Value |
|---|---|
| Data | 20,000 TinyStories stories (stream), encode 529s |
| Windows | train 293,200 / eval 3,812 (ctx 128, 16 crops/story) |
| Config | bs 64, ctx 128, lr 3e-4, warmup 100, steps 4000, T4 |
| Final eval loss | **2.0247** (best, at step 3250) |
| Final train loss | ~2.0 (tracks eval - no memorization) |
| Eval trajectory | 3.42 (250) -> 2.86 (500) -> 2.44 (1000) -> 2.13 (2000) -> 2.07 (3000) -> 2.02 (3250) -> 2.04 (4000) |
| Checkpoint | HF Hub `Manvaniso/forgelm` (pulled to checkpoints/forgelm/) |

Sample generation (best weights, greedy, prompt "Once upon a time there was a little cat"):
"Once upon a time there was a little cat named Tom. Tom loved to play with his toy
cat. One day, Tom saw a big cat in the park. The cat wanted to play with the cat.
Tom said, 'I want to play with the cat. It is fun.'"

Coherent multi-sentence story with names and dialogue. Baseline DONE.
Weights committed to the repo at `models/forgelm-baseline/` (24.3 MB) - no
HF account required to use the model; HF Hub remains the Kaggle transport.
- Notebook: `scripts/kaggle_baseline.ipynb` (import into Kaggle, GPU T4 x2)
- Script: `scripts/kaggle_baseline.py --device cuda --steps 4000 --batch-size 64 --context-length 256 --data stream --max-stories 50000 --hf-repo Manvaniso/forgelm`
- Expected: 4k steps on full TinyStories stream; checkpoint pushed to HF Hub
  `Manvaniso/forgelm`; pull back via `scripts/download_from_hub.py --repo Manvaniso/forgelm`
- This table will be updated with GPU numbers after the run.

## Reproducibility
- Same code path CPU/CUDA (device flag only); seed 0; deterministic windows;
  resume exact. uv.lock pins environment.