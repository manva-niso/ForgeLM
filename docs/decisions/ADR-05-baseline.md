# ADR-05: Baseline Training

Status: accepted
Date: 2026-08-29

## Context
Day 5: produce the first real checkpoint (baseline GPT on TinyStories) with
evidence, and hand the same run to Kaggle GPU.

## Decision
- `scripts/kaggle_baseline.py`: single entry point for CPU (local) and CUDA
  (Kaggle), `--data stream` reads full TinyStories via `datasets` streaming.
- `scripts/kaggle_baseline.ipynb`: importable Kaggle notebook (clone repo,
  install torch cu121, train 4000 steps bs 64 ctx 256, push checkpoint to
  HF Hub `forge-lm/baseline`, zip fallback download).
- `scripts/download_from_hub.py`: pull checkpoint back locally.
- Local evidence run completed: 300 steps, 358s CPU, loss 8.11 -> 3.34 train /
  3.95 eval. Loss still decreasing -> Kaggle 4000-step run is the upgrade.

## Why CPU-first baseline before Kaggle
The vertical-slice rule demands evidence on Day 5 regardless of GPU access;
CPU gives a real checkpoint + curve in 6 minutes, and proves the exact code
path Kaggle will execute (device flag is the only difference).

## Consequences
Baseline checkpoint exists (`checkpoints/baseline/`); Kaggle run pending and
documented in `benchmarks/baseline_train.md` (Run 2 section). Eval harness
(Day 6) will consume this checkpoint.