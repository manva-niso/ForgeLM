# ADR-04: Training Pipeline

Status: accepted
Date: 2026-08-24

## Context
Day 4 requires the training loop that will run the baseline (Day 5, Kaggle T4).

## Decision
- `forger/train/trainer.py` - single-file Trainer: AdamW (0.9/0.95, wd 0.1),
  cosine schedule with linear warmup, gradient accumulation, eval loop,
  checkpoint (model + optimizer + step), TensorBoard, JSON logs.
- `forger/train/dataset.py` - WindowDataset: contiguous token windows,
  labels = input shifted by 1; deterministic indexing by step (resume-safe);
  optional pre-encoded ids to avoid double tokenization.
- `forger/train/config.py` - TrainConfig dataclass + YAML (Hydra-compatible).
- AMP: fp16 + GradScaler on CUDA only. CPU runs plain fp32 - bf16 autocast was
  measured 7x SLOWER on this machine (1.35s vs 0.19s forward) and was dropped.
- Determinism contract: seeded window sampling + get_batch(step) -> exact
  resume; proven by test_resume_matches_uninterrupted (bit-identical losses).

## Benchmarks (CPU, Windows, torch 2.13)
- Smoke: 10 steps, bs 4, ctx 256 = 8.1s (1.23 step/s); loss 8.11 -> 7.36.
- Initial loss ~8.32 matches ln(4096) theory.

## Consequences
Resume is exact; schedule depends on total steps (train(until=N) keeps the
full-run schedule - a test bug caught this). Day 5 switches device to cuda,
bs 64, ctx 256, steps 4000 on Kaggle T4 via scripts/kaggle notebook.