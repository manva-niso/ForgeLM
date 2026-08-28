# Benchmark: Training Smoke - Day 4 (2026-08-29)

## Setup
- Model: GPTConfig(vocab 4096, d 256, 4 heads, 4 layers, ctx 256) - 5.25M params
- Data: TinyStories sample (1000 stories, 90/10 split), tokenizer artifacts/tokenizer
- Config: configs/train/smoke.yaml (bs 4, ctx 256, lr 3e-4, warmup 2, steps 10)
- Device: CPU fp32 (torch 2.13.0, Windows)

## Result
| Metric | Value |
|---|---|
| steps | 10 |
| wall time | 8.1s (1.23 step/s) |
| initial loss | 8.114 (theory: ln 4096 = 8.32) |
| final loss | 7.356 |
| eval loss (step 10) | 7.336 |
| checkpoint | checkpoints/smoke/ |

## Notes
- bf16 CPU autocast measured 7x slower than fp32 (1349ms vs 193ms per 1024-token forward) - CPU runs fp32.
- Encode-once optimization: tokenizer.encode run once, reused by train/eval datasets.
- Resume proven bit-exact (tests/test_trainer.py::test_resume_matches_uninterrupted).