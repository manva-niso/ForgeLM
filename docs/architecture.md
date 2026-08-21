# Architecture

## Decisions
(architect agent fills in during planning)

## Constraints (from PROJECT.md)
- CPU-first dev on Windows (torch CPU build); Kaggle GPU (T4) for baseline/SFT/QLoRA training.
- Target: <=25M params, <=8MB after 4-bit quant, <100ms first-token latency on CPU.
- Vertical-slice rule: each day ships working artifact = code + tests + benchmark/evidence + commit.
- Training data (TinyStories) is provisional and may change later.
- No gh CLI: remote repo added via browser when ready.