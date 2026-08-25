# Changelog
Append one line per significant change - do not rewrite this file, only append.

- 2026-08-21 d1: product + data contract - DatasetExample schema, validate CLI, TinyStories sample (1000 rows, sha256 7cce2e67), CI workflow, ADR-01. Tag v0.0.2.
- 2026-08-22 ci: pushed repo, fixed CI (isort known-first-party, forger/data gitignore), added docs/development-log.md. CI green on run 32558803277.
- 2026-08-22 d2: implemented deterministic byte-level BPE tokenizer with save/load, round-trip tests and reference parity benchmark (10/10 vs HF). Tag v0.0.3.
- 2026-08-22 d3: implemented GPT model core (RMSNorm, RoPE, sdpa attention, SwiGLU, KV-cache stub, tied head), 10 tests, CPU forward benchmark, ADR-03. Tag v0.0.4.
- 2026-08-22 d4: implemented training pipeline (WindowDataset, Trainer with AdamW/cosine+warmup/AMP-CUDA/eval/checkpoint/TensorBoard, TrainConfig+YAML), 6 tests, 10-step CPU smoke (8.1s, loss 8.11->7.36), ADR-04. Tag v0.0.5.
- 2026-08-22 d5: baseline trained on CPU (300 steps, 358s, loss 8.11->3.34/3.95 eval); Kaggle notebook + script + HF Hub downloader shipped; ADR-05. Tag v0.0.6.
- 2026-08-22 d5b: Kaggle debugging + memorization fix - windows_per_story 16/4 (was 1/1), eval-rise warning, kaggle branch, A/B evidence, docs/JOURNEY.md. Tag v0.0.7.
- 2026-08-22 d5c: encoder 20x speedup (single-pass rank scan + piece cache; 37.3s->1.8s, parity 10/10); memorization correction (unique data = stories, not windows); Kaggle max-stories 20000. Tag v0.0.8.
- 2026-08-22 d5d: real root cause - windows_per_story capped at 1 by len//ctx AND ctx 256 discarded 91% of TinyStories; uncapped crops, kaggle ctx 128, best-eval checkpoint. Eval now decreasing (4.63->4.58). Tag v0.0.9.
- 2026-08-22 d5e: Kaggle run 2b healthy - 20K stories, 293K train windows, eval 3.42->2.08 all NEW BEST, no memorization; trainer saves model.pt+config.json for GPT.load; scripts/smoke_infer.py (model speaks coherent TinyStories). Tag v0.0.10.