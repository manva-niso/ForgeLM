# Changelog

Append one line per significant change - do not rewrite this file, only append.
Dates follow the PLAN calendar (D1=2026-08-21 ... D9=2026-08-29, D10=08-30,
D11=08-31, D12=09-01). Detail lives in development-log.md.

## Day 1 — Fri 2026-08-21 — Product + data contract
- 2026-08-21 v0.0.1: repo bootstrap — uv project, dev group, GitHub Actions CI, docs skeleton, smoke tests.
- 2026-08-21 v0.0.2: product contract (<=25M params / <=8MB 4-bit / <100ms CPU), DatasetExample schema + forger-data-validate CLI, TinyStories sample (1000 rows, sha256 7cce2e67), CI made green (isort first-party + /data/ gitignore fix), ADR-01.
- Evidence: `benchmarks/data_contract.md`.

## Day 2 — Sat 2026-08-22 — Tokenizer (BPE from scratch)
- 2026-08-22 v0.0.3: deterministic byte-level BPE (GPT-2 style) — train/encode/decode/save/load, GPT-2 regex pretokenization, CLI, 17 tests, **10/10 parity vs HuggingFace**, ADR-02.
- Evidence: `benchmarks/tokenizer_parity.md` (artifact checksum e20ed6...).

## Day 3 — Sun 2026-08-23 — Model core (GPT blocks)
- 2026-08-23 v0.0.4: GPT — RMSNorm, RoPE, causal sdpa attention, SwiGLU, tied embedding/head, KV-cache stub (proven == full forward), 5,249,280 params, init weights, 10 tests.
- Evidence: `benchmarks/model_forward.md` (211ms forward, 4,855 tok/s CPU).

## Day 4 — Mon 2026-08-24 — Training pipeline
- 2026-08-24 v0.0.5: WindowDataset (shift-by-1 windows, deterministic get_batch), Trainer (AdamW, cosine+warmup, grad accum, eval loop, checkpoints, TensorBoard), TrainConfig+YAML, 6 tests incl. **bit-exact resume**, CPU smoke 10 steps in 8.1s (loss 8.11->7.36), ADR-04. CPU runs fp32 (bf16 measured 7x slower).
- Evidence: `benchmarks/train_smoke.md`.

## Day 5 — Tue 2026-08-25 — Baseline training (the Kaggle saga + memorization war)
- 2026-08-25 v0.0.6: baseline on CPU (300 steps, loss 8.11->3.34/3.95 eval); kaggle_baseline.py + notebook + download_from_hub.py, ADR-05.
- 2026-08-25 v0.0.7: memorization fight — windows_per_story 16/4, eval-rise warning, kaggle branch, A/B evidence, JOURNEY.md.
- 2026-08-25 v0.0.8: encoder 20x speedup (single-pass rank scan + piece cache; 37.3s->1.8s; parity 10/10), memorization correction (unique data = stories, not windows), Kaggle max-stories 20000.
- 2026-08-25 v0.0.9: real root cause — windows_per_story capped by len//ctx AND ctx 256 discards 91% of TinyStories; uncapped crops, ctx 128, best-eval checkpoint.
- 2026-08-25 v0.0.10: checkpoint handoff — model.pt + config.json in trainer saves, smoke_infer script.
- 2026-08-25 v0.1.0: baseline COMPLETE — 20K stories, 293,200 windows, best eval **2.0247**, coherent generations, checkpoint on HF `Manvaniso/forgelm`.
- 2026-08-25 v0.1.1: weights committed in-repo (`models/forgelm-baseline/`, 24.3MB) — usable with zero accounts; README usage.
- Evidence: `benchmarks/baseline_train.md` (incl. A/B memorization table).

## Day 6 — Wed 2026-08-26 — Evaluation harness
- 2026-08-26 v0.2.0: sliding-window ppl + bits-per-byte, generation (greedy/top-k/temp, seeded), distinct-n/repetition metrics, PINNED 500-story eval corpus (sha256 06aa34...), eval report (ppl 8.449 / bpb 1.0366 / 136K tokens), context-overflow guard, ADR-06.
- Evidence: `benchmarks/eval_report.md`.

## Day 7 — Thu 2026-08-27 — Inference engine
- 2026-08-27 v0.2.1: Engine (prefill/decode/generate on the proven KV-cache), int8 dynamic variant (4MB, 5x smaller), torch.compile fallback, benchmark protocol — fp32 123 tok/s, first-token 10ms; ADR-07.
- Evidence: `benchmarks/serve_speedup.{md,json}`.

## Day 8 — Fri 2026-08-28 — LoRA SFT (and the dolly lesson)
- 2026-08-28 v0.3.0: hand-rolled LoRA (r8/a16, 3.3% trainable, freeze-all contract), dolly-15k formatting, merge + convert-to-plain-GPT, local evidence run, ADR-08.
- 2026-08-28 v0.3.2: SFT domain lesson — dolly SFT rejected (domain mismatch, ppl 38 stories); story-domain SFT shipped (loss 1.96, coherent instruction-following); fp32 (21.3MB) + int4 (6.27MB) committed.
- Evidence: `benchmarks/sft_qualitative.md`.

## Day 9 — Sat 2026-08-29 — QLoRA int4
- 2026-08-29 v0.3.1: block-wise int4 (2 codes/byte), Int4Linear, QLoRA-in-spirit training, export/load, artifacts 4-6.27MB (<=8MB spec), ppl +2.9% for 5x size, 4 quantization bugs fixed, ADR-09.
- Evidence: `benchmarks/qlora.md`.

## Upcoming
- Day 10 (Sun 08-30): Optimized serving — ONNX export + ONNX Runtime.
- Day 11 (Mon 08-31): API hardening + cloud deployment.
- Day 12 (Tue 09-01): Security, model card, SRE + release.