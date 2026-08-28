# ADR-06: Evaluation Harness

Status: accepted
Date: 2026-08-29

## Context
Day 6 requires measuring the baseline properly (not just training eval loss).

## Decision
- `forger/eval/` package: perplexity (sliding-window, stride ctx/2), bits-per-byte
  (natural for a byte-level tokenizer: NLL / (ln2 x bytes)), generation
  (greedy/top-k/temperature with seeded rng), metrics (distinct-1/2/3,
  repetition trigram rate, length sanity).
- Pinned eval corpus: 500 held-out TinyStories **validation** stories streamed
  once, contract-validated, written to `data/eval_tinystories.jsonl` +
  sha256 sidecar (`benchmarks/eval_corpus.sha256`) - reproducible, offline
  after first fetch. Production-grade eval requires pinned data, not
  re-streamed data.
- Report: `benchmarks/eval_report.md` with reproducibility block (torch
  version, checkpoint, tokenizer checksum, corpus sha256, seed).
- Model loading centralized in `forger/model/checkpoint.py` (best_model.pt /
  model.pt / Trainer checkpoint layouts) - shared by eval, engine, smoke_infer.

## Bug found by the harness
Generation past `context_length` silently broke RoPE (empty cos/sin slice ->
size-0 tensors). `GPT.forward` now raises on cached+new > context_length;
`generate()` stops at the context limit with a `context_limited` flag.

## Results (baseline)
Perplexity 8.449 / bpb 1.0366 on 136,428 tokens (500 stories); generations
distinct-2 0.89-0.97, trigram repetition 0.00-0.05.