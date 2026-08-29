# ForgeLM v1.0.0 — Release Notes (2026-09-01)

## Summary
A complete from-scratch pipeline for a small language model:
data contract → BPE tokenizer → GPT core → training → baseline →
evaluation → inference engine → LoRA fine-tuning → QLoRA int4 →
ONNX serving → hardened API → public demo → release.

## Highlights
- **5.25M-param GPT trained entirely from scratch** (custom byte-level BPE,
  RMSNorm/RoPE/SwiGLU, tied head, KV-cache) — best baseline eval 2.0247.
- **LoRA fine-tuned** story-instruction model (eval 2.00) with coherent
  instruction-following; dolly domain-mismatch lesson documented.
- **QLoRA int4 ≤8MB** (6.27MB, ppl +2.9%) — edge/on-device ready.
- **Inference engine**: 123-309 tok/s CPU decode, 10ms first token,
  int8 4MB variant, ONNX export (int8 prefill 11.8K tok/s).
- **API**: OpenAI-compatible /v1/completions + auth + rate limit + metrics.
- **Public demo**: https://forgelm-dep.streamlit.app (live).
- **Quality**: 92 tests green, ruff clean, pip-audit 0 findings,
  benchmarks + ADRs + logs for every stage (tags v0.0.1 → v1.0.0).

## Usage
- Demo: https://forgelm-dep.streamlit.app
- Local: `uv sync --group dev` →
  `uv run python scripts/smoke_infer.py --ckpt models/forgelm-sft-story --prompt "..."`
- API: `uv run uvicorn forger.serve.api:app` → POST /v1/completions.

## Known limitations (see MODEL_CARD.md)
- 128-token context; story domain only; no general knowledge.
- Free-tier cold starts; GIL-bound concurrency on single-worker CPU API.

## Release checklist
- [x] pytest green (92)  [x] ruff clean  [x] pip-audit clean
- [x] CI (lint+test+build)  [x] changelog  [x] model card
- [x] SRE/SLOs + runbook  [x] demo live  [x] weights in repo