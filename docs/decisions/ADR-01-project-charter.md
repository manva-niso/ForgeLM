# ADR-01: Project Charter

Status: accepted
Date: 2026-08-21

## Context
Build an edge/on-device assistant LLM from scratch (tokenizer, GPT, training,
eval, serving, LoRA/QLoRA, API, release) in 12 vertical slices, targeting YC
AI-role skills: PyTorch, fine-tuning, inference optimization, APIs, cloud.

## Decision
- CPU-first dev (Windows, torch CPU); Kaggle GPU for baseline/SFT/QLoRA training.
- Model target: <=25M params, ctx 512, vocab 4096, <=8MB 4-bit, <100ms first-token CPU.
- Stack: uv + PyTorch + Pydantic + Hydra + pytest + ruff. Hand-rolled BPE/GPT/LoRA/int4.
- TinyStories is the provisional base dataset (may change).

## Consequences
Small model = headline feature; every day ships tested, evidenced, committed
artifact. Constraint: CPU compute limits baseline scale.