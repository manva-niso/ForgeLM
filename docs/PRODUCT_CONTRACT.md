# Product Contract — ForgeLM

## Purpose
An edge/on-device assistant language model: train, fine-tune (LoRA/QLoRA) and serve a small GPT from scratch, with quantization and a hardened API. The small size is the headline feature, not a limitation.

## Target specification
| Item | Target |
|---|---|
| Parameters | <=25M |
| Context window | 512 tokens |
| Vocab size | 4096 |
| Model size on disk | <=8MB after 4-bit quant |
| First-token latency | <100ms on CPU |
| Base dataset | TinyStories (provisional — may change) |

## Acceptance criteria
1. Custom BPE tokenizer (no HF tokenizer) with round-trip + parity tests.
2. GPT core (causal attention, RMSNorm, SwiGLU) with shape/gradient tests.
3. Training pipeline with deterministic resume + benchmark evidence.
4. Baseline checkpoint with eval report (perplexity + generations).
5. LoRA SFT merged weights with qualitative before/after evidence.
6. QLoRA 4-bit export <=8MB with size/latency report.
7. Optimized serving <100ms first-token on CPU (benchmarked).
8. Hardened OpenAI-compatible API + Docker + CI/CD.
9. Model card, security audit, SLOs, release v1.0.0.

## Non-goals
- Frontier-scale models or multi-GPU training.
- Paying users / multi-tenant production.
- Training data finality — dataset may be swapped later.