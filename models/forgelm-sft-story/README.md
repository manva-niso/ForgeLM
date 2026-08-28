# ForgeLM Story-SFT Weights (v0.3.2)

| Item | Value |
|---|---|
| Files | `model.pt` (fp32 merged, 21.3 MB) · `models/forgelm-sft-story-4bit/` (int4, 6.27 MB) |
| Base | `models/forgelm-baseline` (TinyStories, eval 2.0247) |
| Fine-tune | LoRA r=8 / alpha=16 on 20 layers (3.3% trainable), 300 steps, lr 3e-4 |
| Data | Story instructions over TinyStories ("Write a story about {topic}") — domain-matched to base knowledge |
| Final loss | 1.96 (eval 2.00) |
| Quality | Follows story instructions coherently (see benchmarks/sft_qualitative.md) |

**Why story-domain and not dolly:** the dolly experiment (15K general-KB rows)
produced garbage — SFT teaches response form, not facts, and a 5.25M
TinyStories-only base has no general knowledge. Documented in ADR-08.

Usage:
```powershell
uv run python scripts/smoke_infer.py --ckpt models/forgelm-sft-story --prompt "### Instruction: Write a story about a cat
### Response:"
```