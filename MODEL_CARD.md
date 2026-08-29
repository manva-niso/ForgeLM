---
language: en
license: apache-2.0
tags:
  - tiny-language-model
  - from-scratch
  - pytorch
  - story-generation
datasets:
  - roneneldan/TinyStories
  - databricks/databricks-dolly-15k
model-index:
  - name: forgelm-sft-story
    results:
      - task: language-modeling
        metrics:
          - name: Perplexity (TinyStories eval, 200 stories)
            value: 2.00
          - name: Bits-per-byte
            value: 1.03
---

# ForgeLM — story-SFT (5.25M)

A small decoder-only GPT trained **entirely from scratch** (no pretrained
weights): custom byte-level BPE tokenizer, RMSNorm/RoPE/SwiGLU GPT core,
LoRA fine-tuned on story instructions, QLoRA int4 export ≤8MB.

## Model details

- Architecture: decoder-only GPT — 4 layers, d_model 256, 4 heads
  (head_dim 64), context 128, SwiGLU MLP (4x), RMSNorm, RoPE, tied
  embedding/head.
- Params: 5,249,280 (fp32 20 MB / int4 ~6 MB).
- Tokenizer: custom byte-level BPE, vocab 4096 (0 = `<|endoftext|>`),
  GPT-2-style pre-tokenization. Trained on TinyStories sample.
- Training: 20,000 TinyStories stories (293,200 windows), 4000 steps,
  bs 64, ctx 128, lr 3e-4, AdamW, cosine+warmup, T4 GPU.
  Best eval 2.0247 (baseline).
- Fine-tuning: LoRA r=8/alpha=16 on attention+MLP projections (3.3%
  trainable), story instructions over TinyStories, 300 steps, lr 3e-4.
  Final loss 1.96 / eval 2.00.

## Intended use

- Story generation from short prompts ("Write a story about a cat.").
- Demonstration of a complete from-scratch ML pipeline (tokenizer → model →
  training → eval → serving → fine-tuning → quantization → deployment).
- Edge/on-device experiments: int4 artifact ≈6 MB fits tight budgets.

## Evaluation results

| Metric | Value |
|---|---|
| Baseline eval (TinyStories, 500-story pinned corpus) | ppl 8.449 / bpb 1.0366 |
| Baseline best eval loss (training, 20K stories) | 2.0247 |
| Story-SFT final loss / eval | 1.96 / 2.00 |
| int4 quality delta (baseline fp32 → int4) | ppl 7.88 → 8.11 (+2.9%) |
| Generation metrics (5 prompts, top-k 50, temp 0.8) | distinct-2 0.89–0.97, trigram repetition ≤ 0.05 |

## Limitations

- Tiny model: short-range coherence, repetition under greedy decoding,
  no factual knowledge outside story patterns.
- Training data is TinyStories (synthetic, child-safe English) — the model
  cannot answer general-knowledge questions. SFT on general-KB data (dolly)
  was attempted and rejected (domain mismatch, documented in
  docs/decisions/ADR-08-lora-sft.md).
- Context limited to 128 tokens (stories are short).
- Streaming generation only; no chat template beyond the instruction format.

## Bias & safety

- Data: TinyStories is designed to be safe/child-friendly; LoRA data is the
  same domain. No personal data; dolly rows are general knowledge.
- Known failure mode: the model may produce repetitive or nonsensical text;
  no filters on output (a moderation layer is out of scope).
- Intended for benign story generation and educational use.

## Citation

```bibtex
@misc{forgelm2026,
  title={ForgeLM: A From-Scratch Tiny Language Model},
  author={ForgeLM authors},
  year={2026},
  url={https://github.com/manva-niso/ForgeLM}
}
```

## Try it

- Public demo: https://forgelm-dep.streamlit.app
- Repo: https://github.com/manva-niso/ForgeLM (weights committed in `models/`)