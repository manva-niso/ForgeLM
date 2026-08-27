"""Perplexity and bits-per-byte with sliding-window causal evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer


@torch.inference_mode()
def evaluate_perplexity(
    model: GPT,
    tokenizer: BPETokenizer,
    texts: list[str],
    context_length: int,
    stride: int | None = None,
) -> dict[str, float]:
    stride = stride or max(1, context_length // 2)
    vocab = model.config.vocab_size
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    for text in texts:
        ids = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        n = len(ids)
        if n < 2:
            continue
        for start in range(0, n - 1, stride):
            end = min(start + context_length, n)
            if end - start < 2:
                continue
            x = torch.tensor([ids[start : end - 1]], dtype=torch.long)
            y = torch.tensor([ids[start + 1 : end]], dtype=torch.long)
            logits, _ = model(x)
            per_token = F.cross_entropy(logits.view(-1, vocab), y.view(-1), reduction="none")
            if start == 0:
                selected = per_token
            else:
                selected = per_token[-stride:]
            total_nll += selected.sum().item()
            total_tokens += selected.numel()
    if total_tokens == 0:
        raise ValueError("no tokens evaluated")
    ppl = math.exp(total_nll / total_tokens)
    bpb = total_nll / (math.log(2.0) * total_bytes)
    return {"perplexity": ppl, "bits_per_byte": bpb, "tokens": total_tokens, "bytes": total_bytes}