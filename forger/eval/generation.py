"""Sampling utilities: greedy, top-k, temperature, generation loop."""

from __future__ import annotations

import torch

from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    cutoff = values[..., -1].unsqueeze(-1)
    return torch.where(logits >= cutoff, logits, torch.tensor(float("-inf"), device=logits.device))


def sample_token(
    logits: torch.Tensor,
    top_k: int | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    if temperature != 1.0:
        logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.inference_mode()
def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 64,
    top_k: int | None = 50,
    temperature: float = 0.8,
    stop_id: int | None = 0,
    seed: int | None = None,
) -> tuple[list[int], dict[str, float]]:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    context_limit = model.config.context_length
    for _ in range(max_tokens):
        if len(generated) >= context_limit:
            break
        x = torch.tensor([generated[-1:]], dtype=torch.long)
        logits, cache = model(x, cache=cache)
        next_id = sample_token(logits[0, 0], top_k=top_k, temperature=temperature, generator=rng)
        if stop_id is not None and next_id == stop_id:
            break
        generated.append(next_id)
    new_tokens = generated[len(ids) :]
    return generated, {
        "generated_tokens": len(new_tokens),
        "stopped": len(new_tokens) < max_tokens and len(generated) < context_limit,
        "context_limited": len(generated) >= context_limit,
    }