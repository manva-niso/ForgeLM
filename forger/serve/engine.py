"""Inference engine: KV-cache prefill/decode/generate on top of the proven model cache."""

from __future__ import annotations

import time

import torch

from forger.eval.generation import sample_token
from forger.model.checkpoint import load_model_from_checkpoint
from forger.model.gpt import GPT, Cache
from forger.tokenizer.bpe import BPETokenizer


class Engine:
    def __init__(self, model: GPT, tokenizer: BPETokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cache: Cache = {}

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, tokenizer_dir: str = "artifacts/tokenizer") -> Engine:
        return cls(load_model_from_checkpoint(ckpt_dir), BPETokenizer.load(tokenizer_dir))

    def reset(self) -> None:
        self.cache = {}

    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        x = torch.tensor([prompt_ids], dtype=torch.long)
        logits, cache = self.model(x, cache={})
        self.cache = cache if cache is not None else {}
        return logits

    @torch.inference_mode()
    def decode_next(
        self,
        last_id: int,
        top_k: int | None = 50,
        temperature: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> int:
        x = torch.tensor([[last_id]], dtype=torch.long)
        logits, cache = self.model(x, cache=self.cache)
        self.cache = cache if cache is not None else {}
        return sample_token(logits[0, 0], top_k=top_k, temperature=temperature, generator=generator)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        top_k: int | None = 50,
        temperature: float = 0.8,
        stop_id: int | None = 0,
        seed: int | None = None,
    ) -> tuple[str, list[int], dict[str, float]]:
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.reset()
        prompt_ids = self.tokenizer.encode(prompt)
        if prompt_ids:
            self.prefill(prompt_ids)
        generated = list(prompt_ids)
        context_limit = self.model.config.context_length
        start = time.monotonic()
        while len(generated) < context_limit and len(generated) - len(prompt_ids) < max_tokens:
            next_id = self.decode_next(generated[-1], top_k=top_k, temperature=temperature, generator=rng)
            if stop_id is not None and next_id == stop_id:
                break
            generated.append(next_id)
        elapsed = time.monotonic() - start
        new_tokens = generated[len(prompt_ids) :]
        text = self.tokenizer.decode(generated)
        stats = {
            "generated_tokens": len(new_tokens),
            "tokens_per_sec": len(new_tokens) / elapsed if elapsed > 0 else 0.0,
            "stopped": len(new_tokens) < max_tokens,
            "context_limited": len(generated) >= context_limit,
        }
        return text, generated, stats