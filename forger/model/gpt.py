"""GPT model: embedding, transformer blocks, tied LM head, KV-cache stub."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from forger.model.blocks import Block, RMSNorm, init_weights
from forger.model.config import GPTConfig

Cache = dict[int, tuple[torch.Tensor, torch.Tensor]]


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        init_weights(self)

    def forward(self, idx: torch.Tensor, cache: Cache | None = None) -> tuple[torch.Tensor, Cache | None]:
        _, T = idx.shape
        if T > self.config.context_length:
            raise ValueError(
                f"sequence length {T} exceeds context_length {self.config.context_length}"
            )
        if cache:
            cached_len = cache[0][0].size(2)
            if cached_len + T > self.config.context_length:
                raise ValueError(
                    f"cached {cached_len} + new {T} exceeds context_length {self.config.context_length}"
                )
        x = self.token_embedding(idx)
        new_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, block in enumerate(self.blocks):
            layer_cache = cache.get(i) if cache else None
            x, kv = block(x, layer_cache)
            new_cache[i] = kv
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, (new_cache if cache is not None else None)

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "model.pt")
        (out / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=1), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> GPT:
        d = Path(directory)
        config = GPTConfig.from_dict(json.loads((d / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        return model