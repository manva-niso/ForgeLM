"""Transformer building blocks: RMSNorm, RoPE, causal attention, SwiGLU MLP, Block."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from forger.model.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = q.size(2)
        cos = self.cos_cached[offset : offset + t].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + t].unsqueeze(0).unsqueeze(0)
        return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_heads * self.head_dim, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        past_len = cache[0].size(2) if cache is not None else 0
        q, k = self.rope(q, k, offset=past_len)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        total_len = k.size(2)
        if T > 1 and past_len > 0:
            mask = torch.zeros(T, total_len, dtype=x.dtype, device=x.device)
            mask[:, past_len:] = torch.triu(
                torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device), diagonal=1
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif T > 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.ffn_mult * config.d_model
        self.gate = nn.Linear(config.d_model, hidden, bias=False)
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv = self.attn(self.ln_1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, kv


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


def init_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    for name, module in model.named_modules():
        if isinstance(module, MLP):
            std = 0.02 / math.sqrt(2 * module.down.out_features) if hasattr(module, "down") else 0.02
            nn.init.normal_(module.down.weight, mean=0.0, std=std)
        if isinstance(module, CausalSelfAttention):
            nn.init.normal_(module.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * module.c_proj.in_features))