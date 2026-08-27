"""Optimization variants: dynamic int8 quantization and torch.compile."""

from __future__ import annotations

import torch
from torch import nn


def quantize_dynamic(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def compile_model(model: nn.Module) -> tuple[nn.Module, str]:
    try:
        compiled = torch.compile(model)
        return compiled, "compiled"
    except Exception as exc:  # noqa: BLE001
        return model, f"compile unavailable: {exc}"


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)