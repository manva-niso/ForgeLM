"""Hand-rolled LoRA: low-rank adapters on frozen Linear layers."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRAConfig:
    def __init__(self, r: int = 8, alpha: float = 16.0) -> None:
        if r <= 0:
            raise ValueError("r must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.r = r
        self.alpha = alpha

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        in_f, out_f = self.base.in_features, self.base.out_features
        self.A = nn.Parameter(torch.empty(config.r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, config.r))
        self.scaling = config.scaling
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_suffixes: tuple[str, ...],
    allowed_types: tuple[type, ...] = (nn.Linear,),
) -> list[str]:
    replaced: list[str] = []

    def _walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, allowed_types) and full.endswith(target_suffixes):
                if isinstance(module, nn.ModuleList):
                    module[int(name)] = LoRALinear(child, config)
                else:
                    setattr(module, name, LoRALinear(child, config))
                replaced.append(full)
            else:
                _walk(child, full)

    _walk(model, "")
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)
    return replaced


def count_lora_params(model: nn.Module) -> int:
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and (".A" in name or ".B" in name)
    )


def merge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                merged = module.base.weight + (module.B @ module.A) * module.scaling
            module.base.weight.data = merged
            module.base.weight.requires_grad_(False)
            module.A.requires_grad_(False)
            module.B.requires_grad_(False)


def convert_merged(model: nn.Module) -> None:
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            plain = nn.Linear(module.base.in_features, module.base.out_features, bias=module.base.bias is not None)
            with torch.no_grad():
                plain.weight.copy_(module.base.weight)
                if plain.bias is not None:
                    plain.bias.copy_(module.base.bias)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                if isinstance(parent, nn.ModuleList):
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            if isinstance(parent, nn.ModuleList):
                parent[int(parts[-1])] = plain
            else:
                setattr(parent, parts[-1], plain)