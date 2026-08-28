"""Block-wise symmetric int4 quantization (2 codes per byte) + Int4Linear.

Storage is 4-bit; forward dequantizes to fp32 (weight-only quantization).
QLoRA-in-spirit: base weights quantized + frozen, adapters train in fp32.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

BLOCK_SIZE = 64
MAX_CODE = 7.0  # symmetric int4 range [-8, 7]


def quantize_4bit(t: torch.Tensor, block_size: int = BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    flat = t.reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().max(dim=1, keepdim=True).values / MAX_CODE
    scale = torch.clamp(scale, min=1e-12)
    codes = torch.clamp(torch.round(blocks / scale), min=-8, max=7).to(torch.int8)
    return codes, scale


def dequantize_4bit(codes: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    flat = (codes.float().reshape(-1, BLOCK_SIZE) * scale).reshape(-1)
    return flat[: int(torch.tensor(shape).prod())].reshape(shape)


def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.int8)
    if flat.numel() % 2:
        flat = F.pad(flat, (0, 1))
    lo = (flat[0::2] & 0x0F).to(torch.uint8)
    hi = ((flat[1::2] & 0x0F) << 4).to(torch.uint8)
    return lo | hi


def unpack_4bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    p = packed.to(torch.uint8).reshape(-1)
    lo = (p & 0x0F).to(torch.int8)
    hi = ((p >> 4) & 0x0F).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi)
    lo = torch.where(lo >= 8, lo - 16, lo)
    out = torch.stack([lo, hi], dim=1).reshape(-1)
    return out[:n]


def quantize_params_4bit(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    codes, scale = quantize_4bit(weight.detach().float())
    return {
        "codes": pack_4bit(codes),
        "scales": scale,
        "n": weight.numel(),
        "out": weight.shape[0],
        "in": weight.shape[1],
    }


def dequantize_from_stored(data: dict[str, torch.Tensor]) -> torch.Tensor:
    codes = unpack_4bit(data["codes"], data["n"])
    shape = (data["out"], data["in"])
    return dequantize_4bit(codes, data["scales"], shape)


class Int4Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        data = quantize_params_4bit(weight)
        self.register_buffer("codes", data["codes"])
        self.register_buffer("scales", data["scales"])
        self.n = data["n"]
        self.weight = nn.Parameter(dequantize_from_stored(data), requires_grad=False)
        self.bias = nn.Parameter(bias.clone(), requires_grad=False) if bias is not None else None

    @classmethod
    def from_stored(
        cls,
        codes: torch.Tensor,
        scales: torch.Tensor,
        n: int,
        out: int,
        inp: int,
        bias: torch.Tensor | None,
    ) -> Int4Linear:
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.out_features, instance.in_features = out, inp
        instance.register_buffer("codes", codes.clone())
        instance.register_buffer("scales", scales.clone())
        instance.n = n
        instance.weight = nn.Parameter(
            dequantize_4bit(unpack_4bit(codes, n), scales, (out, inp)), requires_grad=False
        )
        instance.bias = nn.Parameter(bias.clone(), requires_grad=False) if bias is not None else None
        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def storage_bytes(self) -> int:
        total = self.codes.numel() + self.scales.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total


def quantize_model_4bit(model: nn.Module, exclude_suffixes: tuple[str, ...] = ("lm_head",)) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not name.endswith(exclude_suffixes):
            int4 = Int4Linear(module.weight.detach(), module.bias.detach() if module.bias is not None else None)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, nn.ModuleList) else getattr(parent, part)
            if isinstance(parent, nn.ModuleList):
                parent[int(parts[-1])] = int4
            else:
                setattr(parent, parts[-1], int4)
            replaced += 1
    return replaced


def export_4bit(model: nn.Module, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"config": model.config.to_dict()}
    int4_names = {
        f"{name}.{suffix}"
        for name, module in model.named_modules()
        if isinstance(module, Int4Linear)
        for suffix in ("codes", "scales", "weight", "bias")
    }
    payload["params"] = {
        k: v for k, v in model.state_dict().items() if k not in int4_names
    }
    layers: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, Int4Linear):
            layers[name] = {
                "codes": module.codes,
                "scales": module.scales,
                "n": module.n,
                "out": module.out_features,
                "in": module.in_features,
                "bias": module.bias.detach() if module.bias is not None else None,
            }
    payload["int4_layers"] = layers
    torch.save(payload, out / "model_4bit.pt")
    (out / "config.json").write_text(json.dumps(model.config.to_dict(), indent=1), encoding="utf-8")
    return out


def load_4bit(directory: str | Path):
    from forger.model.config import GPTConfig
    from forger.model.gpt import GPT

    d = Path(directory)
    payload = torch.load(d / "model_4bit.pt", map_location="cpu", weights_only=False)
    config = GPTConfig.from_dict(payload["config"])
    model = GPT(config)
    with torch.no_grad():
        model.load_state_dict(payload["params"], strict=False)
        for name, data in payload["int4_layers"].items():
            int4 = Int4Linear.from_stored(
                data["codes"], data["scales"], data["n"], data["out"], data["in"], data["bias"]
            )
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, nn.ModuleList) else getattr(parent, part)
            if isinstance(parent, nn.ModuleList):
                parent[int(parts[-1])] = int4
            else:
                setattr(parent, parts[-1], int4)
    model.eval()
    return model


def storage_size_mb(model: nn.Module) -> float:
    total = sum(m.storage_bytes() for m in model.modules() if isinstance(m, Int4Linear))
    total += model.token_embedding.weight.numel() * 4
    return total / (1024 * 1024)