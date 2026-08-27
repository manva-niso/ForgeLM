"""Load a GPT from any checkpoint layout (best_model.pt / model.pt / Trainer checkpoint)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT


def load_model_from_checkpoint(directory: str | Path) -> GPT:
    d = Path(directory)
    config_path = d / "config.json"
    if config_path.exists():
        config = GPTConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    else:
        ckpt = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config)
    for name in ("best_model.pt", "model.pt"):
        path = d / name
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            break
    else:
        ckpt = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model