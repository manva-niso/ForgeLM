"""Greedy smoke inference on a trained checkpoint.

Usage:
    uv run python scripts/smoke_infer.py --ckpt checkpoints/baseline --prompt "Once upon a time"
    uv run python scripts/smoke_infer.py --ckpt checkpoints/baseline --prompt "The little cat" --max-tokens 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer


def load_model(ckpt_dir: Path) -> GPT:
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        config = GPTConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    else:
        ckpt = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config)
    best_path = ckpt_dir / "best_model.pt"
    model_path = ckpt_dir / "model.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        print(f"loaded best-eval weights (best eval: {json.loads((ckpt_dir / 'best_eval.json').read_text()):.4f})")
    else:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        print("loaded final model.pt")
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke-infer")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args(argv)

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(Path(args.ckpt))

    ids = tokenizer.encode(args.prompt)
    cache = {}
    with torch.inference_mode():
        for _ in range(args.max_tokens):
            x = torch.tensor([ids[-1:]], dtype=torch.long)
            logits, cache = model(x, cache=cache)
            next_id = int(logits[0, 0].argmax().item())
            if next_id == 0:
                break
            ids.append(next_id)
    print("prompt:", args.prompt)
    print("continuation:", tokenizer.decode(ids[len(tokenizer.encode(args.prompt)) :]))
    print("full:", tokenizer.decode(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())