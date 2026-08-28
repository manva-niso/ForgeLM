"""SFT training: freeze base + train LoRA adapters on instruction data.

Usage:
    uv run python -m forger.ft.train_sft --ckpt models/forgelm-baseline \
        --examples 5000 --steps 200 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from forger.ft.lora import LoRAConfig, apply_lora, convert_merged, count_lora_params, merge_lora
from forger.ft.story_sft_data import load_story_sft
from forger.model.checkpoint import load_model_from_checkpoint
from forger.tokenizer.bpe import BPETokenizer
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset
from forger.train.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forger-sft")
    parser.add_argument("--ckpt", default="models/forgelm-baseline")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--examples", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ckpt-dir", default="checkpoints/sft")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model_from_checkpoint(args.ckpt)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable; falling back to cpu")
        args.device = "cpu"

    texts = load_story_sft("data/tinystories_sample.jsonl", args.examples)
    split = int(len(texts) * 0.95)
    encoded = [tokenizer.encode(t) for t in texts]
    train_data = WindowDataset(
        texts[:split], tokenizer, args.context_length,
        windows_per_story=8, encoded_ids=encoded[:split],
    )
    eval_data = WindowDataset(
        texts[split:], tokenizer, args.context_length,
        windows_per_story=4, encoded_ids=encoded[split:],
    )
    train_data.shuffle(args.seed)

    lora = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
    replaced = apply_lora(model, lora, ("c_attn", "c_proj", "gate", "up", "down"))
    n_lora = count_lora_params(model)
    print(f"LoRA on {len(replaced)} layers, {n_lora:,} trainable params "
          f"({n_lora / 5_249_280 * 100:.2f}% of full model)")

    config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        lr=1e-3,
        weight_decay=0.1,
        warmup_steps=max(1, args.steps // 20),
        grad_accum=1,
        eval_every=max(10, args.steps // 8),
        eval_windows=20,
        log_every=max(5, args.steps // 20),
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.ckpt_dir,
        run_name="sft",
    )
    trainer = Trainer(model, config, train_data, eval_data)
    start = time.monotonic()
    trainer.train()
    trainer.save()
    print(f"sft done in {time.monotonic() - start:.0f}s, final loss {trainer.loss_history[-1]:.4f}")

    merge_lora(model)
    convert_merged(model)
    out = Path(args.ckpt_dir) / "merged"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=1), encoding="utf-8"
    )
    print(f"merged model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())