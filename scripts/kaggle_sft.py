"""Kaggle GPU runner for LoRA SFT + QLoRA int4 export.

Usage (Kaggle notebook, kaggle branch):
    python scripts/kaggle_sft.py --device cuda --steps 3000 --examples 15000 \
        --ckpt models/forgelm-baseline --export models/forgelm-sft-int4 \
        --hf-repo Manvaniso/forgelm-sft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forger.ft.lora import LoRAConfig, apply_lora, convert_merged, count_lora_params, merge_lora
from forger.ft.story_sft_data import format_story_instruction
from forger.model.checkpoint import load_model_from_checkpoint
from forger.quant.quantize import export_4bit, quantize_model_4bit, storage_size_mb
from forger.tokenizer.bpe import BPETokenizer
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset
from forger.train.trainer import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_story_stream(examples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        text = format_story_instruction(ex["text"], i)
        texts.append(text)
        if len(texts) >= examples:
            break
    print(f"story-sft: loaded {len(texts)} streamed examples")
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-sft")
    parser.add_argument("--ckpt", default="models/forgelm-baseline")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--examples", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--ckpt-dir", default="/kaggle/working/ckpt_sft")
    parser.add_argument("--export", default=None, help="int4 export dir (QLoRA artifact)")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model_from_checkpoint(args.ckpt)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable; falling back to cpu")
        args.device = "cpu"

    texts = load_story_stream(args.examples)
    split = int(len(texts) * 0.95)
    t1 = time.monotonic()
    encoded: list[list[int]] = []
    for i, text in enumerate(texts):
        encoded.append(tokenizer.encode(text))
        if (i + 1) % 3000 == 0:
            print(f"encoded {i + 1}/{len(texts)} ({time.monotonic() - t1:.0f}s)", flush=True)
    print(f"encoded {len(texts)} stories in {time.monotonic() - t1:.0f}s", flush=True)
    train_data = WindowDataset(texts[:split], tokenizer, args.context_length, windows_per_story=8, encoded_ids=encoded[:split])
    eval_data = WindowDataset(texts[split:], tokenizer, args.context_length, windows_per_story=4, encoded_ids=encoded[split:])
    train_data.shuffle(args.seed)

    lora = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
    replaced = apply_lora(model, lora, ("c_attn", "c_proj", "gate", "up", "down"))
    print(f"LoRA on {len(replaced)} layers, {count_lora_params(model):,} trainable params")

    config = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, context_length=args.context_length,
        lr=1e-3, weight_decay=0.1, warmup_steps=min(200, max(1, args.steps // 20)),
        grad_accum=1, eval_every=max(50, args.steps // 16), eval_windows=20,
        log_every=max(25, args.steps // 40), device=args.device, seed=args.seed,
        checkpoint_dir=args.ckpt_dir, run_name="sft",
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
    (out / "config.json").write_text(json.dumps(model.config.to_dict(), indent=1), encoding="utf-8")

    if args.export:
        quantize_model_4bit(model)
        export_4bit(model, args.export)
        print(f"int4 export at {args.export} ({storage_size_mb(model):.2f} MB)")

    if args.hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.hf_repo, exist_ok=True, private=True)
            for folder in (str(out), args.export) if args.export else (str(out),):
                api.upload_folder(folder_path=folder, repo_id=args.hf_repo, repo_type="model")
            print(f"pushed to hf.co/{args.hf_repo}")
        except Exception as exc:  # noqa: BLE001
            print(f"HF PUSH FAILED: {exc} - artifacts safe on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())