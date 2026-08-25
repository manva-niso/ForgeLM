"""Baseline training: standalone script (CPU local or Kaggle GPU).

Usage:
    python scripts/kaggle_baseline.py --device cpu --steps 300 --data data/tinystories_sample.jsonl
    python scripts/kaggle_baseline.py --device cuda --steps 4000 --data stream --hf-repo forge-lm/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset
from forger.train.trainer import Trainer


def load_texts(data: str, max_stories: int | None) -> list[str]:
    if data == "stream":
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        texts = [ex["text"] for ex in ds.take(max_stories or 50_000)]
        return texts
    texts = []
    with Path(data).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if max_stories is not None and len(texts) >= max_stories:
                break
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-baseline")
    parser.add_argument("--config", default=None, help="TrainConfig yaml; explicit args override it")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--data", default="data/tinystories_sample.jsonl", help="jsonl path or 'stream' for full TinyStories")
    parser.add_argument("--max-stories", type=int, default=2000, help="cap stories (encode is ~0.17s/story on CPU)")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--ckpt-dir", default="checkpoints/baseline")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--windows-per-story-train", type=int, default=None)
    parser.add_argument("--windows-per-story-eval", type=int, default=None)
    parser.add_argument("--hf-repo", default=None, help="HF repo id to push checkpoint (needs HF_TOKEN)")
    args = parser.parse_args(argv)

    if args.config:
        config = TrainConfig.from_yaml(args.config)
        for field in ("steps", "batch_size", "context_length", "device"):
            value = getattr(args, field)
            if value is not None:
                setattr(config, field, value)
    else:
        config = TrainConfig(
            steps=args.steps if args.steps is not None else 4000,
            batch_size=args.batch_size if args.batch_size is not None else 8,
            context_length=args.context_length if args.context_length is not None else 256,
            lr=3e-4,
            weight_decay=0.1,
            warmup_steps=100,
            grad_accum=1,
            eval_every=250,
            eval_windows=20,
            log_every=50,
            device=args.device if args.device is not None else "cpu",
            seed=0,
            checkpoint_dir=args.ckpt_dir,
            run_name=args.run_name,
        )
        steps = config.steps
        config.eval_every = max(25, steps // 16)
        config.log_every = max(10, steps // 40)
        config.warmup_steps = min(100, max(1, steps // 40))
    if args.windows_per_story_train is not None:
        config.windows_per_story_train = args.windows_per_story_train
    if args.windows_per_story_eval is not None:
        config.windows_per_story_eval = args.windows_per_story_eval
    config.checkpoint_dir = args.ckpt_dir
    config.run_name = args.run_name

    t0 = time.monotonic()
    tokenizer = BPETokenizer.load(args.tokenizer)
    texts = load_texts(args.data, args.max_stories)
    print(f"loaded {len(texts)} stories in {time.monotonic() - t0:.0f}s")
    encoded = [tokenizer.encode(t) for t in texts]
    split = int(len(encoded) * 0.95)
    train_data = WindowDataset(
        texts[:split], tokenizer, config.context_length,
        windows_per_story=config.windows_per_story_train, encoded_ids=encoded[:split],
    )
    eval_data = WindowDataset(
        texts[split:], tokenizer, config.context_length,
        windows_per_story=config.windows_per_story_eval, encoded_ids=encoded[split:],
    )
    train_data.shuffle(0)
    print(f"train windows: {len(train_data.windows)}, eval windows: {len(eval_data.windows)}")

    model = GPT(GPTConfig(vocab_size=len(tokenizer.token_bytes), context_length=config.context_length))
    if config.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        config.device = "cpu"

    trainer = Trainer(model, config, train_data, eval_data)
    start = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - start
    trainer.save()
    print(f"done: {config.steps} steps in {elapsed:.0f}s ({config.steps / elapsed:.2f} step/s)")
    print(f"final loss: {trainer.loss_history[-1]:.4f}")

    if args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True, private=True)
        api.upload_folder(folder_path=args.ckpt_dir, repo_id=args.hf_repo, repo_type="model")
        print(f"pushed checkpoint to hf.co/{args.hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())