"""Training loop: AdamW, cosine schedule with warmup, AMP, grad accumulation, eval, checkpoint, TensorBoard."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset


def lr_at_step(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step + 1 - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.lr * 0.1 + 0.5 * (config.lr - config.lr * 0.1) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(
        self,
        model: GPT,
        config: TrainConfig,
        train_data: WindowDataset,
        eval_data: WindowDataset,
    ) -> None:
        self.model = model
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        torch.manual_seed(config.seed)
        self.model.to(self._device())
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
        )
        self.step = 0
        self.scaler = None
        self.use_amp = config.device.startswith("cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        self.writer = SummaryWriter(log_dir=f"runs/{config.run_name}")
        self.loss_history: list[float] = []
        self.eval_history: list[tuple[int, float]] = []
        self.best_eval_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None

    def _device(self) -> torch.device:
        return torch.device(self.config.device)

    def _tensors(self, xs: list[list[int]], ys: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._device()
        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
        )

    def _forward_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = self.model(xs)
                return torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), ys.view(-1)
                )
        logits, _ = self.model(xs)
        return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1))

    def evaluate(self, windows: int = 20) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for i in range(windows):
                xs, ys = self.eval_data.get_batch(self.step + i, self.config.batch_size)
                x, y = self._tensors(xs, ys)
                total += self._forward_loss(x, y).item()
                count += 1
        self.model.train()
        return total / count

    def train(self, until: int | None = None) -> None:
        self.model.train()
        end = self.config.steps if until is None else until
        for step in range(self.step, end):
            xs, ys = self.train_data.get_batch(step, self.config.batch_size)
            x, y = self._tensors(xs, ys)
            loss = self._forward_loss(x, y) / self.config.grad_accum
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % self.config.grad_accum == 0 or (step + 1) == self.config.steps:
                lr = lr_at_step(step, self.config)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            value = loss.item() * self.config.grad_accum
            self.loss_history.append(value)
            self.writer.add_scalar("train/loss", value, step)
            self.writer.add_scalar("train/lr", lr_at_step(step, self.config), step)
            if (step + 1) % self.config.log_every == 0:
                print(f"step {step + 1}/{self.config.steps} loss {value:.4f}")
            if (step + 1) % self.config.eval_every == 0:
                eval_loss = self.evaluate(self.config.eval_windows)
                self.eval_history.append((step + 1, eval_loss))
                self.writer.add_scalar("eval/loss", eval_loss, step)
                print(f"step {step + 1} eval loss {eval_loss:.4f}")
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.best_state = {
                        k: v.detach().clone() for k, v in self.model.state_dict().items()
                    }
                    print(f"step {step + 1} NEW BEST eval loss {eval_loss:.4f}")
                if len(self.eval_history) >= 3 and all(
                    e < eval_loss for _, e in self.eval_history[-3:]
                ):
                    print(
                        "WARNING: eval loss rising 3 evals in a row - possible memorization "
                        "(add more data, raise windows_per_story, or reduce steps)"
                    )
            self.step = step + 1

    def save(self, directory: str | Path | None = None) -> Path:
        out = Path(directory or self.config.checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "model_config": self.model.config.to_dict(),
            },
            out / "checkpoint.pt",
        )
        if self.best_state is not None:
            torch.save(self.best_state, out / "best_model.pt")
            (out / "best_eval.json").write_text(json.dumps(self.best_eval_loss), encoding="utf-8")
        (out / "train_log.json").write_text(json.dumps(self.loss_history), encoding="utf-8")
        (out / "eval_log.json").write_text(json.dumps(self.eval_history), encoding="utf-8")
        return out

    @classmethod
    def resume(
        cls,
        directory: str | Path,
        model: GPT,
        config: TrainConfig,
        train_data: WindowDataset,
        eval_data: WindowDataset,
    ) -> Trainer:
        ckpt = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=False)
        trainer = cls(model, config, train_data, eval_data)
        trainer.step = int(ckpt["step"])
        model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        return trainer


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from forger.tokenizer.bpe import BPETokenizer

    parser = argparse.ArgumentParser(prog="forger-train")
    parser.add_argument("--config", default="configs/train/baseline.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--data", default="data/tinystories_sample.jsonl")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = TrainConfig.from_yaml(args.config)
    if args.steps is not None:
        config.steps = args.steps
    tokenizer = BPETokenizer.load(args.tokenizer)
    texts = [json.loads(line)["text"] for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    split = int(len(texts) * 0.9)
    encoded = [tokenizer.encode(t) for t in texts]
    train_data = WindowDataset(texts[:split], tokenizer, config.context_length, encoded_ids=encoded[:split])
    eval_data = WindowDataset(texts[split:], tokenizer, config.context_length, encoded_ids=encoded[split:])
    train_data.shuffle(config.seed)
    model = GPT(GPTConfig(vocab_size=len(tokenizer.token_bytes), context_length=config.context_length))
    trainer = Trainer(model, config, train_data, eval_data)
    if args.resume:
        trainer = Trainer.resume(config.checkpoint_dir, model, config, train_data, eval_data)
    start = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - start
    trainer.save()
    print(f"done: {config.steps} steps in {elapsed:.1f}s ({config.steps / elapsed:.2f} step/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())