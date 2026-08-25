"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 8
    context_length: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 20
    grad_accum: int = 1
    eval_every: int = 50
    eval_windows: int = 20
    log_every: int = 10
    device: str = "cpu"
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    run_name: str = "baseline"
    windows_per_story_train: int = 8
    windows_per_story_eval: int = 2

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must be within [0, steps]")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be positive")
        if self.windows_per_story_train <= 0 or self.windows_per_story_eval <= 0:
            raise ValueError("windows_per_story must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)