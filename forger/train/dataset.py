"""Contiguous window sampler over tokenized texts."""

from __future__ import annotations

import random
from collections.abc import Sequence

from forger.tokenizer.bpe import BPETokenizer


class WindowDataset:
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: BPETokenizer,
        context_length: int,
        windows_per_story: int = 1,
        encoded_ids: Sequence[list[int]] | None = None,
    ) -> None:
        self.context_length = context_length
        self.tokenizer = tokenizer
        if encoded_ids is not None:
            encoded = [ids for ids in encoded_ids if len(ids) >= context_length + 1]
        else:
            encoded = [tokenizer.encode(text) for text in texts]
        rng = random.Random(0)
        self.windows: list[tuple[list[int], list[int]]] = []
        for ids in encoded:
            n = min(windows_per_story, max(1, len(ids) // context_length))
            for _ in range(n):
                start = rng.randrange(0, len(ids) - context_length)
                x = ids[start : start + context_length]
                y = ids[start + 1 : start + context_length + 1]
                self.windows.append((x, y))
        if not self.windows:
            raise ValueError("no windows: corpus too short for context_length")

    def shuffle(self, seed: int) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.windows)

    def get_batch(self, step: int, batch_size: int) -> tuple[list[list[int]], list[list[int]]]:
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        n = len(self.windows)
        for i in range(batch_size):
            x, y = self.windows[(step * batch_size + i) % n]
            xs.append(x)
            ys.append(y)
        return xs, ys