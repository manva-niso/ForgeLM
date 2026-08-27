"""Generation quality metrics: distinct-n, repetition, length sanity."""

from __future__ import annotations


def distinct_n(ids: list[int], n: int = 2) -> float:
    if len(ids) < n:
        return 0.0
    ngrams = {tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)}
    return len(ngrams) / (len(ids) - n + 1)


def repetition_rate(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    total = 0
    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i : i + n])
        total += 1
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / total


def length_sanity(ids: list[int], max_tokens: int) -> dict[str, object]:
    return {
        "tokens": len(ids),
        "hit_cap": len(ids) >= max_tokens,
    }