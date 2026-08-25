"""Benchmark GPT forward pass on CPU -> benchmarks/model_forward.md."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from forger.model.blocks import count_params
from forger.model.config import GPTConfig
from forger.model.gpt import GPT

CFG = GPTConfig(vocab_size=4096, d_model=256, n_heads=4, n_layers=4, context_length=512)


def main() -> int:
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    params = count_params(model)
    B, T = 2, 512
    x = torch.randint(0, CFG.vocab_size, (B, T))

    with torch.inference_mode():
        model(x)
        n = 10
        start = time.monotonic()
        for _ in range(n):
            model(x)
        elapsed = (time.monotonic() - start) / n
    tokens_per_sec = B * T / elapsed
    md = [
        "# Benchmark: Model Forward - Day 3 (2026-08-22)",
        "",
        "## Setup",
        f"- Config: {CFG.to_dict()}",
        f"- Params: {params:,}",
        "- Device: CPU (torch 2.13.0, Windows)",
        f"- Input: batch {B}, seq {T}",
        f"- Runs: {n}, averaged",
        "",
        "## Results",
        "| Metric | Value |",
        "|---|---|",
        f"| forward time | {elapsed * 1000:.1f} ms |",
        f"| throughput | {tokens_per_sec:,.0f} tok/s |",
        f"| params | {params:,} |",
        "",
        "## Notes",
        "- Tied embedding/head, RMSNorm, RoPE, SwiGLU, sdpa.",
        "- KV-cache stub validated: chunked decode == full forward (test_kv_cache_stub_matches_full_forward).",
        "",
    ]
    Path("benchmarks/model_forward.md").write_text("\n".join(md), encoding="utf-8")
    print(f"params={params:,} forward={elapsed*1000:.1f}ms throughput={tokens_per_sec:,.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())