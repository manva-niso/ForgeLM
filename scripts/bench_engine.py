"""Benchmark inference variants -> benchmarks/serve_speedup.md + JSON.

Protocol: 64-token prompt (prefill) + 50 generated tokens, repeated 3x,
variants: eager fp32 / int8 dynamic / torch.compile.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from forger.model.checkpoint import load_model_from_checkpoint
from forger.serve.engine import Engine
from forger.serve.optimize import compile_model, model_size_mb, quantize_dynamic
from forger.tokenizer.bpe import BPETokenizer

CKPT = "models/forgelm-baseline"
PROMPT_TEXT = (
    "Once upon a time there was a little cat named Tom. Tom loved to play with "
    "his toy cat in the garden behind the house every single day. "
    "One morning, Tom woke up and saw that the sun was shining brightly "
    "through the window and the birds were singing in the tall green trees."
)
MAX_TOKENS = 50
REPEATS = 3


def time_engine(engine: Engine, prompt_ids: list[int]) -> dict[str, float]:
    prefill_times = []
    first_tokens = []
    per_token_times = []
    total_tokens = 0
    with torch.inference_mode():
        for _ in range(REPEATS):
            engine.reset()
            t0 = time.monotonic()
            engine.prefill(prompt_ids)
            prefill_times.append(time.monotonic() - t0)

            t0 = time.monotonic()
            engine.decode_next(prompt_ids[-1])
            first_tokens.append(time.monotonic() - t0)

            t0 = time.monotonic()
            ids = list(prompt_ids)
            while len(ids) - len(prompt_ids) < MAX_TOKENS and len(ids) < engine.model.config.context_length:
                nid = engine.decode_next(ids[-1])
                ids.append(nid)
            per_token_times.append(time.monotonic() - t0)
            total_tokens += len(ids) - len(prompt_ids)
    avg_prefill = sum(prefill_times) / REPEATS
    avg_first = sum(first_tokens) / REPEATS
    avg_decode = sum(per_token_times) / REPEATS
    return {
        "prefill_ms": avg_prefill * 1000,
        "prompt_tokens_per_sec": len(prompt_ids) / avg_prefill,
        "first_token_ms": avg_first * 1000,
        "decode_loop_ms": avg_decode * 1000,
        "per_token_ms": avg_decode / MAX_TOKENS * 1000,
        "tokens_per_sec": total_tokens / sum(per_token_times),
    }


def main() -> int:
    tokenizer = BPETokenizer.load("artifacts/tokenizer")
    prompt_ids = tokenizer.encode(PROMPT_TEXT)[:64]
    print(f"prompt tokens: {len(prompt_ids)}")

    results = {}
    base_model = load_model_from_checkpoint(CKPT)
    size_fp32 = model_size_mb(base_model)
    results["fp32"] = time_engine(Engine(base_model, tokenizer), prompt_ids)
    results["fp32"]["size_mb"] = size_fp32

    int8_model = quantize_dynamic(base_model)
    results["int8"] = time_engine(Engine(int8_model, tokenizer), prompt_ids)
    results["int8"]["size_mb"] = model_size_mb(int8_model)

    compiled_model, status = compile_model(base_model)
    try:
        results["compile"] = time_engine(Engine(compiled_model, tokenizer), prompt_ids)
        results["compile"]["size_mb"] = size_fp32
        results["compile"]["status"] = status
    except Exception as exc:  # noqa: BLE001
        results["compile"] = {"status": f"failed: {exc}", "size_mb": size_fp32}

    base_per_token = results["fp32"]["per_token_ms"]
    md = [
        "# Benchmark: Optimized Serving - Day 7 (2026-08-22)",
        "",
        "## Setup",
        f"- Checkpoint: {CKPT} (best eval 2.0247)",
        f"- Prompt: {len(prompt_ids)} tokens prefill, {MAX_TOKENS} tokens decode, {REPEATS} repeats",
        "- Device: CPU (torch 2.13.0, Windows)",
        "",
        "| Variant | size MB | prefill ms | prompt tok/s | first-token ms | per-token ms | tok/s | vs fp32 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if r.get("status", "").startswith("failed"):
            md.append(f"| {name} | - | - | - | - | - | - | unavailable: {r['status']} |")
            continue
        ratio = "1.0x" if name == "fp32" else f"{r['per_token_ms'] / base_per_token:.2f}x"
        md.append(
            f"| {name} | {r['size_mb']:.1f} | {r['prefill_ms']:.1f} | {r['prompt_tokens_per_sec']:.0f} "
            f"| {r['first_token_ms']:.1f} | {r['per_token_ms']:.1f} | {r['tokens_per_sec']:.0f} | {ratio} |"
        )
    md.append("")
    if results["compile"].get("status") != "compiled":
        md.append(f"Note: torch.compile unavailable - {results['compile']['status']}")
    md.append("")
    Path("benchmarks/serve_speedup.md").write_text("\n".join(md), encoding="utf-8")
    Path("benchmarks/serve_speedup.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, r in results.items():
        if r.get("status", "").startswith("failed"):
            print(f"{name}: unavailable ({r['status'][:60]}...)")
            continue
        print(f"{name}: first-token {r['first_token_ms']:.1f}ms, per-token {r['per_token_ms']:.2f}ms, {r['tokens_per_sec']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())