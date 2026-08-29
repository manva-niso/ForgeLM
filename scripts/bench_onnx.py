"""Benchmark ONNX/ORT serving variants vs Day-7 torch baselines.

Protocol: 59-token prompt prefill, 50 decode tokens, 3 repeats.
Variants: torch fp32 (engine), torch int8, onnx fp32, onnx int8.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from forger.model.checkpoint import load_model_from_checkpoint
from forger.serve.engine import Engine
from forger.serve.onnx_engine import ORTEngine, export_onnx, quantize_onnx_int8
from forger.serve.optimize import model_size_mb, quantize_dynamic
from forger.tokenizer.bpe import BPETokenizer

CKPT = "models/forgelm-sft-story"
PROMPT_TEXT = (
    "### Instruction: Write a short story about a cat named Tom.\n### Response:"
    " Once upon a time there was a little cat named Tom who loved to play in the"
    " garden behind the house every single day with his toy mouse."
)
MAX_TOKENS = 50
REPEATS = 3


def bench_torch_engine(model, tokenizer, prompt_ids) -> dict:
    engine = Engine(model, tokenizer)
    prefill_t, first_t, decode_t, total_tok = [], [], [], 0
    with torch.inference_mode():
        for _ in range(REPEATS):
            engine.reset()
            t0 = time.monotonic()
            engine.prefill(prompt_ids)
            prefill_t.append(time.monotonic() - t0)
            t0 = time.monotonic()
            engine.decode_next(prompt_ids[-1])
            first_t.append(time.monotonic() - t0)
            t0 = time.monotonic()
            ids = list(prompt_ids)
            while len(ids) - len(prompt_ids) < MAX_TOKENS:
                ids.append(engine.decode_next(ids[-1]))
            decode_t.append(time.monotonic() - t0)
            total_tok += len(ids) - len(prompt_ids)
    return {
        "prefill_ms": sum(prefill_t) / REPEATS * 1000,
        "prompt_tokens_per_sec": len(prompt_ids) / (sum(prefill_t) / REPEATS),
        "first_token_ms": sum(first_t) / REPEATS * 1000,
        "per_token_ms": sum(decode_t) / REPEATS / MAX_TOKENS * 1000,
        "tokens_per_sec": total_tok / sum(decode_t),
    }


def bench_ort(engine: ORTEngine, prompt_ids, tokenizer, prompt_text) -> dict:
    prefill_t, first_t, decode_t, total_tok = [], [], [], 0
    for _ in range(REPEATS):
        t0 = time.monotonic()
        engine._forward(prompt_ids)
        prefill_t.append(time.monotonic() - t0)
        t0 = time.monotonic()
        engine._forward(prompt_ids + [prompt_ids[-1]])
        first_t.append(time.monotonic() - t0)
        t0 = time.monotonic()
        ids = list(prompt_ids)
        while len(ids) - len(prompt_ids) < MAX_TOKENS:
            logits = engine._forward(ids)
            ids.append(int(np.argmax(logits[-1])))
        decode_t.append(time.monotonic() - t0)
        total_tok += len(ids) - len(prompt_ids)
    return {
        "prefill_ms": sum(prefill_t) / REPEATS * 1000,
        "prompt_tokens_per_sec": len(prompt_ids) / (sum(prefill_t) / REPEATS),
        "first_token_ms": sum(first_t) / REPEATS * 1000,
        "per_token_ms": sum(decode_t) / REPEATS / MAX_TOKENS * 1000,
        "tokens_per_sec": total_tok / sum(decode_t),
    }


def onnx_size_mb(path: Path) -> float:
    total = path.stat().st_size
    data = path.with_name(path.name + ".data")
    if data.exists():
        total += data.stat().st_size
    return total / (1024 * 1024)


def main() -> int:
    tokenizer = BPETokenizer.load("artifacts/tokenizer")
    prompt_ids = tokenizer.encode(PROMPT_TEXT)[:59]
    print(f"prompt tokens: {len(prompt_ids)}")

    results = {}
    base = load_model_from_checkpoint(CKPT)
    results["torch fp32"] = bench_torch_engine(base, tokenizer, prompt_ids)
    results["torch fp32"]["size_mb"] = model_size_mb(base)
    int8_model = quantize_dynamic(base)
    results["torch int8"] = bench_torch_engine(int8_model, tokenizer, prompt_ids)
    results["torch int8"]["size_mb"] = model_size_mb(int8_model)

    export_dir = Path("artifacts/onnx")
    onnx_path = export_dir / "model.onnx"
    int8_path = export_dir / "model_int8.onnx"
    export_onnx(base, onnx_path)
    quantize_onnx_int8(onnx_path, int8_path)
    ort_fp32 = ORTEngine(onnx_path, tokenizer)
    ort_int8 = ORTEngine(int8_path, tokenizer)
    results["onnx fp32"] = bench_ort(ort_fp32, prompt_ids, tokenizer, PROMPT_TEXT)
    results["onnx fp32"]["size_mb"] = onnx_size_mb(onnx_path)
    results["onnx int8"] = bench_ort(ort_int8, prompt_ids, tokenizer, PROMPT_TEXT)
    results["onnx int8"]["size_mb"] = onnx_size_mb(int8_path)

    base_per_token = results["torch fp32"]["per_token_ms"]
    md = [
        "# Benchmark: Optimized Serving (ONNX) - Day 10 (2026-08-30)",
        "",
        "## Setup",
        f"- Checkpoint: {CKPT} (story-SFT, eval 2.00)",
        f"- Prompt: {len(prompt_ids)} tokens prefill, {MAX_TOKENS} tokens decode, {REPEATS} repeats",
        "- Device: CPU (torch 2.13.0 / onnxruntime 1.29.0, Windows)",
        "",
        "| Variant | size MB | prefill ms | prompt tok/s | first-token ms | per-token ms | tok/s | vs fp32 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        ratio = "1.0x" if name == "torch fp32" else f"{r['per_token_ms'] / base_per_token:.2f}x"
        md.append(
            f"| {name} | {r['size_mb']:.1f} | {r['prefill_ms']:.1f} | {r['prompt_tokens_per_sec']:.0f} "
            f"| {r['first_token_ms']:.1f} | {r['per_token_ms']:.1f} | {r['tokens_per_sec']:.0f} | {ratio} |"
        )
    md.append("")
    md.append("Notes: ONNX decode recomputes attention over the growing sequence (no")
    md.append("KV-cache in the graph); torch variants use the Day-7 cache. The graph")
    md.append("runtime is the portability win (no torch dependency), not necessarily")
    md.append("a decode speedup at this model size.")
    Path("benchmarks/serving_onnx.md").write_text("\n".join(md), encoding="utf-8")
    for name, r in results.items():
        print(f"{name}: first-token {r['first_token_ms']:.1f}ms, per-token {r['per_token_ms']:.2f}ms, {r['tokens_per_sec']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())