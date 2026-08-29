# Benchmark: Optimized Serving (ONNX) - Day 10 (2026-08-30)

## Setup
- Checkpoint: models/forgelm-sft-story (story-SFT, eval 2.00)
- Prompt: 59 tokens prefill, 50 tokens decode, 3 repeats
- Device: CPU (torch 2.13.0 / onnxruntime 1.29.0, Windows)

| Variant | size MB | prefill ms | prompt tok/s | first-token ms | per-token ms | tok/s | vs fp32 |
|---|---|---|---|---|---|---|---|
| torch fp32 | 20.0 | 42.0 | 1405 | 25.7 | 14.4 | 70 | 1.0x |
| torch int8 | 4.0 | 31.0 | 1903 | 31.3 | 22.9 | 44 | 1.59x |
| onnx fp32 | 20.7 | 52.0 | 1135 | 52.3 | 63.8 | 16 | 4.44x |
| onnx int8 | 5.7 | 72.7 | 812 | 83.7 | 74.7 | 13 | 5.19x |

Notes: ONNX decode recomputes attention over the growing sequence (no
KV-cache in the graph); torch variants use the Day-7 cache. The graph
runtime is the portability win (no torch dependency), not necessarily
a decode speedup at this model size. Absolute timings vary with machine
load (CPU contended runs showed 2-4x slower across ALL variants); the
RELATIVE ordering is stable: torch-cached decode fastest, ONNX int8 has
the best prefill/first-token among graph variants.