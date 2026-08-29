# ADR-10: ONNX Optimized Serving

Status: accepted
Date: 2026-08-30

## Context
Day 10: the "real speedup" path promised in ADR-07 (torch.compile needs MSVC).

## Decision
- `forger/serve/onnx_engine.py`: export the GPT's full forward via
  `torch.onnx.export` (dynamo exporter; needs `onnxscript` dep) with dynamic
  batch/seq axes; `ORTEngine` generates by recomputing over the growing
  sequence (no KV-cache in the graph - correct and portable); int8 variant via
  `onnxruntime.quantization.quantize_dynamic` (QInt8, per-channel).
- Benchmark protocol matches Day 7 (59-token prompt, 50 decode, 3 repeats,
  4 variants).

## Results (CPU, story-SFT checkpoint)
| Variant | size | first-token | per-token | prompt tok/s |
|---|---|---|---|---|
| torch fp32 (cached) | 20.0 MB | ~0-26 ms* | 3.2 ms (fastest) | ~3.8K |
| torch int8 (cached) | 4.0 MB | ~15 ms | 7.5 ms | ~3.7K |
| onnx fp32 | 20.7 MB | ~16-52 ms* | 9.5 ms | ~5.7K |
| onnx int8 | 5.7 MB | ~5 ms | 9.2 ms | ~11.8K |

(*absolute timings vary with machine load; ordering stable)

- Decode: torch cached engine remains fastest (KV-cache > recompute).
- Prefill: ONNX int8 wins (~3x torch) - good for prompt-heavy workloads.
- Size: onnx int8 5.7MB (weights in external .data file; measured correctly).
- The ">=3x decode" target is NOT met by a graph runtime at 5.25M params
  (attention recompute + kernel overhead). The wins are: portability
  (no torch), int8 prefill speed, and 5.7MB artifact. Honest conclusion:
  keep the cached torch engine for interactive decode; use ORT int8 for
  batch/prefill-style serving or embedded deployments.

## Consequences
Deps added: onnx, onnxruntime, onnxscript. Re-export needed after any model
or architecture change (documented in MODEL_UPDATE_GUIDE.md).