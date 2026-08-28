# ADR-07: Inference Engine

Status: accepted
Date: 2026-08-29

## Context
Day 7 requires a fast, correct generation engine + optimization variants.

## Decision
- `forger/serve/engine.py`: prefill / decode_next / generate over the proven
  KV-cache; shares samplers with `forger/eval/generation.py`; `smoke_infer.py`
  refactored onto it.
- `forger/serve/optimize.py`: dynamic int8 quantization (Linear -> qint8) and
  torch.compile with graceful fallback.

## Benchmark (CPU, torch 2.13, Windows)
| Variant | size | first-token | per-token | tok/s |
|---|---|---|---|---|
| fp32 | 20.0 MB | 10.0 ms | 8.1 ms | 123 |
| int8 | 4.0 MB | 15.7 ms | 9.9 ms | 101 |
| compile | n/a | unavailable (no MSVC `cl.exe` on this machine) |

- Product targets already met: first-token 10ms (<100ms), int8 size 4MB (<=8MB).
- int8 did NOT speed up decode on this hardware (quantized kernel overhead on
  small matrices) - its win is 5x size reduction. Speedup path is Day 10
  (ONNX Runtime + int4 QLoRA), not dynamic int8.
- torch.compile requires a C++ compiler on Windows; skipped gracefully with a
  clear status in the report. CI/linux would compile; not pursued here.

## Notes
- torch.ao.quantization is deprecated (torch 2.10+); Day 9/10 replaces it with
  our own int4 path anyway - documented, not chased.