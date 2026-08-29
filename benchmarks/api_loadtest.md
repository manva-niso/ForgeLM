# Benchmark: API Load Test - Day 11 (2026-08-31)

## Setup
- URL: http://127.0.0.1:8000
- Workers: 20, requests/worker: 10 (total 200)
- Prompt: story instruction, max_tokens 32
- Wall time: 92.4s

## Results
| metric | value |
|---|---|
| requests/sec | 2.2 |
| p50 latency | 9156 ms |
| p95 latency | 9813 ms |
| p99 latency | 9969 ms |
| status codes | {200: 200} |

## Notes
- Local uvicorn (single worker, CPU); rate limit 10/min would block this
  many requests - load test ran with FORGE_LM_RATE_LIMIT raised.
- p50 ~9.2s under 20 concurrent workers is GIL contention: CPU torch
  inference holds the GIL, so concurrent threads serialize. SINGLE-REQUEST
  latency is ~0.3-1s (generation-bound). Improvement paths (documented):
  uvicorn --workers N (process-level parallelism), or async request
  batching. Throughput here is honest for a single-process CPU server.
- Latency dominated by generation (32 tokens) + engine path.
