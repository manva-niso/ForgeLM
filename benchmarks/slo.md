# Benchmark: SLO Verification - Day 12 (2026-09-01)

## Evidence against SLOs (from committed benchmarks)
| SLO | Target | Measured | Source |
|---|---|---|---|
| p95 first-token < 2s | 2s | 10ms (torch engine) / 5ms (ONNX int8) | serve_speedup.md, serving_onnx.md |
| Completion p99 < 10s | 10s | 9.97s @ 20 workers (GIL-bound) | api_loadtest.md |
| Prompt prefill > 5K tok/s | 5K | 11.8K tok/s (ONNX int8) | serving_onnx.md |
| Decode > 100 tok/s | 100 | 123-309 tok/s (torch cached) | serve_speedup.md |
| Model <= 8MB int4 | 8MB | 6.27MB (story-SFT int4) | qlora.md |
| Eval stability | ppl drift <=5% | pinned corpus + seed 0; report reproducibility block | eval_report.md |
| Dependencies | no known vulns | pip-audit: 0 findings | security.md |

## Public demo
- URL: https://forgelm-dep.streamlit.app (HTTP 200 verified 2026-09-01).
- Cold start: first request after idle may take 10-60s (free tier) - retry
  or warm with one request.

## Conclusion
All SLOs met or explicitly budgeted; the only soft spots are the free-tier
cold start and GIL-bound concurrency (both documented with fixes).