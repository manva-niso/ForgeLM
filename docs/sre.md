# SRE — Service Level Objectives (Day 12, 2026-09-01)

## Objectives (demo-tier, honest for a free single-CPU service)

| SLO | Target | Measured evidence |
|---|---|---|
| Availability (public demo responds) | 99% monthly | Streamlit Cloud status; local runs 200/200 |
| First-token latency (single request) | p95 < 2s | Day-7/10 engine: 10-26ms first token + generation |
| Completion p99 (low concurrency) | < 10s | Load test: p99 9.97s at 20 workers; single-request ~0.3-1s |
| Prompt prefill throughput (ONNX int8) | > 5K tok/s | 11.8K tok/s measured (Day 10) |
| Decode throughput (torch cached engine) | > 100 tok/s | 123-309 tok/s measured (Days 7/10) |
| Model size | <= 8MB (int4) | 6.27MB story-SFT int4 |
| Evaluation stability | ppl drift <= 5% between runs | pinned eval corpus + seed 0 + reproducibility block |

## Error budget
Availability 99% → 7.3h/month allowed downtime. The public demo is a free
tier (Streamlit); treat it as best-effort: cold starts and rebuilds count
against the budget. On-call: owner only.

## Dashboards / metrics (from /metrics)
- `forgelm_http_requests_total{method,path,status}`
- `forgelm_completion_latency_seconds` (histogram → p50/p95/p99)
- `forgelm_generated_tokens_total{model}`
- `forgelm_model_loaded` (gauge)
- SlowAPI rate-limit counters (429s appear in request totals)

## Alerts (prometheus text-file style)
- p99 completion latency > 10s sustained → check worker count / CPU.
- 429 share > 5% of completions → rate limit too low for traffic.
- `forgelm_model_loaded == 0` for > 60s → engine failed to load (check
  checkpoint path / memory).

## Release discipline
- Every release: tests green, pip-audit clean, changelog entry, tag, CI
  image, demo verified. Rollback = previous tag (runbook).