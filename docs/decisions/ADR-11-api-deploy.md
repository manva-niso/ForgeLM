# ADR-11: API + Deployment

Status: accepted
Date: 2026-08-31

## Context
Day 11: expose the model as a hardened, deployable HTTP service.

## Decision
- `forger/serve/api.py`: FastAPI app factory `create_app()`.
  - POST /v1/completions — OpenAI-compatible (model, prompt, max_tokens,
    temperature, top_k, seed) -> {id, choices[], usage}; validated via Pydantic
    (prompt 1..2048, max_tokens 1..256, temp 0..2, top_k 1..100).
  - GET /v1/models, /healthz (liveness), /readyz (engine loaded, else 503),
    /metrics (Prometheus: request counter, latency histogram, token counter,
    model-loaded gauge).
  - Auth: bearer token from FORGE_LM_API_TOKEN env; disabled when unset
    (dev mode).
  - Rate limit: slowapi, per-IP, FORGE_LM_RATE_LIMIT (default 10/minute),
    per-app limiter (test isolation).
  - X-Request-ID middleware + request logging.
- Engine: lazy thread-safe singleton (models/forgelm-sft-story by default,
  FORGE_LM_CKPT override).
- Dockerfile: single-stage slim image (uv sync --no-dev, non-root user,
  healthcheck) — multi-stage rejected: the runtime needs the venv + weights,
  slim keeps it honest and small enough; CI builds + pushes to GHCR
  (build.yml; GITHUB_TOKEN has package:write).
- Load test: `scripts/load_test.py` (threaded concurrency) ->
  benchmarks/api_loadtest.md.

## Results (local CPU, uvicorn 1 worker)
- Functional: 200/200 requests OK; story completion works (40 tokens);
  auth 401 / malformed 422 / rate-limit 429 all tested (10 API tests).
- p50 9.2s / p99 9.9s at 20 concurrent workers — GIL contention (CPU torch
  holds the GIL; threads serialize). Single-request latency ~0.3-1s.
  Improvement paths documented: uvicorn --workers N or async batching.
- 82 tests green.

## Consequences
The service is deployable as a container (GHCR) or directly
(uvicorn forger.serve.api:app). Deploy target (Fly.io/Render) is a manual
step with account credentials - documented in kaggle-notebook.md (local doc)
and the README.