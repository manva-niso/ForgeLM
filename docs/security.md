# Security Notes — ForgeLM (Day 12, 2026-09-01)

## Dependency / supply chain
- `uv run pip-audit` (2026-09-01): **No known vulnerabilities found**
  (152 packages audited; `forge-lm` itself skipped - not on PyPI).
- `uv.lock` pins the full environment (reproducible installs).
- CI runs ruff + pytest on every push; the GHCR image is built from the
  locked environment.

## API hardening (shipped Day 11)
| Control | Status |
|---|---|
| Bearer-token auth (`FORGE_LM_API_TOKEN`) | implemented; disabled only when unset (dev mode) |
| Per-IP rate limit (slowapi, default 10/min) | implemented + tested (429) |
| Input validation (Pydantic: prompt 1..2048, max_tokens 1..256, temp 0..2, top_k 1..100) | implemented + tested (422) |
| Request IDs (X-Request-ID) + request logging | implemented |
| CORS | permissive by default (public demo) |
| Health/ready probes | /healthz, /readyz |

## Model/data security
- No secrets in the repo: HF token lives only in env (User-level) + Kaggle
  secrets; never committed (verified in `.gitignore` + commit history policy).
- Personal/study docs are gitignored and never pushed.
- Prompt injection: the model is a single-turn story generator; the public
  demo accepts free text. No system-prompt separation exists — for any
  future chat use, add instruction-hierarchy handling. Documented as a
  known limitation (not exploitable in this single-turn scope).
- Output safety: no content filter (out of scope; TinyStories domain is
  child-safe).

## Review checklist for releases
- [ ] `uv run pip-audit` clean
- [ ] `uv run ruff check .` clean
- [ ] `uv run pytest` green
- [ ] No secrets in diff (`git diff | Select-String hf_`)
- [ ] Docker image builds (CI)
- [ ] Personal docs still gitignored