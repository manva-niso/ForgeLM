# Runbook — ForgeLM (Day 12, 2026-09-01)

## Deploy (new version)

1. Merge feature work into `main` (tests + lint green, `docs/changelog.md` appended).
2. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z` — CI builds the GHCR image.
3. Public demo (Streamlit Cloud): push to `main` redeploys automatically
   (Streamlit watches the branch). Verify: open the app URL, run one prompt.
4. API (self-hosted): `docker run -p 8000:8000 ghcr.io/manva-niso/forge-lm:<tag>`
   or `uv run uvicorn forger.serve.api:app --port 8000`.
5. Verify: `curl localhost:8000/healthz` then one `/v1/completions` call.

## Rollback

- Streamlit Cloud: Settings → Rebuild (or revert the git commit and push —
  Streamlit redeploys the previous state automatically).
- API: run the previous image tag / `git checkout <prev-tag>` + redeploy.
- Model: weights are versioned in `models/` — point `FORGE_LM_CKPT` at the
  previous model dir. Never delete old model dirs (see MODEL_UPDATE_GUIDE.md).

## On-call checklist (demo-level SLOs)

| Symptom | Check | Fix |
|---|---|---|
| App won't load | Streamlit Cloud status page | Rebuild from UI; confirm `main` is green |
| Story generation hangs | model cold start (first request ~10-60s) | retry; warm with one request |
| Garbage output | prompt outside story domain | expected for tiny model — not an outage |
| API 429 | rate limit hit (10/min/IP) | raise `FORGE_LM_RATE_LIMIT` for the deployment |
| API 401 | wrong/missing token | set `FORGE_LM_API_TOKEN` env |
| p99 latency high | CPU single-worker + GIL (documented) | `uvicorn --workers N` or batch |

## Data/model regeneration (rebuild from scratch)

Follow `docs/MODEL_UPDATE_GUIDE.md` — the full upgrade checklist is there.
Every artifact is reproducible from `uv.lock` + committed configs + pinned
eval corpus.