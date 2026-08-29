"""ForgeLM OpenAI-compatible serving API.

Endpoints:
  POST /v1/completions   generate text (bearer auth + rate limited)
  GET  /v1/models        list available models
  GET  /healthz          liveness
  GET  /readyz           readiness (model loaded)
  GET  /metrics          Prometheus metrics

Run:
  uv run uvicorn forger.serve.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import Response

from forger.serve.engine import Engine

log = logging.getLogger("forgelm.api")

MODEL_ID = "forgelm-sft-story"
DEFAULT_CKPT = os.environ.get("FORGE_LM_CKPT", "models/forgelm-sft-story")
DEFAULT_TOKENIZER = os.environ.get("FORGE_LM_TOKENIZER", "artifacts/tokenizer")
RATE_LIMIT = os.environ.get("FORGE_LM_RATE_LIMIT", "10/minute")

REQUESTS = Counter("forgelm_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("forgelm_completion_latency_seconds", "Completion latency", ["model"])
GENERATED_TOKENS = Counter("forgelm_generated_tokens_total", "Tokens generated", ["model"])
MODEL_LOADED = Gauge("forgelm_model_loaded", "1 if the model engine is loaded")

_engine: Engine | None = None
_engine_lock = __import__("threading").Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                start = time.monotonic()
                _engine = Engine.from_checkpoint(DEFAULT_CKPT, DEFAULT_TOKENIZER)
                MODEL_LOADED.set(1)
                log.info("engine loaded in %.1fs (%s)", time.monotonic() - start, DEFAULT_CKPT)
    return _engine


limiter = Limiter(key_func=get_remote_address)


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str = Field(min_length=1, max_length=2048)
    max_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    seed: int | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def require_auth(authorization: str | None = Header(default=None)) -> str:
    token = os.environ.get("FORGE_LM_API_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return authorization or ""


def create_app(
    ckpt: str | None = None,
    tokenizer_dir: str | None = None,
    rate_limit: str | None = None,
) -> FastAPI:
    global DEFAULT_CKPT, DEFAULT_TOKENIZER, RATE_LIMIT
    if ckpt:
        DEFAULT_CKPT = ckpt
    if tokenizer_dir:
        DEFAULT_TOKENIZER = tokenizer_dir
    if rate_limit:
        RATE_LIMIT = rate_limit

    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI(title="ForgeLM", version="0.5.0")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        start = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        REQUESTS.labels(request.method, request.url.path, response.status_code).inc()
        log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code,
                 (time.monotonic() - start) * 1000)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        try:
            get_engine()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"model not ready: {exc}") from exc
        return {"status": "ready", "model": MODEL_ID}

    @app.get("/v1/models")
    def models(auth: str = Depends(require_auth)) -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "forgelm"}],
        }

    @app.post("/v1/completions")
    @limiter.limit(RATE_LIMIT)
    def completions(
        request: Request,
        body: CompletionRequest,
        auth: str = Depends(require_auth),
    ) -> CompletionResponse:
        engine = get_engine()
        start = time.monotonic()
        text, ids, stats = engine.generate(
            body.prompt,
            max_tokens=body.max_tokens,
            top_k=body.top_k,
            temperature=body.temperature,
            seed=body.seed,
        )
        elapsed = time.monotonic() - start
        LATENCY.labels(body.model).observe(elapsed)
        prompt_tokens = len(ids) - stats["generated_tokens"]
        GENERATED_TOKENS.labels(body.model).inc(stats["generated_tokens"])
        log.info("completion %d tokens in %.1fms", stats["generated_tokens"], elapsed * 1000)
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=body.model,
            choices=[Choice(text=text[len(body.prompt) :], finish_reason="length" if stats["context_limited"] else "stop")],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=stats["generated_tokens"],
                total_tokens=len(ids),
            ),
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()