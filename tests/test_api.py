import os

import pytest
from fastapi.testclient import TestClient

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.serve.api import create_app
from forger.serve.engine import Engine
from forger.tokenizer.bpe import BPETokenizer

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm and the birds sang",
    "once upon a time there was a little girl who loved her dog and they played in the park every day",
] * 6
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)


@pytest.fixture(scope="module")
def client():
    torch_ = __import__("torch")
    torch_.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32))
    model.eval()
    engine = Engine(model, TOKENIZER)
    app = create_app(rate_limit="1000/minute")
    app.dependency_overrides = {}
    from forger.serve import api as api_mod

    api_mod._engine = engine
    api_mod.MODEL_LOADED.set(1)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client():
    torch_ = __import__("torch")
    torch_.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32))
    model.eval()
    engine = Engine(model, TOKENIZER)
    app = create_app(rate_limit="1000/minute")
    from forger.serve import api as api_mod

    api_mod._engine = engine
    api_mod.MODEL_LOADED.set(1)
    os.environ["FORGE_LM_API_TOKEN"] = "test-token"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        del os.environ["FORGE_LM_API_TOKEN"]


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_after_load(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["model"] == "forgelm-sft-story"


def test_completions_happy(client):
    r = client.post("/v1/completions", json={"prompt": "the cat sat"})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"]
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert "X-Request-ID" in r.headers


def test_completions_malformed(client):
    r = client.post("/v1/completions", json={"max_tokens": 10})
    assert r.status_code == 422
    r = client.post("/v1/completions", json={"prompt": ""})
    assert r.status_code == 422


def test_completions_bad_params(client):
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0})
    assert r.status_code == 422


def test_auth_required(auth_client):
    r = auth_client.post("/v1/completions", json={"prompt": "the cat"})
    assert r.status_code == 401
    r = auth_client.post(
        "/v1/completions", json={"prompt": "the cat"}, headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401
    r = auth_client.post(
        "/v1/completions", json={"prompt": "the cat"}, headers={"Authorization": "Bearer test-token"}
    )
    assert r.status_code == 200


def test_rate_limit(client):
    from forger.serve import api as api_mod

    app = create_app(rate_limit="3/minute")
    api_mod.MODEL_LOADED.set(1)
    with TestClient(app) as c:
        for _ in range(3):
            r = c.post("/v1/completions", json={"prompt": "the cat"})
            assert r.status_code == 200
        r = c.post("/v1/completions", json={"prompt": "the cat"})
        assert r.status_code == 429


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "forgelm-sft-story"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "forgelm_http_requests_total" in r.text
    assert "forgelm_completion_latency_seconds" in r.text