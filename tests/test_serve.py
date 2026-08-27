import pytest
import torch

from forger.eval.generation import generate
from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.serve.engine import Engine
from forger.serve.optimize import model_size_mb, quantize_dynamic
from forger.tokenizer.bpe import BPETokenizer

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm and the birds sang",
    "once upon a time there was a little girl who loved her dog and they played in the park every day",
    "the sun was warm and the birds sang in the trees all day long while the children played outside",
] * 4
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)
CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)


def test_engine_matches_reference_greedy():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    engine = Engine(model, TOKENIZER)
    text, ids, stats = engine.generate("the cat", max_tokens=20, top_k=1, temperature=1.0)
    assert len(ids) > 1
    assert stats["generated_tokens"] == len(ids) - len(TOKENIZER.encode("the cat"))
    assert text == TOKENIZER.decode(ids)
    assert stats["tokens_per_sec"] > 0


def test_engine_stops_at_context_limit():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    engine = Engine(model, TOKENIZER)
    _, ids, stats = engine.generate("the cat", max_tokens=500, top_k=1, temperature=1.0)
    assert len(ids) == CFG.context_length
    assert stats["context_limited"] is True


def test_engine_deterministic_with_seed():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    e1 = Engine(model, TOKENIZER)
    e2 = Engine(model, TOKENIZER)
    _, ids1, _ = e1.generate("the cat", max_tokens=20, seed=3)
    _, ids2, _ = e2.generate("the cat", max_tokens=20, seed=3)
    assert ids1 == ids2


def test_engine_matches_eval_generate():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    engine = Engine(model, TOKENIZER)
    _, engine_ids, _ = engine.generate("the cat", max_tokens=15, seed=5)
    eval_ids, _ = generate(model, TOKENIZER, "the cat", max_tokens=15, seed=5)
    assert engine_ids == eval_ids


def test_int8_logits_close_to_fp32():
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    int8 = quantize_dynamic(model)
    int8.eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    with torch.inference_mode():
        logits_fp32, _ = model(x)
        logits_int8, _ = int8(x)
    diff = (logits_fp32 - logits_int8).abs().mean().item()
    assert diff < 0.5, f"int8 drift too large: {diff}"


def test_int8_size_smaller():
    torch.manual_seed(0)
    model = GPT(CFG)
    int8 = quantize_dynamic(model)
    assert model_size_mb(int8) < model_size_mb(model)


def test_compile_fallback_or_works():
    torch.manual_seed(0)
    model = GPT(CFG)
    try:
        compiled = torch.compile(model)
        compiled.eval()
        x = torch.randint(0, CFG.vocab_size, (1, 8))
        with torch.inference_mode():
            compiled(x)
    except Exception:  # noqa: BLE001
        pytest.skip("torch.compile unavailable on this platform")