import math

import pytest
import torch

from forger.eval.generation import generate, sample_token, top_k_filter
from forger.eval.metrics import distinct_n, length_sanity, repetition_rate
from forger.eval.perplexity import evaluate_perplexity
from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm",
    "once upon a time there was a little girl who loved her dog and they played in the park",
    "the sun was warm and the birds sang in the trees all day long while the children played",
] * 3
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)
CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)


def test_perplexity_matches_manual():
    model = GPT(CFG)
    model.eval()
    text = "the cat sat on the mat"
    ids = TOKENIZER.encode(text)
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    logits, _ = model(x)
    nll = torch.nn.functional.cross_entropy(logits.view(-1, CFG.vocab_size), y.view(-1)).item()
    expected_ppl = math.exp(nll)
    result = evaluate_perplexity(model, TOKENIZER, [text], context_length=CFG.context_length)
    assert result["perplexity"] == pytest.approx(expected_ppl, rel=1e-5)


def test_bits_per_byte_math():
    model = GPT(CFG)
    model.eval()
    text = "hello world"
    result = evaluate_perplexity(model, TOKENIZER, [text], context_length=32)
    ids = TOKENIZER.encode(text)
    n_bytes = len(text.encode("utf-8"))
    expected_bpb = math.log(result["perplexity"]) / math.log(2) * (len(ids) - 1) / n_bytes
    assert result["bits_per_byte"] == pytest.approx(expected_bpb, rel=1e-5)


def test_sliding_window_long_text():
    model = GPT(CFG)
    model.eval()
    text = " ".join(CORPUS) * 5
    full = evaluate_perplexity(model, TOKENIZER, [text], context_length=32, stride=16)
    assert full["tokens"] > 100
    assert full["perplexity"] > 0


def test_top_k_filter():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = top_k_filter(logits, 2)
    assert torch.isinf(filtered[0, 0]).item() and torch.isinf(filtered[0, 1]).item()
    assert torch.isfinite(filtered[0, -2:]).all().item()


def test_sample_token_valid():
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    for _ in range(10):
        token = sample_token(logits, temperature=1.0)
        assert token == 2
    token = sample_token(logits, top_k=1)
    assert token == 2


def test_generate_deterministic_with_seed():
    model = GPT(CFG)
    model.eval()
    ids_a, _ = generate(model, TOKENIZER, "the cat", max_tokens=20, seed=7)
    ids_b, _ = generate(model, TOKENIZER, "the cat", max_tokens=20, seed=7)
    assert ids_a == ids_b


def test_generate_stops_at_special_token():
    model = GPT(CFG)
    model.eval()
    _, stats = generate(model, TOKENIZER, "the cat", max_tokens=64)
    assert stats["generated_tokens"] >= 0


def test_distinct_n():
    assert distinct_n([1, 2, 3, 4], 2) == 1.0
    assert distinct_n([1, 2], 3) == 0.0
    assert distinct_n([1, 1, 1, 1], 2) == pytest.approx(1 / 3)


def test_repetition_rate():
    assert repetition_rate([1, 2, 3, 1, 2, 3], 3) == pytest.approx(1 / 4)
    assert repetition_rate([1, 2, 3, 4, 5, 6], 3) == 0.0


def test_length_sanity():
    assert length_sanity([1, 2, 3], 5) == {"tokens": 3, "hit_cap": False}
    assert length_sanity([1, 2, 3, 4, 5], 5) == {"tokens": 5, "hit_cap": True}