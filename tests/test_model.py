import pytest
import torch

from forger.model.blocks import count_params
from forger.model.config import GPTConfig
from forger.model.gpt import GPT

CFG = GPTConfig(vocab_size=4096, d_model=256, n_heads=4, n_layers=4, context_length=512)


def test_config_rejects_bad_dims():
    with pytest.raises(ValueError):
        GPTConfig(d_model=256, n_heads=3)
    with pytest.raises(ValueError):
        GPTConfig(d_model=0)
    with pytest.raises(ValueError):
        GPTConfig(n_layers=0)


def test_forward_shape():
    model = GPT(CFG)
    model.eval()
    x = torch.randint(0, CFG.vocab_size, (2, 32))
    logits, _ = model(x)
    assert logits.shape == (2, 32, CFG.vocab_size)


def test_causal_no_peek():
    model = GPT(CFG)
    model.eval()
    torch.manual_seed(0)
    base = torch.randint(0, CFG.vocab_size, (1, 16))
    later = base.clone()
    later[0, 8:] = 0
    logits_base, _ = model(base)
    logits_later, _ = model(later)
    assert torch.allclose(logits_base[0, :8], logits_later[0, :8], atol=1e-6)


def test_gradients_flow_and_finite():
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 16))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, CFG.vocab_size), x.view(-1))
    loss.backward()
    named = list(model.named_parameters())
    assert len(named) > 0
    for name, param in named:
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_deterministic_forward():
    torch.manual_seed(42)
    model_a = GPT(CFG)
    torch.manual_seed(42)
    model_b = GPT(CFG)
    model_a.eval()
    model_b.eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    logits_a, _ = model_a(x)
    logits_b, _ = model_b(x)
    assert torch.equal(logits_a, logits_b)


def test_tied_embeddings():
    model = GPT(CFG)
    assert model.token_embedding.weight is model.lm_head.weight


def test_param_count():
    model = GPT(CFG)
    n = count_params(model)
    assert n > 0
    expected = (
        CFG.vocab_size * CFG.d_model
        + CFG.n_layers
        * (
            3 * CFG.d_model * CFG.d_model
            + 3 * CFG.d_model
            + CFG.d_model * CFG.d_model
            + CFG.d_model
            + 2 * CFG.d_model * (CFG.ffn_mult * CFG.d_model)
            + CFG.ffn_mult * CFG.d_model * CFG.d_model
            + 2 * CFG.d_model
        )
        + CFG.d_model
    )
    assert n == expected, f"param count {n} != expected {expected}"


def test_context_length_enforced():
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.context_length + 1))
    with pytest.raises(ValueError):
        model(x)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = GPT(CFG)
    model.eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    logits_before, _ = model(x)
    model.save(tmp_path)
    loaded = GPT.load(tmp_path)
    loaded.eval()
    logits_after, _ = loaded(x)
    assert torch.equal(logits_before, logits_after)
    assert loaded.config == CFG


def test_kv_cache_stub_matches_full_forward():
    torch.manual_seed(3)
    model = GPT(CFG)
    model.eval()
    prompt = torch.randint(0, CFG.vocab_size, (1, 10))
    full_logits, _ = model(prompt)

    _, cache = model(prompt[:, :5], cache={})
    assert cache is not None
    logits_next, cache = model(prompt[:, 5:6], cache=cache)
    assert torch.allclose(full_logits[0, 5], logits_next[0, 0], atol=1e-5)
    logits_last, _ = model(prompt[:, 6:], cache=cache)
    assert torch.allclose(full_logits[0, 6:], logits_last[0], atol=1e-5)