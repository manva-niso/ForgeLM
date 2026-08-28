import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.quant.quantize import (
    Int4Linear,
    dequantize_4bit,
    export_4bit,
    load_4bit,
    pack_4bit,
    quantize_4bit,
    quantize_model_4bit,
    storage_size_mb,
    unpack_4bit,
)

CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)


def test_quant_roundtrip_error_small():
    torch.manual_seed(0)
    t = torch.randn(256, 64) * 0.05
    codes, scale = quantize_4bit(t)
    back = dequantize_4bit(codes, scale, t.shape)
    rel_err = (back - t).abs().mean() / t.abs().mean()
    assert rel_err < 0.15, f"rel err {rel_err}"


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    codes = torch.randint(-8, 8, (100,), dtype=torch.int8)
    packed = pack_4bit(codes)
    assert packed.numel() == 50
    restored = unpack_4bit(packed, 100)
    assert torch.equal(restored, codes)


def test_int4_linear_matches_linear():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 128, bias=True)
    int4 = Int4Linear(lin.weight, lin.bias)
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out_lin = lin(x)
        out_int4 = int4(x)
    assert (out_lin - out_int4).abs().mean().item() < 0.5


def test_quantize_model_replaces_linears():
    torch.manual_seed(0)
    model = GPT(CFG)
    n = quantize_model_4bit(model)
    assert n > 0
    assert not any(isinstance(m, torch.nn.Linear) and not m is model.lm_head for m in model.modules())
    assert isinstance(model.lm_head, torch.nn.Linear)


def test_storage_size_reduction():
    torch.manual_seed(0)
    model = GPT(CFG)
    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    quantize_model_4bit(model)
    int4_bytes = storage_size_mb(model) * 1024 * 1024
    assert int4_bytes < fp32_bytes / 2


def test_export_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = GPT(CFG)
    quantize_model_4bit(model)
    model.eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    with torch.inference_mode():
        logits_before, _ = model(x)
    export_4bit(model, tmp_path)
    loaded = load_4bit(tmp_path)
    loaded.eval()
    with torch.inference_mode():
        logits_after, _ = loaded(x)
    diff = (logits_before - logits_after).abs().mean().item()
    assert diff < 0.05, f"export/load drift {diff}"