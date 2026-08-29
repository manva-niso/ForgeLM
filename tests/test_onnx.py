import pytest
import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.serve.onnx_engine import ORTEngine, export_onnx, quantize_onnx_int8
from forger.tokenizer.bpe import BPETokenizer

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm and the birds sang",
    "once upon a time there was a little girl who loved her dog and they played in the park every day",
] * 6
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)
CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)


@pytest.fixture(scope="module")
def onnx_session_path(tmp_path_factory):
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx")
    return path


def test_export_creates_valid_session(onnx_session_path):
    engine = ORTEngine(onnx_session_path, TOKENIZER)
    ids = TOKENIZER.encode("the cat sat")
    logits = engine._forward(ids)
    assert logits.shape == (len(ids), CFG.vocab_size)


def test_onnx_matches_torch(onnx_session_path):
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    engine = ORTEngine(onnx_session_path, TOKENIZER)
    ids = TOKENIZER.encode("the cat sat on the mat")
    x = torch.tensor([ids])
    with torch.inference_mode():
        torch_logits, _ = model(x)
    onnx_logits = torch.tensor(engine._forward(ids))
    assert torch.allclose(onnx_logits, torch_logits[0], atol=1e-4)


def test_onnx_generate_produces_text(onnx_session_path):
    engine = ORTEngine(onnx_session_path, TOKENIZER)
    text, ids, stats = engine.generate("the cat", max_tokens=15, top_k=1, temperature=1.0)
    assert len(ids) > len(TOKENIZER.encode("the cat"))
    assert stats["generated_tokens"] > 0
    assert text == TOKENIZER.decode(ids)


def test_onnx_int8_quantization(onnx_session_path, tmp_path):
    int8_path = quantize_onnx_int8(onnx_session_path, tmp_path / "model_int8.onnx")
    engine = ORTEngine(int8_path, TOKENIZER)
    torch.manual_seed(0)
    model = GPT(CFG)
    model.eval()
    ids = TOKENIZER.encode("the cat sat on the mat")
    x = torch.tensor([ids])
    with torch.inference_mode():
        torch_logits, _ = model(x)
    onnx_logits = torch.tensor(engine._forward(ids))
    diff = (onnx_logits - torch_logits[0]).abs().mean().item()
    assert diff < 0.5, f"int8 drift too large: {diff}"