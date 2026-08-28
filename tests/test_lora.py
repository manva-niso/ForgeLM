import torch

from forger.ft.lora import LoRAConfig, LoRALinear, apply_lora, convert_merged, count_lora_params, merge_lora
from forger.ft.sft_data import format_dolly
from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset
from forger.train.trainer import Trainer

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm and the birds sang",
    "once upon a time there was a little girl who loved her dog and they played in the park every day",
] * 8
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)
CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)
LORA = LoRAConfig(r=4, alpha=8)


def _lora_model() -> tuple[GPT, list[str]]:
    torch.manual_seed(0)
    model = GPT(CFG)
    replaced = apply_lora(model, LORA, ("c_attn", "c_proj", "gate", "up", "down"))
    return model, replaced


def test_lora_zero_init_is_identity():
    model, replaced = _lora_model()
    assert len(replaced) > 0
    model.eval()
    x = torch.randn(2, 8, 64)
    module = model.blocks[0].attn.c_attn
    assert isinstance(module, LoRALinear)
    with torch.no_grad():
        assert torch.allclose(module(x), module.base(x), atol=1e-6)


def test_only_lora_params_trainable():
    model, _ = _lora_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(".A" in n or ".B" in n for n, p in model.named_parameters() if p.requires_grad)
    assert len(trainable) > 0


def test_count_lora_params():
    model, _ = _lora_model()
    n = count_lora_params(model)
    assert n > 0
    total = sum(p.numel() for p in model.parameters())
    assert n < total
    assert n < total // 10  # LoRA must stay a small fraction


def test_merge_then_convert_matches_lora_forward():
    model, _ = _lora_model()
    model.eval()
    ids = torch.randint(0, 300, (1, 4))
    with torch.no_grad():
        logits_lora, _ = model(ids)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, LoRALinear):
                    m.B.data.normal_(0, 0.1)
        logits_lora2, _ = model(ids)
    assert not torch.allclose(logits_lora, logits_lora2)
    merge_lora(model)
    convert_merged(model)
    with torch.no_grad():
        logits_merged, _ = model(ids)
    assert torch.allclose(logits_lora2, logits_merged, atol=1e-5)


def test_lora_sft_loss_decreases():
    torch.manual_seed(1)
    model = GPT(CFG)
    apply_lora(model, LORA, ("c_attn", "c_proj", "gate", "up", "down"))
    texts = [format_dolly("write a story", "the cat sat on the mat", "sunny day")] * 40
    encoded = [TOKENIZER.encode(t) for t in texts]
    train_data = WindowDataset(texts, TOKENIZER, 32, windows_per_story=4, encoded_ids=encoded)
    eval_data = WindowDataset(texts, TOKENIZER, 32, windows_per_story=2, encoded_ids=encoded)
    cfg = TrainConfig(
        steps=15, batch_size=4, context_length=32, lr=1e-3, warmup_steps=2,
        eval_every=8, eval_windows=4, log_every=15, checkpoint_dir="ckpt_sft_test",
        run_name="sft_test",
    )
    trainer = Trainer(model, cfg, train_data, eval_data)
    trainer.train()
    assert trainer.loss_history[-1] < trainer.loss_history[0]


def test_format_dolly():
    assert format_dolly("do x", "ok", "") == "### Instruction: do x\n### Response: ok"
    assert "### Context: c" in format_dolly("do x", "ok", "c")