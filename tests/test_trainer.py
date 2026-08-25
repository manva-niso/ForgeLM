import pytest
import torch

from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer
from forger.train.config import TrainConfig
from forger.train.dataset import WindowDataset
from forger.train.trainer import Trainer, lr_at_step

CORPUS = [
    "the cat sat on the mat and the dog barked at the cat while the sun was warm and the birds sang",
    "once upon a time there was a little girl who loved her dog and they played in the park every day",
    "the sun was warm and the birds sang in the trees all day long while the children played outside",
    "they ran to the park and played with a red ball together until the evening came and it grew dark",
] * 10
TOKENIZER = BPETokenizer.train(CORPUS, vocab_size=300)
CFG = GPTConfig(vocab_size=300, d_model=64, n_heads=4, n_layers=2, context_length=32)
TRAIN_CFG = TrainConfig(
    steps=10,
    batch_size=4,
    context_length=32,
    lr=1e-3,
    warmup_steps=2,
    eval_every=5,
    eval_windows=4,
    log_every=10,
    checkpoint_dir="checkpoints_test",
    run_name="test",
)


def _datasets():
    train_data = WindowDataset(CORPUS[:30], TOKENIZER, 32, windows_per_story=4)
    eval_data = WindowDataset(CORPUS[30:], TOKENIZER, 32, windows_per_story=2)
    train_data.shuffle(0)
    return train_data, eval_data


def test_window_dataset_contiguous_and_shifted():
    train_data, _ = _datasets()
    xs, ys = train_data.get_batch(0, 2)
    assert len(xs) == 2
    assert all(len(x) == 32 for x in xs)
    assert xs[0][1:] == ys[0][:-1]


def test_dataset_rejects_short_corpus():
    with pytest.raises(ValueError):
        WindowDataset(["hi"], TOKENIZER, 32)


def test_windows_per_story_not_capped_by_len():
    ds = WindowDataset(CORPUS[:10], TOKENIZER, 32, windows_per_story=8)
    qualifying = sum(1 for t in CORPUS[:10] if len(TOKENIZER.encode(t)) >= 33)
    assert len(ds.windows) == qualifying * 8


def test_lr_schedule():
    cfg = TrainConfig(steps=100, warmup_steps=10)
    assert lr_at_step(0, cfg) < cfg.lr
    assert lr_at_step(9, cfg) == pytest.approx(cfg.lr)
    assert lr_at_step(99, cfg) == pytest.approx(cfg.lr * 0.1)
    mid = lr_at_step(54, cfg)
    assert cfg.lr * 0.1 < mid < cfg.lr


def test_trainer_loss_decreases_smoke(tmp_path):
    model = GPT(CFG)
    train_data, eval_data = _datasets()
    cfg = TrainConfig(**{**TRAIN_CFG.to_dict(), "checkpoint_dir": str(tmp_path / "ckpt")})
    trainer = Trainer(model, cfg, train_data, eval_data)
    trainer.train()
    trainer.save()
    first = trainer.loss_history[0]
    last = trainer.loss_history[-1]
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert len(trainer.eval_history) == 2
    assert trainer.step == cfg.steps
    assert (tmp_path / "ckpt" / "checkpoint.pt").exists()


def test_resume_matches_uninterrupted(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    torch.manual_seed(42)
    model_a = GPT(CFG)
    torch.manual_seed(42)
    model_b = GPT(CFG)
    train_data_a, eval_data_a = _datasets()
    train_data_b, eval_data_b = _datasets()
    cfg_a = TrainConfig(**{**TRAIN_CFG.to_dict(), "checkpoint_dir": str(ckpt_dir)})
    cfg_b = TrainConfig(**{**TRAIN_CFG.to_dict(), "checkpoint_dir": str(ckpt_dir)})

    trainer_full = Trainer(model_a, cfg_a, train_data_a, eval_data_a)
    trainer_full.train()

    cfg_partial = TrainConfig(**{**TRAIN_CFG.to_dict(), "checkpoint_dir": str(ckpt_dir)})
    trainer_partial = Trainer(model_b, cfg_partial, train_data_b, eval_data_b)
    trainer_partial.train(until=5)
    trainer_partial.save()

    model_c = GPT(CFG)
    model_c.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["model_state"])
    trainer_resumed = Trainer.resume(ckpt_dir, model_c, cfg_b, train_data_b, eval_data_b)
    trainer_resumed.train()

    assert trainer_resumed.loss_history == trainer_full.loss_history[5:]


def test_model_moved_to_device():
    model = GPT(CFG)
    train_data, eval_data = _datasets()
    cfg = TrainConfig(**{**TRAIN_CFG.to_dict(), "device": "cpu"})
    trainer = Trainer(model, cfg, train_data, eval_data)
    assert next(trainer.model.parameters()).device.type == "cpu"


def test_checkpoint_contains_optimizer_state(tmp_path):
    model = GPT(CFG)
    train_data, eval_data = _datasets()
    cfg = TrainConfig(**{**TRAIN_CFG.to_dict(), "checkpoint_dir": str(tmp_path / "ckpt")})
    trainer = Trainer(model, cfg, train_data, eval_data)
    trainer.train()
    trainer.save()
    ckpt = torch.load(tmp_path / "ckpt" / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == cfg.steps
    assert "model_state" in ckpt
    assert "optimizer_state" in ckpt
    assert ckpt["config"]["steps"] == cfg.steps