# Kaggle Notebook Tracker

Living checklist of everything to run/paste in the Kaggle notebook for each GPU phase.
Update this file as the build progresses (Day 4/5 = baseline, Day 8 = LoRA SFT, Day 9 = QLoRA).

## 0. Setup (once per project)

- [ ] Create/verify Kaggle account: https://www.kaggle.com
- [ ] Create a new Notebook: Settings -> Accelerator -> **GPU T4 x2** (or P100)
- [ ] Secrets (Settings -> Add-ons -> Secrets):
      - [ ] `HF_TOKEN` - HuggingFace write token (https://huggingface.co/settings/tokens)
      - [ ] `WANDB_API_KEY` (optional)
- [ ] Pull the repo into the notebook:
      ```python
      import os
      os.system("git clone https://github.com/manva-niso/ForgeLM.git /kaggle/working/forge-lm")
      os.chdir("/kaggle/working/forge-lm")
      ```
- [ ] Install deps:
      ```python
      os.system("pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
      os.system("pip install -q datasets tokenizers tqdm tensorboard hydra-core")
      ```

## 1. Baseline training (Day 5 - Tue 2026-08-25) - READY

Run these cells in the Kaggle notebook (Accelerator: GPU T4 x2):

```python
# Cell 1 - setup
import os
os.system("git clone https://github.com/manva-niso/ForgeLM.git /kaggle/working/forge-lm")
os.chdir("/kaggle/working/forge-lm")
os.system("pip install -q --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu121")
os.system("pip install -q --disable-pip-version-check datasets pyyaml tensorboard tqdm")
```

```python
# Cell 2 - config + tokenizer + full TinyStories stream
from forger.tokenizer.bpe import BPETokenizer
from forger.train.dataset import WindowDataset
from forger.train.config import TrainConfig
from forger.model.config import GPTConfig
from forger.model.gpt import GPT
from forger.train.trainer import Trainer
from datasets import load_dataset

tok = BPETokenizer.load("artifacts/tokenizer")
cfg = TrainConfig(steps=4000, batch_size=64, context_length=256, lr=3e-4,
                  warmup_steps=100, grad_accum=1, eval_every=250, eval_windows=20,
                  log_every=50, device="cuda", seed=0,
                  checkpoint_dir="/kaggle/working/ckpt", run_name="baseline")
# NOTE: WindowDataset needs a list of texts; for full TinyStories build from stream:
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = [ex["text"] for ex in ds.take(50000)]
encoded = [tok.encode(t) for t in texts]
split = int(len(encoded) * 0.95)
train_data = WindowDataset(texts[:split], tok, 256, encoded_ids=encoded[:split])
eval_data = WindowDataset(texts[split:], tok, 256, encoded_ids=encoded[split:])
model = GPT(GPTConfig(vocab_size=len(tok.token_bytes), context_length=256))
```

```python
# Cell 3 - train + save (checkpoint incl. train_log.json / eval_log.json)
!python scripts/kaggle_baseline.py --config configs/train/kaggle.yaml --data stream --max-stories 20000 --ckpt-dir /kaggle/working/ckpt --hf-repo Manvaniso/forgelm
```

```python
# Cell 4 - push checkpoint to HF Hub (secrets: HF_TOKEN with write scope)
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo("Manvaniso/forgelm", exist_ok=True, private=True)
api.upload_folder(folder_path="/kaggle/working/ckpt",
                  repo_id="Manvaniso/forgelm",
                  repo_type="model")
```

```python
# Cell 5 - evidence numbers for benchmarks/baseline_train.md
import time, torch
print("final eval loss:", trainer.eval_history[-1])
print("wall time:", <capture from cell 3 timing>)
```

After the run: download the checkpoint locally, run smoke inference, fill
`benchmarks/baseline_train.md` (config, GPU type, wall-time, loss curves PNG
from TensorBoard `runs/baseline`).

## 2. LoRA SFT (Day 8 - Fri 2026-08-28) - TBD

## 3. QLoRA (Day 9 - Sat 2026-08-29) - TBD

## 4. Pulling artifacts back locally
```powershell
# from V:\Projects\AIML\Project_1
uv run python scripts/download_from_hub.py --repo Manvaniso/forgelm --out checkpoints/   # script TBD
```

## Gotchas learned
- Kaggle sessions auto-stop after ~9h idle; save checkpoints early.
- HF Hub push needs `HF_TOKEN` with write scope.
- Symlinks unsupported on some Windows setups - use `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.