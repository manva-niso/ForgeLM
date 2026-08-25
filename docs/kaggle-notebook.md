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

## 1. Baseline training (Day 5 - Tue 2026-08-25) - TBD after Day 4

Planned contents:
- Pull tokenizer artifact + data contract from repo
- Stream TinyStories (full corpus, not just sample)
- Load `forger.train.Trainer` with `configs/train/baseline.yaml` (GPU params: bs 64, ctx 256, ~4k steps)
- Train ~30-45 min on T4
- Save checkpoint (state dict + config) and push to HF Hub (`forge-lm/baseline`)
- Download `final.pt`, `train_log.json`, `eval_loss.json` back to local repo
- Fill evidence into `benchmarks/baseline_train.md`

## 2. LoRA SFT (Day 8 - Fri 2026-08-28) - TBD

## 3. QLoRA (Day 9 - Sat 2026-08-29) - TBD

## 4. Pulling artifacts back locally
```powershell
# from V:\Projects\AIML\Project_1
uv run python scripts/download_from_hub.py --repo forge-lm/baseline --out checkpoints/   # script TBD
```

## Gotchas learned
- Kaggle sessions auto-stop after ~9h idle; save checkpoints early.
- HF Hub push needs `HF_TOKEN` with write scope.
- Symlinks unsupported on some Windows setups - use `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.