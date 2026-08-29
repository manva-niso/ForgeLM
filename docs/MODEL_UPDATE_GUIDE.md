# Model Update Guide — how to retrain / re-fine-tune / re-quantize safely

*For future-you. Everything you need to improve ForgeLM later without breaking
the pipeline. Read this BEFORE touching any training code.*

## 0. The golden rules

1. **Never delete an old model artifact.** Each version lives in its own dir
   (`models/forgelm-baseline-v2`, not overwrite). Old models keep working.
2. **Never change the tokenizer casually.** It is the axis that everything
   else (model vocab, eval corpus, SFT data, int4 artifacts) depends on.
   Changing it = rerunning the ENTIRE pipeline.
3. **Every model dir carries its own `config.json`** — the code is fully
   config-driven. As long as weights + config ship together, the pipeline
   adapts automatically.
4. **Measure before/after every change:** `forger.eval.run` (ppl + bpb on the
   pinned corpus) + the generation qualitative check. Never trust a loss line
   alone (remember the dolly lesson).
5. **Tag + changelog + commit after every shipped model.** Tests must pass
   (`uv run pytest`) and lint must be clean (`uv run ruff check .`).
6. **Merge `main` → `kaggle`** after any code change that Kaggle uses.

## 1. Pipeline map (what depends on what)

```
tokenizer (artifacts/tokenizer) ──► GPTConfig.vocab_size
        │
        ▼
training (forger/train, configs/) ──► baseline model (models/forgelm-baseline*)
        │
        ▼
LoRA SFT (forger/ft) ──► fine-tuned model (models/forgelm-sft-*)
        │
        ▼
QLoRA int4 (forger/quant) ──► 4-bit artifact (models/*-4bit) ──► serving/API
```

Dependency rules:
- Tokenizer change → retrain EVERYTHING (model vocab must match).
- Baseline retrain → redo SFT + re-quantize (the SFT was trained on the old base).
- SFT rerun only → safe (new adapter on the same base); re-quantize the new merged model.
- Quantization only → safe (same merged weights, new int4 file).
- Architecture change (d_model/layers/ctx) → retrain everything + re-export ONNX.

---

## 2. Retrain the baseline (new data / more steps / bigger model)

**Files to change:**
| File | What to edit |
|---|---|
| `configs/train/kaggle.yaml` | steps, batch_size, context_length, lr, warmup_steps, windows_per_story_train/eval, checkpoint_dir |
| `scripts/kaggle_baseline.py` | `--max-stories` (data volume), `--data stream` source |
| `forger/model/config.py` | ONLY if changing architecture (d_model, n_layers, ffn_mult, context_length, vocab_size) |
| `forger/train/trainer.py` | only if changing the training mechanics (lr schedule, AMP) |

**Steps:**
1. Small local smoke: `uv run python scripts/kaggle_baseline.py --steps 50 --device cpu`
2. Kaggle run (notebook, kaggle branch): full config → checkpoint pushed to HF
3. Pull back: `uv run python scripts/download_from_hub.py --repo <you>/<repo>`
4. Evaluate: `uv run python -m forger.eval.run --ckpt <new_dir> --fetch-eval` (or reuse pinned corpus)
5. Commit new dir as `models/forgelm-baseline-v2/` + README + tag v0.X.0

**Safe parameter ranges (measured):**
- lr 3e-4 (AdamW 0.9/0.95, wd 0.1) — proven. lr 1e-3 diverges in quality.
- context_length 128 — 97% of TinyStories qualifies. **ctx 256 discards 91%** of stories.
- windows_per_story_train 16, eval 4 — uncapped (see memorization war, Day 5).
- batch_size 64 on T4, steps 4000 for ~20K stories.
- max-stories: encode ≈ 0.024s/story → 20K ≈ 8 min CPU. More stories > more steps.

**Known traps:**
- **Memorization** — watch the eval-rise warning; if train loss → ~0 while eval
  rises, your unique data is too small. Unique data = stories × tokens.
- The best-eval checkpoint (`best_model.pt`) is what you ship, not the last step.

---

## 3. Retrain the tokenizer (only when you really need it)

**Files:** `forger/tokenizer/bpe.py` (train/encode/decode), `forger/tokenizer/train.py` CLI.

**Command:**
```
uv run python -m forger.tokenizer.train train --input <corpus.jsonl> --output artifacts/tokenizer --vocab-size 4096
```

**THIS BREAKS EVERYTHING DOWNSTREAM:**
- `GPTConfig.vocab_size` must match the new vocab (edit model config or pass it).
- All existing models (baseline, SFT, int4) become incompatible — their
  vocab_size ≠ new tokenizer → embeddings misaligned. **Retrain from scratch.**
- The pinned eval corpus + every reproducibility block reference the old
  tokenizer checksum — re-run eval with the new one.
- The committed `models/*` become stale.

**Safe procedure:** keep the old tokenizer as `artifacts/tokenizer-v1/` (backup);
train v2; update `artifacts/tokenizer`; update `models/` configs; retrain
baseline (section 2) → SFT (section 4) → quant (section 5).

---

## 4. Re-run LoRA SFT (new instruction data / different hyperparams)

**Files to change:**
| File | What to edit |
|---|---|
| `forger/ft/train_sft.py` | --ckpt (base), --examples, --steps, --batch-size, --context-length, --lora-r, --lora-alpha, --device |
| `forger/ft/story_sft_data.py` | instruction templates, topic extraction, data source |
| `scripts/kaggle_sft.py` | GPU-run equivalents (streams TinyStories by default) |

**Steps:**
1. Decide the base: `models/forgelm-baseline-v2` (or keep v1 if untouched).
2. **Match the data to the base's knowledge** — this is the #1 lesson:
   - TinyStories base → story instructions over TinyStories content.
   - A general-KB base (if you ever train one) → dolly-style QA is fine.
3. Local sanity: `uv run python -m forger.ft.train_sft --ckpt models/forgelm-baseline --examples 2000 --steps 300 --device cpu`
4. Inspect generations (story prompts) — MUST be coherent before GPU spend.
5. Kaggle upgrade (optional): notebook on kaggle branch.
6. Merge + convert; commit as `models/forgelm-sft-story-v2/`.

**Safe parameter ranges (measured):**
- lr 3e-4 (NOT 1e-3 — adapters dominate and the merged model degrades).
- r=8 / alpha=16 → scaling 2.0. Bigger r = more capacity, more forgetting risk.
- ~300-400 steps at 2K examples on CPU; 3000 steps / 15K examples on T4.

**Known traps:**
- **The dolly disaster:** SFT teaches form, not facts. If your base lacks the
  domain knowledge, no amount of fine-tuning fixes it — you get confident
  garbage (ppl 38 on stories, 68 on held-out dolly).
- **Forgetting:** after SFT always re-check ppl on the TinyStories eval —
  story-SFT v1 went 7.88 → 38 on dolly-SFT (rejected); story-SFT stayed ~2.0.
- Keep the OLD adapter/merged model until the new one passes evaluation.

---

## 5. Re-run QLoRA int4 (re-quantize any model)

**Files:** `forger/quant/quantize.py` (unchanged unless format changes),
`forger/quant/qlora.py` (training), export helpers.

**Command (export only, no retraining):**
```python
from forger.model.checkpoint import load_model_from_checkpoint
from forger.quant.quantize import quantize_model_4bit, export_4bit, load_4bit

model = load_model_from_checkpoint("models/forgelm-sft-story-v2")
quantize_model_4bit(model)
export_4bit(model, "models/forgelm-sft-story-v2-4bit")
# ALWAYS verify:
loaded = load_4bit("models/forgelm-sft-story-v2-4bit")
# then run forger.eval.run / ppl on it - expect <= +3% ppl vs fp32
```

**Known traps:**
- **Always re-measure ppl after load** (the ppl-734K bug: a fresh GPT was
  built with random RMSNorms — the toy test masked it). If ppl explodes, the
  export/load is broken, not the model.
- Old int4 files (pre-fix) lack the `params` key — re-export with current code.
- The tied embedding/lm_head stay fp32 by design (lm_head is excluded) — if
  you change that exclusion, re-verify size (<=8MB target) AND quality.

---

## 6. Architecture changes (d_model, layers, ctx, vocab)

**Files:** `forger/model/config.py` (GPTConfig defaults), every `models/*/config.json` (shipped), training configs, `forger/serve/api.py` (defaults), ONNX export (re-export).

**Rules:**
- `d_model % n_heads == 0` (validated by GPTConfig).
- `vocab_size` must equal the tokenizer's vocab (4096 today).
- Context: keep 128 for TinyStories training; the ENGINE's limit comes from
  the model's own config.json, so serving adapts automatically.
- Re-export ONNX after any change: `export_onnx(model, ...)` + int8 variant.
- Update the API's model-id string + default max_tokens if ctx changes.

---

## 7. The full update checklist (copy for every model upgrade)

```
[ ] uv run pytest          (all green, including parity/roundtrip tests)
[ ] uv run ruff check .    (clean)
[ ] local smoke run first  (CPU, few steps)
[ ] GPU run (Kaggle)       (if needed - notebook re-imports + re-clones)
[ ] pull checkpoint back
[ ] eval: ppl + bpb on pinned corpus (forger.eval.run)
[ ] qualitative: 5 story prompts - coherent?
[ ] int4: quantize + export + ppl check (+3% max)
[ ] ONNX: re-export fp32 + int8, tests pass
[ ] commit models/<name-vN>/ + README + tag + changelog
[ ] merge main -> kaggle  (if Kaggle code changed)
[ ] CI green on GitHub
```

## 8. Problems you will hit (head's-up)

| When | Problem | What to do |
|---|---|---|
| Tokenizer change | everything downstream breaks | accept it; full rerun; keep v1 tokenizer backed up |
| ctx > 128 on TinyStories | 91% of stories discarded → memorization | keep 128; more stories instead |
| SFT on knowledge the base lacks | confident garbage (dolly lesson) | switch data domain to base's knowledge |
| lr 1e-3 on LoRA | merged model degrades | use 3e-4 |
| int4 load | ppl explodes | re-measure after load; re-export with current code |
| New architecture | ONNX/int4/API defaults stale | re-export + re-quantize + update API config |
| Kaggle session resets | /kaggle/working wiped | keep artifacts in HF/Output zips |
| HF push 403 | namespace must be YOUR username | Manvaniso/... (or your new handle) |
| torch.compile on Windows | needs MSVC | skip; use torch eager cache or ONNX |
| Git repo size | 20-25MB per model dir | keep a few versions; prune old ones deliberately (never the shipped one) |

## 9. Where each piece lives (quick file map)

| Concern | Files |
|---|---|
| Data | `forger/data/contract.py`, `scripts/download_tinystories.py`, `data/eval_tinystories.jsonl` (+sha256) |
| Tokenizer | `forger/tokenizer/`, `artifacts/tokenizer/` |
| Model | `forger/model/`, `models/*/config.json` |
| Training | `forger/train/`, `configs/train/*.yaml`, `scripts/kaggle_baseline.py` |
| Eval | `forger/eval/`, `benchmarks/eval_report.md` |
| Serving | `forger/serve/` (engine, onnx, api), `scripts/bench_*.py` |
| FT | `forger/ft/`, `scripts/kaggle_sft.py` |
| Quant | `forger/quant/`, `models/*-4bit/` |
| Evidence | `benchmarks/*.md`, `docs/decisions/ADR-*.md` |