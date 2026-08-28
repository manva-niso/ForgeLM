# Document Index

*What every file in this repo is for. Read this first when you're lost.*

## Repo root
| File | Contents |
|---|---|
| `README.md` | One-paragraph project intro + commands |
| `PROJECT.md` | Project config (modules, model tiers) — **local only, not pushed** |
| `AGENTS.md` | Agent context rules — **local only, not pushed** |
| `pyproject.toml` | Dependencies, dev group, ruff/pytest config, console scripts |
| `uv.lock` | Pinned environment (reproducibility) |
| `.github/workflows/ci.yml` | CI: ruff + pytest on every push |

## docs/ — the knowledge base
| File | Contents |
|---|---|
| `JOURNEY.md` | **The story** — plain-language walkthrough from Day 1 to now: what we built, every problem, how we thought about it, what changed. Read this first. |
| `ARCHITECTURE_INTERNALS.md` | **The hardware** — exact tokenizer/model/trainer internals + the full hyperparameter change ledger with reasons. |
| `STUDY_PLAN.md` | **What to read & do per day** — per-day study boxes (read/understand/calculate/completion criteria). |
| `DOCUMENTS.md` | This index. |
| `architecture.md` | Architecture decisions stub (filled by planning). |
| `modules.md` | Module graph: status of each `forger/` module (done/todo). |
| `changelog.md` | One line per shipped change (append only). |
| `development-log.md` | **Error log** — every bug: symptom → root cause → fix path; + implementation log. |
| `thinking-log.md` | **Why log** — design reasoning, alternatives considered, corrections. |
| `PRODUCT_CONTRACT.md` | Product spec + acceptance criteria (≤25M params, ≤8MB 4-bit, <100ms CPU). |
| `kaggle-notebook.md` | Kaggle setup, branch policy, notebook cells, gotchas. |
| `decisions/ADR-01…05` | Architecture decision records (charter, tokenizer, model, train pipeline, baseline). |

## benchmarks/ — evidence
| File | Contents |
|---|---|
| `data_contract.md` | Day 1: 1000/1000 rows valid, sha256 |
| `tokenizer_parity.md` | Day 2: 10/10 exact parity vs HuggingFace + timings |
| `model_forward.md` | Day 3: 211ms forward, 4,855 tok/s, 5.25M params |
| `train_smoke.md` | Day 4: 10-step smoke, loss 8.11→7.36 |
| `baseline_train.md` | Day 5: CPU run + Kaggle run (eval 2.0247) + A/B memorization evidence |

## Reads/
| File | Contents |
|---|---|
| `What to read for what.txt` | Curated study links mapped to build days |

## configs/train/
| File | Contents |
|---|---|
| `smoke.yaml` | 10-step CPU sanity config |
| `baseline.yaml` | Local CPU reference (500 steps) |
| `kaggle.yaml` | GPU config used for the real baseline (the numbers that shipped) |

## scripts/
| File | Contents |
|---|---|
| `download_tinystories.py` | Contract-valid TinyStories sample + checksum |
| `tokenizer_parity.py` | HF parity benchmark → benchmarks/tokenizer_parity.md |
| `benchmark_model.py` | Forward benchmark → benchmarks/model_forward.md |
| `kaggle_baseline.py` | Training entry point (CPU or CUDA, config-driven) |
| `kaggle_baseline.ipynb` | Importable Kaggle notebook (uses `kaggle` branch) |
| `download_from_hub.py` | Pull checkpoint from HF Hub |
| `smoke_infer.py` | Greedy generation from a checkpoint |

## models/
| File | Contents |
|---|---|
| `forgelm-baseline/` | Pretrained weights (24.3 MB, best eval 2.0247) |
| `forgelm-sft-story/` | **LoRA fine-tuned** (21.3 MB fp32 + README) — instruction-following on stories |
| `forgelm-sft-story-4bit/` | int4 export (6.27 MB, ≤8MB spec) |
| `forgelm-4bit/` | int4 export of the baseline (6.86 MB) |

## forger/ — the code
`data` (contract) · `tokenizer` (BPE) · `model` (GPT) · `train` (pipeline) —
each with `__init__.py`; tests live in `tests/`.