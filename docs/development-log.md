# Development Log

Running log of implementations and errors, with file-path references.
Append after every significant work session. Do not rewrite old entries.

## Path map (key files)

| Area | Path |
|---|---|
| Data contract (schema + CLI) | `forger/data/contract.py` |
| Contract tests | `tests/test_contract.py` |
| TinyStories downloader | `scripts/download_tinystories.py` |
| Product contract | `docs/PRODUCT_CONTRACT.md` |
| Module status | `docs/modules.md` |
| Benchmarks / evidence | `benchmarks/` |
| Architecture decisions | `docs/decisions/` |
| CI workflow | `.github/workflows/ci.yml` |
| Project config (deps, ruff, pytest, scripts) | `pyproject.toml` |
| Reproducibility pins | `uv.lock` |
| Gitignore (artifacts + local agent context) | `.gitignore` |

## Implementation log

| Date | Task | What shipped (path refs) | Verified by |
|---|---|---|---|
| 2026-08-21 | Bootstrap + env | Repo scaffold, `AGENTS.md`, `.opencode/agent/*`, `docs/{modules,architecture,changelog}.md`, uv env (`pyproject.toml`, `uv.lock`), commit + tag v0.0.1 | `uv run ruff check .`, `uv run pytest` (2 smoke) |
| 2026-08-21 | Day 1: product + data contract | `docs/PRODUCT_CONTRACT.md`, `docs/decisions/ADR-01-project-charter.md`, `forger/data/contract.py`, `tests/test_contract.py`, `scripts/download_tinystories.py`, `.github/workflows/ci.yml` | 6 tests green; sample 1000/1000 valid; commit 8f330f0, tag v0.0.2 |
| 2026-08-21 | Day 1 evidence | `benchmarks/data_contract.md` (validation run, sha256, test counts) | commit 4b6a2a2 |
| 2026-08-22 | Push + CI validation | Remote added (`origin`), `main` + tags pushed; CI made green (see error log) | GitHub Actions run 32558803277 success |
| 2026-08-22 | Day 2: BPE tokenizer | `forger/tokenizer/bpe.py` (byte-level BPE: train/encode/decode/save/load), `forger/tokenizer/train.py` CLI, `tests/test_bpe.py` (17 tests), `scripts/tokenizer_parity.py`, artifact `artifacts/tokenizer/`, `docs/decisions/ADR-02-tokenizer.md`, `benchmarks/tokenizer_parity.md` (10/10 parity), `docs/kaggle-notebook.md` | 23 tests green; parity 10/10; commit + tag v0.0.3 |
| 2026-08-22 | Day 3: model core | `forger/model/{config,blocks,gpt}.py` (GPTConfig, RMSNorm, RoPE, CausalSelfAttention, SwiGLU MLP, Block, GPT, KV-cache stub), `tests/test_model.py` (10 tests), `scripts/benchmark_model.py`, `benchmarks/model_forward.md` (211ms/512tok, 4,855 tok/s), `docs/decisions/ADR-03-model-core.md` | 33 tests green; commit + tag v0.0.4 |
| 2026-08-22 | Day 4: training pipeline | `forger/train/{config,dataset,trainer}.py`, `configs/train/{baseline,smoke}.yaml`, `tests/test_trainer.py` (6 tests), `docs/decisions/ADR-04-train-pipeline.md`, `benchmarks/train_smoke.md` | 39 tests green; smoke 10 steps in 8.1s (loss 8.11->7.36); commit + tag v0.0.5 |
| 2026-08-22 | Day 5: baseline training | `scripts/kaggle_baseline.py` (CPU/CUDA entry), `scripts/kaggle_baseline.ipynb` (Kaggle import), `scripts/download_from_hub.py`, `benchmarks/baseline_train.md`, `docs/decisions/ADR-05-baseline.md`; checkpoint `checkpoints/baseline/` | 300-step CPU run: 358s, loss 8.11->3.34 train / 3.95 eval; commit + tag v0.0.6 |
| 2026-08-22 | Kaggle run 1 failed | `ModuleNotFoundError: forger` - `python scripts/x.py` puts `scripts/` on sys.path, package not installed on Kaggle | `sys.path.insert` repo root in `scripts/kaggle_baseline.py`; also capped max-stories 2000 (encode ~0.17s/story -> 50k stories would take ~2.4h) |

## Error log

| Date | Error / symptom | Root cause | Fix (path) |
|---|---|---|---|
| 2026-08-21 | `test_validate_file_missing` KeyError `'ok'` | missing-file branch returned early without `ok` key | `forger/data/contract.py:38` added `report["ok"] = False` |
| 2026-08-21 | `forger-data-validate: program not found` | `[project.scripts]` entry never present in pyproject | `pyproject.toml` added `forger-data-validate = "forger.data.contract:main"` |
| 2026-08-21 | 1/1000 TinyStories rows rejected (text > 4096 chars) | dataset contains over-long stories; contract works as designed | `scripts/download_tinystories.py` filters out-of-contract rows at ingest (skipped: 1) |
| 2026-08-22 | CI fail 1: ruff `I001` import sort ×2 (Linux only) | ruff treats `forger` as third-party (not recognized as first-party) | `pyproject.toml` added `[tool.ruff.lint.isort] known-first-party = ["forger"]` |
| 2026-08-22 | CI fail 2: `ModuleNotFoundError: forger.data` on fresh checkout | `.gitignore` bare `data/` pattern ignored the `forger/data/` package dir | `.gitignore` changed `data/` -> `/data/`; tracked `forger/data/{__init__,contract}.py` |
| 2026-08-22 | (Non-issue) ruff `EXE002` in Docker repro | NTFS mount exposes files as executable (mount artifact, not real) | none — reproduced via in-container git clone instead |
| 2026-08-22 | `KeyError: 104` in save/load test | `byte_to_id` in `load()` keyed by bytes object instead of byte value | `forger/tokenizer/bpe.py` `byte_to_id = {b[0]: i ...}` |
| 2026-08-22 | `KeyError: -1` in training | pair counter only skipped `pair[1] == -1` (piece separator), not `pair[0] == -1` | `forger/tokenizer/bpe.py` skip both |
| 2026-08-22 | Training timeout >15min on 800K chars | naive per-merge full-corpus rescans in pure Python | flattened corpus + local-var loops + `--max-chars` 200K default (89s) |
| 2026-08-22 | `models.BPE(vocab=...)` TypeError (tokenizers 0.23) | new API expects `Dict[token, int]`, not id->token dict | `scripts/tokenizer_parity.py` passes inverted dict + merges tuples |
| 2026-08-22 | encode 12.6s/pass slow | `_ranks` dict rebuilt per piece inside `_bpe` | cached `self._ranks` in `__init__` (8.3s/pass) |
| 2026-08-22 | RoPE shape error (64 vs 32) | cos/sin cached at dim/2, applied to full head_dim | duplicate freqs along pair dim (blocks.py) |
| 2026-08-22 | KV-cache decode mismatch (1e-3) | RoPE rotated decode tokens by position 0, not global offset | `rope(q, k, offset=past_len)` |
| 2026-08-22 | KV-cache chunk mismatch (0.5) | `is_causal=True` with T_q != T_k masks cached keys; also bool attn_mask means True=attend | explicit float(-inf) triu mask over the new-chunk columns |
| 2026-08-22 | resume test: empty loss_history | test mutated shared config (cfg_b.steps=5) -> resumed trainer range(5,5) | separate cfg_partial object |
| 2026-08-22 | resume test: loss mismatch | partial run used 5-step LR schedule (different cosine denominator) | `train(until=N)` keeps the full-run schedule |
| 2026-08-22 | resume test: 0.02 loss diff | test models created from different random inits | seed both model creations identically |
| 2026-08-22 | smoke CLI: 15-min timeout, no output | encode 1000 stories twice (~6 min) + bf16 CPU autocast 7x slower than fp32 (1.35s vs 0.19s/forward) | encode once via `encoded_ids`; CPU = fp32 (AMP only on CUDA); `configs/train/smoke.yaml` (bs 4, ctx 256) -> 10 steps in 8.1s |

## Rules going forward

- Every bug gets a row here: symptom, root cause, fix path.
- Every shipped module gets a row in `docs/modules.md` + one line in `docs/changelog.md`.
- Benchmark/evidence files land in `benchmarks/`.
- Design reasoning (alternatives + why) goes to `docs/thinking-log.md`.