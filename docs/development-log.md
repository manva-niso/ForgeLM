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

## Error log

| Date | Error / symptom | Root cause | Fix (path) |
|---|---|---|---|
| 2026-08-21 | `test_validate_file_missing` KeyError `'ok'` | missing-file branch returned early without `ok` key | `forger/data/contract.py:38` added `report["ok"] = False` |
| 2026-08-21 | `forger-data-validate: program not found` | `[project.scripts]` entry never present in pyproject | `pyproject.toml` added `forger-data-validate = "forger.data.contract:main"` |
| 2026-08-21 | 1/1000 TinyStories rows rejected (text > 4096 chars) | dataset contains over-long stories; contract works as designed | `scripts/download_tinystories.py` filters out-of-contract rows at ingest (skipped: 1) |
| 2026-08-22 | CI fail 1: ruff `I001` import sort ×2 (Linux only) | ruff treats `forger` as third-party (not recognized as first-party) | `pyproject.toml` added `[tool.ruff.lint.isort] known-first-party = ["forger"]` |
| 2026-08-22 | CI fail 2: `ModuleNotFoundError: forger.data` on fresh checkout | `.gitignore` bare `data/` pattern ignored the `forger/data/` package dir | `.gitignore` changed `data/` -> `/data/`; tracked `forger/data/{__init__,contract}.py` |
| 2026-08-22 | (Non-issue) ruff `EXE002` in Docker repro | NTFS mount exposes files as executable (mount artifact, not real) | none — reproduced via in-container git clone instead |

## Rules going forward

- Every bug gets a row here: symptom, root cause, fix path.
- Every shipped module gets a row in `docs/modules.md` + one line in `docs/changelog.md`.
- Benchmark/evidence files land in `benchmarks/`.