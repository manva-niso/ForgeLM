# AGENTS.md

## Project overview
ForgeLM - train, fine-tune & serve a small edge/on-device language model from scratch.

## Directory layout
- `forger/data` - data contracts + TinyStories ingestion
- `forger/tokenizer` - BPE from scratch
- `forger/model` - GPT blocks
- `forger/train` - training pipeline
- `forger/eval` - evaluation harness
- `forger/serve` - inference engine + API
- `forger/ft` - LoRA SFT
- `forger/quant` - QLoRA int4
- `scripts` / `tests` / `benchmarks` / `configs` / `docs` / `notebooks`

## Commands
- Install: `uv sync --group dev`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Run: `uv run python -m forger.<module>` (per-module)

## Conventions
- Prefer small, reviewable changes over large ones
- Run tests before marking any module done
- Do not edit lockfiles / vendor / generated folders unless asked
- Vertical-slice rule: every change ships tests + benchmark/evidence + commit

## Agent notes
- Read docs/architecture.md before cross-module or schema changes
- Read docs/modules.md before touching module boundaries or interfaces
- Log every significant change in docs/changelog.md (append only)
- Use the `architect` agent for planning, `coder` for implementation,
  `approver` for final sign-off, `quick` for one-offs - see
  .opencode/agent/ for definitions