# PROJECT.md — Project Config

## Name
ForgeLM

## One-line description
Edge/on-device assistant LLM: train, fine-tune (LoRA/QLoRA) and serve a small language model from scratch.

## Modules
| Module | Folder | Depends on | Exposes |
|---|---|---|---|
| Data contract + ingestion | forger/data | — | DatasetExample schema, validate CLI, TinyStories loader |
| Tokenizer (BPE from scratch) | forger/tokenizer | data | train/encode/decode/save/load |
| Model core (GPT blocks) | forger/model | tokenizer | GPTConfig, GPT, CausalSelfAttention, KV-cache |
| Training pipeline | forger/train | model, tokenizer, data | Trainer, configs (Hydra) |
| Baseline training | scripts | train | trained checkpoint, benchmark report |
| Evaluation harness | forger/eval | model | perplexity, generation, metrics |
| Inference engine | forger/serve | model, quant | engine, optimization, benchmarks |
| Fine-tuning (LoRA SFT) | forger/ft | train, model | SFT data prep, adapter train/merge |
| Quantization (QLoRA int4) | forger/quant | model | int4 quant, export |
| API + deployment | forger/serve/api.py | serve | FastAPI /v1/completions, Docker, CI/CD |

## Commands
- Install: `uv sync --group dev`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Run: `uv run python -m forger.<module>` (per-module, TBD)

## Model tiers
| Tier | Model |
|---|---|
| Premium | nvidia-nim/z-ai/glm-5.2 |
| Cheap | opencode-go/deepseek-v4-flash |

## Skills needed
(none yet — add later if a workflow repeats)

## Architecture notes
- CPU-first dev on Windows (torch CPU build); Kaggle GPU (T4) for baseline/SFT/QLoRA training.
- Target: <=25M params, <=8MB after 4-bit quant, <100ms first-token latency on CPU.
- Vertical-slice rule: each day ships working artifact = code + tests + benchmark/evidence + commit.
- Repo: github.com/<your-handle>/forge-lm (remote added later via browser, no gh CLI).

## Anything BOOTSTRAP.md should NOT do
- Skip the reviewer agent (two-tier setup only: architect/approver premium, coder/quick cheap).
- Do not start implementing any module during bootstrap.