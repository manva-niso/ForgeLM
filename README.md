# ForgeLM

Train, fine-tune & serve a small edge/on-device language model from scratch.

## Use the model (3 steps, no accounts needed)

```powershell
git clone https://github.com/manva-niso/ForgeLM.git
cd ForgeLM
uv sync --group dev
uv run python scripts/smoke_infer.py --ckpt models/forgelm-baseline --prompt "Once upon a time"
```

The weights ship inside this repo (`models/forgelm-baseline/`), so nothing is
downloaded at runtime. Change `--prompt` and `--max-tokens` to generate more.

## Development

- Install: `uv sync --group dev` | Test: `uv run pytest` | Lint: `uv run ruff check .`
- Docs: `docs/DOCUMENTS.md` is the index of everything.
- Every artifact is a vertical slice: code + tests + benchmark evidence + commit.