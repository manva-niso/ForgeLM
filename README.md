# ForgeLM

Train, fine-tune & serve a small edge/on-device language model from scratch.

## Try it (public, no install)

**Hosted demo (free, no card):** deploy via Streamlit Community Cloud —
sign in at https://share.streamlit.io with GitHub → **Create app** →
repo `manva-niso/ForgeLM` → branch `main` → main file `app/streamlit_ui.py`
→ Deploy. Public URL: `https://forgelm.streamlit.app` (type a prompt, get a
story). Optional alternative: HF Space (free CPU basic) via
`deploy/space/` + `scripts/prep_space.py`.

**Locally (3 steps):**
```powershell
git clone https://github.com/manva-niso/ForgeLM.git
cd ForgeLM
uv sync --group dev
uv run python scripts/smoke_infer.py --ckpt models/forgelm-sft-story --prompt "### Instruction: Write a story about a cat
### Response:"
```

**API (any HTTP client):**
```powershell
uv run uvicorn forger.serve.api:app --port 8000
curl -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d "{\"prompt\": \"### Instruction: Write a story about a dog.\\n### Response:\", \"max_tokens\": 40}"
```

## Development

- Install: `uv sync --group dev` | Test: `uv run pytest` | Lint: `uv run ruff check .`
- Docs: `docs/DOCUMENTS.md` is the index of everything.
- Every artifact is a vertical slice: code + tests + benchmark evidence + commit.