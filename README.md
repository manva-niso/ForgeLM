# ForgeLM

Train, fine-tune & serve a small edge/on-device language model from scratch.

## Try it (public, no install)

**Hosted demo (LIVE):** **https://forgelm-dep.streamlit.app/** — type a prompt,
get a story. No install, no account. Free (Streamlit Community Cloud).
Alternative hosting path (HF Space, free CPU basic): `deploy/space/` +
`scripts/prep_space.py`.

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