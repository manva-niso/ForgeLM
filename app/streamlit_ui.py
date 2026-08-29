"""ForgeLM public demo - Streamlit app (deploys free on Streamlit Community Cloud).

Anyone with the URL can type a prompt and get a story.
Runs the story-SFT model in-process; weights ship in the repo.
"""

from __future__ import annotations

import streamlit as st

from forger.serve.engine import Engine

st.set_page_config(page_title="ForgeLM", page_icon="🪄")

EXAMPLES = [
    "Write a story about a cat who learns to fly.",
    "Tell me a story about a brave little mouse.",
    "Write a story about two friends and a magic tree.",
    "Once upon a time there was a dragon who...",
]


@st.cache_resource
def load_engine() -> Engine:
    return Engine.from_checkpoint("models/forgelm-sft-story", "artifacts/tokenizer")


st.title("ForgeLM")
st.caption(
    "A 5.25M-parameter language model trained from scratch — own BPE tokenizer, "
    "own GPT (RMSNorm/RoPE/SwiGLU), LoRA fine-tuned, QLoRA int4 export ≤8MB."
)

prompt = st.text_area("Prompt", placeholder=EXAMPLES[0], height=100)
col1, col2 = st.columns(2)
max_tokens = col1.slider("Max tokens", 16, 128, 64, 8)
temperature = col2.slider("Temperature", 0.2, 1.5, 0.8, 0.1)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please type a prompt first.")
    else:
        with st.spinner("Writing..."):
            engine = load_engine()
            text, _, stats = engine.generate(
                prompt, max_tokens=int(max_tokens), top_k=50,
                temperature=float(temperature), seed=0,
            )
        st.markdown(text)
        st.caption(f"generated {stats['generated_tokens']} tokens")

with st.expander("Example prompts"):
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state["prompt"] = ex
            st.rerun()

st.divider()
st.caption("Source + docs: github.com/manva-niso/ForgeLM")