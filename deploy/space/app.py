"""ForgeLM public demo - Gradio app running the model in-process.

Deployed as a HuggingFace Space (free CPU basic). Anyone with the URL can
type a prompt and get a story - no install, no accounts.
"""

from __future__ import annotations

import gradio as gr

from forger.serve.engine import Engine

engine = Engine.from_checkpoint("models/forgelm-sft-story", "artifacts/tokenizer")

EXAMPLES = [
    "Write a story about a cat who learns to fly.",
    "Tell me a story about a brave little mouse.",
    "Write a story about two friends and a magic tree.",
    "Once upon a time there was a dragon who...",
]


def generate_story(prompt: str, max_tokens: int, temperature: float) -> str:
    if not prompt.strip():
        return "Please type a prompt first."
    text, _, stats = engine.generate(
        prompt,
        max_tokens=int(max_tokens),
        top_k=50,
        temperature=float(temperature),
        seed=0,
    )
    return text + f"\n\n_(generated {stats['generated_tokens']} tokens)_"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ForgeLM - story generator") as demo:
        gr.Markdown(
            "# ForgeLM\n"
            "A 5.25M-parameter language model trained from scratch"
            " (own BPE tokenizer, own GPT, LoRA fine-tuned, int4-capable).\n"
            "Type any story prompt below."
        )
        prompt = gr.Textbox(label="Prompt", lines=3, placeholder=EXAMPLES[0])
        with gr.Row():
            max_tokens = gr.Slider(16, 128, value=64, step=8, label="Max tokens")
            temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
        output = gr.Textbox(label="Story", lines=10)
        button = gr.Button("Generate", variant="primary")
        button.click(generate_story, [prompt, max_tokens, temperature], output)
        gr.Examples(EXAMPLES, inputs=prompt)
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()