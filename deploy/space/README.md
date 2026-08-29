---
title: ForgeLM
emoji: 🪄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ForgeLM — train, fine-tune & serve a small language model

A 5.25M-parameter decoder-only GPT trained from scratch:
custom byte-level BPE tokenizer, RMSNorm/RoPE/SwiGLU GPT core, LoRA
fine-tuned on story instructions, QLoRA int4 export (≤8MB).

Type a prompt → get a story. Model runs in-process on the Space's CPU.

Source + full documentation: https://github.com/manva-niso/ForgeLM
Model weights ship inside this Space (models/forgelm-sft-story).