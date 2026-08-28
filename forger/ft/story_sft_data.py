"""Story-writing SFT data: instructions + TinyStories responses (matches base knowledge)."""

from __future__ import annotations

import json
from pathlib import Path

from forger.data.contract import DatasetExample

INSTRUCTION_TEMPLATES = [
    "Write a story about {topic}.",
    "Tell me a short story about {topic}.",
    "Can you write a story about {topic}?",
    "Create a story about {topic}.",
]


def topic_from_story(text: str) -> str:
    words = [w.lower().strip(".,!?") for w in text.split()[:8] if w.strip()]
    for word in words:
        if len(word) > 3 and word not in {"once", "upon", "there", "with", "that", "from", "they", "them", "when", "then"}:
            return word
    return "a little animal"


def format_story_instruction(text: str, template_index: int = 0) -> str:
    topic = topic_from_story(text)
    instruction = INSTRUCTION_TEMPLATES[template_index % len(INSTRUCTION_TEMPLATES)].format(topic=topic)
    return f"### Instruction: {instruction}\n### Response: {text}"


def load_story_sft(jsonl_path: str | Path, examples: int = 5000) -> list[str]:
    texts = []
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            story = json.loads(line)["text"]
            text = format_story_instruction(story, len(texts))
            try:
                DatasetExample(text=text, split="train")
            except Exception as exc:  # noqa: BLE001
                print(f"skipped story-sft row: {exc}")
                continue
            texts.append(text)
            if len(texts) >= examples:
                break
    print(f"story-sft: loaded {len(texts)} examples from {jsonl_path}")
    return texts