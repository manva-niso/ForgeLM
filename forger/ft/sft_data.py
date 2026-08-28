"""SFT data: dolly-15k instruction formatting."""

from __future__ import annotations

from forger.data.contract import DatasetExample


def format_dolly(instruction: str, response: str, context: str = "") -> str:
    if context:
        return (
            f"### Instruction: {instruction}\n"
            f"### Context: {context}\n"
            f"### Response: {response}"
        )
    return f"### Instruction: {instruction}\n### Response: {response}"


def load_dolly(examples: int = 5000, split: str = "train") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    texts = []
    skipped = 0
    for ex in ds:
        text = format_dolly(ex["instruction"], ex["response"], ex.get("context") or "")
        try:
            DatasetExample(text=text, split="train")
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        texts.append(text)
        if len(texts) >= examples:
            break
    print(f"dolly: loaded {len(texts)} rows (skipped {skipped} out-of-contract)")
    return texts