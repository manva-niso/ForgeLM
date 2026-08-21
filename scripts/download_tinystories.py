"""Download a TinyStories sample, write contract-valid jsonl + sha256 checksum."""

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

from forger.data.contract import DatasetExample


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading TinyStories (train split, streaming)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    rows = []
    skipped = 0
    for ex in ds:
        if len(rows) >= 1000:
            break
        try:
            example = DatasetExample(text=ex["text"], split="train")
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        rows.append(example.model_dump())
    out = out_dir / "tinystories_sample.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"Wrote {len(rows)} examples to {out} (skipped {skipped} out-of-contract)")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())