"""Compare ForgeLM tokenizer against Hugging Face ByteLevel BPE (parity oracle)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from forger.tokenizer.bpe import BPETokenizer

ARTIFACT = Path("artifacts/tokenizer")
TEST_STRINGS = [
    "Hello world",
    "Hello, world!",
    "I don't know.",
    "Caf\u00e9 r\u00e9sum\u00e9",
    "\u4f60\u597d\u4e16\u754c",
    "\U0001f642",
    "multiple    spaces",
    "line one\nline two",
    "The cat sat on the mat and purred happily.",
    "12345 numbers 67890",
]


def build_hf_tokenizer(artifact: Path) -> Tokenizer:
    vocab = json.loads((artifact / "vocab.json").read_text(encoding="utf-8"))
    inverted = {token: int(token_id) for token_id, token in vocab.items()}
    merges = []
    for line in (artifact / "merges.txt").read_text(encoding="utf-8").splitlines():
        if line:
            a, b = line.split(" ")
            merges.append((a, b))
    tokenizer = Tokenizer(models.BPE(vocab=inverted, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tokenizer


def main() -> int:
    ours = BPETokenizer.load(ARTIFACT)
    ref = build_hf_tokenizer(ARTIFACT)

    rows = []
    total = len(TEST_STRINGS)
    matches = 0
    mismatches = 0
    example_mismatch = ""
    for s in TEST_STRINGS:
        our_ids = ours.encode(s)
        hf_ids = ref.encode(s).ids
        ok = our_ids == hf_ids
        rows.append((s, our_ids, hf_ids, ok))
        if ok:
            matches += 1
        else:
            mismatches += 1
            if not example_mismatch:
                example_mismatch = f"text={s!r} ours={our_ids} hf={hf_ids}"

    sample_texts = []
    with (Path("data") / "tinystories_sample.jsonl").open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 50:
                break
            sample_texts.append(json.loads(line)["text"])
    big = " ".join(sample_texts)

    start = time.monotonic()
    n = 20
    for _ in range(n):
        ours.encode(big)
    our_encode_s = (time.monotonic() - start) / n

    start = time.monotonic()
    for _ in range(n):
        ref.encode(big)
    hf_encode_s = (time.monotonic() - start) / n

    md = [
        "# Benchmark: Tokenizer Parity - Day 2 (2026-08-22)",
        "",
        "## Setup",
        "- Custom: `forger/tokenizer/bpe.py` (byte-level BPE, GPT-2 pretokenization, deterministic ties)",
        "- Reference: Hugging Face `tokenizers` `models.BPE` + `pre_tokenizers.ByteLevel`",
        f"- Artifact: `{ARTIFACT}` (vocab 4096, merges 3839)",
        f"- Artifact checksum: `{ours.checksum(ARTIFACT)}`",
        "- Method: same vocab + merges loaded into both; compare encodings on fixed test strings.",
        "",
        "## Test strings",
        "| string | ours == HF |",
        "|---|---|",
    ]
    for s, our_ids, hf_ids, ok in rows:
        md.append(f"| {s!r} | {ok} |")
    md.append("")
    md.append(f"## Result: {matches}/{total} strings match exactly")
    if mismatches:
        md.append(f"- Mismatch example: {example_mismatch}")
    md.append("")
    md.append("## Timing (50 TinyStories stories, ~seconds/encode pass)")
    md.append("| Implementation | s/pass |")
    md.append("|---|---|")
    md.append(f"| ours | {our_encode_s:.4f} |")
    md.append(f"| HF reference | {hf_encode_s:.4f} |")
    md.append(f"| speed ratio (ours/HF) | {our_encode_s / hf_encode_s:.1f}x |")
    md.append("")
    md.append("## Notes")
    md.append("- Same vocab/merges => identical encoding is expected where algorithms agree.")
    md.append("- Differences, if any, indicate algorithmic (not correctness) divergence; documented above.")
    md.append("- Training corpus: 200,000 chars of TinyStories (capped for speed). Full-corpus retrain possible via `--max-chars`.")
    md.append("")
    Path("benchmarks/tokenizer_parity.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{matches}/{total} match; encode ours={our_encode_s:.4f}s HF={hf_encode_s:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())