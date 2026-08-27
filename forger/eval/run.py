"""Evaluation harness CLI.

Usage:
    uv run python -m forger.eval.run --ckpt models/forgelm-baseline --fetch-eval
    uv run python -m forger.eval.run --ckpt models/forgelm-baseline --eval-data data/eval_tinystories.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from forger.data.contract import DatasetExample
from forger.eval.generation import generate
from forger.eval.metrics import distinct_n, length_sanity, repetition_rate
from forger.eval.perplexity import evaluate_perplexity
from forger.model.checkpoint import load_model_from_checkpoint
from forger.tokenizer.bpe import BPETokenizer

EVAL_CORPUS = Path("data/eval_tinystories.jsonl")
DEFAULT_PROMPTS = [
    "Once upon a time there was a little cat",
    "The little girl went to the park",
    "Tom and his dog were playing",
    "It was a rainy day and",
    "The rabbit wanted to find",
]


def fetch_eval_corpus(count: int = 500) -> Path:
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    EVAL_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with EVAL_CORPUS.open("w", encoding="utf-8") as fh:
        for ex in ds:
            try:
                example = DatasetExample(text=ex["text"], split="validation")
            except Exception as exc:  # noqa: BLE001
                print(f"skipped out-of-contract story: {exc}")
                continue
            fh.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
            rows.append(example)
            if len(rows) >= count:
                break
    digest = hashlib.sha256(EVAL_CORPUS.read_bytes()).hexdigest()
    Path("benchmarks/eval_corpus.sha256").write_text(f"{digest}  {EVAL_CORPUS}\n", encoding="utf-8")
    print(f"wrote {len(rows)} eval stories -> {EVAL_CORPUS}")
    print(f"sha256: {digest}")
    return EVAL_CORPUS


def load_eval_texts(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def reproducibility_block(
    tokenizer: BPETokenizer, ckpt_dir: str, eval_corpus: Path, seed: int
) -> list[str]:
    return [
        f"- torch: {torch.__version__}",
        f"- checkpoint: {ckpt_dir}",
        f"- tokenizer checksum: {tokenizer.checksum('artifacts/tokenizer')}",
        f"- eval corpus: {eval_corpus} sha256: {hashlib.sha256(eval_corpus.read_bytes()).hexdigest()}",
        f"- seed: {seed}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forger-eval")
    parser.add_argument("--ckpt", default="models/forgelm-baseline")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--eval-data", default=str(EVAL_CORPUS))
    parser.add_argument("--fetch-eval", action="store_true", help="download pinned eval corpus (500 held-out stories)")
    parser.add_argument("--tasks", default="perplexity,generation", help="comma-separated: perplexity,generation")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmarks/eval_report.md")
    args = parser.parse_args(argv)

    if args.fetch_eval or not Path(args.eval_data).exists():
        fetch_eval_corpus()
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model_from_checkpoint(args.ckpt)
    texts = load_eval_texts(Path(args.eval_data))
    tasks = [t.strip() for t in args.tasks.split(",")]

    md: list[str] = [
        "# Evaluation Report",
        "",
        f"date: 2026-08-22 | ckpt: `{args.ckpt}` | tasks: {', '.join(tasks)}",
        "",
        "## Reproducibility",
    ]
    md += reproducibility_block(tokenizer, args.ckpt, Path(args.eval_data), args.seed)

    if "perplexity" in tasks:
        result = evaluate_perplexity(model, tokenizer, texts, model.config.context_length)
        md += [
            "",
            "## Perplexity (sliding window, pinned eval corpus)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| perplexity | {result['perplexity']:.3f} |",
            f"| bits-per-byte | {result['bits_per_byte']:.4f} |",
            f"| tokens evaluated | {result['tokens']:,} |",
            f"| stories | {len(texts)} |",
        ]

    if "generation" in tasks:
        md += ["", "## Generations (top-k 50, temp 0.8, max 64 tokens)", ""]
        for prompt in DEFAULT_PROMPTS:
            ids, _ = generate(
                model, tokenizer, prompt, max_tokens=args.max_tokens, seed=args.seed
            )
            continuation = tokenizer.decode(ids[len(tokenizer.encode(prompt)) :])
            generated_ids = ids[len(tokenizer.encode(prompt)) :]
            length = length_sanity(generated_ids, args.max_tokens)
            md += [
                f"### Prompt: {prompt}",
                "",
                f"**Model:** {continuation.strip()}",
                "",
                "| metric | value |",
                "|---|---|",
                f"| tokens | {length['tokens']} |",
                f"| hit cap | {length['hit_cap']} |",
                f"| distinct-1 | {distinct_n(generated_ids, 1):.3f} |",
                f"| distinct-2 | {distinct_n(generated_ids, 2):.3f} |",
                f"| repetition (trigram) | {repetition_rate(generated_ids, 3):.3f} |",
                "",
            ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())