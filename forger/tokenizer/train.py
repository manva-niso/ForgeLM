"""CLI for tokenizer training and encoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forger.data.contract import DatasetExample
from forger.tokenizer.bpe import BPETokenizer


def _train(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}")
        return 1
    texts: list[str] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            example = DatasetExample.model_validate(json.loads(line))
            texts.append(example.text)
    if not texts:
        print("error: no valid examples in input file")
        return 1
    corpus_chars = sum(len(t) for t in texts)
    print(f"corpus: {len(texts)} examples, {corpus_chars} chars")
    start = time.monotonic()
    tokenizer = BPETokenizer.train(
        texts, vocab_size=args.vocab_size, max_chars=args.max_chars, verbose=True
    )
    elapsed = time.monotonic() - start
    out_dir = tokenizer.save(args.output)
    print(f"vocab size: {len(tokenizer.token_bytes)}")
    print(f"merges: {len(tokenizer.merges)}")
    print(f"training time: {elapsed:.1f}s")
    print(f"saved to: {out_dir}")
    print(f"checksum: {tokenizer.checksum(out_dir)}")
    return 0


def _encode(args: argparse.Namespace) -> int:
    tokenizer = BPETokenizer.load(args.tokenizer)
    ids = tokenizer.encode(args.text)
    decoded = tokenizer.decode(ids)
    print(f"ids: {ids}")
    print(f"round-trip ok: {decoded == args.text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forger-tokenizer", description="ForgeLM BPE tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train a tokenizer on a jsonl corpus")
    train_p.add_argument("--input", required=True, help="path to contract-valid jsonl corpus")
    train_p.add_argument("--output", default="artifacts/tokenizer", help="output directory")
    train_p.add_argument("--vocab-size", type=int, default=4096, help="target vocabulary size")
    train_p.add_argument("--max-chars", type=int, default=200_000, help="cap training characters (default 200000 for speed)")
    train_p.set_defaults(func=_train)

    enc_p = sub.add_parser("encode", help="encode and decode one text string")
    enc_p.add_argument("--tokenizer", required=True, help="tokenizer directory")
    enc_p.add_argument("--text", required=True, help="text to encode")
    enc_p.set_defaults(func=_encode)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())