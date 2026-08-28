"""Greedy smoke inference using the Engine (KV-cache).

Usage:
    uv run python scripts/smoke_infer.py --ckpt models/forgelm-baseline --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from forger.serve.engine import Engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke-infer")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    engine = Engine.from_checkpoint(Path(args.ckpt), args.tokenizer)
    text, _, stats = engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
    )
    print("prompt:", args.prompt)
    print("full:", text)
    print(f"({stats['generated_tokens']} tokens, {stats['tokens_per_sec']:.0f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())