"""Download a checkpoint folder from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download-from-hub")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. forge-lm/baseline")
    parser.add_argument("--out", default="checkpoints", help="local output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.repo, local_dir=out / Path(args.repo).name, repo_type="model")
    print(f"downloaded to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())