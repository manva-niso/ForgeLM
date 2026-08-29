"""Assemble the HuggingFace Space folder (deploy/space/) for deployment.

Copies the package, tokenizer artifact and story-SFT weights into
deploy/space/ (ignored by the main repo's git). Run before pushing the
Space:

    uv run python scripts/prep_space.py
    cd deploy/space
    git init && git remote add origin https://huggingface.co/spaces/Manvaniso/forgelm
    git add -A && git commit -m "ForgeLM demo" && git push -u origin main
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "deploy" / "space"

COPIES = [
    ("forger", "forger"),
    ("artifacts/tokenizer", "artifacts/tokenizer"),
    ("models/forgelm-sft-story", "models/forgelm-sft-story"),
]

KEEP = ["app.py", "README.md", "requirements.txt"]


def main() -> int:
    SPACE.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        src = SPACE / name
        if not src.exists():
            print(f"missing space file: {src}")
            return 1
    for src_rel, dst_rel in COPIES:
        src = ROOT / src_rel
        dst = SPACE / dst_rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"copied {src_rel} -> deploy/space/{dst_rel}")
    total = sum(f.stat().st_size for f in SPACE.rglob("*") if f.is_file())
    print(f"space folder ready ({total / 1024 / 1024:.1f} MB), push it to HF Spaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())