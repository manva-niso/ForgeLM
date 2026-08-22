"""Data contract for ForgeLM: typed schema + validation CLI."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetExample(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    split: Split = Split.TRAIN
    meta: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


def validate_file(path: Path) -> dict[str, Any]:
    report = {"file": str(path), "total": 0, "valid": 0, "violations": []}
    if not path.exists():
        report["violations"].append({"error": "file not found"})
        report["ok"] = False
        return report
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            report["total"] += 1
            try:
                DatasetExample.model_validate(json.loads(line))
                report["valid"] += 1
            except Exception as exc:  # noqa: BLE001
                report["violations"].append({"line": line_no, "error": str(exc)})
    report["ok"] = report["total"] > 0 and report["valid"] == report["total"]
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: forger-data-validate <examples.jsonl>")
        return 2
    report = validate_file(Path(argv[0]))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())