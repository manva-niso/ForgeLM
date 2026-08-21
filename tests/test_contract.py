import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from forger.data.contract import DatasetExample, Split, validate_file

non_blank = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())


@given(text=non_blank, split=st.sampled_from(list(Split)), has_id=st.booleans())
@settings(max_examples=50)
def test_roundtrip_model_dump(text, split, has_id):
    ex = DatasetExample(text=text, split=split, id="x" if has_id else None)
    assert DatasetExample.model_validate(ex.model_dump()) == ex


@given(text=st.text().filter(lambda s: not s.strip()))
def test_blank_text_rejected(text):
    with pytest.raises(ValidationError):
        DatasetExample(text=text)


def test_validate_file_reports_violations(tmp_path):
    f = tmp_path / "examples.jsonl"
    f.write_text('{"text": "hello", "split": "train"}\n{"text": "   "}\n', encoding="utf-8")
    report = validate_file(f)
    assert report["total"] == 2
    assert report["valid"] == 1
    assert report["ok"] is False


def test_validate_file_missing(tmp_path):
    report = validate_file(tmp_path / "nope.jsonl")
    assert report["ok"] is False