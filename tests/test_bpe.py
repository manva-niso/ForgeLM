import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forger.tokenizer.bpe import (
    BASE_VOCAB,
    SPECIAL_TOKEN,
    BPETokenizer,
    count_pairs,
    merge_ids,
    pretokenize,
)

TINY_CORPUS = [
    "low low low low",
    "low lower lowest",
    "lower the lowest low",
    "Caf\u00e9 r\u00e9sum\u00e9 \u4f60\u597d",
]

safe_text = st.text(max_size=200).filter(
    lambda s: all(not 0xD800 <= ord(c) <= 0xDFFF for c in s)
)


def test_count_pairs():
    assert count_pairs([10, 20, 10, 20]) == {(10, 20): 2, (20, 10): 1}


def test_count_pairs_single():
    assert count_pairs([7]) == {}


def test_merge_ids_basic():
    assert merge_ids([10, 20, 10, 20], (10, 20), 256) == [256, 256]


def test_merge_ids_non_overlapping():
    assert merge_ids([10, 20, 20], (10, 20), 256) == [256, 20]


def test_merge_ids_absent():
    assert merge_ids([1, 2, 3], (4, 5), 256) == [1, 2, 3]


def test_merge_ids_at_start_and_end():
    assert merge_ids([10, 20, 5], (10, 20), 256) == [256, 5]
    assert merge_ids([5, 10, 20], (10, 20), 256) == [5, 256]


def test_pretokenize_roundtrip_pieces():
    text = "Hello, world! I don't know. Tab\tand\nnewline."
    pieces = pretokenize(text)
    assert "".join(pieces) == text


def test_train_deterministic():
    first = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    second = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    assert first.merges == second.merges
    assert first.token_bytes == second.token_bytes


def test_train_rejects_small_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train(TINY_CORPUS, vocab_size=BASE_VOCAB + 1)


def test_train_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BPETokenizer.train([], vocab_size=300)


@given(text=safe_text)
@settings(max_examples=100)
def test_roundtrip_hypothesis(text):
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_roundtrip_target_strings():
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    texts = [
        "hello",
        "Hello, world!",
        "Caf\u00e9",
        "\u4f60\u597d\u4e16\u754c",
        "\U0001f642",
        "line one\nline two",
        "multiple    spaces",
        "I don't know.",
    ]
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_token():
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    ids = tokenizer.encode(f"a{SPECIAL_TOKEN}b")
    assert ids[0] == 0 or ids[1] == 0 or ids[2] == 0
    assert tokenizer.decode(ids) == f"a{SPECIAL_TOKEN}b"


def test_encode_shape():
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    ids = tokenizer.encode("Hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_decode_unknown_id():
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        tokenizer.decode([9999])


def test_save_load(tmp_path):
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    tokenizer.save(tmp_path)
    loaded = BPETokenizer.load(tmp_path)
    assert loaded.merges == tokenizer.merges
    for text in ["hello", "Caf\u00e9", "\u4f60\u597d", "a  b\nc"]:
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


def test_checksum_stable(tmp_path):
    tokenizer = BPETokenizer.train(TINY_CORPUS, vocab_size=300)
    tokenizer.save(tmp_path)
    assert tokenizer.checksum(tmp_path) == tokenizer.checksum(tmp_path)