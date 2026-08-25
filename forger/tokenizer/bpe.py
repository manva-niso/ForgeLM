"""Deterministic byte-level BPE tokenizer (GPT-2 style), implemented from scratch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import regex

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKEN = "<|endoftext|>"
BASE_VOCAB = 256
SPECIAL_ID = 0


def bytes_to_unicode() -> dict[int, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(BASE_VOCAB):
        if b not in bs:
            bs.append(b)
            cs.append(BASE_VOCAB + n)
            n += 1
    return dict(zip(bs, cs))


BYTE_TO_CHAR = {b: chr(c) for b, c in bytes_to_unicode().items()}
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def pretokenize(text: str) -> list[str]:
    return regex.findall(GPT2_PATTERN, text)


def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    stats: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if ids[i] == a and i + 1 < n and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        token_bytes: dict[int, bytes],
        merges: dict[tuple[int, int], int],
        byte_to_id: dict[int, int],
    ) -> None:
        self.token_bytes = token_bytes
        self.merges = merges
        self.byte_to_id = byte_to_id
        self.merges_ranked = sorted(merges.items(), key=lambda kv: kv[1])
        self._ranks = {pair: rank for pair, rank in merges.items()}

    @classmethod
    def train(cls, texts: list[str], vocab_size: int, max_chars: int | None = None, verbose: bool = False) -> BPETokenizer:
        if vocab_size <= BASE_VOCAB + 1:
            raise ValueError(f"vocab_size must be > {BASE_VOCAB + 1}")
        corpus = "".join(texts)
        if max_chars is not None:
            corpus = corpus[:max_chars]
        if not corpus:
            raise ValueError("empty corpus")
        pieces = [c.encode("utf-8") for c in pretokenize(corpus)]
        ids: list[int] = []
        for piece in pieces:
            ids.extend(b + 1 for b in piece)
            ids.append(-1)
        byte_to_id = {b: b + 1 for b in range(BASE_VOCAB)}
        token_bytes = {i: bytes([i - 1]) for i in range(1, BASE_VOCAB + 1)}
        merges: dict[tuple[int, int], int] = {}
        num_merges = vocab_size - BASE_VOCAB - 1
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            stats_get = stats.get
            ids_local = ids
            for j in range(len(ids_local) - 1):
                pair = (ids_local[j], ids_local[j + 1])
                if pair[0] == -1 or pair[1] == -1:
                    continue
                stats[pair] = stats_get(pair, 0) + 1
            if not stats:
                break
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            new_id = BASE_VOCAB + 1 + i
            merges[best_pair] = new_id
            token_bytes[new_id] = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
            ids = cls._merge_all(ids, best_pair, new_id)
            if verbose and (i + 1) % 500 == 0:
                print(f"merge {i + 1}/{num_merges} best={best_pair}")
        token_bytes[SPECIAL_ID] = SPECIAL_TOKEN.encode("utf-8")
        return cls(token_bytes=token_bytes, merges=merges, byte_to_id=byte_to_id)

    @staticmethod
    def _merge_all(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        out_append = out.append
        a, b = pair
        i = 0
        n = len(ids)
        while i < n:
            if ids[i] == a and i + 1 < n and ids[i + 1] == b:
                out_append(new_id)
                i += 2
            else:
                out_append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        parts = text.split(SPECIAL_TOKEN)
        for i, part in enumerate(parts):
            if part:
                ids.extend(self._encode_piece(part))
            if i < len(parts) - 1:
                ids.append(SPECIAL_ID)
        return ids

    def _encode_piece(self, text: str) -> list[int]:
        piece_ids = [self.byte_to_id[b] for b in text.encode("utf-8")]
        return self._bpe(piece_ids)

    def _bpe(self, ids: list[int]) -> list[int]:
        ranks = self._ranks
        while len(ids) > 1:
            stats = count_pairs(ids)
            best_pair = min(stats, key=lambda p: ranks.get(p, 10**9))
            if best_pair not in ranks:
                break
            ids = merge_ids(ids, best_pair, ranks[best_pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            try:
                raw += self.token_bytes[token_id]
            except KeyError:
                raise ValueError(f"unknown token id {token_id}") from None
        return raw.decode("utf-8", errors="replace")

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        vocab = {str(i): self._to_chars(b) for i, b in self.token_bytes.items()}
        vocab_path = out / "vocab.json"
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        merges_path = out / "merges.txt"
        lines = [
            f"{self._to_chars(self.token_bytes[a])} {self._to_chars(self.token_bytes[b])}"
            for (a, b), rank in sorted(self.merges.items(), key=lambda kv: kv[1])
        ]
        merges_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        config = {"vocab_size": len(self.token_bytes), "pretokenize": "gpt2", "version": 1}
        (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        d = Path(directory)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        token_bytes = {int(k): cls._from_chars(v) for k, v in vocab.items()}
        merges: dict[tuple[int, int], int] = {}
        id_to_int = {v: int(k) for k, v in vocab.items()}
        for rank, line in enumerate((d / "merges.txt").read_text(encoding="utf-8").splitlines()):
            if not line:
                continue
            a, b = line.split(" ")
            pair = (id_to_int[a], id_to_int[b])
            merges[pair] = BASE_VOCAB + 1 + rank
        byte_to_id = {b[0]: i for i, b in token_bytes.items() if len(b) == 1}
        return cls(token_bytes=token_bytes, merges=merges, byte_to_id=byte_to_id)

    @staticmethod
    def _to_chars(data: bytes) -> str:
        return "".join(BYTE_TO_CHAR[b] for b in data)

    @classmethod
    def _from_chars(cls, text: str) -> bytes:
        return bytes(CHAR_TO_BYTE[c] for c in text)

    def checksum(self, directory: str | Path) -> str:
        d = Path(directory)
        h = hashlib.sha256()
        for name in ("config.json", "vocab.json", "merges.txt"):
            h.update((d / name).read_bytes())
        return h.hexdigest()