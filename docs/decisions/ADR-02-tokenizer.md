# ADR-02: Tokenizer Design

Status: accepted
Date: 2026-08-29

## Context
Day 2 requires a tokenizer built from scratch for a 4096-vocab small GPT.

## Decision
- Byte-level BPE, GPT-2 style (`forger/tokenizer/bpe.py`).
- IDs: 0 = `<|endoftext|>`, 1..256 = bytes, 257.. = merge results (target 4096, 3839 merges).
- Pretokenization: GPT-2 regex (`regex` package, `\p{L}\p{N}`) - spaces kept inside word tokens.
- Tie-breaking: highest pair frequency, then lexicographically smallest pair. Deterministic.
- Encoding: GPT-2-style rank loop (merge lowest-rank pair until none apply).
- Decoding: concatenate token bytes, then UTF-8 decode once (errors=replace).
- Serialization: `vocab.json` (id -> byte chars), `merges.txt` (rank order), `config.json`.
- HF `tokenizers` used ONLY as parity oracle in `scripts/tokenizer_parity.py`.

## Known limitations
- Training corpus capped at 200,000 chars for CPU speed (89s). Retrain with
  `--max-chars` for a larger corpus if needed.
- Encoder is ~400x slower than HF reference (rank loop rebuilds stats per piece);
  acceptable for Day 2, candidate for optimization on Day 7/10.
- Decoding uses errors=replace, so invalid UTF-8 byte sequences do not crash.

## Consequences
Deterministic, byte-complete tokenizer; parity 10/10 vs HF on test strings
(document in `benchmarks/tokenizer_parity.md`). Artifact at `artifacts/tokenizer/`.