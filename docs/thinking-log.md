# Thinking Log

Reasoning trail: design choices, alternatives considered, trade-offs and why.
Complements docs/development-log.md (what happened) - this is the "why".
Append per work session; do not rewrite old entries.

## Day 1 - Product + data contract (2026-08-21)

- **Why Pydantic for the contract:** runtime validation at the ingestion boundary
  beats unit-test-only checks; model_validate + model_dump give a lossless
  round-trip invariant we can property-test.
- **Why jsonl over csv/parquet:** line-oriented = streamable, diff-able in git,
  one-record-per-line matches "one validation error per line" reporting.
- **Why sha256 on the sample:** any later change to data becomes detectable;
  reproducibility anchor before any training.
- **Why filter over-long stories at ingest instead of loosening the contract:**
  the 4096-char limit matches the model's future context budget; the data must
  bend to the product spec, not the other way.
- **max_length = 4096 (not 2048):** stories sometimes run long; generous enough
  for TinyStories, still small enough to catch genuinely broken rows.

## Day 2 - BPE tokenizer (2026-08-22)

- **Byte-level over character-level:** character-level breaks on unknown Unicode
  and huge vocabularies; byte-level (GPT-2 style) guarantees completeness - any
  UTF-8 input is representable. No unk problem.
- **bytes_to_unicode instead of raw bytes in the artifact:** makes vocab.json
  text-safe (no binary), and - critically - makes parity testing against HF's
  ByteLevel model a 1:1 comparison rather than a mapping exercise.
- **ID layout (0=special, 1..256=bytes, 257+=merges):** mirrors GPT-2; the
  special token at 0 is cheap insurance for later SFT/chat formatting.
- **Tie-breaking (freq desc, then lexicographic pair):** determinism is a
  contract for the whole pipeline - same corpus must always give the same
  artifact. Chose lexicographic because it is simple to reason about and test.
- **Naive O(merges x corpus) loop first, optimize after:** correctness before
  speed; the 15-min timeout proved the naive version unusable at 800K chars,
  so I flattened the corpus (separator -1), bound locals, and capped default
  training chars at 200K (89s). Trade-off: smaller training corpus, documented
  in ADR-02; retraining with --max-chars is a one-liner.
- **Why merge-into-HF for parity instead of train-both-and-compare:**
  independently trained tokenizers can legitimately differ (tie-breaking,
  corpus order) - that would test nothing. Loading MY vocab+merges into HF
  isolates "does my algorithm apply the same rules the same way" = 10/10.
- **Decoder concatenates bytes THEN decodes once:** decoding per-token would
  corrupt multi-byte characters split across token boundaries.
- **Encode slowness (400x vs HF) accepted for now:** the rank loop rebuilds
  per-piece stats; correctness and determinism matter more on Day 2. Flagged
  as a Day 7/10 optimization candidate.

## Rules going forward

- Every significant design decision gets a "why" entry here with alternatives
  considered.
- Keep entries short (3-8 lines); this is a pointer doc, not a transcript.