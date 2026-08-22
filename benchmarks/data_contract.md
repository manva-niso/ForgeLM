# Benchmark: Data Contract — Day 1 (2026-08-21)

## Artifacts
- Schema: `forger/data/contract.py` -> `DatasetExample` (text 1..4096 chars, split enum, meta, id)
- CLI: `forger-data-validate <file.jsonl>`
- Sample: `scripts/download_tinystories.py` -> `data/tinystories_sample.jsonl`

## Validation run (2026-08-21)
```
$ forger-data-validate data/tinystories_sample.jsonl
total:      1000
valid:      1000
violations: []
ok:         true
sha256:     7cce2e67d72c5bdd4fe4dbf95aef0dce261c947b88fe4a11eb6b1a085342c610
```

## Contract filtering effect
Streaming ingest attempted 1001 rows; 1 rejected at the contract boundary
(text length > 4096 chars). Filtered at source so the sample is 100% in-contract.

## Tests
`uv run pytest tests/test_contract.py` -> 6 passed
- hypothesis round-trip `model_dump()` -> `model_validate()`: 50 random cases
- blank/whitespace text rejected
- mixed-validity file -> correct report (total/valid/violations/ok)
- missing file -> ok=false (regression: `ok` key bug fixed)

## Reproducibility
- `uv.lock` pins the environment; data sample pinned by sha256 above.
- Re-run: `uv run python scripts/download_tinystories.py && uv run forger-data-validate data/tinystories_sample.jsonl`