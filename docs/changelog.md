# Changelog
Append one line per significant change - do not rewrite this file, only append.

- 2026-08-21 d1: product + data contract - DatasetExample schema, validate CLI, TinyStories sample (1000 rows, sha256 7cce2e67), CI workflow, ADR-01. Tag v0.0.2.
- 2026-08-22 ci: pushed repo, fixed CI (isort known-first-party, forger/data gitignore), added docs/development-log.md. CI green on run 32558803277.
- 2026-08-22 d2: implemented deterministic byte-level BPE tokenizer with save/load, round-trip tests and reference parity benchmark (10/10 vs HF). Tag v0.0.3.
- 2026-08-22 d3: implemented GPT model core (RMSNorm, RoPE, sdpa attention, SwiGLU, KV-cache stub, tied head), 10 tests, CPU forward benchmark, ADR-03. Tag v0.0.4.