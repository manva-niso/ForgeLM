# Development Log

Running log of implementations and errors, with file-path references.
Append after every significant work session. Do not rewrite old entries.
Dates follow the PLAN calendar: D1=2026-08-21 (Fri) ... D9=2026-08-29 (Sat),
D10=08-30, D11=08-31, D12=09-01.

## Path map (key files)

| Area | Path |
|---|---|
| Data contract (schema + CLI) | `forger/data/contract.py` |
| Contract tests | `tests/test_contract.py` |
| TinyStories downloader | `scripts/download_tinystories.py` |
| Product contract | `docs/PRODUCT_CONTRACT.md` |
| Module status | `docs/modules.md` |
| Benchmarks / evidence | `benchmarks/` |
| Architecture decisions | `docs/decisions/` |
| CI workflow | `.github/workflows/ci.yml` |
| Project config (deps, ruff, pytest, scripts) | `pyproject.toml` |
| Reproducibility pins | `uv.lock` |
| Gitignore (artifacts + local agent context) | `.gitignore` |

---

# Implementation Log (detailed)

## Day 1 — Fri 2026-08-21 — Product + data contract (tags v0.0.1, v0.0.2)

**What shipped:**
- Repo bootstrap: uv project (Python 3.12, hatchling), dev group
  (ruff/pytest/hypothesis/pytest-benchmark), GitHub Actions CI (ruff + pytest
  on every push), `.gitignore`, README. Tag v0.0.1.
- `docs/PRODUCT_CONTRACT.md`: the product spec — edge assistant, <=25M params,
  <=8MB after 4-bit quant, <100ms first-token on CPU, 9 acceptance criteria.
- `docs/decisions/ADR-01-project-charter.md`: charter (CPU-first dev + Kaggle
  GPU, provisional TinyStories data, vertical-slice rule).
- `forger/data/contract.py`: Pydantic `DatasetExample` (text 1..4096 chars,
  split enum train/validation/test, meta dict, optional id) with a
  `text_not_blank` field validator; `validate_file()` producing a machine
  report {total, valid, violations[], ok}; CLI `forger-data-validate`
  (exit 0 ok / 1 invalid / 2 usage).
- `scripts/download_tinystories.py`: streams 1000 TinyStories rows,
  contract-validates each, writes `data/tinystories_sample.jsonl` + prints a
  sha256 checksum (7cce2e67...) as the reproducibility anchor.
- `tests/test_contract.py`: hypothesis round-trip (50 cases), blank-text
  rejection, file-level validation reporting, missing-file handling.
- Evidence: `benchmarks/data_contract.md` — 1000/1000 rows valid.

**Notable decisions:** jsonl over csv (streamable, one record per line = one
validation error per line); sha256 on the sample (any change detectable);
filter over-long stories instead of loosening the contract (data bends to the
spec). Pushed repo + made CI green (see error log).

## Day 2 — Sat 2026-08-22 — BPE tokenizer from scratch (tag v0.0.3)

**What shipped:**
- `forger/tokenizer/bpe.py`: deterministic byte-level BPE (GPT-2 style).
  - ID layout: 0 = `<|endoftext|>`, 1..256 = raw bytes, 257.. = merge results
    (target 4096 → 3839 merges).
  - GPT-2 `bytes_to_unicode` mapping (artifact is text-safe + HF-parity-ready).
  - GPT-2 pre-tokenization regex (`regex` package, `\p{L}\p{N}`).
  - Training: flattened corpus with -1 separators, pair counting per merge,
    tie-break = highest frequency then lexicographically smallest pair
    (determinism contract).
  - Encoding: rank-based merge application; Decoding: concatenate ALL token
    bytes then UTF-8 decode ONCE (multi-byte characters span tokens).
  - Save/load: `vocab.json`, `merges.txt` (rank order), `config.json`;
    `checksum()` for reproducibility.
- `forger/tokenizer/train.py` CLI: `train --input --output --vocab-size
  --max-chars` and `encode` modes.
- `tests/test_bpe.py` (17 tests): pair counting, merge ops (start/end/
  repeated/absent), determinism, hypothesis round-trips (100 unicode cases),
  special tokens, save/load, invalid inputs.
- `scripts/tokenizer_parity.py` + `benchmarks/tokenizer_parity.md`: loaded OUR
  vocab + merges into a HuggingFace BPE (same pre-tokenizer) — **10/10 exact
  match** on fixed strings; encode-speed table.
- Artifact trained on 200K chars of TinyStories in 89s (checksum
  e20ed6346c49fc5d...).

**Notable decisions:** byte-level (no unknown-character problem); parity via
"load my artifact into HF" (not independent training — that would test
nothing); slowness accepted initially (correctness first), optimized on Day 5.

## Day 3 — Sun 2026-08-23 — Model core (GPT blocks) (tag v0.0.4)

**What shipped:**
- `forger/model/config.py`: GPTConfig (vocab 4096, d_model 256, n_heads 4,
  n_layers 4, context 512, ffn_mult 4) with __post_init__ validation
  (d % heads == 0, positive dims), to_dict/from_dict.
- `forger/model/blocks.py`:
  - RMSNorm (weight only, eps 1e-5) — cheaper than LayerNorm, LLaMA-style.
  - RotaryEmbedding (RoPE): base 10,000, inv-freq duplicated across the pair
    dimension, precomputed cos/sin tables, `offset` parameter for decode.
  - CausalSelfAttention: c_attn (d→3d, bias) → Q/K/V split → (B,H,T,head_dim)
    → RoPE → `F.scaled_dot_product_attention` (causal) → c_proj; KV-cache
    concat + explicit float(-inf) triu mask when T_q != T_k.
  - MLP: SwiGLU (gate/up/down, SiLU, hidden 4d, no bias).
  - Block: pre-norm residuals.
- `forger/model/gpt.py`: GPT (tied token_embedding/lm_head, ln_f, KV-cache
  dict, context-length guard), `init_weights` (N(0,0.02), residual projections
  scaled 1/sqrt(2n)), save/load.
- `tests/test_model.py` (10): shapes [B,T,V], causal no-peek, gradient flow
  finite, determinism, tied head, exact param count (5,249,280), context
  enforcement, save/load roundtrip, KV-cache == full forward (prefill + decode
  chunks).
- `scripts/benchmark_model.py` → `benchmarks/model_forward.md`: 210.9ms
  forward for B=2/T=512, 4,855 tok/s CPU.

**Notable decisions:** RoPE over learned positional embeddings (no params,
generalizes); SwiGLU cost the 3rd projection (params 5.25M vs 3.6M estimate —
documented, still 5x under the 25M cap); sdpa for fused kernels; the cache
proven bit-identical to full forward on Day 3 so Day 7 could trust it.

## Day 4 — Mon 2026-08-24 — Training pipeline (tag v0.0.5)

**What shipped:**
- `forger/train/config.py`: TrainConfig dataclass + from_yaml (steps, bs, ctx,
  lr, wd, warmup, grad_accum, eval/log intervals, device, seed, checkpoint
  dir, run name, windows_per_story).
- `forger/train/dataset.py`: WindowDataset — contiguous ctx-windows from
  encoded stories, labels = input shifted by 1 (next-token prediction);
  deterministic `get_batch(step, bs)` indexing → EXACT resume; optional
  pre-encoded ids (avoid double tokenization).
- `forger/train/trainer.py`: Trainer — AdamW (0.9/0.95, wd 0.1), linear
  warmup → cosine decay to 10% floor, gradient accumulation, eval loop on
  held-out windows, checkpoint {model+optimizer+step+configs}, TensorBoard,
  JSON logs, `train(until=N)` for partial runs, model.to(device), optimizer
  over requires_grad params only (enables LoRA later).
- `configs/train/{smoke,baseline}.yaml`.
- `tests/test_trainer.py` (6): window contiguity, LR schedule math, loss
  decreases on 10-step smoke, BIT-EXACT resume (partial+restore == full run),
  checkpoint contents, model device placement.
- Evidence: `benchmarks/train_smoke.md` — 10 steps in 8.1s, loss 8.11 → 7.36
  (initial loss matches theory ln(4096) ≈ 8.32).

**Notable decisions:** deterministic indexing over shuffled iterators (resume
is trivially exact); AMP fp16+CUDA only — CPU measured 7x SLOWER with bf16
(see errors); encode once, reuse for train+eval.

## Day 5 — Tue 2026-08-25 — Baseline training + the Kaggle saga (tags v0.0.6..v0.1.1)

**What shipped:**
- `scripts/kaggle_baseline.py`: one entry point for CPU (local) and CUDA
  (Kaggle); `--data stream` reads full TinyStories via datasets streaming;
  sys.path fix; encode progress; config-driven.
- `scripts/kaggle_baseline.ipynb`: importable Kaggle notebook (wipe+re-clone
  the repo, `git checkout kaggle`, install deps, load HF_TOKEN via
  kaggle_secrets, run, zip fallback).
- `scripts/download_from_hub.py`: checkpoint pull from HF Hub.
- Branch architecture: `main` (local CPU) vs `kaggle` (GPU config + notebook
  edition), merged after every change.
- CPU evidence run: 300 steps / 358s / loss 8.11 → 3.34 (train), 3.95 (eval).
- **Kaggle T4 run (the good one):** 20,000 streamed stories, 293,200 train
  windows, 4000 steps bs64 ctx128 → best eval **2.0247**, no memorization;
  checkpoint pushed to HF `Manvaniso/forgelm` and pulled back locally.
- Weights committed to the repo: `models/forgelm-baseline/` (24.3MB) so the
  model is usable with zero accounts (README 3-step usage).
- Memorization fixes shipped: windows_per_story 16/4, eval-rise warning,
  best-eval checkpoint (best_model.pt), encoder 20x speedup, ctx 128.
  All detailed in the error log — this day was mostly debugging.

**Notable decisions:** CPU-first baseline (vertical-slice rule: evidence today,
not "whenever GPU works"); one script two targets (device flag is the only
difference); HF Hub as the artifact bus; unique-data volume is the lever
(more stories beats more steps — established after two wrong theories).

## Day 6 — Wed 2026-08-26 — Evaluation harness (tag v0.2.0)

**What shipped:**
- `forger/eval/perplexity.py`: sliding-window causal perplexity (stride =
  ctx/2, only fresh tokens counted per window) + bits-per-byte
  (NLL / (ln2 × bytes) — natural for a byte-level tokenizer).
- `forger/eval/generation.py`: top_k_filter, temperature, seeded
  `sample_token`, `generate()` with KV-cache + context-limit stop +
  stopped/context_limited flags.
- `forger/eval/metrics.py`: distinct-1/2/3, trigram repetition rate, length
  sanity.
- `forger/eval/run.py` CLI: tasks perplexity,generation; `--fetch-eval`
  builds the PINNED eval corpus (500 held-out TinyStories VALIDATION stories,
  contract-validated, sha256 sidecar committed in benchmarks/eval_corpus.sha256).
- `forger/model/checkpoint.py`: shared loader (best_model.pt > model.pt >
  Trainer checkpoint) — used by eval, engine, smoke_infer.
- `tests/test_eval.py` (10): ppl matches manual CE math, bpb math, sliding
  window on long text, top-k/temperature validity, seeded determinism,
  distinct/repetition/length metrics.
- Evidence: `benchmarks/eval_report.md` — **ppl 8.449 / bpb 1.0366** on
  136,428 tokens (500 stories) + 5-prompt generation table with metrics
  (distinct-2 0.89-0.97, repetition <=0.05).

**Notable decisions:** production-grade eval needs PINNED data (checksummed),
not re-streamed random data; report carries a reproducibility block (torch
version, checkpoint, tokenizer checksum, corpus sha256, seed).

## Day 7 — Thu 2026-08-27 — Inference engine (tag v0.2.1)

**What shipped:**
- `forger/serve/engine.py`: Engine — prefill (whole prompt, builds cache),
  decode_next (one token, ~constant cost via cache), generate (sampling
  params, stats incl. tok/s); `from_checkpoint`; shares samplers with eval.
- `forger/serve/optimize.py`: `quantize_dynamic` (Linear → qint8) and
  `torch.compile` with graceful fallback; `model_size_mb`.
- `scripts/bench_engine.py`: fixed protocol (59-token prompt prefill + 50
  decode tokens × 3 repeats × 3 variants) → `benchmarks/serve_speedup.{md,json}`
  + `scripts/smoke_infer.py` refactored onto Engine.
- `tests/test_serve.py` (7): engine == reference greedy, context-limit stop,
  seeded determinism, engine == eval.generate, int8 logits within tolerance,
  int8 smaller, compile skip-if-unavailable.
- Results: fp32 **123 tok/s, first-token 10ms, 20MB**; int8 **4MB (5x smaller,
  meets <=8MB spec)** but 1.22x SLOWER decode (quantized kernel overhead on
  small matrices — size win, not speed); compile unavailable (no MSVC).

**Notable decisions:** int8 kept for size (product spec), speed path deferred
to Day 10 (ONNX Runtime); honest benchmark table including the slower variant;
the engine reuses the Day-3-proven cache.

## Day 8 — Fri 2026-08-28 — LoRA SFT (+ the dolly lesson) (tags v0.3.0, v0.3.2)

**What shipped:**
- `forger/ft/lora.py`: LoRALinear (frozen base + A (r×in, kaiming) + B
  (out×r, zeros) + alpha/r scaling), apply_lora (targets c_attn/c_proj/gate/
  up/down; ModuleList-safe recursive walk; freezes EVERYTHING else incl.
  RMSNorms), count_lora_params, merge_lora (fold W + (alpha/r)·B·A),
  convert_merged (LoRALinear → plain Linear so checkpoints load as normal GPT).
- `forger/ft/sft_data.py` (dolly, later superseded) + `forger/ft/story_sft_data.py`
  (THE shipped data: TinyStories formatted as
  "### Instruction: Write a story about {topic}\n### Response: {story}").
- `forger/ft/train_sft.py`: loads baseline, applies LoRA, trains (CPU/CUDA),
  saves Trainer checkpoint + merged model. Trainer's requires_grad-only
  optimizer reused unchanged.
- `scripts/kaggle_sft.py`: GPU runner (streams TinyStories, story-domain
  instructions, 15K examples, 3000 steps, int4 export, HF push).
- Models committed: `models/forgelm-sft-story/` (fp32 21.3MB) +
  `models/forgelm-sft-story-4bit/` (int4 6.27MB) — instruction-following
  works: "Write a story about a cat" → coherent story (loss 1.96, eval 2.00).
- Evidence: `benchmarks/sft_qualitative.md` (dolly failure + story-SFT success).

**Notable decisions:** LoRA over full FT (pattern + plain-GPT artifact);
story-domain data over dolly (THE lesson of the day — SFT teaches form, not
facts; a TinyStories-only base has no general knowledge); dolly experiment
documented as evidence, not deleted.

## Day 9 — Sat 2026-08-29 — QLoRA int4 (tag v0.3.1)

**What shipped:**
- `forger/quant/quantize.py`: block-wise symmetric int4 — per-64 absmax
  scale, codes in [-8, 7], PACKED 2 codes/byte (pack_4bit/unpack_4bit);
  `Int4Linear` (stores codes+scales, dequantizes to fp32 for forward;
  `from_stored` reconstructs without re-quantizing); `quantize_model_4bit`
  (replaces Linears except lm_head — keeps the fp32 tied head); export_4bit /
  load_4bit (full non-int4 state carried); storage_size_mb.
- `forger/quant/qlora.py`: QLoRA-in-spirit training (int4 frozen base + fp32
  LoRA adapters; merged result re-exported to int4).
- `tests/test_quant.py` (6): roundtrip error bound, pack/unpack roundtrip,
  Int4Linear ≈ Linear, replacement coverage, storage halving, export/load
  roundtrip.
- Models: `models/forgelm-4bit/` (baseline int4) + the story-SFT int4
  (6.27MB, <=8MB spec met).
- Evidence: `benchmarks/qlora.md` — size table (20MB → 4-6.3MB) + quality:
  **ppl 7.88 → 8.11 (+2.9%)** for 5x size; 4 quantization bugs found & fixed.

**Notable decisions:** symmetric int4 over NF4 (measured +2.9% acceptable;
NF4 = documented refinement, not a rewrite); storage-4-bit/compute-fp32
(weight-only; the paper's on-the-fly kernels matter only for >1B models —
honestly documented); export carries the FULL non-int4 state (RMSNorms!).

---

# Error Log (detailed)

## Day 1 — Fri 2026-08-21

### E1.1 `forger-data-validate: program not found`
- **Symptom:** CLI installed in venv but the console script wasn't created.
- **Investigation:** `[project.scripts]` was never in pyproject.toml; the
  entry point existed only in the plan.
- **Root cause:** pyproject.toml missing the `[project.scripts]` table.
- **Fix:** added `forger-data-validate = "forger.data.contract:main"` and
  re-synced the package (`uv sync --reinstall-package forge-lm`).
- **Impact:** CLI works from the shell; tests never caught it (they import the
  function directly).

### E1.2 `test_validate_file_missing` — KeyError 'ok'
- **Symptom:** test for a nonexistent file crashed with KeyError.
- **Root cause:** the missing-file branch returned early without setting the
  `ok` key the report contract promises.
- **Fix:** `report["ok"] = False` before the early return
  (`forger/data/contract.py`).
- **Impact:** contract completeness enforced — every report has total/valid/
  violations/ok.

### E1.3 1/1000 TinyStories rows rejected (text > 4096 chars)
- **Symptom:** the sample download produced a row the contract rejected.
- **Root cause:** the dataset contains over-long stories; the contract is
  stricter than the data.
- **Fix:** the downloader filters out-of-contract rows at ingest (skipped: 1)
  instead of loosening the contract — data bends to the spec.
- **Impact:** sample is 1000/1000 contract-valid; lesson reused on Day 8.

### E1.4 CI red: ruff I001 ×2 + `forger.data` missing on fresh checkout
- **Symptom:** GitHub Actions failed on (a) import-order lint, (b) tests with
  `ModuleNotFoundError: No module named 'forger.data'` — while everything
  passed locally.
- **Investigation:** reproduced CI in Docker (fresh Linux clone): ruff treated
  `forger` as third-party (not first-party) → I001; and `forger/data/*.py`
  were NOT tracked at all.
- **Root cause:** (a) no isort known-first-party config; (b) the `.gitignore`
  pattern `data/` matched ANY directory named data — including
  `forger/data/` — so the package never got committed. (c) EXE002 seen in
  Docker was an NTFS-mount artifact, not real.
- **Fix:** `[tool.ruff.lint.isort] known-first-party = ["forger"]`;
  `.gitignore` `data/` → `/data/`; committed `forger/data/{__init__,contract}.py`.
- **Impact:** CI green (run 32558803277). Big lesson: a bare ignore pattern
  can silently exclude source; always test on a fresh clone.

## Day 2 — Sat 2026-08-22

### E2.1 save/load test — KeyError 104
- **Symptom:** loading the tokenizer then encoding "hello" crashed with
  KeyError: 104 (the byte value of 'h').
- **Root cause:** `load()` built `byte_to_id` keyed by the 1-byte BYTES
  object (`b`) instead of the byte VALUE (`b[0]`).
- **Fix:** `byte_to_id = {b[0]: i for i, b in token_bytes.items() if len(b) == 1}`.
- **Impact:** save/load now round-trips exactly (deterministic artifact).

### E2.2 training — KeyError: -1
- **Symptom:** tokenizer training crashed mid-merge.
- **Root cause:** the pair counter only skipped pairs where the SEPARATOR
  (-1) was the second element; pairs like (-1, x) got counted and selected.
- **Fix:** skip any pair containing -1.
- **Impact:** piece boundaries never merge across stories.

### E2.3 training timeout > 15 minutes on 800K chars
- **Symptom:** training the 4096-vocab tokenizer on the sample exceeded the
  15-min command timeout with no output.
- **Investigation:** the naive loop re-scanned the whole corpus per merge —
  3839 merges × ~1M positions in pure Python.
- **Fix:** flattened corpus into one list with -1 separators, bound local
  variables in hot loops, default `--max-chars 200000` (documented trade-off).
- **Impact:** 89s training; retraining with more chars = one flag (used for
  the real artifact).

### E2.4 HF `models.BPE(vocab=...)` TypeError (tokenizers 0.23)
- **Symptom:** parity script crashed: `'str' object cannot be interpreted as
  an integer`.
- **Root cause:** the new tokenizers API expects `Dict[token, int]`, we passed
  id→token.
- **Fix:** inverted vocab + merges as tuple list.
- **Impact:** parity harness works; 10/10 match achieved.

### E2.5 encode 12.6s/pass slow
- **Symptom:** encoding 50 stories took 12.6s.
- **Root cause:** the merge-rank dict was rebuilt inside `_bpe` PER PIECE.
- **Fix:** cached `self._ranks` once in `__init__`.
- **Impact:** 8.3s/pass (~1.5x). (Further 20x came on Day 5.)

## Day 3 — Sun 2026-08-23

### E3.1 RoPE shape error (64 vs 32)
- **Symptom:** attention crashed: tensor a (64) vs b (32) in `q * cos`.
- **Root cause:** cos/sin cached at dim/2 (32) but applied to the full
  head_dim (64).
- **Fix:** duplicate each frequency across the pair dimension
  (`torch.cat([freqs, freqs], dim=-1)`).
- **Impact:** RoPE correct; all forward tests green.

### E3.2 KV-cache decode mismatch (~1e-3)
- **Symptom:** single-token decode with cache differed from full forward.
- **Root cause:** RoPE rotated the decode token as if it were position 0
  instead of its global position.
- **Fix:** `rope(q, k, offset=past_len)`.
- **Impact:** cache decode == full forward (test proves it).

### E3.3 KV-cache chunk mismatch (~0.5)
- **Symptom:** decoding a multi-token chunk with cache diverged badly.
- **Investigation:** two traps: (1) `is_causal=True` builds a mask assuming
  T_q == T_k — with cached keys present it masks the PAST; (2) PyTorch's
  sdpa BOOLEAN mask means True = ALLOWED (opposite of intuition).
- **Fix:** explicit float(-inf) triu mask over the new-chunk columns only;
  documented forever in ADR-03.
- **Impact:** chunked decode ≡ full forward (atol 1e-5).

### E3.4 test param count mismatch (5,249,280 vs 5,250,304)
- **Root cause:** test formula counted 3 RMSNorms per block; blocks have 2
  (ln_1, ln_2) + final ln_f.
- **Fix:** corrected the test formula (the model was right).
- **Impact:** exact param accounting is now enforced by test.

## Day 4 — Mon 2026-08-24

### E4.1 smoke CLI: 15-min timeout, zero output
- **Symptom:** `python -m forger.train.trainer --steps 10` produced nothing
  and timed out.
- **Investigation:** two compounding costs — (a) the sample was encoded
  TWICE (once in main, once inside WindowDataset) ≈ 6 min; (b) bf16 CPU
  autocast measured **7x slower** than fp32 (1.35s vs 0.19s per 1024-token
  forward).
- **Root cause:** assumed "bf16 is faster on CPU" (it isn't on this
  hardware); double tokenization.
- **Fix:** CPU runs plain fp32 (AMP stays CUDA-only); `encoded_ids` param
  (encode once, reuse); `configs/train/smoke.yaml` (bs 4, ctx 256).
- **Impact:** 10-step smoke in 8.1s. Convention lost to measurement.

### E4.2 resume test: empty loss_history
- **Symptom:** resumed trainer ran ZERO steps.
- **Root cause:** the test mutated the SHARED config object
  (`trainer_partial.config.steps = 5`) which also set the resumed trainer's
  steps → `range(5, 5)`.
- **Fix:** separate `cfg_partial` object.
- **Impact:** test isolation lesson — never mutate a shared config.

### E4.3 resume test: loss mismatch (5.4476 vs 5.2914)
- **Root cause:** the partial run used a 5-step LR schedule (cosine
  denominator = total steps), so its weights diverged from the full run.
- **Fix:** `train(until=N)` keeps the full-run schedule; partial runs don't
  change config.steps.
- **Impact:** exact-resume semantics now tested properly.

### E4.4 resume test: 0.02 residual diff
- **Root cause:** test models created from DIFFERENT random inits — after
  load, state matched but the comparison was invalid by construction.
- **Fix:** seed both model creations identically.
- **Impact:** bit-exact resume proven (loss histories identical).

## Day 5 — Tue 2026-08-25 — the Kaggle saga (biggest debugging day)

### E5.1 Kaggle run 1: `ModuleNotFoundError: No module named 'forger'`
- **Symptom:** `python scripts/kaggle_baseline.py` failed to import forger.
- **Root cause:** running a script from scripts/ puts `scripts/` on sys.path,
  not the repo root; locally it worked because the package is INSTALLED in
  the venv (uv sync); on Kaggle it isn't.
- **Fix:** `sys.path.insert(0, repo_root)` at the top of the script.
- **Impact:** scripts are import-robust anywhere. (Also capped max-stories at
  2000 — encoding 50K stories would have taken ~2.4h at the then-encoder
  speed.)

### E5.2 Kaggle run 2: same error again
- **Symptom:** identical ModuleNotFoundError after the fix was pushed.
- **Investigation:** the notebook runs `git clone` every time; the folder from
  run 1 still existed → clone failed SILENTLY → old code ran.
- **Fix:** notebook wipes `/kaggle/working/forge-lm` before cloning + an
  explicit `import forger` check.
- **Impact:** every run starts from a clean checkout.

### E5.3 Kaggle run 3: `index on cuda:0, weights on cpu`
- **Symptom:** training crashed on device mismatch.
- **Root cause:** Trainer moved input tensors to config.device but never the
  MODEL — worked on CPU (everything same device), broke on GPU.
- **Fix:** `model.to(device)` in Trainer.__init__ + device regression test.
- **Impact:** one code path for CPU/GPU.

### E5.4 HF still "unauthenticated" despite the secret
- **Symptom:** "You are sending unauthenticated requests to the HF Hub" and
  push failures.
- **Root cause:** `UserSecretsClient().get_secret()` RETURNS a value but does
  NOT export it to the environment — the `!python` subprocess never saw
  HF_TOKEN.
- **Fix:** notebook sets `os.environ["HF_TOKEN"] = get_secret(...)` before
  training (with fallback secret name `forgelm`).
- **Impact:** pushes work end-to-end.

### E5.5 `getcwd() failed: No such file or directory` during git clone
- **Symptom:** clone crashed; os.chdir failed after it.
- **Root cause:** the notebook kernel's CURRENT DIRECTORY was INSIDE the
  folder we `rmtree`d — every child process inherited a dead cwd.
- **Fix:** `os.chdir("/kaggle/working")` BEFORE deleting anything.
- **Impact:** recovery step for any session that deletes its own cwd.

### E5.6 Kaggle run 4: catastrophic memorization (train → 0.01, eval → 7.7)
- **Symptom:** train loss collapsed toward zero while eval loss ROSE past the
  random baseline (5.1 → 7.7).
- **Investigation:** 2000 stories × 1 window = ~2K unique windows; bs 64 ×
  4000 steps cycled them ~128 times. A 5.25M model memorizes 512K tokens
  easily; once memorized it is overconfident on unseen text.
- **Fix attempt 1:** windows_per_story 16/4 + eval-rise warning in Trainer.
- **Impact:** (didn't fully work — see E5.8; the real root cause was deeper).

### E5.7 A/B experiments — and the first wrong theory
- **Experiment:** same data/steps, only windows-per-story differs (A=1, B=16).
  - 50 stories: A and B BOTH collapse (0.14/7.46 and 0.15/7.84).
  - 400 stories: both "healthy" at 600 steps (2.71/4.92, 2.80/4.89).
- **Wrong conclusion (corrected):** "16x unique data" — FALSE. Windows cropped
  from the same story overlap almost completely; unique data = stories ×
  tokens, not windows × tokens. B at 1200 steps still showed eval rising
  (5.09 → 5.66). The real lever is STORY COUNT, gated by encoder speed.

### E5.8 encoder 37.3s/20 stories — the 20x speedup
- **Symptom:** encoding 20 stories took 37.3s (0.62s/story) — making large
  corpora infeasible.
- **Investigation (cProfile):** 22.6M dict gets + 6.6M lambda calls per 3
  encodes — `_bpe` rebuilt pair dicts and ran `min(...)` every merge pass.
- **Fix:** single-pass rank scan with inline best-pair search + per-tokenizer
  piece cache (100K entries).
- **Impact:** 37.3s → 1.8s (**20x**); parity still 10/10; tests green.
  20,000 stories now encode in ~8 min instead of ~2.5h.

### E5.9 Kaggle run 5 STILL memorizing — `train windows: 378`
- **Symptom:** even after the fixes, the run printed `train windows: 378`
  (should be ~30K) and memorized again.
- **Root cause:** TWO stacked bugs —
  1. `n = min(windows_per_story, len(ids)//ctx)` silently capped crops at 1
     for any story shorter than 2× ctx — the 16-window multiplier NEVER fired.
  2. ctx 256 discards every story < 257 tokens — **91% of TinyStories**
     (average story ≈ 150 tokens). Only 378/1900 stories qualified.
- **Fix:** windows_per_story now unconditional (16 crops per qualifying
  story); kaggle config ctx 256 → 128 (97% qualify; 6,176 vs 386 windows for
  400 stories); best-eval checkpoint as overfit safeguard.
- **Impact:** validation run showed eval DECREASING (4.63 → 4.58) with NEW
  BEST prints; Kaggle rerun healthy.

### E5.10 HF push: 403 Forbidden "namespace forge-lm"
- **Root cause:** the token belongs to `Manvaniso`; it cannot create repos
  under the `forge-lm` namespace.
- **Fix:** repo id `Manvaniso/forgelm`; push wrapped in try/except (fail
  non-fatally, checkpoint stays on disk); token loaded from env with
  `forgelm` secret-name fallback.
- **Impact:** push works; failures no longer lose training results.

### E5.11 "stuck at loaded 20000 stories in 10s"
- **Symptom:** no output for ~8 minutes after the load message.
- **Root cause:** the encode phase (20K stories) ran silently.
- **Fix:** progress print every 2000 stories (flush=True).
- **Impact:** no more phantom hangs.

## Day 6 — Wed 2026-08-26

### E6.1 generation crash past context_length (size-0 tensors)
- **Symptom:** `generate(..., max_tokens=64)` on a ctx-32 test model crashed:
  "shape [1,1,64] is invalid for input of size 0".
- **Root cause:** the cache grew past context_length; RoPE's
  `cos_cached[offset:offset+t]` returned an EMPTY slice → empty q/k/v.
- **Fix:** GPT.forward raises when cached + new > ctx; `generate()` stops at
  the limit with a `context_limited` flag.
- **Impact:** generation is context-safe by construction.

### E6.2 distinct_n test expectation wrong
- **Root cause:** my expected value was wrong — [1,1,1,1] has 1 unique bigram
  over 3 positions = 1/3, not 0.
- **Fix:** corrected the test (metric was right).
- **Impact:** metrics math verified.

## Day 7 — Thu 2026-08-27

### E7.1 torch.compile fails on Windows CPU
- **Symptom:** InductorError: `InvalidCxxCompiler: Compiler: cl is not found`
  at the FIRST forward (not at torch.compile()).
- **Root cause:** Inductor's C++ backend needs MSVC on Windows.
- **Fix:** bench_engine wraps the compile timing in try/except → "unavailable"
  row in the report; test skips gracefully.
- **Impact:** benchmark still complete; Day 10's ONNX/ORT path is the real
  speedup (documented in ADR-07).

### E7.2 int8 slower than fp32 on decode (1.22x)
- **Symptom:** int8 was SLOWER, not faster.
- **Root cause:** quantized kernel overhead dominates on small matrices
  (d=256).
- **Fix:** none needed — accepted; int8 kept for the 5x SIZE win (4MB <= 8MB
  spec); documented. (Benchmark script also had two KeyError bugs — status
  clobber and print loop on failed dicts — fixed.)

## Day 8 — Fri 2026-08-28

### E8.1 LoRA trainable-check failed
- **Symptom:** non-A/B params were still trainable after apply_lora.
- **Root cause:** apply_lora only froze the Linear bases; RMSNorms etc.
  remained trainable.
- **Fix:** apply_lora freezes ALL params then re-enables A/B (the classic
  LoRA freeze contract).
- **Impact:** Trainer's requires_grad-only optimizer now trains adapters only
  (~3.3% of params).

### E8.2 kaggle_sft.py `ModuleNotFoundError: forger` (2nd time)
- **Root cause:** the sys.path.insert was placed AFTER the forger imports.
- **Fix:** moved it before them (same trap as kaggle_baseline.py — checklist
  item in docs/kaggle-notebook.md).
- **Impact:** SFT runs on Kaggle.

### E8.3 "it's skipping everything" scare
- **Symptom:** the run printed a wall of "skipped out-of-contract dolly row".
- **Investigation:** local reproduction: 14,760 of 15,011 rows load; only
  251 skipped (1.7%) — the old code printed one line PER skipped row, then
  hit a SILENT ~6-min encode phase.
- **Fix:** one summary line ("loaded N rows, skipped M") + encode progress.
- **Impact:** transparent runs; no more false alarms.

### E8.4 THE DOLLY DISASTER — SFT model broken (ppl 38 stories / 68 dolly)
- **Symptom:** fine-tuned model generated confident garbage
  ("Starch 1: Computoh, Robert Lego, Titmia, New").
- **Investigation (layer by layer, in order):**
  1. State dict = clean plain GPT (47 keys, no leftover LoRA) — structure ok.
  2. Merge verified EXACT: unmerged vs merged logits diff = 8e-6;
     zero-LoRA (B=0) == baseline (diff 0.0) — merge math correct.
  3. Adapter deltas small (max |B@A·scale| = 0.041 vs base scale 0.131) —
     yet logits shifted by 7.55 (residual compounding, plausible).
  4. Training converged (eval loss dropping, loss 1.91 at 4000 steps) —
     but that's dolly's RESPONSE distribution, not knowledge.
  - **Root cause: DOMAIN MISMATCH.** A 5.25M model trained only on
    TinyStories has no general knowledge; dolly asks "Who is / Explain..."
    questions. SFT teaches the FORM of responses, never facts.
- **Fix:** switched SFT data to the model's own domain — story instructions
  over TinyStories content → loss 1.96, coherent instruction-following
  ("Write a story about a cat" → proper story).
- **Impact:** THE lesson of the project: domain alignment is the
  highest-leverage fine-tuning choice, above lr/rank/steps. Dolly experiment
  kept as evidence in ADR-08 + benchmarks/sft_qualitative.md.

### E8.5 UnicodeEncodeError in smoke_infer (cp1252 console)
- **Root cause:** Windows console codepage can't print some generated chars.
- **Fix:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

## Day 9 — Sat 2026-08-29

### E9.1 int4 dequant allocation 4.3GB (OOM-style crash)
- **Symptom:** "not enough memory: you tried to allocate 4294967296 bytes".
- **Root cause:** `codes` from unpack is 1-D; broadcasting with the 2-D
  scale (blocks, 1) produced a (blocks, n) product — for the gate layer that
  is 4096 × 262144 floats = 4.3GB.
- **Fix:** reshape codes to (-1, BLOCK_SIZE) BEFORE scaling.
- **Impact:** dequant cost bounded by the true weight size.

### E9.2 int4 model catastrophically broken (ppl 734K)
- **Symptom:** the loaded int4 model had ppl 734,541 (vs fp32 7.88).
- **Root cause:** `load_4bit` built a FRESH GPT and restored only the
  embedding + int4 layers — RMSNorm weights stayed RANDOM. The toy roundtrip
  test passed only because identical RNG state happened to reproduce the
  random norms.
- **Fix:** export now carries the full non-int4 state dict (`params`);
  load_4bit restores it.
- **Impact:** after fix: ppl 8.112 (+2.9%). Lesson: toy roundtrip tests can
  pass for the WRONG reason — always re-measure the real artifact.

### E9.3 double-quantization drift (1.08)
- **Root cause:** load_4bit fed already-dequantized weights back through
  Int4Linear.__init__, quantizing them a second time.
- **Fix:** `Int4Linear.from_stored` reconstructs from codes/scales directly.
- **Impact:** in-memory == loaded (drift ~0).

### E9.4 `nn.Parameter(None)` size-0 bias
- **Symptom:** RuntimeError: "expanded size (256) must match existing size
  (0)" on bias-less Linears (gate/up/down).
- **Root cause:** `nn.Parameter(None)` creates an EMPTY parameter.
- **Fix:** create the Parameter only when bias is not None.
- **Impact:** bias-less layers quantize correctly.

### E9.5 export/load KeyError 'params' on the old artifact
- **Root cause:** the committed 4-bit artifact predated the params fix.
- **Fix:** re-exported with the fixed exporter.
- **Impact:** shipped artifacts load cleanly.

---

## Rules going forward

- Every bug gets a detailed row here: symptom, investigation, root cause,
  fix path, impact — never one-liners.
- Every shipped module gets a row in `docs/modules.md` + one line in
  `docs/changelog.md`.
- Benchmark/evidence files land in `benchmarks/`.
- Design reasoning (alternatives + why) goes to `docs/thinking-log.md`.