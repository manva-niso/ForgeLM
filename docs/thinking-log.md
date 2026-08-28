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

## Day 2 - BPE tokenizer (2026-08-29)

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

## Day 3 - Model core (2026-08-29)

- **RoPE over learned absolute embeddings:** no extra params, length-generalizes
  beyond trained context, standard in modern LMs; absolute embeddings were the
  fallback if RoPE proved buggy (it did - twice - but was worth fixing).
- **RMSNorm over LayerNorm:** no bias, one learnable scale, ~20% cheaper; modern
  LLMs (LLaMA) use it. Tiny downside: no shift learnable - irrelevant here.
- **SwiGLU over ReLU/GELU MLP:** better quality per param, standard since
  LLaMA; costs a third projection (2d -> 4d -> d instead of d -> 4d -> d) -
  the main reason params came to 5.25M vs the 3.6M estimate. Kept: quality
  matters more than the plan's estimate; still 5x under the 25M cap.
- **sdpa instead of hand-rolled attention:** flash-attention kernels for free on
  CUDA, fused softmax masking, less code to get wrong. The mask semantics trap
  (bool=True means ATTEND in sdpa) is now documented in ADR-03 for posterity.
- **KV-cache stub on Day 3, real caching Day 7:** the stub proves the cache
  contract (identical logits) while keeping Day 3 scope small. Real perf work
  belongs with the inference day.
- **Causal validation strategy:** "change later tokens -> earlier logits must
  not change" is the cleanest, most direct test of causality - catches mask
  bugs that shape tests miss.

## Day 4 - Training pipeline (2026-08-29)

- **AMP only on CUDA, fp32 on CPU:** measured bf16 autocast 7x SLOWER on this
  CPU (1.35s vs 0.19s forward) - the "bf16 on CPU" assumption from the plan
  was wrong for this hardware. Measurement over convention; Kaggle T4 keeps
  fp16 + GradScaler where AMP actually wins.
- **Deterministic get_batch(step) instead of a shuffled iterator:** makes
  resume trivially exact (test proves bit-identical losses). Cost: windows
  repeat after the pool is exhausted - fine for small runs; Kaggle will use
  the full corpus stream.
- **Schedule depends on total steps:** resume must reuse the ORIGINAL steps
  total (cosine denominator), so training to a partial count goes through
  `train(until=N)` - a test bug (different denominator) exposed this.
- **Resume test = full + partial + compare:** the strongest form; three
  separate test bugs (shared config mutation, LR-schedule mismatch, unseeded
  inits) had to be fixed before it passed - each one documented in the dev log.

## Day 5 - Baseline training (2026-08-29)

- **CPU-first baseline before Kaggle:** the vertical-slice rule wants evidence
  today, not "whenever GPU access happens". 300 steps on CPU (6 min) produced a
  real checkpoint + curves and proved the exact code path Kaggle will run.
  GPU only changes `--device`.
- **One script, two targets:** `kaggle_baseline.py` doubles as the Kaggle cell
  body and the local command - no divergent code paths to debug later.
- **Why bs 64/ctx 256 for the T4 run (not bs 8/ctx 512):** bigger batch = more
  stable gradients + faster wall-clock; ctx 256 matches the 4x smaller memory
  per sample and TinyStories stories are short; 50K stories stream keeps
  encode time sane on the notebook.
- **HF Hub as the artifact bus:** checkpoint push (Kaggle) / pull (local) via
  huggingface_hub - no zip downloads, versioned, private by default.

## Day 5b - The memorization investigation (2026-08-29)

- **First instinct was wrong:** I blamed "too few windows per story" and
  changed windows_per_story to 16 before proving anything. The A/B tests then
  showed 50 stories collapse EVEN at 16 windows - the real lever is TOTAL
  UNIQUE DATA vs repetition count, not the window multiplier alone.
  At Kaggle scale (2K stories) 16x windows pushes unique tokens from 512K to
  ~8M - beyond a 5.25M model's memorization range at 8 epochs. Two levers
  were documented; only one is free (windows), the other is data/encode speed.
- **Why eval loss RISES (not just stays):** once memorized, the model's
  distribution is peaked on train tokens; on unseen text it is confidently
  wrong -> cross-entropy can exceed the random-init ln(4096)=8.3 ceiling.
  Reading "eval > initial loss" as a signal is a useful trick.
- **The warning beats the postmortem:** eval-rise detection in the Trainer
  means this class of failure announces itself at step ~500, not after a
  45-minute run.
- **Second correction (2026-08-29):** the "16x unique data" claim was wrong -
  windows from the same story overlap ~completely (130-token story cropped
  into 16x128 windows = 16 near-identical views). Unique data = stories x
  tokens, not windows x tokens. B at 1200 steps showed eval rising 5.09->
  5.66, proving memorization still wins at 400 stories. The REAL lever is
  story count, gated by encoder speed. Profiling showed the _bpe loop
  rebuilding dicts per pass (22.6M dict gets / 3 encodes); the fix was a
  single-pass rank scan + piece cache: 20x faster, parity unchanged. Kaggle
  plan: 20000 stories (~2.6M unique tokens), not 2000.
- **Kaggle debugging taxonomy:** the 5 failures were (1) import-path
  semantics, (2) state leaking between runs (stale clone), (3) device
  placement, (4) secret env semantics (get_secret != export), (5) filesystem
  cwd lifetime. Each is a known "works locally, breaks in cloud notebooks"
  class - worth a checklist in docs/kaggle-notebook.md. |

## Day 8 - LoRA SFT (2026-08-29)

- **Why LoRA over full fine-tuning:** 5.25M params is small, but the *lesson*
  and the *pattern* (freeze base, adapters, merge) is what carries over to
  real models. Also keeps the artifact a plain GPT (merge+convert), which the
  rest of the pipeline (eval, engine, int4) consumes unchanged.
- **LoRA target set = all projections except the tied head:** c_attn/c_proj/
  gate/up/down. The embedding/head stays fp32 and tied - LoRA there would
  break the tie semantics for no benefit.
- **Freeze contract:** apply_lora freezes EVERYTHING (incl. RMSNorms) except
  A/B - the classic LoRA setup. Trainer now optimizes only requires_grad
  params, so the same Trainer runs pretraining, SFT, and QLoRA unchanged.
- **SFT is just next-token loss on formatted text:** reusing WindowDataset +
  Trainer (vs a bespoke chat-loss loop) was the right call - one code path,
  everything else (resume, eval, TensorBoard) inherited.

## Day 9 - QLoRA int4 (2026-08-29)

- **Symmetric block int4 over NF4:** NF4's normal-float codebook buys maybe
  1% ppl over absmax int4; implementing it correctly is a day on its own.
  Measured int4 costs +2.9% ppl for 5x size - acceptable, and NF4 is
  documented as a refinement, not a rewrite.
- **QLoRA-in-spirit, honestly documented:** we store base as int4 but
  dequantize for forward (no on-the-fly int4 matmul kernels). At 5.25M params
  memory is irrelevant - the paper's machinery exists for >1B models. What we
  kept is the *contract*: quantized frozen base + fp32 adapters + int4 export.
- **The ppl-734K debugging session:** the toy test passed while the real
  model was broken (fresh-GPT RNG state masked the missing RMSNorm restore).
  Lesson: roundtrip tests on toy models can pass for the WRONG reason -
  always re-measure the real artifact (ppl delta) after loading.
## Day 8b - The dolly domain lesson (2026-08-29)

- **SFT teaches form, not facts:** a 5.25M model trained only on TinyStories
  cannot answer general-knowledge questions no matter how well you fine-tune
  it - there is no knowledge to draw on. The dolly model learned dolly's
  response *distribution* (ppl 4.7 on dolly windows) while producing
  confident garbage - the clearest possible demonstration that loss is not
  understanding.
- **Debug order matters:** before suspecting the pipeline (merge, lr, env),
  we verified each layer: merge exact, zero-LoRA == baseline, deltas small.
  The pipeline was never wrong; the *data distribution* was. The vertical
  slice discipline (tests + direct measurement) made this provable instead
  of debatable.
- **Match SFT data to base knowledge:** story instructions over TinyStories
  content => loss 1.96 and real instruction following. Domain alignment is
  the highest-leverage choice in fine-tuning, above lr/rank/steps.
