# ForgeLM Journey — What We Built, What Broke, What We Learned

*A plain-language walkthrough of the project from start to now. Written for
re-reading a week from now — no jargon assumed.*

---

## 1. What We Are Building

ForgeLM is a **small language model** — the kind of thing that could run on a
phone or a laptop. We are building it from scratch: our own tokenizer, our own
Transformer model, our own training loop, and later our own API to serve it.

The whole point is the journey: every piece is written by us, tested, measured,
and committed with evidence. No black boxes.

## 2. The Plan (12 Slices)

We split the build into 12 one-day "vertical slices":
data contract → tokenizer → model → training → baseline → evaluation →
inference → LoRA fine-tuning → QLoRA quantization → optimized serving →
API + deployment → security + release.

Every day ends with: working code + passing tests + a benchmark/evidence file
+ a git commit with a version tag. This rule ("vertical-slice rule") is what
keeps the project honest — nothing is "done" until it is measured.

## 3. Day 1 — The Data Contract

**What:** Before feeding text to a model, we defined exactly what a "valid
piece of data" looks like. We used Pydantic to declare: a text field
(1 to 4096 characters), a split name (train / validation / test), and optional
metadata. A small command-line tool (`forger-data-validate`) checks any
JSON-lines file against this contract and reports line-by-line violations.

**Problem we hit:** When we downloaded a sample of TinyStories (1,000 stories),
one story was longer than 4096 characters and got rejected by our own contract.

**How we dealt with it:** This was the contract *working* — we kept the limit
and changed the downloader to filter out-of-contract stories at the source.
The lesson: the data bends to the spec, not the other way around.

**Evidence:** `benchmarks/data_contract.md` — 1000/1000 rows valid, with a
sha256 checksum so the sample is reproducible forever.

## 4. Day 2 — The Tokenizer (BPE from Scratch)

**What:** A tokenizer turns text into integer IDs. We built a **byte-level BPE**
tokenizer — the same family of algorithm that GPT-2 uses — with no tokenizer
library underneath.

**How it works, simply:**
- Text is split into word-like pieces first (pre-tokenization, using a
  GPT-2-style regex that keeps spaces and punctuation).
- Each piece becomes bytes. Every byte (0–255) is a token ID to start with.
- We count which *pairs* of bytes/IDs appear together most often, merge the
  most frequent pair into one new ID, and repeat until we reach our target
  vocabulary (4,096 IDs). The merge *rules* are saved in a `merges.txt` file
  (in training order), and the byte→ID dictionary in a `vocab.json` file.
- **encode(text)** runs the merges in rank order → IDs.
- **decode(IDs)** joins all the token bytes together *first*, then decodes the
  whole byte sequence as UTF-8 once. (Decoding token-by-token would corrupt
  multi-byte characters like emojis or Chinese.)

**Problems we hit:**
- *Determinism:* two pairs can tie in frequency. We pick the
  lexicographically smallest pair — so the same corpus always produces the
  same tokenizer.
- *Speed:* the naive training loop timed out after 15 minutes on the full
  sample. We flattened the corpus into one list, tightened the loops, and
  capped default training characters at 200K → 89 seconds.
- *Encoder slowness:* our encoder is ~400× slower than HuggingFace's (the
  merge loop rebuilds stats per piece). We accepted it for now — it is a
  documented optimization target for the inference days.

**Proof:** 10/10 test strings encoded **identically** to a HuggingFace
tokenizer built from our exact vocab + merges (`benchmarks/tokenizer_parity.md`).

## 5. Day 3 — The Model Core (GPT Blocks)

**What:** A decoder-only Transformer. We built:
- **RMSNorm** — normalizes each token's hidden vector (cheaper than LayerNorm).
- **RoPE** — rotary positional embeddings, so the model knows token order.
- **Causal self-attention** — every token looks at itself and the tokens
  before it (never the future), using PyTorch's fused attention kernel.
- **SwiGLU MLP** — the "thinking" layer with a gated activation.
- **Tied embedding/head** — the same weight matrix turns tokens in AND
  predictions out (saves ~20% parameters).
- **KV-cache stub** — the plumbing for fast generation later, proven to give
  identical results to a full forward pass.

Sizes: input `[B, T]` → embedding `[B, T, 256]` → logits `[B, T, 4096]`.
The model has **5,249,280 parameters** (~5.25M).

**Three bugs we found the hard way (all in RoPE/attention):**
1. RoPE's cosine/sine tables were half the width they needed — the rotation
   hit only 32 of 64 dimensions. Fixed by repeating each frequency for the
   pair dimension.
2. With a KV-cache, a decoding token was rotated as if it were *position 0*,
   not its real global position. Fixed by passing a position offset.
3. The sneakiest one: PyTorch's `scaled_dot_product_attention` treats a
   **boolean mask as "True = allowed to attend"** (the opposite of what you'd
   guess). Our first cache mask let future tokens through. Switched to an
   explicit float `-inf` mask — this is now documented forever in ADR-03.

**Evidence:** `benchmarks/model_forward.md` — 211 ms forward for 512 tokens on
CPU (4,855 tokens/sec).

## 6. Day 4 — The Training Pipeline

**What:** The loop that makes the model learn.

- **WindowDataset:** each story is encoded once, then we cut out contiguous
  windows of `context_length` tokens. For every window `X`, the *labels* `Y`
  are the same tokens **shifted one to the right** — exactly like your example:
  if `X` is "low low", `Y` is "low lower". The model learns to predict the
  next token.
- **Trainer:** AdamW optimizer, a learning-rate schedule that warms up then
  decays like a cosine curve, gradient accumulation, an evaluation loop on
  held-out stories every N steps, checkpoints that save model + optimizer +
  step, and TensorBoard logging.
- **Deterministic resume:** batches are picked by step number, so resuming
  from a checkpoint continues *bit-identically* to an uninterrupted run —
  proven by a test that compares loss histories exactly.

**Problem we hit:** the plan said "use bfloat16 on CPU for speed". We measured:
**bf16 was 7× slower than plain fp32** on this machine (1.35s vs 0.19s per
forward). Convention lost to measurement: CPU now runs fp32; AMP (fp16 +
GradScaler) is used only on the GPU where it actually wins.

**Evidence:** `benchmarks/train_smoke.md` — 10 steps in 8.1 s, loss dropping
from 8.11 (theory says a random model should start near ln(4096) ≈ 8.32 —
we matched theory).

## 7. Day 5 — Baseline Training (and the Kaggle Saga)

**What:** Train the real 5.25M model on TinyStories and get a checkpoint.

First we trained **on our own CPU** as guaranteed evidence: 300 steps, 358
seconds, loss 8.11 → 3.34 (train) / 3.95 (eval). Real checkpoint, real curve.

Then we tried **Kaggle's free GPU (T4)** via a notebook. This is where things
got fun — five failures in a row, each one a different class of bug:

1. **`ModuleNotFoundError: forger`** — running `python scripts/x.py` puts
   `scripts/` on the import path, not the repo root; locally it worked only
   because the package is installed in the venv. → Fix: add the repo root to
   `sys.path` inside the script.
2. **Same error again** — the notebook's `git clone` failed *silently*
   because the folder from run 1 still existed, so the old code ran.
   → Fix: delete the folder before cloning, and verify `import forger`.
3. **"index on cuda:0, weights on cpu"** — the model was never moved to the
   GPU; the Trainer only moved the input tensors. → Fix: `model.to(device)`
   in the Trainer + a regression test. (A classic: works on CPU, breaks on
   GPU.)
4. **HF still "unauthenticated"** even though the secret was set — Kaggle's
   `get_secret()` returns a value but does **not** put it in the environment,
   so the training subprocess never saw it. → Fix: explicitly
   `os.environ["HF_TOKEN"] = get_secret(...)`.
5. **`getcwd() failed`** — the notebook kernel's current directory was
   *inside the folder we deleted*. Every child process inherited a dead cwd.
   → Fix: `os.chdir("/kaggle/working")` before deleting anything.

**Branch architecture:** because local and Kaggle want different configs
(device, batch size, steps), the repo now has two branches:
`main` (local CPU development) and `kaggle` (GPU runs, pinned by the
notebook). Same code, one config file each, merged after every change.

## 8. The Big Lesson — Memorization (the model cheating)

**What happened:** the first successful Kaggle run produced *terrifying*
numbers:

```
step 500/4000 loss 0.2996    <- train loss collapsing toward zero
step 500 eval loss 5.9804    <- eval loss RISING
step 1000 eval loss 7.1448   <- getting worse and worse
```

Train loss ≈ 0.01 while eval loss climbs past 7.5. This is **catastrophic
memorization**: the model didn't learn language — it learned to recite.

**Why it happened:** 2,000 stories × 1 window each = only ~2,000 unique token
windows. At batch size 64, the model cycled through the *same* 2,000 windows
~128 times over 4,000 steps. A 5.25M-parameter model can easily memorize
512K tokens. Once memorized, it is overconfident on unseen text → eval
cross-entropy rises.

**How we proved it (A/B experiments on CPU):**

| Setup | final train loss | final eval loss | verdict |
|---|---|---|---|
| 50 stories, 1 window/story, 400 steps | 0.14 | 7.46 | memorized |
| 50 stories, 16 windows/story, 400 steps | 0.15 | 7.84 | still memorized |
| 400 stories, 1 window/story, 600 steps | 2.71 | 4.92 | not yet collapsed |
| 400 stories, 16 windows/story, 600 steps | 2.80 | 4.89 | not yet collapsed |
| 400 stories, 16 windows/story, 1200 steps | ~2.5 | 5.09 -> 5.66 | **memorization creeping back** |

**Correction (learned the hard way):** the "16× unique data" idea was wrong.
Windows cropped from the *same* story overlap almost completely — 400 stories
contain only ~52K unique tokens whether you take 1 window or 16. The window
multiplier only adds redundant views. The real lever for unique data is
**the number of stories**, and that was gated by encoder speed.

**The fix that actually matters:** we profiled the encoder and found the
merge loop rebuilding dictionaries on every pass (22.6M dict lookups for three
encodes). Rewrote it as a single-pass rank scan with a piece cache:
**37.3s → 1.8s (20× faster), parity still 10/10.** That makes large corpora
affordable: 20,000 stories now encode in ~8 minutes instead of ~2.5 hours.
The Kaggle run now uses 20,000 stories (~2.6M unique tokens) — beyond the
5.25M model's memorization range even at 4,000 steps.

Other changes that stayed useful: `windows_per_story` as a config knob
(16/4), and the Trainer's eval-rise warning.

**Hyperparameters that changed during Day 5:**
- `windows_per_story_train`: 1 → **16** (unique data volume)
- `windows_per_story_eval`: 1 → **4**
- Kaggle batch size 64, context 256, steps 4000, lr 3e-4, warmup 100 — unchanged
- CPU runs: adaptive eval interval (`steps/16`), warmup `steps/40`

## 9. Where We Are Now

- **Done (tagged v0.0.1 … v0.0.6):** data contract, tokenizer (10/10 parity),
  model core (KV-cache proven), training pipeline (bit-exact resume), baseline
  checkpoint on CPU + Kaggle pipeline fixed end-to-end.
- **Next:** Day 6 evaluation harness (perplexity + sample generations),
  Day 7 inference engine (fast KV-cache generation + `torch.compile` +
  int8), then LoRA/QLoRA, serving, API, and release.
- **The Kaggle run to do:** re-run the notebook (it now uses 16 windows/story
  and the kaggle branch) → checkpoint lands on HuggingFace Hub
  (`forge-lm/baseline`) → we pull it back and measure it properly.

## 10. The Rules We Learned (short version)

1. Measure before trusting a convention (bf16 "should" be faster — it wasn't).
2. The data bends to the contract, not the other way.
3. More unique data beats more steps, always.
4. Every bug gets written down with its root cause and fix path
   (`docs/development-log.md`), and every design decision gets its "why"
   (`docs/thinking-log.md`).
5. Evidence over claims: every number in this document came from a run we can
   re-execute.