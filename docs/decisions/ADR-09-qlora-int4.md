# ADR-09: QLoRA int4 Quantization

Status: accepted
Date: 2026-08-29

## Context
Day 9: shrink the model to <=8MB (product spec) and enable 4-bit fine-tuning.

## Decision
- Block-wise symmetric int4: per-64-element absmax scale, codes in [-8, 7],
  packed 2 codes/byte (pack_4bit/unpack_4bit). Dequantize to fp32 for the
  forward (weight-only quantization - storage is 4-bit, compute is fp32).
- `Int4Linear`: stores packed codes + scales; `quantize_model_4bit` replaces
  all Linears EXCEPT lm_head (kept fp32 + tied to embedding to avoid drift).
- QLoRA-in-spirit training: base stored int4 + frozen; LoRA adapters train in
  fp32 on top of the dequantized base; merged result re-exported to int4.
  Documented simplification vs the paper: we dequantize for forward (no
  on-the-fly dequant kernels) - irrelevant at 5.25M params, matters only for
  >1B models; no NF4/double-quant (symmetric int4 chosen for simplicity;
  NF4 would tighten the ppl delta, noted as a refinement).
- Export format: model_4bit.pt {config, params (non-int4 state), int4_layers};
  load_4bit reconstructs a full GPT (embedding, RMSNorms, int4 layers).

## Bugs found during validation
1. `nn.Parameter(None)` creates a size-0 bias for bias-less Linears (gate/up/down).
2. `dequantize_4bit` broadcast: 1-D codes x (blocks,1) scales made a
   (blocks, n) product -> 4.3GB alloc. Fixed by reshaping codes to blocks.
3. `load_4bit` re-quantized already-dequantized weights (double quantization).
4. `load_4bit` left RMSNorm weights random (fresh GPT init) - export now
   carries the full non-int4 state dict.

## Results
4.00 MB (5x smaller, <=8MB spec met); ppl 7.88 -> 8.11 (+2.9%); export/load
bit-exact; QLoRA local training run healthy (loss 5.06).