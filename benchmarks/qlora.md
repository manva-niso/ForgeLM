# Benchmark: QLoRA int4 — Day 9 (2026-08-22)

## Size table
| Variant | Storage | vs fp32 |
|---|---|---|
| fp32 baseline | 20.0 MB | 1.0x |
| int8 dynamic (Day 7) | 4.0 MB | 5.0x |
| **int4 (this day)** | **4.00 MB** | **5.0x** (<=8MB product target MET) |

## Quality (200 eval stories, sliding window, ctx 128)
| Model | ppl | bpb | vs fp32 |
|---|---|---|---|
| fp32 baseline | 7.882 | 1.0069 | 1.0x |
| int4 baseline | 8.112 | 1.0209 | **+2.9%** |

Block-wise symmetric int4 (per-64 scale) costs ~3% perplexity for 5x size —
the expected quantization trade-off. (NF4 would shave the delta further; noted
in ADR-09 as a refinement.)

## QLoRA training (local evidence)
- Base: merged SFT checkpoint, all Linear weights stored int4 + frozen.
- LoRA r=8/alpha=16 on 20 int4 layers (fp32 adapters), 60 steps, dolly-1k,
  CPU 56s, final loss 5.06 / best eval 4.95.
- Merged result re-exported to int4 (same 4.00 MB format).

## Files
- `forger/quant/quantize.py` - block int4 quant/dequant, packed 2 codes/byte,
  Int4Linear, export/load.
- `models/forgelm-4bit/` - committed int4 artifact (4.00 MB).