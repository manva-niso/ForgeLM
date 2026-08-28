# Benchmark: Optimized Serving - Day 7 (2026-08-27)

## Setup
- Checkpoint: models/forgelm-baseline (best eval 2.0247)
- Prompt: 59 tokens prefill, 50 tokens decode, 3 repeats
- Device: CPU (torch 2.13.0, Windows)

| Variant | size MB | prefill ms | prompt tok/s | first-token ms | per-token ms | tok/s | vs fp32 |
|---|---|---|---|---|---|---|---|
| fp32 | 20.0 | 21.3 | 2766 | 10.0 | 8.1 | 123 | 1.0x |
| int8 | 4.0 | 20.3 | 2902 | 15.7 | 9.9 | 101 | 1.22x |
| compile | - | - | - | - | - | - | unavailable: failed: InvalidCxxCompiler: Compiler: cl is not found.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
 |

Note: torch.compile unavailable - failed: InvalidCxxCompiler: Compiler: cl is not found.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

