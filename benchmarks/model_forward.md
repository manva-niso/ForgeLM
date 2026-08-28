# Benchmark: Model Forward - Day 3 (2026-08-23)

## Setup
- Config: {'vocab_size': 4096, 'd_model': 256, 'n_heads': 4, 'n_layers': 4, 'context_length': 512, 'ffn_mult': 4}
- Params: 5,249,280
- Device: CPU (torch 2.13.0, Windows)
- Input: batch 2, seq 512
- Runs: 10, averaged

## Results
| Metric | Value |
|---|---|
| forward time | 210.9 ms |
| throughput | 4,855 tok/s |
| params | 5,249,280 |

## Notes
- Tied embedding/head, RMSNorm, RoPE, SwiGLU, sdpa.
- KV-cache stub validated: chunked decode == full forward (test_kv_cache_stub_matches_full_forward).
