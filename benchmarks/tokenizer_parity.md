# Benchmark: Tokenizer Parity - Day 2 (2026-08-29)

## Setup
- Custom: `forger/tokenizer/bpe.py` (byte-level BPE, GPT-2 pretokenization, deterministic ties)
- Reference: Hugging Face `tokenizers` `models.BPE` + `pre_tokenizers.ByteLevel`
- Artifact: `artifacts\tokenizer` (vocab 4096, merges 3839)
- Artifact checksum: `e20ed6346c49fc5def2da9314d06cfb68fdf83b38dbdd5ef29d72f2fe1b7609b`
- Method: same vocab + merges loaded into both; compare encodings on fixed test strings.

## Test strings
| string | ours == HF |
|---|---|
| 'Hello world' | True |
| 'Hello, world!' | True |
| "I don't know." | True |
| 'Café résumé' | True |
| '你好世界' | True |
| '🙂' | True |
| 'multiple    spaces' | True |
| 'line one\nline two' | True |
| 'The cat sat on the mat and purred happily.' | True |
| '12345 numbers 67890' | True |

## Result: 10/10 strings match exactly

## Timing (50 TinyStories stories, ~seconds/encode pass)
| Implementation | s/pass |
|---|---|
| ours | 0.3570 |
| HF reference | 0.0437 |
| speed ratio (ours/HF) | 8.2x |

## Notes
- Same vocab/merges => identical encoding is expected where algorithms agree.
- Differences, if any, indicate algorithmic (not correctness) divergence; documented above.
- Training corpus: 200,000 chars of TinyStories (capped for speed). Full-corpus retrain possible via `--max-chars`.
