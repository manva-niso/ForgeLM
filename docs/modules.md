# Module Graph
| Module | Status | Depends on | Exposes |
|---|---|---|---|
| data (contract + ingestion) | done | — | DatasetExample schema, validate CLI, TinyStories loader |
| tokenizer (BPE from scratch) | done | data | train/encode/decode/save/load |
| model (GPT blocks) | done | tokenizer | GPTConfig, GPT, CausalSelfAttention, KV-cache |
| train (pipeline) | todo | model, tokenizer, data | Trainer, Hydra configs |
| baseline training | todo | train | checkpoint, benchmark report |
| eval (harness) | todo | model | perplexity, generation, metrics |
| serve (inference engine) | todo | model, quant | engine, optimization, benchmarks |
| ft (LoRA SFT) | todo | train, model | SFT data prep, adapter train/merge |
| quant (QLoRA int4) | todo | model | int4 quant, export |
| api + deployment | todo | serve | FastAPI /v1/completions, Docker, CI/CD |