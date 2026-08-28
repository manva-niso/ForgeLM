# Benchmark: LoRA SFT — Final (2026-08-29, tag v0.3.2)

## The dolly experiment (important lesson, artifact rejected)
- SFT on databricks-dolly-15k (15K rows, 3000 steps T4) produced a model with
  **ppl 38 on TinyStories** and **garbage generations**.
- Root cause: **domain mismatch**, NOT the pipeline. A 5.25M model trained only
  on TinyStories has no general knowledge; dolly asks "Who is / Explain..."
  questions. SFT teaches the *form* of responses, not *facts* -> confident
  garbage (ppl 68 even on held-out dolly).
- Verified pipeline correct: merge exact (unmerged vs merged diff 8e-6),
  zero-LoRA == baseline, LoRA deltas small (max 0.041 on 0.13-scale weights).
- Action: switched SFT data to the model's own domain.

## Final model — story-domain SFT (forgelm-sft-story)
- Data: TinyStories sample formatted as
  "### Instruction: Write a story about {topic}\n### Response: {story}"
  (topic extracted from story; 4 template variants), 2000 examples.
- LoRA r=8/alpha=16 on 20 layers, 300 steps, bs 8, ctx 128, lr 3e-4, CPU, 103s.
- Final loss 1.96 / eval 2.00 (vs dolly 5.08) - model stays in its knowledge domain.

## Generations (story-SFT, temp 0.7, top-k 40)
| Prompt | Output |
|---|---|
| Write a story about a cat. | "Once upon a time, there was a little girl who loved to explore. One day, she went outside and saw an old lady. She had a big bag of candy that she did..." |
| Tell me a story about a dog. | "Once upon a time there was a big, red ball. Pache was all alone in the forest when the dog met a small girl. The girl wanted to play with the ball, but..." |
| Write a story about a girl. | "Once upon a time, there was a little girl who loved to explore. One day, she went outside and saw an old man. He had a big coat and a big bag of candy..." |
| Once upon a time there was | "Once upon a time, there was a little girl who loved to explore. She had many friends, but one day she met a new friend. 'Hello, I'm time yous!'..." |

Instruction format followed; coherent TinyStories English. Cat/dog/girl prompts
all produce appropriate story starts (topic keyword absorbed into the story).

## Artifacts (committed)
- `models/forgelm-sft-story/` - fp32 merged (21.3 MB)
- `models/forgelm-sft-story-4bit/` - int4 export (6.27 MB, <=8MB spec)
- Kaggle runner now uses streamed story-domain data (`scripts/kaggle_sft.py`,
  15K stories for the bigger run - pending optional GPU run).