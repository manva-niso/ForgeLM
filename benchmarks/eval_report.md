# Evaluation Report

date: 2026-08-22 | ckpt: `models/forgelm-baseline` | tasks: perplexity, generation

## Reproducibility
- torch: 2.13.0+cpu
- checkpoint: models/forgelm-baseline
- tokenizer checksum: e20ed6346c49fc5def2da9314d06cfb68fdf83b38dbdd5ef29d72f2fe1b7609b
- eval corpus: data\eval_tinystories.jsonl sha256: 06aa340c1e27c31d3ded27b82f414102b73c1ef46623683e2adfe346c2d4d94e
- seed: 0

## Perplexity (sliding window, pinned eval corpus)

| Metric | Value |
|---|---|
| perplexity | 8.449 |
| bits-per-byte | 1.0366 |
| tokens evaluated | 136,428 |
| stories | 500 |

## Generations (top-k 50, temp 0.8, max 64 tokens)

### Prompt: Once upon a time there was a little cat

**Model:** . "I want to hear a funny noise. It is a joke. I want to sing." Anna says, "OK, Ben, but you have to be good. Do you want to join me?" Ben says, "Yes, I want to ride a bike too!"

They

| metric | value |
|---|---|
| tokens | 64 |
| hit cap | True |
| distinct-1 | 0.625 |
| distinct-2 | 0.889 |
| repetition (trigram) | 0.048 |

### Prompt: The little girl went to the park

**Model:** . Timmy's mom said they had to go to the park where they were going. 

After a while, Timmy saw a big rock in the park. He wanted to touch it, but his mom said no. Timmy was sad because he loved it. He said to his mom, "Mommy, can we go

| metric | value |
|---|---|
| tokens | 64 |
| hit cap | True |
| distinct-1 | 0.625 |
| distinct-2 | 0.921 |
| repetition (trigram) | 0.000 |

### Prompt: Tom and his dog were playing

**Model:** . Timmy's mom said, "Look at the yellow orange jacket! I want to chew the yellow jacket too!" Timmy was very happy and said, "Yes, Mommy! I will try!"

After dinner, Timmy and his mom went to the park to play. Timmy saw a big

| metric | value |
|---|---|
| tokens | 64 |
| hit cap | True |
| distinct-1 | 0.641 |
| distinct-2 | 0.889 |
| repetition (trigram) | 0.032 |

### Prompt: It was a rainy day and

**Model:** made up them. He looked at his mom and dad with a smile. He smiled and said, "That's great, Lily. You're very kind. You can be my friend. Do you want to play with us?"

Lily was so happy to hear that. She liked her crayons.

| metric | value |
|---|---|
| tokens | 64 |
| hit cap | True |
| distinct-1 | 0.797 |
| distinct-2 | 0.968 |
| repetition (trigram) | 0.000 |

### Prompt: The rabbit wanted to find

**Model:** out. They ran over to help them, but they were scared. They tried to get away, but the dog was faster. 

Then, a kind woman came to the pond. She asked the woman why she was crying. The man told her he was sorry and that she wanted to help, but it made

| metric | value |
|---|---|
| tokens | 64 |
| hit cap | True |
| distinct-1 | 0.688 |
| distinct-2 | 0.937 |
| repetition (trigram) | 0.000 |
