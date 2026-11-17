# Benchmark protocol and current status

!!! note "Historical record"
    This page records the pipeline-validation stage of 2026-11-17 (small models, byte-level tokenizer, WikiText-2). It confirms that training and evaluation run end to end and says nothing about model capability. The actual experiments and conclusions are in the [TTT-E2E technical report](../technical_report_en.md).

A benchmark number is meaningful only when it is tied to a checkpoint, data revision, tokenizer, prompt, number of shots, context limit, adaptation setting, and valid sample count. This page records the evaluation tiers used in the project and the protocol already run on H200.

## Evaluation tiers

| Tier | Tasks | Question | Status |
| --- | --- | --- | --- |
| Training health | held-out NLL/PPL, mixture proportions, repetition | Did the model learn the target distribution and remain stable? | Measured |
| Knowledge and commonsense | MMLU, ARC-E/ARC-C, OpenBookQA, PIQA, HellaSwag, WinoGrande | Does a small model transfer knowledge and commonsense judgments? | MMLU and HellaSwag diagnostics |
| Mathematics | GSM8K, MATH, AIME/AMC | Can the model perform multi-step symbolic reasoning? | Run after tokenizer and scale are fixed |
| Code | HumanEval, MBPP, LiveCodeBench | Does generated code execute correctly and generalize across time? | Planned |
| Hard reasoning | BBH, GPQA, TruthfulQA | Long-chain reasoning, expert knowledge, and factuality | Planned |
| Long context | RULER, LongBench, DROP | Retrieval and composition as context grows | Synthetic probe exists; suite pending |
| Chinese | CMMLU, C-Eval | Chinese knowledge and exam-style reasoning | After bilingual tokenizer freeze |

These tiers are complementary. A lower PPL does not imply a higher GSM8K score, and MMLU answer selection does not replace execution-based code evaluation. A release table should cover knowledge/commonsense, mathematics or code, and long context, together with held-out NLL.

## Current MMLU diagnostic

`scripts/eval_mmlu_loglikelihood.py` reads MMLU test CSV files and constructs:

```text
question

A. choice_0
B. choice_1
C. choice_2
D. choice_3
Answer:
```

For each option it scores the continuation ` space + letter` and selects the largest conditional log likelihood:

\[
\hat y=\arg\max_{c\in\{A,B,C,D\}}
\log p_\theta(\text{space}+c\mid\text{prompt}).
\]

The protocol uses the byte tokenizer, a 512-token limit, and no TTT session adaptation. Invalid rows and prompts exceeding the fixed context are counted as `skipped`; they are never silently truncated. JSON output includes per-subject correct, valid, skipped, and accuracy counts, plus data and checkpoint hashes.

This is an engineering diagnostic for the current 15M, byte-token, non-instruction-tuned model. It is not directly comparable with vendor scores using subword tokenizers, few-shot prompts, or instruction tuning. A formal result requires a frozen tokenizer, scale, training corpus, and prompt template, at least three seeds, and a Wilson or bootstrap confidence interval.

The H200 command is:

```bash
CUDA_VISIBLE_DEVICES=5 JAX_PLATFORMS=cuda PYTHONPATH=. \
  .venv/bin/python scripts/eval_mmlu_loglikelihood.py \
  --checkpoint outputs/h200-15m-ttt-mixture-85-15/step_00010000.orbax \
  --data-dir /home/wang.21020/mmlu-data/mmlu \
  --subjects abstract_algebra high_school_mathematics college_physics computer_security professional_law \
  --limit 100 --output outputs/benchmarks/mmlu_ttt_mixture_5x100.json
```

Run the baseline on GPU7 by changing the checkpoint and device. Keep the experiment configuration, dataset hash, script hash, and checkpoint metadata with every result.

On the H200 15M mixture pair, the five-subject MMLU subset gave 116/392 (29.59%) for the plain baseline and 104/392 (26.53%) for TTT no-adapt. On the first 100 HellaSwag validation examples, both models gave 32/100 (32%). These are byte-token, non-instruction-tuned pilot measurements, not leaderboard scores.

The same pair gave ARC-Challenge 12/100 for the baseline and 11/100 for TTT no-adapt, and PIQA 57/100 for both. Greedy GSM8K numeric exact match was 0/100 for both; HumanEval official test execution was 0/50 pass@1 for both. A synthetic key-value diagnostic at requested lengths 256, 512, and 1024 produced 0/16 at every length for baseline, TTT no-adapt, and TTT adapt. The synthetic test is a RULER/LongBench preflight, not a public-suite score.

On the first 50 GPQA examples from a public un-gated mirror, the fixed 512-token protocol yielded 9/32 (28.13%) for the baseline and 5/32 (15.63%) for TTT no-adapt; 18 examples were skipped in each run for context length. Because the original GPQA release has strict anti-redistribution and contamination requirements, only result JSON is stored here, not question text, and this is not an official full GPQA leaderboard score.

For BBH, the first 50 examples of the `sports_understanding` subtask were run with zero-shot greedy normalized exact match: 0/50 for both baseline and TTT no-adapt. Official BBH scoring uses subtask-specific answer parsers and a shot protocol, so this only validates the generation path.

The current 15M configuration has `max_position_embeddings=2048`, while the standard RULER context bands begin at 4K. Truncating those tasks to 2048 would change the task, so no official RULER score is claimed. LongBench also requires task-specific input preservation, answer normalization, and length policies; until the tokenizer and long-context configuration are frozen, we keep only the synthetic preflight.

## Contamination and comparability

Before formal evaluation, compare the training corpus, tokenizer training text, and benchmark questions/answers with exact matching, NFC-normalized matching, long-substring matching, and approximate matching. Fix the revision, split, prompt, shot count, normalization, timeout rule, and repeat count for every benchmark. Closed models are external references; unequal protocols must not be reported as a victory.

## Release gate

The current results remain pilot evidence. Before entering the main technical-report table, add:

1. Same-seed, same-token-budget paired plain and TTT results;
2. At least three seeds and uncertainty intervals;
3. A mathematics or code benchmark beyond MMLU;
4. A public long-context benchmark;
5. Contamination audit, complete version information, and reproducible commands.
