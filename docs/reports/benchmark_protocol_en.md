# Benchmark protocol and current status

!!! note "Version note"
    This page keeps both the 15M pipeline-validation record from 2026-11-17 and the H200 results for the 1.0 recipe. Their model scale, tokenizer, data, and context length differ, so each subsection must be read with its protocol. The actual experiments and conclusions are in the [TTT-E2E technical report](../technical_report_en.md).

A benchmark number is meaningful only when it is tied to a checkpoint, data revision, tokenizer, prompt, number of shots, context limit, adaptation setting, and valid sample count. This page records the evaluation tiers used in the project and the protocol already run on H200.

## 0. Relation to the 1.0 report

The repository's current experiment pipeline has moved to the 1.0 technical-report recipe. It uses the GPT-2 BPE vocabulary (\(V=50{,}257\)), a 50/50 FineWeb-Edu and PG-19 mixture, a 4,096-token context, and four matched models of about 130M parameters. The raw-results entry point is [`docs/assets/report/results.json`](../assets/report/results.json); the [TTT-E2E technical report](../technical_report_en.md) records the method, data revisions, and reproduction commands.

The 15M byte-token numbers retained below are an earlier pipeline-validation record and do not replace the 1.0 main experiment. The 1.0 training curves, ablations, NIAH results, extended-benchmark JSON files, and resource-failure records are now stored in `docs/assets/report/results.json`. Valid-sample counts, skipped counts, and protocol versions for the extended benchmarks are the values shown in Table 9a of the technical report.

## Evaluation tiers

| Tier | Tasks | Question | Status |
| --- | --- | --- | --- |
| Training health | held-out NLL/PPL, mixture proportions, repetition | Did the model learn the target distribution and remain stable? | Measured |
| Knowledge and commonsense | MMLU, ARC-E/ARC-C, OpenBookQA, PIQA, HellaSwag, WinoGrande | Does a small model transfer knowledge and commonsense judgments? | 1.0 includes ARC-Challenge and PIQA. MMLU, ARC-E, OpenBookQA, and WinoGrande are not in the main table |
| Mathematics | GSM8K, MATH, AIME/AMC | Can the model perform multi-step symbolic reasoning? | Full GSM8K is complete for the main conditions. MATH and AIME/AMC are not run |
| Code | HumanEval, MBPP, LiveCodeBench | Does generated code execute correctly and generalize across time? | Full HumanEval is complete for the main conditions. MBPP and LiveCodeBench are not run |
| Hard reasoning | BBH, GPQA, TruthfulQA | Long-chain reasoning, expert knowledge, and factuality | GPQA and the `sports_understanding` BBH diagnostic are complete. Full BBH and TruthfulQA are not run |
| Long context | RULER, LongBench, DROP | Retrieval and composition as context grows | NIAH and the 256–8192 synthetic preflight are complete. Official RULER, LongBench, and DROP suites are not run |
| Chinese | CMMLU, C-Eval | Chinese knowledge and exam-style reasoning | After bilingual tokenizer freeze |

The reproducible extension runner is `scripts/run_extended_benchmarks.sh`. It runs ARC-Challenge, PIQA, GPQA, GSM8K, HumanEval, the `sports_understanding` BBH diagnostic, and a synthetic key-value long-context preflight for a chosen final checkpoint. The synthetic row is an engineering check for RULER/LongBench integration and is never presented as an official score.

These tiers are complementary. A lower PPL does not imply a higher GSM8K score, and MMLU answer selection does not replace execution-based code evaluation. A release table should cover knowledge/commonsense, mathematics or code, and long context, together with held-out NLL.

## 2. 15M MMLU diagnostic

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

## 3. 1.0 extended benchmarks

`scripts/run_extended_benchmarks.sh` ran three main conditions one card at a time on GPU5 and GPU7. `main-full` and `main-ttt adapt` use the complete public splits. `main-ttt no-adapt` and `main-ck no-adapt` use 100-example diagnostics, with GPQA fixed at 65 valid examples. ARC, PIQA, and GPQA use zero-shot answer-text continuation log-likelihood. GSM8K and HumanEval use greedy generation, and HumanEval executes the official tests. BBH is the single `sports_understanding` subtask, so it remains a diagnostic rather than a BBH aggregate score.

The complete main-model and TTT-adapt results are in Table 9a. The main model scores 1158/1165 on ARC, 1800/1819 on PIQA, 58/256 on GPQA, 12/1319 on GSM8K, 0/164 on HumanEval, and 0/250 on BBH. TTT adapt scores 1158/1165, 980/1819, 63/256, 10/1319, 0/164, and 0/250. The large number of skipped GPQA items comes from the fixed 512-token limit and must not be read as an accuracy denominator.

The TTT no-adapt 100-example diagnostic gives 24/100 on ARC, 55/100 on PIQA, 17/65 on GPQA, 2/100 on GSM8K, 0/100 on HumanEval, and 0/100 on BBH. The checkpoint-only condition gives 20/100, 51/100, 15/65, 3/100, 0/100, and 0/100. These small samples check paired conditions and the evaluation path. They are not directly comparable with the complete-split scores.

The synthetic long-context row uses 32 examples at each requested length of 256, 512, 1024, 2048, 4096, and 8192. All four main conditions score 0/32 at every length. This is an engineering preflight for RULER/LongBench, not an official suite score. Full-attention NIAH retry results at 1K, 2K, 4K, 8K, and 16K are stored in `main_full_retry.json`. The 32K request triggered `RESOURCE_EXHAUSTED` while allocating about 72.3 GiB. The report keeps the failure record and does not convert the missing 32K result into a zero score.

## 4. Contamination and comparability

Before formal evaluation, compare the training corpus, tokenizer training text, and benchmark questions/answers with exact matching, NFC-normalized matching, long-substring matching, and approximate matching. Fix the revision, split, prompt, shot count, normalization, timeout rule, and repeat count for every benchmark. Closed models are external references; unequal protocols must not be reported as a victory.

## 5. Release gate

The current results remain pilot evidence. Before entering the main technical-report table, add:

1. Same-seed, same-token-budget paired plain and TTT results;
2. At least three seeds and uncertainty intervals;
3. A mathematics or code benchmark beyond MMLU;
4. A public long-context benchmark;
5. Contamination audit, complete version information, and reproducible commands.
