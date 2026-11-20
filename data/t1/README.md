# T1 proxy results (LordKarV's Mac)

Per-run teacher-forced disagreement between Qwen3.5 BF16 and its 4-bit and 3-bit MLX copies, on
successful SWE-agent trajectories written by other models. This is proxy evidence for the `04` P0d
score choice, not on-policy data: no Qwen-written runs and no energy measurements. Discussion and
summaries are in issue #1.

## Files

| File | Model | Runs | Tasks | Selection |
|---|---|---|---|---|
| `t1b_4b_rest.jsonl` | Qwen3.5-4B | 327 | 57 | entries 263-589 of the 589-row job list in issue #1 (comment 7004324394); the owner's agent scored entries 1-262 |
| `t1b_9b.jsonl` | Qwen3.5-9B | 589 | 100 | the full 589-row job list, so every row pairs with a 4B row by `row` |
| `t1b_long.jsonl` | Qwen3.5-4B | 393 | 100 | D1: `--min-tokens 16384 --max-tokens 65536 --tasks 100 --runs-per-task 6 --min-runs 2` |
| `t1b_xcheck_4b.jsonl` | Qwen3.5-4B | 3 | 1 | rows 9984, 9997, 10001, re-scored for the two-Mac check |

`row` is the index into `successes.parquet`, built by `t1/fetch_data.py` from
`nebius/SWE-agent-trajectories` (successful runs only, 13,314 rows). Use it to join files and to pair
4B with 9B.

## How they were made

- Scorer: `t1b_score.py`, the fixed version posted in issue #1 comment 7004324394, unmodified
  (sha256 prefix `0f93e6ff9d57602c`).
- Models: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and
  `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, converted with
  `mlx_lm convert --dtype bfloat16`, plus `-q --q-bits {4,3} --q-group-size 64` for the quantized
  copies (MLX affine quantization, not llama.cpp `Q4_K_M`/`Q3_K_M`).
- Software: MLX 0.32.4, mlx-lm 0.32.2, transformers 5.23.0, Python 3.12.
- Hardware: Apple M5 Pro, 48 GiB unified memory.
- Commands, run from `t1/`:
  - 4B rest: `t1b_score.py --jobs-file results/jobs_4b_remaining_rows.json --out results/t1b_4b_rest.jsonl`
  - 9B: `t1b_score.py --models-root models9b --jobs-file results/jobs_4b_main_rows.json --out results/t1b_9b.jsonl`
  - Long: the D1 flags above, `--out results/t1b_long.jsonl`

## Two-Mac check

Rows 9984, 9997 and 10001 re-scored here match the owner's agent's values to every printed digit
(`kl4_mean`, `kl3_mean`, `kl3_decision`, `flip3_mean`, `ent_mean`; largest difference 0.03% on a
value printed to 4 digits). The two Macs produce the same numbers, so the 4B halves can be pooled.

## Fields

One JSON object per run:

- `instance_id`, `model_name` (the model that wrote the trajectory), `run_key`, `row`
- `n_tokens` (rendered length), `turns`, `n_scored` (assistant-written tokens scored)
- `kl4_mean`, `kl3_mean`: mean KL(BF16 || Q) over scored tokens, nats per token
- `flip4_mean`, `flip3_mean`: top-1 disagreement rate
- `ent_mean`: mean BF16 entropy; `nll16_mean`, `nll4_mean`, `nll3_mean`: NLL of the written token
- `kl4_decision`, `kl3_decision`, `n_decision`: the decision-point score (T1's version: command
  starts and top-10% entropy positions)
- `by_span`: sums by span (`action` is the command inside the ``` fence, the T1 equivalent of
  `kl_command`; `text` is reasoning; `end` is the end-of-turn token). Divide each sum by `n`.
- `by_bucket`: sums by context-position bucket; `by_pos_ent`: sums by position bucket within fixed
  BF16-entropy bins (`ent<0.1`, `ent0.1-1`, `ent>=1`)

## Caveats

The trajectories were written by SWE-agent with Llama models and rendered in Qwen's chat format
without thinking blocks, so these numbers measure Qwen's view of someone else's agent text. In the
long set, scored tokens past 32K are few (11,587 of 1,531,404), so the 32K-64K bucket is thin.
