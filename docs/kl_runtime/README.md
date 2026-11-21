# KL runtime check: episode and result

`metal_t1_11426.json` is a row written by `scripts/check_kl_runtime.py`. `t1_11426.npz` is the
episode it scored, committed so the check can be rerun without the T1 data.

## Source and license

The episode is one agent run from
[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories),
licensed CC-BY-4.0 (dataset card), revision `68195a1450865274106246d0d0296a1d6807b88e` (the
dataset's main branch, last modified 2024-12-23; the fetch below read main without a pin).

- Row id: 11426, a 0-based index into `successes.parquet` as written by the T1 fetch script
  (`q-local/t1/fetch_data.py`, outside this repo). That script keeps only successful runs
  (`target` true) of instances with at least two of them, in the dataset's file order, so the
  index is stable for a fixed revision.
- That row: `instance_id` `bryanyang0528__ksql-python-73`, `model_name` `swe-agent-llama-70b`,
  `exit_status` `submitted`.

## Changes from the source

The trajectory text is unchanged. It was rendered and tokenized:

- Tokenizer: Qwen3.5-4B (`Qwen/Qwen3.5-4B`) at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Rendering: the T1 render function (`render` in `q-local/t1/t1b_score.py`). Each turn becomes
  `<|im_start|>{role}\n{text}<|im_end|>\n`, with roles system, user, and assistant (the dataset's
  "ai"); turns with other roles are dropped. `model_mask` is true on the assistant's text tokens
  and on the `<|im_end|>` that closes each assistant turn, false everywhere else.
- Result: 16,542 tokens, 3,057 of them model-written. `input_ids` is int32 and `model_mask` bool,
  the episode store's format. The file also holds `command_spans` and `turn_spans` (start, end
  pairs for the 33 code-fenced commands and the 33 assistant turns) and an empty
  `tool_call_positions`; the check reads only `input_ids` and `model_mask`.

## Rerun the check

Build `kl_score` as `qtrace/kl_gguf.py` describes, then, from the repo root:

    python scripts/check_kl_runtime.py docs/kl_runtime/t1_11426.npz --hf path/to/Qwen3.5-4B \
        --kl-score path/to/kl_score --gguf-bf16 qwen3.5-4b-bf16.gguf --gguf-q4 qwen3.5-4b-q4_k_m.gguf \
        [--device cuda] [--machine mac-48gb] --out row.json

`--hf` is the HF BF16 checkpoint the GGUFs were converted from. The row records the platform
string, devices and the optional `--machine` label, never a hostname. The committed row is
informative only (Metal and MPS, see its "note"); the deciding run is CUDA at the llama.cpp pin on
a 64K trajectory.
