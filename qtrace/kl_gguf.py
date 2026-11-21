"""The KL scorer for GGUF models, run through llama.cpp.

P0d's probes are the `Q3_K_M` and `Q4_K_M` GGUF files that llama.cpp serves (docs/04), so the score is
computed in that runtime against the `BF16` GGUF, not against a PyTorch stand-in for the quantizer.
qtrace/kl_score.cpp does the model work and writes the per-position fields of kl.compare(); scoring
from there is kl.score_stats(), shared with the PyTorch path.

Build the helper once against a llama.cpp build tree (the same command works on Linux):

    LLAMA=path/to/llama.cpp
    c++ -std=c++17 -O3 qtrace/kl_score.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
        -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o kl_score

Per-position values depend on the backend and on n_batch (llama.cpp picks matmul kernels by the
number of tokens and output rows in a batch; on Metal KL moved by up to 6e-2 between n_batch 7 and
512 in the PR #11 review), so P0d scores at n_batch 512 and every result carries the settings used
under "runtime". scripts/check_kl_runtime.py holds the cross-runtime acceptance criterion.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import torch

from qtrace.kl import logit_index, score_stats

FIELDS = ("kl", "flip", "entropy_p", "nll_p", "nll_q")
INT32_MAX = 2**31 - 1


def check_run(i, input_ids, idx):
    """input_ids and idx of run i as int32 arrays, or ValueError naming the run.

    Checked here so one bad run fails before the helper loads two models for the whole batch. Token
    ids are checked against the vocabulary size by the helper, which reads it from the GGUF.
    """
    ids = np.asarray(input_ids)
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"run {i}: input_ids must be integer, shaped [n] or [1, n], not {ids.dtype} {ids.shape}")
    if len(ids) and (ids.min() < 0 or ids.max() > INT32_MAX):
        raise ValueError(f"run {i}: token ids must be in [0, 2**31)")
    pos = np.asarray(idx)
    if pos.ndim != 1 or (len(pos) and not np.issubdtype(pos.dtype, np.integer)):
        raise ValueError(f"run {i}: idx must be a 1-D integer array")
    if len(pos) and (pos[0] < 0 or pos[-1] >= len(ids) - 1 or (np.diff(pos) <= 0).any()):
        raise ValueError(f"run {i}: idx must be sorted, unique and in [0, n_tokens - 1)")
    return ids.astype(np.int32), pos.astype(np.int32)


def compare_gguf(binary, gguf_p, gguf_q, runs, n_batch=512, cpu=False, top_k=0):
    """Per-position stats with the keys of kl.compare(), one dict per (input_ids, idx) in runs.

    Both models load once for all runs. idx are logit positions from kl.logit_index(); input_ids and
    idx may be torch tensors or numpy arrays. cpu=True runs on the CPU backend only. top_k > 0 adds
    "topk_ids" and "topk_logprob" ([n, top_k], most likely first) of P at each position, for the
    cross-runtime noise floor. Each dict also has "runtime", the helper's settings (devices, n_batch,
    KV-cache types, flash attention). A non-finite value in any row raises ValueError.
    """
    checked = [check_run(i, ids, idx) for i, (ids, idx) in enumerate(runs)]
    row = np.dtype([("stats", "<f4", len(FIELDS)), ("ids", "<i4", top_k), ("lp", "<f4", top_k)])
    with tempfile.TemporaryDirectory() as d:
        jobs = []
        for i, (ids, idx) in enumerate(checked):
            tok, pos, out = (os.path.join(d, f"{i}.{ext}") for ext in ("tok", "idx", "out"))
            ids.tofile(tok)
            idx.tofile(pos)
            jobs.append((tok, pos, out))
        jobs_file = os.path.join(d, "jobs.txt")
        with open(jobs_file, "w") as f:
            f.writelines(" ".join(j) + "\n" for j in jobs)
        flags = (["--cpu"] if cpu else []) + (["--top-k", str(top_k)] if top_k else [])
        # stderr is kept for failures only: llama.cpp warns on every load about tensors it skips.
        r = subprocess.run([binary, *flags, gguf_p, gguf_q, jobs_file, str(n_batch)], capture_output=True,
                           text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"kl_score exited with {r.returncode}:\n{r.stderr[-4000:]}")
        runtime = json.loads(r.stdout.strip().splitlines()[-1])
        stats = []
        for i, ((_, _, out), (_, idx)) in enumerate(zip(jobs, checked)):
            rows = np.fromfile(out, dtype=row)
            if len(rows) != len(idx):
                raise RuntimeError(f"run {i}: kl_score wrote {len(rows)} rows for {len(idx)} positions")
            bad = ~np.isfinite(rows["stats"]).all(1)
            if top_k:
                bad |= ~np.isfinite(rows["lp"]).all(1)
            if bad.any():
                raise ValueError(f"run {i}: non-finite stats at logit positions {idx[bad][:10].tolist()}")
            s = {k: torch.from_numpy(rows["stats"][:, j].copy()) for j, k in enumerate(FIELDS)}
            if top_k:
                s["topk_ids"], s["topk_logprob"] = torch.from_numpy(rows["ids"].copy()), torch.from_numpy(rows["lp"].copy())
            stats.append({**s, "runtime": runtime})
    return stats


def score_trajectories_gguf(binary, gguf_p, gguf_q, episodes, n_batch=512, entropy_threshold=None, cpu=False):
    """kl.score_trajectory() for each episode, with GGUF models, plus the helper's "runtime" settings.

    Each episode is a dict of score_trajectory's episode arguments: input_ids and model_mask (torch
    tensors or numpy arrays), plus optionally tool_call_positions, command_spans and turn_spans.
    """
    idxs = [logit_index(torch.as_tensor(e["model_mask"])) for e in episodes]
    stats = compare_gguf(binary, gguf_p, gguf_q, [(e["input_ids"], i) for e, i in zip(episodes, idxs)], n_batch,
                         cpu=cpu)
    return [{**score_stats(s, i, e.get("tool_call_positions", ()), e.get("command_spans", ()), entropy_threshold,
                           e.get("turn_spans", ())), "runtime": s["runtime"]}
            for s, i, e in zip(stats, idxs, episodes)]
