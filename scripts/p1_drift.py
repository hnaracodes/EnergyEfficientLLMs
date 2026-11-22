"""Drift of one trained P1 arm from the base model (docs/04, "Training": KL to base on held-out text).

Run on the box after scripts/p1_train_lora.py, once per arm and seed:
    python scripts/p1_drift.py --run-dir outputs/runs/qtrace_seed0 --attempts outputs/attempts/<p0b run>.jsonl
Writes <run-dir>/drift.json next to train_meta.json, whose "model" (base) and "merged" (arm) it reads.

Primary (owner rule, issue #1 and the PR #15 review): HF teacher-forced KL(base || arm) in float32,
kl.score_trajectory with model_p = base and model_q = the merged arm, over the model-written tokens, pooled
over episodes (token-weighted). Secondary: the same KL between the two GGUF files on llama.cpp's CPU backend,
reported only when its mean is above GGUF_REPORT_KL, because HF and llama.cpp already disagree by about 0.04
nats per position on the base alone. It is not computed here: pass --gguf-kl, a JSON list with one row per
held-out episode ({"attempt_id", "n_scored", "kl_mean"}, optionally "runtime"), for example
qtrace.kl_gguf.score_trajectories_gguf(kl_score, base_gguf, arm_gguf, eps, cpu=True) with each row's
attempt_id added, where eps = [{"input_ids": e["input_ids"], "model_mask": e["model_mask"]} for e in
heldout_episodes(attempts)]. Do not pass the episodes whole: their turn_spans is a numpy [m, 2] array, and
kl.score_stats tests it with `if turn_spans:`, which raises on any real episode.

Not done here, still open: the other half of docs/04's drift item, each arm's BF16 resolve rate minus base's.
It needs the P2 eval runs, and P2's pick_headline and b_metrics do not report it yet (they give each arm's
resolve_rate_bf16 and the arm minus random; their only "_vs_base" pair is the estimator check's log energy
ratio). It belongs in pick_headline's output, in a follow-up to the P2 analysis PR.

Held-out text, an owner decision (docs/04 does not name the set): the heldout100 BF16 P0b episodes, the first
attempt per task by attempt_id, resolved or not. Held-out repos never enter a pool, so no arm trained on them;
one attempt per task keeps the cost at 100 episodes x 2 forward passes per arm. Both models in float32 take
about 32 GB for the 4B; this has only been run on the tiny test model, not at 64K tokens on the box.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.kl import score_trajectory
from qtrace.models import pick_device
from qtrace.sft import load_episode

HELDOUT_LIST, HELDOUT_PROFILE, PER_TASK = "heldout100", "BF16", 1
GGUF_REPORT_KL = 1e-3
HELDOUT_NOTE = (
    "docs/04 does not name the held-out text set; default (owner to confirm): the first heldout100 "
    "BF16 P0b attempt per task by attempt_id, resolved or not"
)


def heldout_episodes(attempts_path):
    """The held-out set: per task, the first HELDOUT_LIST HELDOUT_PROFILE attempt by attempt_id, loaded.

    episode_path is resolved against the attempts file's folder, as qtrace.p1_select (PR #14, read_attempts) reads it. Returns
    dicts with task_id, attempt_id and load_episode's arrays, ordered by task_id.
    """
    base = os.path.dirname(os.path.abspath(attempts_path))
    by_task = {}
    for r in sorted(read_jsonl(attempts_path), key=lambda r: r["attempt_id"]):
        if r["list"] == HELDOUT_LIST and r["profile"] == HELDOUT_PROFILE:
            by_task.setdefault(r["task_id"], []).append(r)
    if not by_task:
        raise ValueError(
            f"{attempts_path}: no {HELDOUT_LIST} {HELDOUT_PROFILE} attempts"
        )
    return [
        {
            "task_id": t,
            "attempt_id": r["attempt_id"],
            **load_episode(os.path.join(base, r["episode_path"])),
        }
        for t in sorted(by_task)
        for r in by_task[t][:PER_TASK]
    ]


def pooled(rows, key):
    """Token-weighted mean of rows[key] over rows with scored tokens; ValueError if none has any."""
    n = sum(r["n_scored"] for r in rows)
    if n == 0:
        raise ValueError("no scored tokens in the held-out episodes")
    return sum(r[key] * r["n_scored"] for r in rows if r["n_scored"]) / n


def hf_drift(base, arm, episodes):
    """Pooled KL(base || arm), top-1 flip rate and NLLs of the actual tokens, plus kl_mean per episode."""
    rows = []
    for e in episodes:
        s = score_trajectory(
            base,
            arm,
            torch.from_numpy(e["input_ids"]).unsqueeze(0),
            torch.from_numpy(e["model_mask"]),
        )
        rows.append({"task_id": e["task_id"], "attempt_id": e["attempt_id"], **s})
    return {
        "method": "hf_f32",
        "dtype": "float32",
        "n_scored": sum(r["n_scored"] for r in rows),
        "kl_mean": pooled(rows, "kl_mean"),
        "flip_rate": pooled(rows, "flip_rate"),
        "nll_base_mean": pooled(rows, "nll_p_mean"),
        "nll_arm_mean": pooled(rows, "nll_q_mean"),
        "per_episode": [
            {k: r[k] for k in ("task_id", "attempt_id", "n_scored", "kl_mean")}
            for r in rows
        ],
    }


def gguf_drift(path, attempt_ids):
    """The secondary GGUF drift from a precomputed result file; ValueError unless it covers exactly attempt_ids."""
    with open(path) as f:
        rows = json.load(f)
    got = [r["attempt_id"] for r in rows]
    if sorted(got) != sorted(attempt_ids):
        raise ValueError(
            f"{path}: must score the same held-out episodes as the HF drift, got {got[:5]} "
            f"for {list(attempt_ids)[:5]}"
        )
    kl = pooled(rows, "kl_mean")
    return {
        "method": "gguf_llama_cpp",
        "source": path,
        "n_scored": sum(r["n_scored"] for r in rows),
        "kl_mean": kl,
        "threshold": GGUF_REPORT_KL,
        "reported": kl > GGUF_REPORT_KL,
        "runtime": rows[0].get("runtime"),
    }


def load_f32(path, device):
    from transformers import AutoModelForCausalLM

    return (
        AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        .to(device)
        .eval()
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="a P1 run folder with train_meta.json and merged/",
    )
    ap.add_argument(
        "--attempts",
        required=True,
        help="the P0b main run's attempts file, outputs/attempts/<run>.jsonl",
    )
    ap.add_argument(
        "--gguf-kl",
        default=None,
        help="precomputed GGUF drift rows (see the module docstring)",
    )
    ap.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    args = ap.parse_args(argv)
    with open(os.path.join(args.run_dir, "train_meta.json")) as f:
        meta = json.load(f)
    episodes = heldout_episodes(args.attempts)
    gguf = (
        gguf_drift(args.gguf_kl, [e["attempt_id"] for e in episodes])
        if args.gguf_kl
        else None
    )
    device = args.device or pick_device()
    primary = hf_drift(
        load_f32(meta["model"], device), load_f32(meta["merged"], device), episodes
    )
    drift = {
        "arm": meta["arm"],
        "seed": meta["seed"],
        "base": meta["model"],
        "arm_model": meta["merged"],
        "device": device,
        "heldout": {
            "attempts": args.attempts,
            "list": HELDOUT_LIST,
            "profile": HELDOUT_PROFILE,
            "per_task": PER_TASK,
            "n_episodes": len(episodes),
            "note": HELDOUT_NOTE,
        },
        "primary": primary,
        "secondary_gguf": gguf,
    }
    out = os.path.join(args.run_dir, "drift.json")
    with open(out, "w") as f:
        json.dump(drift, f, indent=2)
    print(
        f"saved {out}: KL(base || arm) {primary['kl_mean']:.4g} nats per position over {primary['n_scored']} tokens"
    )


if __name__ == "__main__":
    main()
