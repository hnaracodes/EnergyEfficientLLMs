"""P0d step 1: score the pool's successful BF16 runs against the Q3_K_M and Q4_K_M probes (docs/04, P0d).

    python scripts/p0d_score_pool.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json \
        --kl-score bin/kl_score --gguf-p bf16.gguf --gguf-q3 q3_k_m.gguf --gguf-q4 q4_k_m.gguf \
        [--lists pool200] [--batch 25] [--positions-dir outputs/p0d/positions] [--llama-commit b1b770d9] \
        [--machine ec2-g6e.16xlarge]

Every resolved BF16 attempt on --lists whose run_id is one of the --manifest's P0b runs is scored
(the manifest of scripts/p0_analyze.py); P0c's BF16 runs on pool200 are a different draw. An attempt
that ended in end_reason error is an infrastructure fault, rerun as a fresh attempt (docs/04), so it
is skipped and counted, as qtrace.p1_select.candidates drops it. A run flagged by the test-file tripwire is skipped and counted, because it can never be a selection
candidate (docs/04, "Test-file tripwire"). Held-out lists are refused.

Each (attempt, probe) becomes one row of --out: attempt_id, task_id, probe, gguf_p, gguf_q and the
keys of qtrace.kl.score_stats, plus "runtime", the helper's backend settings (devices, n_batch,
KV-cache types, flash attention). One helper call scores --batch episodes, so each load of the two
GGUF files is shared by many runs, and the rows are appended after every call. A restart skips the
pairs already written. A torn last line left by a crash is cut off and that run is scored again.

So the scoring hours are stated, not rebuilt (docs/04, P0 step 3), every call also appends one line
to <out stem>.batches.jsonl: probe, attempt ids, token count, wall seconds, GPU joules (EnergyMeter:
NVIDIA only, idle draw included, else null), the helper's runtime settings, the platform, the
sha256 of the helper and both GGUFs, n_batch, --llama-commit and the optional free-text --machine
label as given ("machine_label"); no hostname is recorded (PR #11 review). Restarts and machine
moves append to the same file, so each line describes its own call. With --positions-dir, each (attempt, probe)
also gets <attempt_id>.<probe>.npz holding idx and the per-position kl, flip, entropy_p, nll_p and
nll_q, written before its row, so the S2 entropy threshold and any re-analysis need no rescoring.
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import append_jsonl, read_jsonl
from qtrace.energy import EnergyMeter
from qtrace.kl_gguf import score_trajectories_gguf
from qtrace.p0d import HELD_OUT_LISTS, flagged

EPISODE_KEYS = ("input_ids", "model_mask", "tool_call_positions", "command_spans", "turn_spans")


def pool_runs(attempt_paths, lists, run_ids):
    """(record, absolute episode path) for each run to score, the flagged count and the error count excluded."""
    runs, seen, n_flagged, n_error = [], set(), 0, 0
    for path in attempt_paths:
        base = os.path.dirname(os.path.abspath(path))
        for r in read_jsonl(path):
            missing = [k for k in ("attempt_id", "run_id", "task_id", "list", "profile", "end_reason", "resolved")
                       if k not in r]
            if missing:
                raise ValueError(f"{path}: attempt {r.get('attempt_id')!r} lacks {missing}")
            if r["run_id"] not in run_ids or r["profile"] != "BF16" or r["list"] not in lists:
                continue
            if r["end_reason"] == "error":
                n_error += 1
                continue
            if not r["resolved"]:
                continue
            if flagged(r):
                n_flagged += 1
                continue
            if r["attempt_id"] in seen:
                raise ValueError(f"duplicate attempt_id {r['attempt_id']!r} (is an attempts file passed twice?)")
            seen.add(r["attempt_id"])
            if "episode_path" not in r:
                raise ValueError(f"{path}: attempt {r['attempt_id']!r} has no episode_path")
            runs.append((r, os.path.join(base, r["episode_path"])))  # join keeps an absolute path as is
    return runs, n_flagged, n_error


def load_episode(path):
    """An episode .npz as the episode dict of score_trajectories_gguf, checked against the agreed format."""
    with np.load(path) as z:
        missing = [k for k in EPISODE_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        ids, mask = z["input_ids"], z["model_mask"]
        if ids.ndim != 1 or mask.shape != ids.shape or mask.dtype != np.bool_:
            raise ValueError(f"{path}: need input_ids [n] and a bool model_mask [n]")
        spans = {}
        for k in ("command_spans", "turn_spans"):
            a = z[k].reshape(0, 2) if z[k].size == 0 else z[k]  # an empty list may be saved with shape (0,)
            if a.ndim != 2 or a.shape[1] != 2 or (a[:, 0] < 0).any() or (a[:, 0] > a[:, 1]).any() or \
                    (a[:, 1] > len(ids)).any():
                raise ValueError(f"{path}: {k} must be [m, 2] (start, end) spans inside the episode")
            spans[k] = [tuple(s) for s in a.tolist()]
        calls = z["tool_call_positions"].reshape(-1)
        if ((calls < 0) | (calls >= len(ids))).any():
            raise ValueError(f"{path}: tool_call_positions outside the episode")
        if not mask[calls].all():  # a decision point must be a scored, model-written token
            raise ValueError(f"{path}: tool_call_positions outside model_mask")
        return {"input_ids": torch.from_numpy(ids.astype(np.int64)).reshape(1, -1),
                "model_mask": torch.from_numpy(mask.copy()), "tool_call_positions": calls.tolist(), **spans}


def resume(out, gguf_p, probes):
    """(attempt_id, probe) pairs already in out.

    Every row is checked before anything is written, so a wrong --out is refused, not cut. A row
    without probe or the GGUF names means out is not a scores file. A row scored with a GGUF of
    another file name raises: one probe's ranking must come from one file. Names, not full paths, are
    compared, so a run can move between machines. A last line without its newline is then kept if it
    parses (the crash fell between a row and its newline) and cut if not, so that run is scored again.
    """
    if not os.path.exists(out):
        return set()
    with open(out, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n") + 1
    rows = [json.loads(line) for line in data[:cut].splitlines() if line.strip()]
    try:
        tail = json.loads(data[cut:]) if data[cut:].strip() else None
    except ValueError:
        tail = None
    rows += [tail] if tail is not None else []
    done = set()
    for r in rows:
        missing = [k for k in ("attempt_id", "probe", "gguf_p", "gguf_q") if not isinstance(r, dict) or k not in r]
        if missing:
            raise ValueError(f"{out} is not a P0d scores file (a row lacks {missing}); use another --out")
        want = probes.get(r["probe"])
        if want is None or os.path.basename(r["gguf_q"]) != os.path.basename(want) or \
                os.path.basename(r["gguf_p"]) != os.path.basename(gguf_p):
            raise ValueError(f"{out}: {r['attempt_id']} {r['probe']} was scored with a different GGUF "
                             f"({r['gguf_p']}, {r['gguf_q']}); use another --out")
        done.add((r["attempt_id"], r["probe"]))
    if cut < len(data):
        with open(out, "rb+") as f:
            if tail is None:
                f.truncate(cut)
                print(f"cut a torn last line ({len(data) - cut} bytes) from {out}")
            else:
                f.seek(0, os.SEEK_END)
                f.write(b"\n")  # or the next appended row would join this line
    return done


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def batches_path(out):
    return os.path.splitext(out)[0] + ".batches.jsonl"


def score_pool(attempt_paths, binary, gguf_p, probes, lists, out, batch=25, n_batch=512, *, run_ids,
               positions_dir=None, llama_commit=None, machine=None):
    """Score every pool run of the P0b run_ids under each probe ({probe name: GGUF path}), appending to out."""
    bad = sorted(set(lists) & set(HELD_OUT_LISTS))
    if bad:
        raise ValueError(f"{bad} are held-out lists and never enter a pool")
    runs, n_flagged, n_error = pool_runs(attempt_paths, lists, run_ids)
    done = resume(out, gguf_p, probes)
    print(f"{len(runs)} pool runs, {n_flagged} flagged by the test-file tripwire excluded, "
          f"{n_error} infrastructure errors excluded, {len(done)} rows already in {out}")
    if positions_dir:
        os.makedirs(positions_dir, exist_ok=True)
    hashes = {}  # computed once, only for files a call uses
    scored, seconds, joules = 0, 0.0, None
    for probe, gguf_q in probes.items():
        todo = [(r, ep) for r, ep in runs if (r["attempt_id"], probe) not in done]
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            for f in (binary, gguf_p, gguf_q):
                hashes[f] = hashes.get(f) or sha256(f)
            episodes = [load_episode(ep) for _, ep in chunk]
            with EnergyMeter() as meter:
                stats = score_trajectories_gguf(binary, gguf_p, gguf_q, episodes, n_batch=n_batch,
                                                positions=bool(positions_dir))
            if positions_dir:
                for (r, _), st in zip(chunk, stats):
                    np.savez(os.path.join(positions_dir, f"{r['attempt_id']}.{probe}.npz"), **st["positions"])
            stats = [{k: v for k, v in st.items() if k != "positions"} for st in stats]
            append_jsonl(out, [{"attempt_id": r["attempt_id"], "task_id": r["task_id"], "probe": probe,
                                "gguf_p": gguf_p, "gguf_q": gguf_q, **st} for (r, _), st in zip(chunk, stats)])
            append_jsonl(batches_path(out), [{
                "probe": probe, "attempt_ids": [r["attempt_id"] for r, _ in chunk],
                "n_tokens": sum(e["input_ids"].shape[-1] for e in episodes), "seconds": meter.seconds,
                "joules": meter.joules, "runtime": stats[0].get("runtime"), "platform": platform.platform(),
                "n_batch": n_batch, "llama_commit": llama_commit, "machine_label": machine,
                "sha256": {"kl_score": hashes[binary], "gguf_p": hashes[gguf_p], "gguf_q": hashes[gguf_q]}}])
            scored += len(chunk)
            seconds += meter.seconds
            joules = None if meter.joules is None else (joules or 0.0) + meter.joules
            print(f"{probe}: {s + len(chunk)}/{len(todo)}, {meter.seconds:.0f} s")
    return {"kept": len(runs), "flagged_excluded": n_flagged, "infra_error_excluded": n_error,
            "already_scored": len(done), "scored": scored, "seconds": seconds, "joules": joules}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--manifest", required=True, help="scripts/p0_analyze.py's manifest; its P0b run ids are scored")
    ap.add_argument("--kl-score", required=True, help="the kl_score helper binary (qtrace/kl_gguf.py)")
    ap.add_argument("--gguf-p", required=True, help="BF16 GGUF")
    ap.add_argument("--gguf-q3", required=True)
    ap.add_argument("--gguf-q4", required=True)
    ap.add_argument("--lists", nargs="+", default=["pool200"])
    ap.add_argument("--batch", type=int, default=25, help="episodes per helper call (one model load each)")
    ap.add_argument("--n-batch", type=int, default=512, help="llama.cpp batch size")
    ap.add_argument("--out", default="outputs/p0d/scores.jsonl")
    ap.add_argument("--positions-dir", help="write each run's per-position values here as <attempt_id>.<probe>.npz")
    ap.add_argument("--llama-commit", help="the llama.cpp commit kl_score was built against, recorded per batch")
    ap.add_argument("--machine", help="free-text machine label, e.g. mac-48gb; recorded per batch as given")
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        p0b = json.load(f).get("P0b")
    if not p0b:
        ap.error(f"{args.manifest} names no P0b run ids")
    report = score_pool(args.attempts, args.kl_score, args.gguf_p, {"Q3_K_M": args.gguf_q3, "Q4_K_M": args.gguf_q4},
                        args.lists, args.out, batch=args.batch, n_batch=args.n_batch,
                        run_ids=set(p0b), positions_dir=args.positions_dir, llama_commit=args.llama_commit,
                        machine=args.machine)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
