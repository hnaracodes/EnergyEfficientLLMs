"""Build the P1 training sets: one run per task for each arm and seed (docs/03 Method step 3, docs/04 Selection).

    python scripts/p1_select.py --config configs/p1_select.yaml

Reads the pool runs' attempts (<attempts_dir>/<run_id>.jsonl for each id in pool_runs), the P0d scores
and score choice, and the P0a energy model and joule estimates. Writes <out_dir>/<arm>_seed<s>.jsonl and
<out_dir>/summary.json and prints a table. A P0d or P0a file that does not exist is passed on as absent;
qtrace/p1_select.py decides whether that is a pre-registered fallback or an error. Arm files from an
earlier run that this run did not build are removed, so training never picks up a stale arm.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl, write_jsonl
from qtrace.p1_select import ARMS, read_attempts, select


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load(path, reader):
    return reader(path) if os.path.exists(path) else None


def write(out_dir, picks, summary):
    os.makedirs(out_dir, exist_ok=True)
    names = {f"{arm}_seed{s}.jsonl" for arm in picks for s in picks[arm]}
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r"(\w+)_seed\d+\.jsonl", name)
        if m and m.group(1) in ARMS and name not in names:
            os.remove(os.path.join(out_dir, name))
            print("removed stale", name)
    for arm in picks:
        for s, rows in picks[arm].items():
            write_jsonl(os.path.join(out_dir, f"{arm}_seed{s}.jsonl"), rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, allow_nan=False)


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def report(summary):
    pool, score, fj, ov = summary["pool"], summary["score"], summary["fewest_joules"], summary["overlap"]
    print(f"pool: {pool['n_tasks']} tasks, {pool['n_candidates']} candidates; {len(pool['dropped_tasks'])} tasks "
          f"below min_candidates {summary['config']['min_candidates']}; infrastructure errors excluded "
          f"{pool['infra_error_excluded']}; flagged successes excluded {pool['flagged_excluded']}")
    print(f"score: {score['used']} (score_choice.json: {score['file_chosen'] if score['file_present'] else 'missing'}"
          f", override: {score['override']}); fewest_joules picks as {fj['picks_as']}"
          + (f" ({fj['reason']})" if fj["reason"] else ""))
    if summary["h3"]:
        print("H3:", summary["h3"])
    print(f"\n{'arm':15s} {'seed':>4s} {'tasks':>6s} {'turns':>7s} {'tokens':>9s} {'joules':>10s} {'score':>8s}")
    for arm, by_seed in summary["arms"].items():
        for s, st in by_seed.items():
            print(f"{arm:15s} {s:>4s} {st['n_tasks']:6d} {fmt(st['mean_turns'], '.2f'):>7s} "
                  f"{fmt(st['mean_output_tokens'], '.1f'):>9s} {fmt(st['mean_est_joules_q4'], '.1f'):>10s} "
                  f"{fmt(st['mean_score'], '.4f'):>8s}")
    close = [f"{a}~{b} {p['share_same']:.2f} (seed {s})" for s, pairs in ov["by_seed"].items() for p in pairs
             if p["not_separable"] for a, b in [p["arms"]]]
    print(f"\noverlap: chance rate {ov['chance_rate']:.3f}; not separable (above {ov['threshold']:.0%}): "
          + (", ".join(close) or "none"))
    rule = summary["arm_overlap_rule"]
    print("P0b fewest_joules/fewest_turns rule:", rule["note"])
    for s, r in (rule["by_seed"] or {}).items():
        regret = ", ".join(f"{n} {x:.1%}" for n, x in r["median_joule_regret"].items())
        print(f"  seed {s}: median joule regret {regret}; calibrated token-mix cost: {r['calibrated_token_mix']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_select.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths, runs = cfg["paths"], cfg.get("pool_runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("pool_runs is empty: list the run ids whose BF16 attempts form the pool (P0b's main run)")

    attempts = [r for run in runs for r in read_attempts(os.path.join(paths["attempts_dir"], f"{run}.jsonl"))]
    inputs = {k: paths[k] for k in ("scores", "score_choice", "energy_model", "est_joules")}
    picks, summary = select(attempts, load(inputs["scores"], read_jsonl), load(inputs["score_choice"], read_json),
                            load(inputs["energy_model"], read_json), load(inputs["est_joules"], read_jsonl),
                            **cfg["select"])
    summary = {**summary, "pool_runs": runs,
               "inputs": {k: {"path": p, "present": os.path.exists(p)} for k, p in inputs.items()}}
    write(paths["out_dir"], picks, summary)
    report(summary)
    print("wrote", paths["out_dir"])


if __name__ == "__main__":
    main()
