"""P0d steps 2-3: choose the stability score, pick the replay pairs and, given continuation results, run
the DiD test (docs/04, P0d).

    python scripts/p0d_analyze.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json \
        --scores outputs/p0d/scores.jsonl [--est-joules outputs/p0a/est_joules.jsonl] [--power-rates BF16 Q3]
    python scripts/p0d_analyze.py --continuations FILE

The manifest is scripts/p0_analyze.py's ({"P0b": ["run-..."], "P0a": [...], "P0c": [...]}); only
attempts whose run_id is one of its P0b runs are read, so P0c's BF16 runs on pool200 never enter.
The first form writes outputs/p0d/score_choice.json: the chosen score (or null), every candidate's
numbers, the replay pairs under the chosen score and, with --power-rates, the simulated power of the
DiD test at 4 and 8 seeds per prefix. --power-rates are the expected BF16 and Q3_K_M continuation
success rates (from the pilot and P0c). Without --est-joules the joules check does not run and the
rule stays undecided unless a candidate fails another check first. A did.json already in --out-dir
belongs to the score_choice.json this run replaces, so it is removed.

The second form runs the DiD test on the pairs frozen in --out-dir/score_choice.json, never on pairs
recomputed from inputs that may have changed since the replays, and writes did.json. Continuation
rows are {task_id, prefix, profile, replay_valid, low_attempt_id, high_attempt_id, successes,
attempts}; the two ids must be the frozen pair's runs, and a replay_valid false row needs no results.
It also writes docs/04's secondary results, which never change the decision: did_q4.json, the same
test on the Q4_K_M continuation rows (when there are any), and did_kl_mean.json, the Q3_K_M test with
each pair's low and high prefix set by kl_mean (it counts the pairs that flip, drops and counts the
pairs tied on kl_mean, and lists both kl_mean values of every pair). All three files are removed
before the test runs, so each one left in --out-dir comes from this run's continuations. A secondary that cannot be
computed from them (no Q4_K_M rows, a pair with Q4_K_M rows on one prefix only, or pairs frozen
without kl_mean) is not written, the reason is printed, and the other files are still written.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0d import (
    CANDIDATES,
    CONFOUNDS,
    POOL_LISTS,
    choose_score,
    did_kl_mean,
    did_q4,
    did_test,
    join_runs,
    power,
    select_pairs,
)

POWER_EFFECTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
POWER_SEEDS = (4, 8)  # docs/04: 4 seeded attempts per prefix, extra budget goes to 8
DID_FILES = ("did.json", "did_q4.json", "did_kl_mean.json")  # all test the pairs of one score_choice.json


def rates(text):
    """A success rate, or comma-separated per-task rates (paired by position across BF16 and Q3)."""
    return [float(v) for v in text.split(",")]


def fmt(x):
    return "n/a" if x is None else f"{x:+.2f}"


def print_table(choice):
    cols = ("turns", "tokens", "joules", "entropy", "split-half", "agreement")
    print(f"{'candidate':<12}" + "".join(f"{c:>11}" for c in cols) + "  result")
    for name in CANDIDATES:
        c = choice["candidates"][name]
        vals = [c["against"][k].get("mean") for k in CONFOUNDS]
        vals += [c["split_half"].get("spearman_brown_mean"), c["probe_agreement"].get("mean")]
        result = "pass" if c["pass"] else f"fails {', '.join(c['failed'])}" if c["failed"] else \
            f"not run: {', '.join(c['not_run'])}"
        print(f"{name:<12}" + "".join(f"{fmt(v):>11}" for v in vals) + f"  {result}")
    print(choice["verdict"])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="*")
    ap.add_argument("--manifest", help="scripts/p0_analyze.py's manifest; its P0b run ids pick the pool attempts")
    ap.add_argument("--scores", default="outputs/p0d/scores.jsonl")
    ap.add_argument("--lists", nargs="+", default=list(POOL_LISTS), help="the pool lists p0d_score_pool.py scored")
    ap.add_argument("--est-joules", help="outputs/p0a/est_joules.jsonl; omit until P0a's energy model exists")
    ap.add_argument("--continuations", help="continuation results; runs the DiD test on the frozen pairs")
    ap.add_argument("--power-rates", nargs=2, type=rates, metavar=("BF16", "Q3"),
                    help="expected continuation success rates: two numbers, or two comma-separated per-task lists")
    ap.add_argument("--replay-drop-rate", type=float, default=0.0,
                    help="expected pair drop rate (pilot replay match rate), so power counts pairs after drops")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out-dir", default="outputs/p0d")
    args = ap.parse_args(argv)
    if args.continuations:
        if args.attempts:
            ap.error("--continuations reads the pairs frozen in --out-dir/score_choice.json; pass no attempts files")
        return did_step(args)
    if not args.attempts or not args.manifest:
        ap.error("attempts files and --manifest are required to choose the score")
    with open(args.manifest) as f:
        manifest = json.load(f)
    if not manifest.get("P0b"):
        ap.error(f"{args.manifest} names no P0b run ids")

    attempts = [r for path in args.attempts for r in read_jsonl(path)]
    joules = read_jsonl(args.est_joules) if args.est_joules else None
    runs, info = join_runs(attempts, read_jsonl(args.scores), joules, lists=args.lists,
                           run_ids=set(manifest["P0b"]))
    choice = choose_score(runs, have_joules=joules is not None, missing_q3=info["n_eligible_without_q3"])
    pairs = select_pairs(runs, choice["chosen"]) if choice["chosen"] else None
    out = {**choice, "inputs": {**info, "manifest": args.manifest, "scores": args.scores,
                                "est_joules": args.est_joules}, "pairs": pairs}
    if args.power_rates and pairs and pairs["n_tasks"] >= 2:
        out["power"] = [power(*args.power_rates, n_tasks=pairs["n_tasks"], seeds=s, effects=POWER_EFFECTS,
                              n_boot=args.n_boot, seed=args.seed, drop_rate=args.replay_drop_rate,
                              covariates=[p["diff"] for p in pairs["tasks"]]) for s in POWER_SEEDS]
    write_json(os.path.join(args.out_dir, "score_choice.json"), out)
    for name in DID_FILES:
        stale = os.path.join(args.out_dir, name)
        if os.path.exists(stale):
            os.remove(stale)
            print("removed", stale, "(it tested the pairs of the score_choice.json just replaced)")
    print_table(choice)
    if pairs:
        print(f"pairs: {pairs['n_tasks']} tasks, spread {pairs['spread']}")
    for p in out.get("power", []):
        print(f"power at {p['seeds']} seeds/prefix, {p['n_tasks_after_drops']} of {p['n_tasks']} tasks after drops:",
              ", ".join(f"{e['effect']:.2f}: {e['power']:.2f}" for e in p["power"]), f"(MDE80 {p['mde_80']})")


def did_step(args):
    path = os.path.join(args.out_dir, "score_choice.json")
    with open(path) as f:
        frozen = json.load(f)
    if not frozen["pairs"]:
        why = "H3 was answered negatively" if frozen["decided"] else "the score rule is undecided"
        raise SystemExit(f"{path} has no chosen score ({why}: {frozen['verdict']}), so there are no replay pairs")
    rows, pairs = read_jsonl(args.continuations), frozen["pairs"]["tasks"]
    for name in DID_FILES:
        stale = os.path.join(args.out_dir, name)
        if os.path.exists(stale):
            os.remove(stale)
            print("removed", stale, "(from an earlier continuations file)")
    did = did_test(rows, pairs=pairs, n_boot=args.n_boot, seed=args.seed)
    did["score"], did["score_choice"] = frozen["chosen"], path
    write_json(os.path.join(args.out_dir, "did.json"), did)
    adj = did["adjusted"]["estimate"] if did["adjusted"] else None
    print(f"DiD {did['did_mean']:+.3f} (p one-sided {did['p_one_sided']:.4f}, "
          f"upper95 {did['upper95_one_sided']:+.3f}), adjusted {fmt(adj)}, "
          f"pair drop rate {did['pair_drop_rate']:.2f} of {did['n_pairs_selected']}: {did['decision']}")
    if did["adjusted"] and did["adjusted"]["note"]:
        print(did["adjusted"]["note"])
    if did["pair_tasks_without_continuations"]:
        print(f"{len(did['pair_tasks_without_continuations'])} pairs have no BF16 and Q3_K_M continuation rows")
    secondary = {"did_q4.json": (did_q4, "no Q4_K_M rows"),
                 "did_kl_mean.json": (did_kl_mean, "the frozen pairs have no low_kl_mean")}
    for name, (test, why) in secondary.items():
        try:
            sec = test(rows, pairs, n_boot=args.n_boot, seed=args.seed)
        except ValueError as e:  # bad secondary rows never undo the primary result already written
            sec, why = None, str(e)
        if sec is None:
            print(f"secondary {name} not written: {why}")
            continue
        sec["pairs_score"], sec["score_choice"] = frozen["chosen"], path
        write_json(os.path.join(args.out_dir, name), sec)
        print(f"secondary {name}: DiD {sec['did_mean']:+.3f} (p one-sided {sec['p_one_sided']:.4f}) on "
              f"{sec['n_tasks']} tasks: {sec['decision']}")


if __name__ == "__main__":
    main()
