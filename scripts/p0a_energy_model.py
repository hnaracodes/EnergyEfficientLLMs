"""Fit the P0a per-turn energy model (docs/04) and estimate Q4_K_M joules for every pool attempt.

    python scripts/p0a_energy_model.py --subrun outputs/attempts/sub-*.jsonl \
        --pool outputs/attempts/p0b-*.jsonl [--lists pool200] \
        [--windows outputs/p0a/q4_metrics.jsonl --window-attempts outputs/attempts/p0c-*.jsonl] --out outputs/p0a

--subrun: attempt files holding the energy sub-run; only concurrency-1 attributable attempts are fitted,
          per profile, and Q4_K_M must be among them.
--pool:   candidate pool attempt files (P0b). Rows on lists other than --lists (default pool200), such as
          P0b's heldout100 attempts, are skipped and counted; a held-out repo under a pool list fails.
          end_reason "error" is dropped.
          Every attempt gets an estimate; the pick-agreement checks use the arms' candidates (resolved,
          unflagged BF16 runs) on tasks with at least --min-candidates of them (4; the P0b fallback is 2).
--windows: the Q4_K_M block's 1 s llama-server metrics log, cumulative values
          {t, counter_j, prompt_tokens_total, predicted_tokens_total}.
--window-attempts: the attempt files holding that block; the batched fit spreads their per-turn tokens over
          the windows with gen_t_start, gen_t_first_token and gen_t_end (the server counters are a
          cross-check). Only Q4_K_M rows are used, so a P0c run file holding every llama.cpp profile can be
          passed as is; the other profiles' rows are skipped and counted per profile in
          batched.n_skipped_other_profiles.

Writes energy_model.json ({"profiles": {...}, "fewest_joules_fallback": bool, ...}) and
est_joules.jsonl (one {attempt_id, est_joules_q4, est_gen_joules_q4} per pool attempt, written even under
the fallback; selection obeys the flag). The fallback reads the nested LOAO error of the raw Q4_K_M fit.
est_joules_q4 is the provisional whole-attempt estimate used for ranking (generation-window model plus
the profile's idle_w x tool_seconds); est_gen_joules_q4 is its generation-window part. The pick-agreement
checks compare coefficient sets (docs/04), so both sides are generation-window estimates: a shared
idle_w x tool_seconds term would pull both rankings toward the run with the fewest tool seconds.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl, write_jsonl
from qtrace.energy_model import (
    AGREEMENT_MIN,
    FALLBACK_MAX_REL_ERR,
    MIN_CANDIDATES,
    POOL_LISTS,
    RANK_PROFILE,
    RANK_TARGET,
    TARGETS,
    TERMS,
    estimate_attempt_joules,
    estimate_batched_joules,
    fit_attempts,
    fit_batched,
    pick_agreement,
    pool_records,
    profile_report,
)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subrun", nargs="+", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--windows")
    ap.add_argument("--window-attempts", nargs="+")
    ap.add_argument("--out", default="outputs/p0a")
    ap.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    ap.add_argument("--lists", nargs="+", default=list(POOL_LISTS), help="pool lists to estimate")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()
    if bool(args.windows) != bool(args.window_attempts):
        ap.error("--windows and --window-attempts go together")

    by_profile, excluded = fit_attempts([r for p in args.subrun for r in read_jsonl(p)])
    if RANK_PROFILE not in by_profile:
        raise SystemExit(f"no usable {RANK_PROFILE} sub-run attempts; runs are ranked with its coefficients")
    profiles = {p: profile_report(a, n_boot=args.n_boot, seed=args.seed) for p, a in sorted(by_profile.items())}
    model = profiles[RANK_PROFILE][RANK_TARGET]
    if "error" in model:
        raise SystemExit(f"the {RANK_PROFILE} {RANK_TARGET} fit used for ranking failed: {model['error']}")
    idle_w = profiles[RANK_PROFILE]["idle_w"]
    if idle_w is None:
        raise SystemExit(f"no valid energy.idle_w_ref in the {RANK_PROFILE} sub-run; the whole-attempt estimate "
                         "charges idle_w per tool second")
    pool, n_error, skipped = pool_records([r for p in args.pool for r in read_jsonl(p)], lists=args.lists)
    q4_gen = {r["attempt_id"]: estimate_attempt_joules(r, model) for r in pool}
    q4 = {r["attempt_id"]: estimate_attempt_joules(r, model, idle_w=idle_w) for r in pool}
    by_q4_gen = lambda r: q4_gen[r["attempt_id"]]

    bf16 = None
    b = profiles.get("BF16")
    if b and "error" not in b[RANK_TARGET]:
        bf16 = pick_agreement(pool, lambda r: estimate_attempt_joules(r, b[RANK_TARGET]), by_q4_gen,
                              args.min_candidates)
    batched = None
    if args.windows:
        rows = [r for p in args.window_attempts for r in read_jsonl(p)]
        block = [r for r in rows if r["profile"] == RANK_PROFILE]
        if not block:
            raise SystemExit(f"--window-attempts hold no {RANK_PROFILE} attempts (the --windows block)")
        other = dict(sorted(Counter(r["profile"] for r in rows if r["profile"] != RANK_PROFILE).items()))
        fit = {**fit_batched(read_jsonl(args.windows), block), "n_skipped_other_profiles": other}
        agree = pick_agreement(pool, lambda r: estimate_batched_joules(r, fit), by_q4_gen, args.min_candidates)
        share = agree["share_same_pick"]
        batched = {**fit, "agreement_with_c1": agree, "below_85pct": None if share is None else share < AGREEMENT_MIN}

    report = {
        "ranking": {"profile": RANK_PROFILE, "target": RANK_TARGET, "quantity": "whole_attempt_raw_joules",
                    "idle_w": idle_w, "provisional": True,
                    "rule": "generation-window model + idle_w x tool_seconds; provisional until the owner "
                            "confirms the endpoint (issue #1, question 2)"},
        "fewest_joules_fallback": model["fallback"],
        "fallback_rule": f"nested LOAO median absolute relative error on attempt joules > {FALLBACK_MAX_REL_ERR} "
                         f"({RANK_PROFILE}, {RANK_TARGET}); fewest_joules then falls back to fewest_turns",
        "fallback_error": model["cv"]["nested_loao"],
        "terms_available": list(TERMS),
        "profiles": profiles,
        "excluded_subrun_attempts": excluded,
        "pool": {"n_attempts": len(pool), "n_dropped_error": n_error, "lists": args.lists,
                 "n_skipped_other_lists": skipped},
        "bf16_vs_q4": bf16,
        "batched": batched,
        "n_boot": args.n_boot, "seed": args.seed,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "energy_model.json"), "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    write_jsonl(os.path.join(args.out, "est_joules.jsonl"),
                [{"attempt_id": aid, "est_joules_q4": j, "est_gen_joules_q4": q4_gen[aid]} for aid, j in q4.items()])

    print(f"{'profile':8} {'target':16} {'nested':>7} {'loao':>7} {'pairs':>6}  terms")
    for p, rep in profiles.items():
        for t in TARGETS:
            r = rep[t]
            if "error" in r:
                print(f"{p:8} {t:16} not fitted: {r['error']}")
                continue
            pairs = f"{r['pairs']['n_correct']}/{r['pairs']['n_pairs']}"
            print(f"{p:8} {t:16} {pct(r['cv']['nested_loao']):>7} {pct(r['cv']['loao']):>7} {pairs:>6}  "
                  + "+".join(r["terms"]))
    print(f"fewest_joules_fallback: {model['fallback']} ({RANK_PROFILE} {RANK_TARGET} nested LOAO "
          f"{pct(model['cv']['nested_loao'])} vs {pct(FALLBACK_MAX_REL_ERR)})")
    if bf16:
        print(f"BF16 vs {RANK_PROFILE} same cheapest run: {pct(bf16['share_same_pick'])} of {bf16['n_tasks']} tasks, "
              f"mean Spearman {'n/a' if bf16['mean_spearman'] is None else round(bf16['mean_spearman'], 3)}")
    if batched:
        a = batched["agreement_with_c1"]
        print(f"batched vs concurrency-1 same cheapest run: {pct(a['share_same_pick'])} of {a['n_tasks']} tasks"
              f"{' (below 85%: name the regime)' if batched['below_85pct'] else ''}")
    print("wrote", os.path.join(args.out, "energy_model.json"), "and est_joules.jsonl,", len(q4), "rows")


if __name__ == "__main__":
    main()
