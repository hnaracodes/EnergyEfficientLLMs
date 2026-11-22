"""Simulate the P2 minimum detectable effect for P0b criterion 3 and the guard's resolve-rate SE (docs/04).

    python scripts/p2_mde.py outputs/attempts/<P0b run>.jsonl ... [--est-joules outputs/p0a/est_joules.jsonl] \
        [--profile BF16] [--ratios 0.8 0.9 ...] [--n-sim 500] [--n-boot 1000] [--seed 0] [--guard-margin M] \
        [--gates outputs/p0/gates.json | --q4-resolve-ratio R] [--out outputs/p0b/mde.json]

Per-task pass rates come from the heldout100 rows at --profile: P0b's BF16 rates at k = 16 by default, as docs/04
criterion 3 is worded (owner's agent on PR #17, answer 2). Since P2's primary runs at Q4_K_M, a variant scales
them by P0a's pooled Q4_K_M/BF16 resolve ratio, read from --gates (outputs/p0/gates.json: the re-test's
P0a.retest.factors.resolve_rate when one ran, else P0a.factors.resolve_rate) or given as --q4-resolve-ratio; it
is written under variant_q4_resolve_scaled, null when neither is given. --profile Q4_K_M with P0a's file reads
P0a's k = 3 rates instead. Those rows must all come from one runtime: P0a also runs BF16 on heldout100, on
llama.cpp, so passing P0a's file with --profile BF16 fails rather than pooling its attempts into P0b's vLLM
counts (docs/04 P0b criterion 3 reads P0b's 16 attempts per task). Costs come from the pool200 rows at
--pool-profile, per pool task and split by outcome: each attempt's est_joules_q4 from est_joules.jsonl when
--est-joules is given, else its processed tokens (output plus every input token), which the output names. The
primary draws each simulated task's costs from one pool task's own attempts (matched to tasks);
sensitivity_independent_costs draws them from all pool attempts regardless of the task. An attempt is resolved
when resolved is true and end_reason is "submitted", as P2 counts it; end_reason "error" rows are left out.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.energy_model import pool_records
from qtrace.p2_analysis import MDE_POWER, MDE_RATIOS, mde


def resolved(rec):
    if type(rec.get("resolved")) is not bool:
        raise ValueError(f"attempt {rec.get('attempt_id')!r}: resolved must be true or false")
    return rec["resolved"] and rec["end_reason"] == "submitted"


def heldout_tasks(rows, profile, lst="heldout100"):
    acc = {}
    rows = [r for r in rows if r.get("list") == lst and r.get("profile") == profile and r["end_reason"] != "error"]
    runtimes = sorted({r["runtime"] for r in rows})
    if len(runtimes) > 1:
        raise ValueError(f"{lst} attempts at {profile} come from runtimes {runtimes}; pass the files of one runtime "
                         "(P0b's vLLM run for BF16, P0a's llama.cpp runs for Q4_K_M)")
    for r in rows:
        a = acc.setdefault(r["task_id"], {"repo": r["repo"], "successes": 0, "attempts": 0})
        a["successes"] += int(resolved(r))
        a["attempts"] += 1
    if not acc:
        raise ValueError(f"no {lst} attempts at {profile}")
    return [acc[t] for t in sorted(acc)]


def pool_costs(rows, profile, est_joules=None):
    """Per pool task (sorted by task_id), its attempts' costs by outcome: [{"resolved": [...], "failed": [...]}]."""
    kept, _, _ = pool_records(rows)
    kept = [r for r in kept if r["profile"] == profile]
    if est_joules is not None:
        missing = [r["attempt_id"] for r in kept if r["attempt_id"] not in est_joules]
        if missing:
            raise ValueError(f"{len(missing)} pool attempts have no est_joules row, e.g. {missing[:3]}")
    by_task = {}
    for r in kept:
        c = by_task.setdefault(r["task_id"], {"resolved": [], "failed": []})
        c["resolved" if resolved(r) else "failed"].append(
            est_joules[r["attempt_id"]] if est_joules is not None
            else sum(r["tokens"][k] for k in ("output", "input_uncached", "input_cached")))
    return [by_task[t] for t in sorted(by_task)]


def q4_resolve_ratio(path):
    """(P0a's pooled Q4_K_M/BF16 resolve rate ratio, where in gates.json it came from)."""
    with open(path) as f:
        p0a = json.load(f)["P0a"]
    look = p0a.get("retest") or p0a  # retest is null unless the inconclusive first look was re-tested
    where = "P0a.retest.factors.resolve_rate" if look is not p0a else "P0a.factors.resolve_rate"
    rr = look["factors"]["resolve_rate"]
    q4, bf16 = rr["Q4_K_M"], rr["BF16"]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in (q4, bf16)):
        raise ValueError(f"{path} {where}: need positive Q4_K_M and BF16 rates for the resolve ratio, got {rr!r}")
    return q4 / bf16, f"{path} {where}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--est-joules", help="est_joules.jsonl from scripts/p0a_energy_model.py")
    ap.add_argument("--profile", default="BF16", help="profile of the heldout100 pass rates")
    ap.add_argument("--pool-profile", default="BF16", help="profile of the pool200 cost rows")
    ap.add_argument("--ratios", type=float, nargs="+", default=list(MDE_RATIOS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-sim", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-margin", type=float)
    scale = ap.add_mutually_exclusive_group()
    scale.add_argument("--gates", help="gates.json from scripts/p0_analyze.py, for P0a's pooled resolve ratio")
    scale.add_argument("--q4-resolve-ratio", type=float, help="P0a's pooled Q4_K_M/BF16 resolve rate ratio")
    ap.add_argument("--out", default="outputs/p0b/mde.json")
    args = ap.parse_args(argv)

    rows = [r for p in args.attempts for r in read_jsonl(p)]
    est = None if args.est_joules is None else {e["attempt_id"]: e["est_joules_q4"] for e in read_jsonl(args.est_joules)}
    try:
        tasks = heldout_tasks(rows, args.profile)
        costs = pool_costs(rows, args.pool_profile, est)
        kw = {"ratios": args.ratios, "seeds": args.seeds, "n_sim": args.n_sim, "n_boot": args.n_boot,
              "alpha": args.alpha, "seed": args.seed, "guard_margin": args.guard_margin}
        out = mde(tasks, costs, **kw)
        out["sensitivity_independent_costs"] = mde(tasks, costs, matched=False, **kw)
        ratio, src = (q4_resolve_ratio(args.gates) if args.gates else
                      (args.q4_resolve_ratio, "--q4-resolve-ratio") if args.q4_resolve_ratio is not None else
                      (None, None))
        out["variant_q4_resolve_scaled"] = None if ratio is None else {
            "resolve_ratio": ratio, "resolve_ratio_source": src, **mde(tasks, costs, rate_scale=ratio, **kw)}
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"p2_mde: {e!r}") from e
    variant = out["variant_q4_resolve_scaled"]
    for o in (out, out["sensitivity_independent_costs"], variant or {"cost": {}}):
        o["cost"]["source"] = "est_joules_q4" if est is not None else "processed_tokens"
    out["inputs"] = {"attempts": args.attempts, "est_joules": args.est_joules, "profile": args.profile,
                     "pool_profile": args.pool_profile}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    se = out["guard_se"]
    print(f"{args.profile} mean pass@1 {out['mean_pass1']:.3f}, expected resolves per arm-seed run "
          f"{out['expected_resolves_per_arm_seed']:.1f} on {out['n_tasks']} tasks")
    for p in out["power"]:
        print(f"  true JPR ratio {p['ratio']:.3f}: power {p['power']:.3f}"
              + (f", with guard {p['power_with_guard']:.3f}" if "power_with_guard" in p else ""))
    sens = out["sensitivity_independent_costs"]
    print(f"independent costs (sensitivity): MDE at 80% power ratio {sens['mde_ratio_80']}"
          + (f", with the guard {sens['mde_ratio_80_with_guard']}" if args.guard_margin is not None else ""))
    print(f"MDE at 80% power: ratio {out['mde_ratio_80']} (cost: {out['cost']['source']}, matched to tasks over "
          f"{out['cost']['n_pool_tasks']} pool tasks); se_rr {se['se_rr']:.4f}, "
          f"se_rr_diff {se['se_rr_diff']:.4f} ({se['formula']})")
    print(f"guard margin = k x se_rr_diff = k x {se['margin_se']:.4f} (the guard bounds the paired difference; "
          "k is the owner's decision)")
    if args.guard_margin is not None:
        print(f"MDE at 80% power with the guard at margin {args.guard_margin}: ratio {out['mde_ratio_80_with_guard']}")
        if out["guard_pass_rate"] < MDE_POWER:
            print(f"  the guard alone passes in {out['guard_pass_rate']:.3f} of simulated studies, so power with the "
                  "guard stays below 0.8 at every ratio")
    if variant is not None:
        print(f"Q4_K_M-scaled variant (resolve ratio {variant['resolve_ratio']:.3f}, {variant['resolve_ratio_source']})"
              f": mean pass@1 {variant['mean_pass1']:.3f}, MDE at 80% power ratio {variant['mde_ratio_80']}, "
              f"se_rr_diff {variant['guard_se']['se_rr_diff']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
