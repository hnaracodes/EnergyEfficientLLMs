"""Compute the P0 gates (docs/04) from attempt records and write outputs/p0/gates.json.

    python scripts/p0_analyze.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json --inrepo-size N

The manifest maps each gate to the run ids that belong to it, so a gate never mixes in another
run's attempts (the pilot, the energy sub-run, P0d continuations):
    {"P0b": ["run-..."], "P0a": ["run-..."], "P0c": ["run-..."], "P0a_topup": ["run-..."]}
P0a_topup is listed only after an inconclusive first look: 04 then adds 3 attempts per task at BF16
and Q4_K_M on heldout100 and re-tests once on all 6. P0a.final_verdict is the verdict P1 reads (the
re-test's when it ran); it stays null while a required re-test has not run.
Records with end_reason "error" are dropped (04 reruns them as fresh attempts) and duplicate
attempt ids are dropped; both are counted per gate. Attempt counts per task and profile are
checked against the design (P0b 16, P0a 3 on heldout100 and verified100, P0c 4). Every gate also
reports its result excluding flagged attempts (test-file tripwire), the flagged share per profile, and
a sensitivity analysis excluding infra_timeout hits (04). P0a.secondary holds 04's secondaries: every
profile against BF16 on both lists, turns per resolved attempt, and a paired permutation test.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0_analysis import (
    clean,
    design_mismatches,
    flagged,
    flagged_share,
    p0a_factors,
    p0a_primary,
    p0a_secondary,
    p0b_report,
    p0c_breakage,
)

DESIGN = {"P0b": 16, "P0a": 3, "P0c": 4}
TOPUP_PROFILES = ("BF16", "Q4_K_M")
OUTCOME = {"pass": "B continues", "fail": "drop B (03 kill rule)",
           "inconclusive": "B continues; Q3_K_M is reported as the stress profile and P1 measures B directly"}


def gate_rows(rows, run_ids, lst):
    picked, dropped = clean([r for r in rows if r.get("run_id") in set(run_ids)])
    return [r for r in picked if lst is None or r["list"] == lst], dropped


def unflagged(rows):
    return [r for r in rows if not flagged(r)]


def no_timeouts(rows):
    return [r for r in rows if r["end_reason"] != "infra_timeout"]


def p0a_look(sel, args):
    """The primary test on heldout100 rows, with its flagged, factor and infra_timeout variants."""
    kw = {"n_boot": args.n_boot, "seed": args.seed}
    return {"primary": p0a_primary(sel, **kw),
            "primary_excluding_flagged": p0a_primary(unflagged(sel), **kw),
            "flagged_share": flagged_share(sel),
            "factors": p0a_factors(sel),
            "factors_excluding_flagged": p0a_factors(unflagged(sel)),
            "sensitivity_excluding_infra_timeout": p0a_primary(no_timeouts(sel), **kw)}


def p0a_retest(rows, manifest, first, args):
    """04's single re-test after an inconclusive first look. The second look is the last one.

    Refuses a top-up after a decisive first look, a top-up holding anything but BF16 and Q4_K_M on
    heldout100, and more than 3 top-up attempts per task and profile (a third look).
    """
    if "P0a_topup" not in manifest:
        if first == "inconclusive":
            return {"retest": None, "final_verdict": None,
                    "final": {"look": 1, "b_continues": None, "stress_profile": None,
                              "note": "re-test required: add 3 attempts per task at BF16 and Q4_K_M on heldout100 "
                                      "and list their run ids under P0a_topup"}}
        return {"retest": None, "final_verdict": first,
                "final": {"look": 1, "b_continues": first == "pass", "stress_profile": None, "note": OUTCOME[first]}}
    if first != "inconclusive":
        raise SystemExit(f"P0a_topup is listed but the first look is {first!r}; 04 re-tests only an inconclusive one")
    if set(manifest["P0a"]) & set(manifest["P0a_topup"]):
        raise SystemExit("a run id is in both P0a and P0a_topup")
    topup, _ = gate_rows(rows, manifest["P0a_topup"], None)
    off = sorted({(r["list"], r["profile"]) for r in topup} - {("heldout100", p) for p in TOPUP_PROFILES})
    if off:
        raise SystemExit(f"P0a_topup holds {off}; the top-up is BF16 and Q4_K_M on heldout100 only")
    over = [m for m in design_mismatches(topup, DESIGN["P0a"]) if m[2] > DESIGN["P0a"]]
    if over:
        raise SystemExit(f"more than {DESIGN['P0a']} top-up attempts for {over}: 04 allows one re-test, "
                         "not a third look")
    sel, dropped = gate_rows(rows, manifest["P0a"] + manifest["P0a_topup"], "heldout100")
    sel = [r for r in sel if r["profile"] in TOPUP_PROFILES]
    look = p0a_look(sel, args)
    v = look["primary"]["verdict"]
    return {"retest": {**look, "dropped": dropped, "design_mismatches": design_mismatches(sel, 2 * DESIGN["P0a"])},
            "final_verdict": v,
            "final": {"look": 2, "b_continues": v != "fail",
                      "stress_profile": "Q3_K_M" if v == "inconclusive" else None, "note": OUTCOME[v]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inrepo-size", type=int, required=True, help="set by the owner (04, Owner decisions)")
    ap.add_argument("--min-candidates", type=int, default=4, help="P0b fallback 1 sets 2 (04, P0b)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default="outputs/p0/gates.json")
    args = ap.parse_args()

    rows = [r for path in args.attempts for r in read_jsonl(path)]
    with open(args.manifest) as f:
        manifest = json.load(f)
    gates = {}

    if "P0b" in manifest:
        sel, dropped = gate_rows(rows, manifest["P0b"], None)
        kw = {"inrepo_size": args.inrepo_size, "min_candidates": args.min_candidates, "n_sub": args.n_boot,
              "seed": args.seed}
        sens = p0b_report(no_timeouts(sel), **kw)
        gates["P0b"] = {**p0b_report(sel, **kw),
                        "flagged_share": flagged_share(sel),
                        "sensitivity_excluding_infra_timeout": {
                            k: v for k, v in sens.items()
                            if k not in ("per_task", "heldout_per_task", "tasks_below_design_attempts")},
                        "dropped": dropped, "design_mismatches": design_mismatches(sel, DESIGN["P0b"])}
    if "P0a" in manifest:
        both, dropped = gate_rows(rows, manifest["P0a"], None)
        sel = [r for r in both if r["list"] == "heldout100"]
        gates["P0a"] = {**p0a_look(sel, args),
                        "secondary": p0a_secondary(both, n_boot=args.n_boot, seed=args.seed),
                        "dropped": dropped, "design_mismatches": design_mismatches(both, DESIGN["P0a"])}
        gates["P0a"].update(p0a_retest(rows, manifest, gates["P0a"]["primary"]["verdict"], args))
    if "P0c" in manifest:
        if "P0b" not in gates:
            raise SystemExit("P0c bins tasks by P0b's success counts; add P0b's run ids to the manifest")
        counts = {t: v["successes"] for t, v in gates["P0b"]["per_task"].items()}
        sel, dropped = gate_rows(rows, manifest["P0c"], "pool200")
        kw = {"n_boot": args.n_boot, "seed": args.seed}
        gates["P0c"] = {p: {**p0c_breakage(sel, counts, p, **kw),
                            "excluding_flagged": p0c_breakage(unflagged(sel), counts, p, **kw),
                            "sensitivity_excluding_infra_timeout": p0c_breakage(no_timeouts(sel), counts, p, **kw)}
                        for p in ("Q4_K_M", "Q3_K_M") if any(r["profile"] == p for r in sel)}
        gates["P0c"]["flagged_share"] = flagged_share(sel)
        gates["P0c"]["dropped"] = dropped
        gates["P0c"]["design_mismatches"] = design_mismatches(sel, DESIGN["P0c"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(gates, f, indent=1, allow_nan=False)
    for name, g in gates.items():
        if name == "P0a":
            verdicts = {"first_look": g["primary"]["verdict"], "final": g["final_verdict"]}
        else:
            verdicts = g.get("pass") or {k: v["verdict"] for k, v in g.items() if isinstance(v, dict) and "verdict" in v}
        print(name, json.dumps(verdicts))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
