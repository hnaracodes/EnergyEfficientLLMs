"""Pick the P1 headline arm by the docs/03 rule (Phases, P1 row) and write outputs/p1/headline.json.

    python scripts/p1_pick_headline.py outputs/attempts/<P1 run>.jsonl ... --models outputs/p1/models.json \
        --energy-model outputs/p0a/energy_model.json --p0a-verdict pass|fail|inconclusive \
        [--score-choice outputs/p0d/score_choice.json] [--did outputs/p0d/did.json] \
        [--not-separable ARM ...] [--config configs/p2_analysis.yaml] [--out outputs/p1/headline.json]

--models: {model value in the attempt records: {"arm": ..., "seed": ...}} (seed = training seed; arm "base"
for the untouched model). Rows must be inrepo_test tasks, one seed model per arm. --p0a-verdict is P0a's
final gate verdict (gates.json P0a.primary.verdict, after the re-test if one ran); "fail" drops stable and
qtrace (docs/03 kill rule). Otherwise P0d's score_choice.json, required then (and did.json when it chose a
score), decides whether they are eligible. --not-separable names eligible arms P0b's arm-overlap rule
flagged as not separable and P1 did not train (docs/04 P0b); they are dropped. The config's
guard_margin must be set, though the pick itself does not use it, so P1 cannot start before it is frozen.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl
from qtrace.p2_analysis import HEADLINE_ARMS, pick_headline


def read_json(path):
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--models", required=True)
    ap.add_argument("--energy-model", required=True)
    ap.add_argument("--p0a-verdict", required=True, choices=("pass", "fail", "inconclusive"))
    ap.add_argument("--score-choice")
    ap.add_argument("--did")
    ap.add_argument("--not-separable", nargs="+", default=[], choices=HEADLINE_ARMS)
    ap.add_argument("--config", default="configs/p2_analysis.yaml")
    ap.add_argument("--out", default="outputs/p1/headline.json")
    args = ap.parse_args(argv)

    try:
        out = pick_headline([r for p in args.attempts for r in read_jsonl(p)], read_json(args.models),
                            read_json(args.energy_model), read_json(args.score_choice), read_json(args.did),
                            load_config(args.config), args.p0a_verdict, args.not_separable)
    except ValueError as e:
        raise SystemExit(f"p1_pick_headline: {e}") from e
    out["inputs"] = {"attempts": args.attempts, "models": args.models, "energy_model": args.energy_model,
                     "p0a_verdict": args.p0a_verdict, "score_choice": args.score_choice, "did": args.did,
                     "not_separable": args.not_separable,
                     "config": args.config}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    en = out["energy"]
    print(f"JPR: {en['quantity']} ({en['rule']}){', provisional' if en['provisional'] else ''}")
    for r in out["arms"]:
        lo, hi = r["bf16_rr_diff_ci95"]
        print(f"{r['arm']:14} JPR_Q4 {r['jpr_q4']}  BF16 RR diff vs random {r['bf16_rr_diff_vs_random']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  {'eligible' if r['eligible'] else 'not eligible'}")
    for arm, why in out["dropped"].items():
        print(f"{arm:14} dropped: {why}")
    print(f"headline: {out['headline'] or 'none (' + out['reason'] + ')'}"
          + (f", tied with {out['tied_with']}" if out["tied_with"] else ""), "->", args.out)


if __name__ == "__main__":
    main()
