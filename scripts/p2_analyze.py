"""Run the P2 analysis (docs/04, "Analysis plan for P1 and P2") and write outputs/p2/analysis.json and analysis.md.

    python scripts/p2_analyze.py outputs/attempts/<P2 run>.jsonl ... --models outputs/p2/models.json \
        --energy-model outputs/p0a/energy_model.json --headline outputs/p1/headline.json \
        [--estimator-check outputs/attempts/<concurrency-1 run>.jsonl ...] \
        [--p1-attempts outputs/attempts/<P1 run>.jsonl ... --p1-models outputs/p1/models.json] \
        [--config configs/p2_analysis.yaml] [--out-dir outputs/p2]

The attempt files hold the held-out-repo runs, one attempt per task per seed model per profile, plus H4's
grammar-constrained runs (grammar: true), which only the H4 family reads.
--estimator-check holds the separate concurrency-1 Q4_K_M runs of the headline arm, random and base (one seed
model each) with per-turn counter reads, on estimator_check30 or another held-out list.
--p1-attempts and --p1-models are P1's inrepo_test runs and manifest, for the exploratory within-repo split. The config's guard_margin, host_cpu_tdp_w and host_vcpus must be set.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl
from qtrace.p2_analysis import analyze


def read_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, spec=".3f"):
    return "n/a" if x is None else x if isinstance(x, str) else format(x, spec)


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def contrast_row(name, c):
    g, ub = c["guard"], c["upper_bound"]
    supported = c.get("supported", c["wins"] and c["holds_under_upper_bound"])  # the primary is an energy claim
    return row(name, f"{c['arm']} vs {c['comparator']}", c["profile"], c["n_tasks"], fmt(c["jpr"]["arm"], ".0f"),
               fmt(c["jpr"]["comparator"], ".0f"), fmt(c["ratio"]), fmt(c["gpu"]["p_one_sided"], ".4f"),
               fmt(c.get("holm_p"), ".4f"), f"{g['rr_diff']:+.3f} ({g['lower95_one_sided']:+.3f})",
               "pass" if g["pass"] else "FAIL",
               f"{fmt(ub['ratio'])} / {fmt(c.get('holm_p_upper_bound', ub['p_one_sided']), '.4f')}", c["wins"],
               supported)


def b_row(arm, b):
    d = b.get("bf16_rr_diff_vs_random")
    diff = "-" if d is None else f"{d['value']:+.3f} [{fmt(d['ci95'][0], '+.3f')}, {fmt(d['ci95'][1], '+.3f')}]"
    rr, jpr = b["resolve_rate"], b["jpr"]
    return row(arm, f"{rr['BF16']:.3f}", f"{rr['Q4_K_M']:.3f}", f"{b['quant_gap']:+.3f}", fmt(jpr["BF16"], ".0f"),
               fmt(jpr["Q4_K_M"], ".0f"), fmt(b["jpr_ratio"]["value"]), fmt(b["processed_tokens_ratio"]["value"]), diff)


def markdown(out):
    p, en = out["primary"], out["energy"]
    ts, sign, sens = p["two_stage"], p["sign_holds"], p["sensitivity_without_infra_timeout"]
    idle = ", ".join(f"{k} {fmt(v, '.1f')} W" for k, v in en["idle_w"].items())
    gen, sub = p["alongside"]["generation_window"], p["alongside"]["idle_subtracted"]
    lines = ["# P2 analysis", "",
             (f"Headline arm: {out['headline']}. GPU joules are the P0a whole-attempt estimate ({en['rule']}; "
              f"idle_w {idle}){', provisional' if en['provisional'] else ''}. UB adds container CPU-seconds x TDP "
              "per vCPU."), "",
             (f"Primary alongside (point estimates): generation windows only, ratio {fmt(gen['ratio'])}; "
              f"idle-subtracted, ratio {fmt(sub and sub['ratio'])}."), "",
             row("test", "contrast", "profile", "tasks", "JPR arm", "JPR comp", "ratio", "p", "Holm p",
                 "RR diff vs random (lower95)", "guard", "UB ratio / p", "wins", "supported"),
             "|" + "---|" * 14, contrast_row("primary", p)]
    if out["token_matched"]:
        lines.append(contrast_row("primary, token-matched", out["token_matched"]))
    m = out["measured_subset"]
    if m:  # docs/04: when the estimate does not stand, only this row is claimed
        lines.append(contrast_row(f"primary, measured subset ({m['scope']})", m))
    for fam, f in out["families"].items():
        lines += [contrast_row(fam, c) for c in f["contrasts"]]
    h4 = out["h4"]
    if h4:
        lines += [contrast_row("H4 (grammar rows)", c) for c in h4["contrasts"]]
    lines += ["", (f"Primary, two-stage (seed, then task) bootstrap: p {ts['p_one_sided']:.4f}, upper95 T "
                   f"{fmt(ts['upper95_one_sided'])}. Sign holds in {sign['n_same_sign']} of {sign['n_repos']} repos "
                   f"(needs {sign['min_repos']}). Without infra_timeout attempts: ratio {fmt(sens['ratio'])}."), "",
              row("arm", "RR BF16", "RR Q4", "gap", "JPR BF16", "JPR Q4", "JPR Q4/BF16", "tokens/resolved Q4/BF16",
                  "BF16 RR diff vs random [95%]"),
              "|" + "---|" * 9]
    lines += [b_row(arm, b) for arm, b in out["b_metrics"].items()]
    for c in out["tokens_ratio_contrasts"]:
        lo, hi = c["ci95"]
        lines += ["", (f"{c['family']} secondary, Q4_K_M/BF16 processed tokens per resolved task: {c['arm']} "
                       f"{fmt(c['arm_ratio'])} vs {c['comparator']} {fmt(c['comparator_ratio'])}, ratio "
                       f"{fmt(c['ratio_of_ratios'])}, 95% [{fmt(lo)}, {fmt(hi)}].")]
    if h4:
        lines += ["", f"H4: {h4['reading']}"]
        lines += [(f"- {c['arm']} vs {c['comparator']} at {c['profile']}: T {fmt(c['T'])} with grammar, "
                   f"{fmt(c['without_grammar']['T'])} without; keeps sign: {c['keeps_sign']}; effect ratio "
                   f"{fmt(c['effect_ratio'])}; survives: {c['supported']}.") for c in h4["contrasts"]]
    est, fr = out["estimator_check"], out["turn_cap_frontier"]
    if est:
        pairs = "; ".join(f"{k} {est['pairs'][k]['ratio']:.3f}, 95% [{est['pairs'][k]['ratio_ci95'][0]:.3f}, "
                          f"{est['pairs'][k]['ratio_ci95'][1]:.3f}]" for k in est["decides"])
        lines += ["", f"Estimator check ({est['scope']}), measured/estimated ratio: {pairs}; stands: {est['stands']}."]
    if fr:
        top, dom = fr["headline_uncapped"], fr["base_points_dominating"]
        at = "no base cap reaches" if not dom else f"base caps {[p['cap'] for p in dom]} reach"
        lines += ["", (f"Turn-cap frontier: headline uncapped RR {top['resolve_rate']:.3f} at JPR "
                       f"{fmt(top['jpr'], '.0f')}; {at} that RR at that JPR or less; pass: {fr['pass']}.")]
    bt = out["batched_regime"]
    if bt:
        bp = bt["primary"]
        lines += ["", (f"Batched regime (P0a batched and concurrency-1 picks agree on "
                       f"{fmt(en['batched_share_same_pick'])} of tasks, below 0.85): primary on {bt['estimate']}, "
                       f"ratio {fmt(bp['ratio'])}, p {bp['gpu']['p_one_sided']:.4f}, guard "
                       f"{'pass' if bp['guard']['pass'] else 'FAIL'}. Run-level measured GPU totals:"), "",
                  row("arm", "profile", "blocks", "attempts", "joules", "resolved", "J per resolved"),
                  "|" + "---|" * 7]
        lines += [row(r["arm"], r["profile"], r["n_blocks"], r["n_attempts"], fmt(r["joules"], ".0f"),
                      r["n_resolved"], fmt(r["jpr"], ".0f")) for r in bt["run_level"]]
    w = out["within_repo"]
    if w:
        c = w["headline_vs_random"]
        lines += ["", (f"Within-repo split (inrepo_test, P1, one seed, exploratory): {c['arm']} vs random at "
                       f"{c['profile']} on {c['n_tasks']} tasks, JPR ratio {fmt(c['ratio'])}, p "
                       f"{c['gpu']['p_one_sided']:.4f}, RR diff vs random {c['guard']['rr_diff']:+.3f}.")]
    lines += ["", f"Claim A on estimates: {out['claim_A']['claimed_on_estimates']} ({out['claim_A']})", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--models", required=True)
    ap.add_argument("--energy-model", required=True)
    ap.add_argument("--headline", required=True, help="outputs/p1/headline.json from scripts/p1_pick_headline.py")
    ap.add_argument("--estimator-check", nargs="+")
    ap.add_argument("--p1-attempts", nargs="+")
    ap.add_argument("--p1-models")
    ap.add_argument("--config", default="configs/p2_analysis.yaml")
    ap.add_argument("--out-dir", default="outputs/p2")
    args = ap.parse_args(argv)

    est = [r for p in args.estimator_check for r in read_jsonl(p)] if args.estimator_check else None
    p1 = [r for p in args.p1_attempts for r in read_jsonl(p)] if args.p1_attempts else None
    try:
        out = analyze([r for p in args.attempts for r in read_jsonl(p)], read_json(args.models),
                      read_json(args.energy_model), read_json(args.headline).get("headline"), load_config(args.config),
                      estimator_rows=est, p1_records=p1, p1_manifest=args.p1_models and read_json(args.p1_models))
    except ValueError as e:
        raise SystemExit(f"p2_analyze: {e}") from e
    out["inputs"] = {"attempts": args.attempts, "models": args.models, "energy_model": args.energy_model,
                     "headline": args.headline, "estimator_check": args.estimator_check,
                     "p1_attempts": args.p1_attempts, "p1_models": args.p1_models, "config": args.config}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    text = markdown(out)
    with open(os.path.join(args.out_dir, "analysis.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
