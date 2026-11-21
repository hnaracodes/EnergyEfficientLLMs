"""Tests for the P1 headline pick and the P2 analysis on synthetic attempt records with planted answers.

The energy model charges 1 J per output token, so an attempt's estimated joules are its output tokens.
Run:  python -m pytest tests/test_p2_analysis.py -q
"""

import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace.p2_analysis import (
    NOT_SEPARABLE_WHY,
    analyze,
    b_metrics,
    contrast,
    estimator_check,
    holm,
    p0d_dropped,
    pick_headline,
    turn_cap_frontier,
)
from qtrace.p2_data import BASE_ARMS, MALFORMED, arrays, check_config, load, strat_weights

Q4, BF = "Q4_K_M", "BF16"
HELD = ("scrapy", "pyramid", "datalad", "coveragepy")
POOL = ("pandas", "numpy", "pillow", "orange3", "aiohttp", "tornado")
# idle_w 0 keeps every known answer in output tokens; tests of the tool-window charge set their own.
EM = {"ranking": {"profile": Q4, "target": "raw", "quantity": "whole_attempt_raw_joules", "idle_w": 0.0,
                  "provisional": True},
      "profiles": {p: {"raw": {"coef": {"output": 1.0}}, "idle_w": 0.0, "idle_subtracted": {"coef": {"output": 0.5}}}
                   for p in (Q4, BF, "Q3_K_M")},
      "batched": {"agreement_with_c1": {"share_same_pick": 0.9}, "below_85pct": False}}
CFG = {"guard_margin": 0.05, "infra_error_cap": 0.05, "n_boot": 400, "seed": 1, "alpha": 0.05,
       "host_cpu_tdp_w": 280.0, "host_vcpus": 128,
       "families": {"H1a": [["fewest_turns", "random"]], "H1b": [["fewest_joules", "random"]],
                    "H3": [["stable", "low_entropy"]]},
       "upper_bound_required": ["H1b"]}
CHOSEN = {"chosen": "kl_decision", "decided": True}


def rec(aid, task, model, profile, resolved, joules, seed=0, lst="heldout691", turns=2, end=None, cpu=10.0,
        conc=16, measured=None, cached=0.0, malformed=None, tool_s=1.0, tool_j=None, flagged=False, grammar=False):
    """One attempt; joules (= output tokens), cached input tokens (processed but free under EM) and the measured
    gen_energy_j and tool_energy_j are spread evenly over its turns, each with tool_s tool seconds."""
    per_turn = [{"turn": i, "output_tokens": joules / turns, "input_uncached": 0, "input_cached": cached / turns,
                 "gen_s": 1.0, "tool_s": tool_s, "gen_energy_j": None if measured is None else measured / turns,
                 "tool_energy_j": None if tool_j is None else tool_j / turns} for i in range(turns)]
    return {"attempt_id": aid, "task_id": task, "list": lst, "repo": task.split("/")[0], "profile": profile,
            "model": model, "seed": seed, "resolved": resolved, "grammar": grammar,
            "end_reason": end or ("submitted" if resolved else "turn_limit"), "turns": turns,
            "tokens": {"output": joules, "input_uncached": 0, "input_cached": cached}, "per_turn": per_turn,
            "tool_seconds": tool_s * turns, "grading": {"test_files_touched": flagged},
            "container_cpu_seconds": cpu, "concurrency": conc, "energy": {"attributable": conc == 1},
            "malformed": {k: 0 for k in MALFORMED} | (malformed or {})}


def build(arms, fn, profiles=(Q4, BF), seeds=(0,), n=5, repos=HELD, lst="heldout691", **kw):
    """Every arm x seed x profile x task. fn(arm, seed, profile, repo, i) -> (resolved, joules) or a dict of rec kwargs.

    Trained arms get one model per seed; a base arm is one model with seed null, its replicates the sampling seeds."""
    recs, manifest = [], {}
    for arm in arms:
        for s in seeds:
            model = arm if arm in BASE_ARMS else f"{arm}_seed{s}"
            manifest[model] = {"arm": arm, "seed": None if arm in BASE_ARMS else s}
            for prof in profiles:
                for repo in repos:
                    for i in range(n):
                        got = fn(arm, s, prof, repo, i)
                        args = got if isinstance(got, dict) else {"resolved": got[0], "joules": got[1]}
                        task = f"{repo}/t{i}"
                        recs.append(rec(f"{model}-s{s}-{prof}-{task}", task, model, prof, seed=s, lst=lst,
                                        **{**kw, **args}))
    return recs, manifest


def base_joules(repo, i):
    return 1000.0 + 100 * i + 37 * HELD.index(repo) if repo in HELD else 1000.0 + 50 * i


def scaled(factor_by_arm):
    def fn(arm, s, prof, repo, i):
        return (i + s) % 2 == 0 or i == 1, factor_by_arm.get(arm, 1.0) * base_joules(repo, i)
    return fn


def test_known_answer_and_report_shape():
    recs, manifest = build(["fewest_turns", "random"], scaled({"fewest_turns": 0.8}), seeds=(0, 1, 2))
    out = analyze(recs, manifest, EM, "fewest_turns", CFG)
    p = out["primary"]
    assert p["T"] == pytest.approx(math.log(0.8), abs=1e-12)
    assert p["gpu"]["p_one_sided"] < 0.01 and p["guard"]["pass"] and p["wins"]
    assert p["guard"]["rr_diff"] == 0 and p["guard"]["lower95_one_sided"] == 0
    assert p["sign_holds"]["n_same_sign"] == 4 and p["sign_holds"]["holds"]
    assert all(r["T"] == pytest.approx(math.log(0.8)) for r in p["per_repo"].values())
    assert [s["seeds"] for s in p["per_seed"]] == [[0, 0], [1, 1], [2, 2]]
    assert p["two_stage"]["n_draws_undefined"] == 0
    # Upper bound by hand: every attempt adds 10 CPU-seconds x 280 W / 128 vCPUs to both arms.
    E = sum(base_joules(r, i) for r in HELD for i in range(5))
    R = sum(((i + s) % 2 == 0 or i == 1) for r in HELD for i in range(5) for s in range(3)) / 3
    cpu = 20 * 10.0 * 280.0 / 128
    assert p["upper_bound"]["jpr"]["arm"] == pytest.approx((0.8 * E + cpu) / R)
    assert p["upper_bound"]["T"] == pytest.approx(math.log((0.8 * E + cpu) / (E + cpu)))
    assert p["holds_under_upper_bound"]
    h1a = out["families"]["H1a"]
    assert [c["profile"] for c in h1a["contrasts"]] == [BF, Q4] and h1a["contrasts"][1]["supported"]
    assert out["families"]["H3"]["n_tested"] == 0 and len(out["families"]["H3"]["untested"]) == 2
    assert out["claim_A"]["frontier_pass"] is None and not out["claim_A"]["claimed_on_estimates"]
    json.dumps(out, allow_nan=False)


def test_guard_pass_and_fail():
    def fn(arm, s, prof, repo, i):
        return (arm == "random" or i < 4), (0.5 if arm == "fewest_turns" else 1.0) * base_joules(repo, i)
    recs, manifest = build(["fewest_turns", "random"], fn, n=5)
    out = analyze(recs, manifest, EM, "fewest_turns", CFG)["primary"]
    assert out["gpu"]["p_one_sided"] < 0.05  # wins on joules ...
    assert out["guard"]["rr_diff"] == pytest.approx(-0.2) and not out["guard"]["pass"]  # ... but fails the guard
    assert not out["wins"] and not out["two_stage"]["guard_pass"]
    wide = analyze(recs, manifest, EM, "fewest_turns", {**CFG, "guard_margin": 0.5})["primary"]
    assert wide["guard"]["pass"] and wide["wins"] and wide["two_stage"]["guard_pass"]


@pytest.mark.parametrize("cpu_j, holds", [(10.0, True), (2000.0, False)])
def test_energy_win_needs_the_upper_bound(cpu_j, holds):
    # fewest_joules spends 0.8x random's GPU joules. At 2000 container CPU-seconds (4375 J at 280 W / 128 vCPUs)
    # against random's 10, it loses under the upper bound, so H1b and the primary energy claim do not stand.
    def fn(arm, s, prof, repo, i):
        fj = arm == "fewest_joules"
        return {"resolved": (i + s) % 2 == 0 or i == 1, "joules": (0.8 if fj else 1.0) * base_joules(repo, i),
                "cpu": cpu_j if fj else 10.0}
    recs, manifest = build(["fewest_joules", "random"], fn, seeds=(0, 1))
    out = analyze(recs, manifest, EM, "fewest_joules", CFG)
    p, h1b = out["primary"], out["families"]["H1b"]["contrasts"]
    assert p["wins"] and p["holds_under_upper_bound"] is holds and out["claim_A"]["holds_under_upper_bound"] is holds
    assert all(c["wins"] and c["holds_under_upper_bound"] is holds and c["supported"] is holds for c in h1b)
    assert not holds or [c["profile"] for c in h1b if c["supported"]] == [BF, Q4]


def test_holm_decides_family_wins():
    # Two H1a contrasts (BF16 and Q4_K_M) with noisy effects. alpha is set between the smaller raw p and its Holm
    # p (2x), so the contrast wins on its raw p and must not win on its Holm p.
    def fn(arm, s, prof, repo, i):
        f = (0.7, 1.25, 0.8, 1.15, 0.95)[i] if arm == "fewest_turns" else 1.0
        return True, f * base_joules(repo, (i + (prof == BF)) % 5)
    recs, manifest = build(["fewest_turns", "random"], fn)
    cs = analyze(recs, manifest, EM, "fewest_turns", CFG)["families"]["H1a"]["contrasts"]
    raw = [c["gpu"]["p_one_sided"] for c in cs]
    assert 0 < min(raw) and 2 * min(raw) < 1 and all(c["guard"]["pass"] for c in cs)
    alpha = 1.5 * min(raw)
    cs = analyze(recs, manifest, EM, "fewest_turns", {**CFG, "alpha": alpha})["families"]["H1a"]["contrasts"]
    low = min(cs, key=lambda c: c["gpu"]["p_one_sided"])
    assert low["gpu"]["p_one_sided"] < alpha <= low["holm_p"] and not low["wins"]


def test_per_repo_test_uses_its_own_draws():
    # scrapy keeps 2 tasks, the other repos 5, and every task is resolved by both arms at 0.8x the joules, so T is
    # log(0.8) in every draw of the scrapy block; draws taken from another repo's columns can weigh nothing.
    recs, manifest = build(["fewest_turns", "random"], scaled({"fewest_turns": 0.8}), profiles=(Q4,))
    recs = [{**r, "resolved": True, "end_reason": "submitted"} for r in recs
            if not (r["repo"] == "scrapy" and r["task_id"][-1] in "234")]
    p = analyze(recs, manifest, EM, "fewest_turns", {**CFG, "families": {}, "upper_bound_required": []})["primary"]
    assert p["per_repo"]["scrapy"]["n_tasks"] == 2
    assert all(r["n_draws_undefined"] == 0 and r["p_one_sided"] == 0 for r in p["per_repo"].values())


def test_holm_hand_example():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_stratification_keeps_repo_counts():
    repos = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    W = strat_weights(repos, 500, np.random.default_rng(0))
    for r, k in (("a", 3), ("b", 5), ("c", 2)):
        assert (W[:, repos == r].sum(1) == k).all()
    assert W[:, repos == "b"].std() > 0  # tasks really are resampled


def test_seed_averaging_and_infinite_jpr():
    def fn(arm, s, prof, repo, i):
        if arm == "random":
            return True, 200.0
        return (True, 100.0) if s == 0 else (False, 300.0)
    recs, manifest = build(["fewest_turns", "random"], fn, seeds=(0, 1), n=2)
    p = analyze(recs, manifest, EM, "fewest_turns", CFG)["primary"]
    assert p["jpr"]["arm"] == pytest.approx(400.0)  # (100 + 300) / 2 J per task over 0.5 resolves per task
    assert p["T"] == pytest.approx(math.log(2))
    assert p["per_seed"][0]["T"] == pytest.approx(math.log(0.5)) and p["per_seed"][1]["T"] == "inf"
    two = p["two_stage"]
    assert 0.5 < two["p_one_sided"] < 1 and two["upper95_one_sided"] == "inf"  # draws of seed 1 twice: JPR inf


def test_infra_error_drop_cap_rerun_and_timeouts():
    fn = scaled({"fewest_turns": 0.8})
    recs, manifest = build(["fewest_turns", "random"], fn)  # 20 tasks
    err = next(r for r in recs if r["model"] == "fewest_turns_seed0" and r["profile"] == Q4)
    one = [{**r, "end_reason": "error"} if r is err else r for r in recs]
    p = analyze(one, manifest, EM, "fewest_turns", CFG)["primary"]
    assert p["n_tasks"] == 19 and p["n_tasks_dropped_infra_error"] == 1  # 1/20 is at the cap
    two_errs = [{**r, "end_reason": "error"} if r["model"] == "random_seed0" and r["profile"] == Q4
                and r["task_id"] in ("scrapy/t0", "pyramid/t0") else r for r in recs]
    with pytest.raises(ValueError, match="infra_error_cap"):
        analyze(two_errs, manifest, EM, "fewest_turns", CFG)
    rerun = one + [{**err, "attempt_id": "rerun-1"}]
    data = load(rerun, manifest, EM, "p2")
    assert data["n_reruns_used"] == 1
    assert analyze(rerun, manifest, EM, "fewest_turns", CFG)["primary"]["n_tasks"] == 20
    timeout = [{**r, "end_reason": "infra_timeout", "resolved": False} if r is err else r for r in recs]
    p = analyze(timeout, manifest, EM, "fewest_turns", CFG)["primary"]
    assert p["n_tasks"] == 20  # charged as a failure
    assert p["sensitivity_without_infra_timeout"]["n_tasks"] == 19


def test_upper_bound_arithmetic():
    recs, manifest = build(["random"], lambda *a: {"resolved": True, "joules": 500.0, "cpu": 36.0}, n=1)
    arr = arrays(load(recs, manifest, EM, "p2"), [("random", Q4)], 0.05, cpu_w=280.0 / 128)
    assert arr["groups"]["random", Q4]["U"] == pytest.approx(np.full((1, 4), 500 + 36 * 280 / 128))  # 578.75 J


def est_rows(factor_by_arm, n=8, unmeasured=None):
    def fn(arm, s, prof, repo, i):
        noise = 1 + 0.01 * ((i * 7 + HELD.index(repo)) % 5)
        j = base_joules(repo, i)
        return {"resolved": True, "joules": j, "conc": 1, "measured": j * factor_by_arm[arm] * noise}
    recs, manifest = build(["fewest_turns", "random", "base"], fn, profiles=(Q4,), n=n)
    if unmeasured:
        recs = [copy.deepcopy(r) for r in recs]
        hit = next(r for r in recs if r["model"] == unmeasured)
        hit["per_turn"][0]["gen_energy_j"] = None
    return recs, manifest


def test_estimator_check_pass_and_fail():
    recs, manifest = est_rows({"fewest_turns": 1.0, "random": 1.04, "base": 0.96}, unmeasured="base")
    out = estimator_check(load(recs, manifest, EM, "p2"), "fewest_turns", CFG)
    assert out["stands"] and out["pairs"]["fewest_turns_vs_random"]["ratio"] == pytest.approx(1 / 1.04)
    assert out["excluded"]["base"]["unmeasured_turn"] == 1 and out["pairs"]["fewest_turns_vs_base"]["n_tasks"] == 31
    assert not out["pairs"]["random_vs_base"]["within_band"]  # 1.04 / 0.96, reported only
    # The frontier compares estimated headline and base joules, so a headline-vs-base bias also stops it.
    recs, manifest = est_rows({"fewest_turns": 1.1, "random": 1.1, "base": 1.0})
    out = estimator_check(load(recs, manifest, EM, "p2"), "fewest_turns", CFG)
    assert out["pairs"]["fewest_turns_vs_random"]["within_band"] and not out["stands"]
    recs, manifest = est_rows({"fewest_turns": 1.2, "random": 1.0, "base": 1.0})
    out = estimator_check(load(recs, manifest, EM, "p2"), "fewest_turns", CFG)
    assert not out["stands"] and out["pairs"]["fewest_turns_vs_random"]["ratio_ci95"][0] > 1.05


def test_estimator_check_one_task_per_repo_does_not_collapse():
    # One paired task per repo, measured/estimated 0.74 to 1.35 for the headline and exact for the others: a
    # bootstrap within repo has nothing to resample and would give a zero-width interval around 1.0.
    off = {"scrapy": 0.74, "pyramid": 1.35, "datalad": 0.80, "coveragepy": 1.25}
    def fn(arm, s, prof, repo, i):
        return {"resolved": True, "joules": 1000.0, "conc": 1, "measured": 1000.0 * (off[repo] if arm ==
                                                                                      "fewest_turns" else 1.0)}
    recs, manifest = build(["fewest_turns", "random", "base"], fn, profiles=(Q4,), n=1)
    out = estimator_check(load(recs, manifest, EM, "p2"), "fewest_turns", CFG)
    lo, hi = out["pairs"]["fewest_turns_vs_random"]["ratio_ci95"]
    assert lo < 0.95 and hi > 1.05 and not out["stands"]


P1_JOULES = {"random": 1000.0, "fewest_turns": 800.0, "fewest_joules": 800.0, "stable": 700.0, "qtrace": 500.0,
             "low_entropy": 100.0}


def p1(joules=P1_JOULES, arms=None):
    def fn(arm, s, prof, repo, i):
        return not (arm == "qtrace" and prof == BF), joules[arm] + 10 * i  # qtrace never resolves at BF16
    return build(arms or list(joules), fn, repos=POOL, n=4, lst="inrepo_test")


def test_headline_pick_tie_ineligible_and_p0d_drop():
    recs, manifest = p1()
    out = pick_headline(recs, manifest, EM, CHOSEN, {"score": "kl_decision", "decision": "pass"}, CFG, "pass")
    rows = {r["arm"]: r for r in out["arms"]}
    assert out["headline"] == "stable" and rows["qtrace"]["clearly_below_random"] and not rows["qtrace"]["eligible"]
    assert "low_entropy" not in rows  # controls are never eligible, however cheap
    for decision in ("inconclusive", "fail"):
        out = pick_headline(recs, manifest, EM, CHOSEN, {"score": "kl_decision", "decision": decision}, CFG, "pass")
        assert out["headline"] == "fewest_turns" and out["tied_with"] == ["fewest_joules"]
        assert set(out["dropped"]) == {"stable", "qtrace"}
    out = pick_headline(recs, manifest, EM, {"chosen": None, "decided": True}, None, CFG, "inconclusive")
    assert out["headline"] == "fewest_turns" and [r["arm"] for r in out["arms"]] == ["fewest_turns", "fewest_joules"]
    with pytest.raises(ValueError, match="undecided"):
        pick_headline(recs, manifest, EM, {"chosen": None, "decided": False}, None, CFG, "pass")
    with pytest.raises(ValueError, match="did.json"):
        p0d_dropped(CHOSEN, None)


def test_p0a_fail_drops_stability_arms():
    recs, manifest = p1()  # stable is the cheapest eligible arm when P0a and P0d pass
    out = pick_headline(recs, manifest, EM, None, None, CFG, "fail")
    assert out["headline"] == "fewest_turns" and set(out["dropped"]) == {"stable", "qtrace"}
    recs, manifest = p1(arms=["random", "fewest_turns", "fewest_joules"])  # stability arms never trained
    assert pick_headline(recs, manifest, EM, None, None, CFG, "fail")["headline"] == "fewest_turns"
    with pytest.raises(ValueError, match="p0a_verdict"):
        pick_headline(recs, manifest, EM, None, None, CFG, None)


def test_not_separable_arm_is_dropped_not_required():
    passed = {"score": "kl_decision", "decision": "pass"}
    recs, manifest = p1(arms=["random", "fewest_turns", "stable", "qtrace"])  # fewest_joules merged, not trained
    with pytest.raises(ValueError, match="no attempts for arm 'fewest_joules'"):
        pick_headline(recs, manifest, EM, CHOSEN, passed, CFG, "pass")
    out = pick_headline(recs, manifest, EM, CHOSEN, passed, CFG, "pass", not_separable=["fewest_joules"])
    assert out["headline"] == "stable" and out["dropped"] == {"fewest_joules": NOT_SEPARABLE_WHY}
    with pytest.raises(ValueError, match="untrained arms"):
        pick_headline(recs, manifest, EM, CHOSEN, passed, CFG, "pass", not_separable=["fewest_turns"])


def test_tripwire_covers_every_p1_group():
    recs, manifest = p1()
    recs = [{**r, "grading": {"test_files_touched": True}} if r["model"] in ("random_seed0", "low_entropy_seed0")
            and r["profile"] == Q4 else r for r in recs]
    out = pick_headline(recs, manifest, EM, CHOSEN, {"score": "kl_decision", "decision": "pass"}, CFG, "pass")
    assert sorted(out["tripwire"]) == sorted(f"{a}/{p}" for a in P1_JOULES for p in (BF, Q4))
    assert out["tripwire"]["random/Q4_K_M"]["flagged_share"] == out["tripwire"]["low_entropy/Q4_K_M"][
        "flagged_share"] == 1


def test_tokens_ratio_contrast():
    # random processes 1.5x the tokens at Q4_K_M, fewest_turns the same at both: H1a's secondary ratio is 1 / 1.5.
    def fn(arm, s, prof, repo, i):
        return True, base_joules(repo, i) * (1.5 if arm == "random" and prof == Q4 else 1.0)
    recs, manifest = build(["fewest_turns", "random"], fn, seeds=(0, 1))
    (c,) = analyze(recs, manifest, EM, "fewest_turns", CFG)["tokens_ratio_contrasts"]
    assert (c["family"], c["arm"], c["comparator"], c["n_tasks"]) == ("H1a", "fewest_turns", "random", 20)
    assert c["arm_ratio"] == pytest.approx(1.0) and c["comparator_ratio"] == pytest.approx(1.5)
    assert c["ratio_of_ratios"] == pytest.approx(1 / 1.5) and c["ci95"] == pytest.approx([1 / 1.5, 1 / 1.5])


def test_headline_pick_none_eligible():
    joules = {"random": 1000.0, "fewest_turns": 800.0, "fewest_joules": 800.0}
    recs, manifest = build(list(joules), lambda arm, s, prof, repo, i: (arm == "random" or prof == Q4, joules[arm]),
                           repos=POOL, n=4, lst="inrepo_test")
    out = pick_headline(recs, manifest, EM, {"chosen": None}, None, CFG, "pass")
    assert out["headline"] is None and out["reason"] == "no eligible arm"


def frontier(base_turns, base_resolved, head=lambda i: {"resolved": True, "joules": 300.0, "turns": 3}):
    def fn(arm, s, prof, repo, i):
        if arm == "fewest_turns":
            return head(i)
        t = base_turns(i)
        return {"resolved": base_resolved(i), "joules": 100.0 * t, "turns": t}
    recs, manifest = build(["fewest_turns", "base"], fn, profiles=(Q4,), n=4)
    return turn_cap_frontier(load(recs, manifest, EM, "p2"), "fewest_turns", EM, CFG)


def test_turn_cap_frontier_pass_and_fail():
    out = frontier(lambda i: 2, lambda i: True)  # base resolves everything in 2 turns at 200 J: dominates
    assert out["base_points_dominating"][0] == {"cap": 2, "resolve_rate": 1.0, "mean_joules": 200.0, "jpr": 200.0}
    assert not out["pass"]
    out = frontier(lambda i: 10, lambda i: True)  # base needs 10 turns, 1000 J, to match
    assert out["pass"] and out["base_points_dominating"] == []
    assert out["points"]["base"][4] == {"cap": 5, "resolve_rate": 0.0, "mean_joules": 500.0, "jpr": "inf"}
    assert out["headline_uncapped"] == {"cap": 10, "resolve_rate": 1.0, "mean_joules": 300.0, "jpr": 300.0}
    out = frontier(lambda i: 1 + i, lambda i: i < 2)  # base never reaches the headline's rate
    assert out["pass"] and out["base_points_dominating"] == []
    # Headline: 2 of 4 at 260 J per attempt (JPR 520). Base capped at 3 turns: 3 of 4 at 275 J per attempt
    # (JPR 367), so it spends more per attempt but beats the headline on both axes of goal A.
    out = frontier(lambda i: (2, 3, 3, 10)[i], lambda i: i < 3,
                   head=lambda i: {"resolved": i < 2, "joules": 260.0, "turns": 2})
    assert out["base_points_dominating"][0]["cap"] == 3 and not out["pass"]
    assert out["points"]["base"][-1]["jpr"] == pytest.approx(600.0)  # uncapped base alone would not catch it


def test_b_metrics_and_breakdown_sum():
    def fn(arm, s, prof, repo, i):
        # Q4_K_M: 1.3x the processed tokens at 0.6x the joules per token, and 3 of 5 resolved against 4 of 5.
        bj, q4 = base_joules(repo, i), prof == Q4
        return {"resolved": i < (3 if q4 else 4), "joules": 0.78 * bj if q4 else bj, "cached": 0.52 * bj if q4 else 0.0}
    recs, manifest = build(["random", "fewest_turns"], fn)
    b = b_metrics(load(recs, manifest, EM, "p2"), "fewest_turns", check_config(CFG, True))
    assert b["quant_gap"] == pytest.approx(0.2) and b["bf16_rr_diff_vs_random"]["value"] == 0
    bd = b["breakdown"]
    assert bd["sum_of_terms"] == pytest.approx(bd["log_total"])
    assert bd["joules_per_token"] == pytest.approx(math.log(0.6))
    assert bd["tokens_per_attempt"] == pytest.approx(math.log(1.3))
    assert bd["attempts_per_resolved"] == pytest.approx(math.log(4 / 3))
    assert b["processed_tokens_ratio"]["value"] == pytest.approx(1.3 * 4 / 3)
    assert b["jpr_ratio"]["value"] == pytest.approx(math.exp(bd["log_total"]))
    lo, hi = b["jpr_ratio"]["ci95"]
    assert lo <= b["jpr_ratio"]["value"] <= hi


def drop(recs, k=0):
    return recs[:k] + recs[k + 1:]


def edit(recs, k, **kw):
    return [{**r, **kw} if i == k else r for i, r in enumerate(recs)]


@pytest.mark.parametrize("mutate, phase, match", [
    (lambda r: edit(r, 0, tokens={"output": 1}), "p2", "missing field tokens.input_uncached"),
    (lambda r: r + [r[0]], "p2", "duplicate attempt_id"),
    (lambda r: drop(r, 3), "p2", "incomplete design"),
    (lambda r: r + [{**r[0], "attempt_id": "again"}], "p2", "1 per task per seed model"),
    (lambda r: edit(r, 0, model="mystery"), "p2", "not in the models manifest"),
    (lambda r: edit(r, 0, profile="AWQ"), "p2", "no raw model for profile 'AWQ'"),
    (lambda r: r, "p1", "P1 rows are inrepo_test"),
    (lambda r: edit(r, 0, repo="pandas", task_id="pandas/x"), "p2", "P2 rows are tasks of the held-out repos"),
    (lambda r: edit(r, 0, list="pool200"), "p2", "P2 rows are tasks of the held-out repos"),
    (lambda r: edit(r, 0, turns=5), "p2", "one entry per turn"),
    (lambda r: edit(r, 0, resolved=1), "p2", "resolved must be true or false"),
    (lambda r: edit(r, 0, malformed={}), "p2", "missing field malformed.misnested_tag"),
    (lambda r: [x for x in r if x["model"] != "random_seed0"], "p2", r"models \['random_seed0'\] have no attempts"),
    (lambda r: edit(r, 0, tool_seconds=99.0), "p2", "not the per-turn tool_s sum"),
    (lambda r: edit(r, 0, tool_seconds=2.0 + 1e-6), "p2", "not the per-turn tool_s sum"),  # the harness sums exactly
    (lambda r: [{k: v for k, v in x.items() if k != "grammar"} for x in r], "p2", "grammar must be true or false"),
    (lambda r: edit(r, 0, grammar=0), "p2", "grammar must be true or false"),
    (lambda r: edit(r, 0, list="estimator_check30"), "p2", "--estimator-check input"),
    (lambda r: edit(r, 0, grading={"resolved": False}), "p2", "grading.test_files_touched must be true or false"),
])
def test_validation_errors(mutate, phase, match):
    recs, manifest = build(["fewest_turns", "random"], scaled({}), n=2)
    with pytest.raises(ValueError, match=match):
        load(mutate(recs), manifest, EM, phase)


def test_design_and_config_errors():
    recs, manifest = build(["random"], scaled({}), n=2, seeds=(0, 1))
    with pytest.raises(ValueError, match="different seed models per profile"):
        load([r for r in recs if not (r["model"] == "random_seed1" and r["profile"] == BF)], manifest, EM, "p2")
    with pytest.raises(ValueError, match="both random seed 0"):
        load(recs, {**manifest, "dup": {"arm": "random", "seed": 0}}, EM, "p2")
    with pytest.raises(ValueError, match="unknown arm"):
        load(recs, {**manifest, "x": {"arm": "nope", "seed": 0}}, EM, "p2")
    with pytest.raises(ValueError, match="guard_margin is required"):
        check_config({**CFG, "guard_margin": None}, upper_bound=False)
    with pytest.raises(ValueError, match="host_cpu_tdp_w is required"):
        check_config({**CFG, "host_cpu_tdp_w": None}, upper_bound=True)
    assert check_config({**CFG, "host_cpu_tdp_w": None}, upper_bound=False)["guard_margin"] == 0.05


def test_contrast_draws_are_paired():
    # Identical arms give T = 0 in every draw, so the shared draw is what makes the null exact.
    recs, manifest = build(["fewest_turns", "random"], scaled({}), seeds=(0, 1))
    arr = arrays(load(recs, manifest, EM, "p2"), [("fewest_turns", Q4), ("random", Q4)], 0.05, cpu_w=1.0)
    out = contrast(arr, ("fewest_turns", Q4), ("random", Q4), check_config(CFG, True))
    assert out["T"] == 0 and out["gpu"]["p_one_sided"] == 1.0 and out["gpu"]["upper95_one_sided"] == 0


def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT, check=False)


def test_scripts_end_to_end_and_missing_margin(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    (tmp_path / "em.json").write_text(json.dumps(EM))
    recs, manifest = p1()
    write_jsonl(tmp_path / "p1.jsonl", recs)
    (tmp_path / "m1.json").write_text(json.dumps(manifest))
    (tmp_path / "sc.json").write_text(json.dumps({"chosen": None, "decided": True}))
    head = tmp_path / "headline.json"
    common = ["--models", tmp_path / "m1.json", "--energy-model", tmp_path / "em.json"]
    res = run("p1_pick_headline.py", tmp_path / "p1.jsonl", *common, "--score-choice", tmp_path / "sc.json",
              "--p0a-verdict", "pass", "--config", cfg, "--out", head)
    assert res.returncode == 0, res.stderr
    assert json.loads(head.read_text())["headline"] == "fewest_turns"
    res = run("p1_pick_headline.py", tmp_path / "p1.jsonl", *common, "--score-choice", tmp_path / "sc.json",
              "--p0a-verdict", "pass", "--not-separable", "fewest_joules", "--config", cfg, "--out", head)
    assert res.returncode != 0 and "untrained arms" in res.stderr  # fewest_joules has P1 rows here

    recs, manifest = build(["fewest_turns", "random", "base"], scaled({"fewest_turns": 0.8}), seeds=(0, 1))
    write_jsonl(tmp_path / "p2.jsonl", recs + grammar_copy(recs, {"fewest_turns": 0.9 / 0.8}))
    erows, _ = est_rows({"fewest_turns": 1.0, "random": 1.0, "base": 1.0})
    write_jsonl(tmp_path / "est.jsonl", [{**r, "attempt_id": "e-" + r["attempt_id"], "list": "estimator_check30"}
                                         for r in erows])
    (tmp_path / "m2.json").write_text(json.dumps(manifest))
    common = ["--models", tmp_path / "m2.json", "--energy-model", tmp_path / "em.json", "--headline", head]
    res = run("p2_analyze.py", tmp_path / "p2.jsonl", *common, "--estimator-check", tmp_path / "est.jsonl",
              "--config", cfg, "--out-dir", tmp_path / "p2")
    assert res.returncode == 0, res.stderr
    out = json.loads((tmp_path / "p2" / "analysis.json").read_text())
    assert out["primary"]["T"] == pytest.approx(math.log(0.8)) and out["estimator_check"]["stands"]
    md = (tmp_path / "p2" / "analysis.md").read_text()
    assert out["turn_cap_frontier"] is not None and "| primary |" in md and "| H4 (grammar rows) |" in md
    assert out["h4"]["contrasts"][0]["keeps_sign"] and "keeps sign: True" in md

    cfg.write_text(yaml.safe_dump({**CFG, "guard_margin": None}))
    for res in (run("p1_pick_headline.py", tmp_path / "p1.jsonl", *common[:4], "--score-choice", tmp_path / "sc.json",
                    "--p0a-verdict", "pass", "--config", cfg),
                run("p2_analyze.py", tmp_path / "p2.jsonl", *common, "--config", cfg)):
        assert res.returncode != 0 and "guard_margin is required" in res.stderr


@pytest.mark.parametrize("end", ["infra_timeout", "turn_limit", "context_limit"])
def test_budget_hit_is_a_failure_even_if_graded_resolved(end):
    def fn(arm, s, prof, repo, i):
        hit = arm == "fewest_turns" and i == 0
        return {"resolved": hit or i < 2, "joules": 1000.0, "end": end if hit else None}
    recs, manifest = build(["fewest_turns", "random"], fn, n=4)
    out = analyze(recs, manifest, EM, "fewest_turns", CFG)
    assert out["primary"]["resolve_rate"] == {"arm": 0.25, "comparator": 0.5}
    assert out["n_resolved_after_budget_hit"] == 8  # 1 task per repo x 4 repos x 2 profiles


def test_guard_is_against_random_in_every_family():
    def fn(arm, s, prof, repo, i):
        # random resolves 9 of 10, qtrace and fewest_joules 4 of 10; qtrace spends half of fewest_joules' joules.
        return (i < 9 if arm == "random" else i < 4), (0.5 if arm == "qtrace" else 1.0) * base_joules(repo, i % 5)
    recs, manifest = build(["qtrace", "fewest_joules", "random"], fn, n=10)
    cfg = {**CFG, "families": {"H5": [["qtrace", "fewest_joules"]]}, "upper_bound_required": []}
    h5 = analyze(recs, manifest, EM, "qtrace", cfg)["families"]["H5"]["contrasts"]
    assert [c["guard"]["rr_diff"] for c in h5] == pytest.approx([-0.5, -0.5])
    assert all(c["gpu"]["p_one_sided"] < 0.05 and not c["guard"]["pass"] and not c["supported"] for c in h5)


def test_missing_seed_model_is_an_error():
    recs, manifest = build(["fewest_turns", "random"], scaled({}), seeds=(0, 1, 2))
    kept = [r for r in recs if r["model"] != "random_seed2"]
    with pytest.raises(ValueError, match="have no attempts"):
        analyze(kept, manifest, EM, "fewest_turns", CFG)
    two_seed = {m: v for m, v in manifest.items() if m != "random_seed2"}  # per-seed pairs would be truncated
    with pytest.raises(ValueError, match="seed model by seed model"):
        analyze(kept, two_seed, EM, "fewest_turns", CFG)


def test_secondary_tier_subset_and_thinking_off_base():
    conf, m1 = build(["fewest_turns", "random"], scaled({"fewest_turns": 0.8}), n=6)
    sec, m2 = build(["stable", "low_entropy"], scaled({"stable": 0.7}), profiles=(Q4,), n=3, lst="heldout100")
    nothink, m3 = build(["base_nothink"], scaled({}), n=3, lst="heldout100")
    manifest = {**m1, **m2, **m3}
    cfg = {**CFG, "families": {"H3": [["stable", "low_entropy"]]}, "upper_bound_required": []}
    out = analyze(conf + sec + nothink, manifest, EM, "fewest_turns", cfg)
    (h3,) = out["families"]["H3"]["contrasts"]
    assert out["primary"]["n_tasks"] == 24 and h3["n_tasks"] == 12 and h3["T"] == pytest.approx(math.log(0.7))
    assert out["b_metrics"]["base_nothink"]["n_tasks"] == 12
    with pytest.raises(ValueError, match="incomplete design: stable seed 0"):  # a subset cell covers its own list
        load(conf + sec[1:] + nothink, manifest, EM, "p2")


def test_sign_rule_two_of_four_blocks_claim():
    def fn(arm, s, prof, repo, i):
        f = {"base": 3.0, "random": 1.0}.get(arm, 0.4 if repo in ("scrapy", "pyramid") else 1.05)
        return (i + s) % 2 == 0 or i == 1, f * base_joules(repo, i)
    recs, manifest = build(["fewest_turns", "random", "base"], fn, seeds=(0, 1, 2), n=8)
    erows, _ = est_rows({"fewest_turns": 1.0, "random": 1.0, "base": 1.0})
    out = analyze(recs, manifest, EM, "fewest_turns", CFG,
                  estimator_rows=[{**r, "attempt_id": "e-" + r["attempt_id"]} for r in erows])
    assert out["primary"]["sign_holds"] == {"n_repos": 4, "n_same_sign": 2, "min_repos": 3, "holds": False}
    claim = out["claim_A"]
    assert claim["primary_wins"] and claim["frontier_pass"] and claim["estimator_stands"]
    assert claim["sign_holds"] is False and not claim["claimed_on_estimates"]


def test_contrast_resamples_tasks_within_repo():
    # Per-task values are constant within a repo and differ across repos, so draws that keep each repo's task
    # count leave T and the resolve-rate difference fixed; unstratified draws would move both.
    def fn(arm, s, prof, repo, i):
        return (repo != "scrapy", 1000.0) if arm == "random" else (True, 300.0 * (HELD.index(repo) + 1))
    recs, manifest = build(["fewest_turns", "random"], fn, profiles=(Q4,))
    arr = arrays(load(recs, manifest, EM, "p2"), [("fewest_turns", Q4), ("random", Q4)], 0.05)
    out = contrast(arr, ("fewest_turns", Q4), ("random", Q4), check_config(CFG, False))
    assert out["T"] == pytest.approx(math.log(750 / (20000 / 15)))
    assert out["gpu"]["upper95_one_sided"] == pytest.approx(out["T"])
    assert out["guard"]["lower95_one_sided"] == pytest.approx(out["guard"]["rr_diff"]) == pytest.approx(0.25)


def test_analyze_input_refusals_and_malformed_table():
    recs, manifest = build(["low_entropy", "random"], scaled({}))
    for head in ("random", "low_entropy"):
        with pytest.raises(ValueError, match="not one of the arms the P1 rule picks from"):
            analyze(recs, manifest, EM, head, CFG)
    def fn(arm, *_):
        return {"resolved": True, "joules": 100.0, "malformed": {"wrong_name": int(arm == "random")}}
    recs, manifest = build(["fewest_turns", "random"], fn)
    with pytest.raises(ValueError, match="both regimes"):
        analyze(recs, manifest, {**EM, "batched": {"below_85pct": True}}, "fewest_turns", CFG)
    no_check = {k: v for k, v in EM.items() if k != "batched"}
    for em in (no_check, {**EM, "batched": None}, {**EM, "batched": {"below_85pct": None}}):
        with pytest.raises(ValueError, match="batched.below_85pct must be true or false"):
            analyze(recs, manifest, em, "fewest_turns", CFG)
    energy = analyze(recs, manifest, EM, "fewest_turns", CFG)["energy"]
    assert energy["batched_below_85pct"] is False and energy["batched_share_same_pick"] == 0.9
    rows = analyze(recs, manifest, {**EM, "batched": {"below_85pct": False}}, "fewest_turns", CFG)["arms"]
    assert {(r["arm"], r["malformed_per_attempt"]["wrong_name"]) for r in rows} == {("fewest_turns", 0), ("random", 1)}


def em_idle(idle_w=10.0, **idle_by_profile):
    """EM with idle_w W charged per tool second, per profile where given."""
    em = copy.deepcopy(EM)
    for prof, v in em["profiles"].items():
        v["idle_w"] = idle_by_profile.get(prof, idle_w)
    return em


def test_endpoint_is_whole_attempt_with_idle_tool_time():
    # fewest_turns spends 0.8x random's generation joules but waits 3x as long on tools: 1000 x 0.8 + 10 W x 60 s
    # = 1400 J against 1000 + 10 W x 20 s = 1200 J at Q4_K_M, where BF16 charges 20 W per tool second.
    def fn(arm, s, prof, repo, i):
        ft = arm == "fewest_turns"
        return {"resolved": i < 4, "joules": 800.0 if ft else 1000.0, "tool_s": 30.0 if ft else 10.0}
    recs, manifest = build(["fewest_turns", "random"], fn)
    out = analyze(recs, manifest, em_idle(10.0, BF16=20.0), "fewest_turns", CFG)
    p = out["primary"]
    assert p["jpr"] == pytest.approx({"arm": 1400 / 0.8, "comparator": 1200 / 0.8})
    assert p["T"] == pytest.approx(math.log(1400 / 1200)) and not p["wins"]
    gen, idle = p["alongside"]["generation_window"], p["alongside"]["idle_subtracted"]
    assert gen["T"] == pytest.approx(math.log(0.8)) and gen["jpr"]["arm"] == pytest.approx(800 / 0.8)
    assert idle["T"] == pytest.approx(math.log(0.8)) and idle["jpr"]["arm"] == pytest.approx(400 / 0.8)
    cpu = 10.0 * 280.0 / 128  # the upper bound adds CPU energy to the whole-attempt estimate
    assert p["upper_bound"]["T"] == pytest.approx(math.log((1400 + cpu) / (1200 + cpu)))
    rows = {(r["arm"], r["profile"]): r for r in out["arms"]}
    assert rows["random", BF]["jpr"] == pytest.approx((1000 + 20 * 20) / 0.8)
    assert rows["random", BF]["jpr_generation_window"] == pytest.approx(1000 / 0.8)
    assert out["energy"]["provisional"] is True and out["energy"]["quantity"] == "whole_attempt_raw_joules"
    assert out["energy"]["idle_w"] == {BF: 20.0, "Q3_K_M": 10.0, Q4: 10.0}


def test_failed_raw_fit_and_missing_idle_w_fail_loudly():
    recs, manifest = build(["fewest_turns", "random"], scaled({}), n=2)
    bad = copy.deepcopy(EM)
    bad["profiles"][BF]["raw"] = {"error": "nested LOAO needs at least 3 attempts per profile"}
    with pytest.raises(ValueError, match=r"raw fit for profile 'BF16' failed \(nested LOAO"):
        load(recs, manifest, bad, "p2")
    assert load([r for r in recs if r["profile"] == Q4], manifest, bad, "p2")["n_rows"] == 16  # BF16 rows only
    no_idle = copy.deepcopy(EM)
    no_idle["profiles"][Q4]["idle_w"] = None
    with pytest.raises(ValueError, match="profiles.Q4_K_M.idle_w"):
        load(recs, manifest, no_idle, "p2")
    for ranking in (None, {**EM["ranking"], "quantity": "generation_window_raw_joules"}):
        with pytest.raises(ValueError, match="ranking"):
            load(recs, manifest, {**EM, "ranking": ranking}, "p2")


def test_estimator_check_compares_like_with_like():
    # Generation windows are estimated exactly. The measured tool windows match 10 W x 2 s for fewest_turns and
    # base, and random draws 20 W while tools run, which only the whole-attempt comparison can see.
    def fn(arm, s, prof, repo, i):
        j = base_joules(repo, i)
        return {"resolved": True, "joules": j, "conc": 1, "measured": j, "tool_j": 40.0 if arm == "random" else 20.0}
    recs, manifest = build(["fewest_turns", "random", "base"], fn, profiles=(Q4,), n=8)
    em = em_idle(10.0)
    out = estimator_check(load(recs, manifest, em, "p2"), "fewest_turns", CFG)
    want = math.exp(np.mean([-math.log((base_joules(r, i) + 40) / (base_joules(r, i) + 20))
                             for r in HELD for i in range(8)]))
    assert out["scope"] == "whole_attempt" and out["n_attempts_without_tool_energy"] == 0
    assert out["pairs"]["fewest_turns_vs_random"]["ratio"] == pytest.approx(want)
    assert out["pairs"]["fewest_turns_vs_base"]["ratio"] == pytest.approx(1.0)
    recs[0] = copy.deepcopy(recs[0])
    recs[0]["per_turn"][1]["tool_energy_j"] = None  # one unmeasured tool window: generation windows on both sides
    out = estimator_check(load(recs, manifest, em, "p2"), "fewest_turns", CFG)
    assert out["scope"] == "generation_window" and out["n_attempts_without_tool_energy"] == 1
    assert all(v["ratio"] == pytest.approx(1.0) for v in out["pairs"].values()) and out["stands"]


def test_turn_cap_frontier_charges_tool_time_of_kept_turns():
    def fn(arm, s, prof, repo, i):
        t = 3 if arm == "fewest_turns" else 10
        return {"resolved": True, "joules": 100.0 * t, "turns": t, "tool_s": 5.0}
    recs, manifest = build(["fewest_turns", "base"], fn, profiles=(Q4,), n=4)
    em = em_idle(10.0)
    out = turn_cap_frontier(load(recs, manifest, em, "p2"), "fewest_turns", em, CFG)
    # Each kept turn: 100 J of generation plus 10 W x its own 5 s of tools.
    assert [p["mean_joules"] for p in out["points"]["base"]] == pytest.approx([150.0 * c for c in range(1, 11)])
    assert out["headline_uncapped"]["mean_joules"] == pytest.approx(450.0) and out["pass"]


def test_tripwire_rates_in_pick_and_arm_table():
    recs, manifest = p1()
    hit = [r for r in recs if r["model"] == "fewest_turns_seed0" and r["profile"] == BF]
    flag = {r["attempt_id"] for r in hit if r["task_id"].endswith("t0")}
    miss = {r["attempt_id"] for r in hit if r["task_id"].endswith("t1")}
    recs = [{**r, "grading": {"test_files_touched": True}} if r["attempt_id"] in flag else
            {**r, "resolved": False, "end_reason": "turn_limit"} if r["attempt_id"] in miss else r for r in recs]
    out = pick_headline(recs, manifest, EM, CHOSEN, {"score": "kl_decision", "decision": "pass"}, CFG, "pass")
    assert out["tripwire"]["fewest_turns/BF16"] == pytest.approx(
        {"resolve_rate": 0.75, "flagged_share": 0.25, "resolve_rate_unflagged": 12 / 18})
    assert out["tripwire"]["random/BF16"]["flagged_share"] == 0 and out["energy"]["provisional"] is True
    recs, manifest = build(["fewest_turns", "random"], lambda *a: {"resolved": True, "joules": 100.0, "flagged": True})
    rows = analyze(recs, manifest, EM, "fewest_turns", CFG)["arms"]
    assert all(r["tripwire"]["flagged_share"] == 1 and r["tripwire"]["resolve_rate_unflagged"] is None for r in rows)


def test_p2_lists():
    recs, manifest = build(["fewest_turns", "random"], scaled({}), n=2)
    for lst in ("heldout691", "heldout100", "some_frozen_subset"):
        assert load([{**r, "list": lst} for r in recs], manifest, EM, "p2")["n_rows"] == len(recs)
    for lst in ("pool200", "inrepo_test", "verified100"):
        for phase in ("p2", "estimator"):
            with pytest.raises(ValueError, match="P2 rows are tasks of the held-out repos"):
                load(edit(recs, 0, list=lst), manifest, EM, phase)
    erows, emanifest = est_rows({"fewest_turns": 1.0, "random": 1.0, "base": 1.0})
    erows = [{**r, "list": "estimator_check30"} for r in erows]
    with pytest.raises(ValueError, match="--estimator-check input"):
        load(erows, emanifest, EM, "p2")
    assert estimator_check(load(erows, emanifest, EM, "estimator"), "fewest_turns", CFG)["stands"]
    main, manifest = build(["fewest_turns", "random", "base"], scaled({"fewest_turns": 0.8}), seeds=(0, 1))
    out = analyze(main, manifest, EM, "fewest_turns", CFG,
                  estimator_rows=[{**r, "attempt_id": "e-" + r["attempt_id"]} for r in erows])
    assert out["estimator_check"]["stands"]


def grammar_copy(recs, factor_by_arm, profiles=(Q4,)):
    """Grammar-constrained reruns of recs at profiles, joules scaled per arm."""
    return [{**r, "attempt_id": "g-" + r["attempt_id"], "grammar": True,
             "tokens": {**r["tokens"], "output": r["tokens"]["output"] * factor_by_arm.get(r["model"].split("_seed")[0], 1)},
             "per_turn": [{**t, "output_tokens": t["output_tokens"] * factor_by_arm.get(r["model"].split("_seed")[0], 1)}
                          for t in r["per_turn"]]}
            for r in recs if r["profile"] in profiles]


def test_grammar_rows_stay_out_of_the_main_analysis():
    recs, manifest = build(["fewest_turns", "random", "base"], scaled({"fewest_turns": 0.8}), seeds=(0, 1))
    erows, _ = est_rows({"fewest_turns": 1.0, "random": 1.02, "base": 1.0})
    erows = [{**r, "attempt_id": "e-" + r["attempt_id"]} for r in erows]
    # Grammar rows that would move every number: the headline spends 3x random's joules under grammar.
    noisy = grammar_copy(recs, {"fewest_turns": 3.0}, profiles=(Q4, BF))
    cfg = {**CFG, "h4_contrasts": [["headline", "random"]]}
    plain = analyze(recs, manifest, EM, "fewest_turns", cfg, estimator_rows=erows)
    mixed = analyze(recs + noisy, manifest, EM, "fewest_turns", cfg,
                    estimator_rows=erows + grammar_copy(erows, {"fewest_turns": 2.0}))
    for k in ("primary", "families", "arms", "b_metrics", "tokens_ratio_contrasts", "estimator_check",
              "turn_cap_frontier", "claim_A", "n_rows"):
        assert mixed[k] == plain[k], k
    assert plain["h4"] is None and mixed["n_rows_grammar"] == len(noisy)
    p1recs, p1manifest = p1()
    pick = pick_headline(p1recs + grammar_copy(p1recs, {"fewest_turns": 0.01}, profiles=(Q4, BF)), p1manifest, EM,
                         CHOSEN, {"score": "kl_decision", "decision": "pass"}, CFG, "pass")
    assert pick["headline"] == "stable"


@pytest.mark.parametrize("factor, keeps", [(0.9, True), (1.1, False)])
def test_h4_contrast_on_grammar_rows(factor, keeps):
    recs, manifest = build(["fewest_turns", "random"], scaled({"fewest_turns": 0.8}), seeds=(0, 1))
    g = grammar_copy(recs, {"fewest_turns": factor / 0.8})  # under grammar the headline spends factor x random's
    out = analyze(recs + g, manifest, EM, "fewest_turns", {**CFG, "h4_contrasts": [["headline", "random"]]})
    (c,) = out["h4"]["contrasts"]
    assert (c["arm"], c["comparator"], c["profile"], c["n_tasks"]) == ("fewest_turns", "random", Q4, 20)
    assert c["T"] == pytest.approx(math.log(factor)) and c["without_grammar"]["T"] == pytest.approx(math.log(0.8))
    assert c["keeps_sign"] is keeps and c["effect_ratio"] == pytest.approx(math.log(factor) / math.log(0.8))
    assert c["supported"] is keeps and c["wins"] is keeps and c["holm_p"] == c["gpu"]["p_one_sided"]
    assert out["primary"]["T"] == pytest.approx(math.log(0.8))
    assert "H4" in out["h4"]["reading"]


def test_h4_holm_and_required_contrast():
    def fn(arm, s, prof, repo, i):
        f = {"fewest_turns": (0.7, 1.25, 0.8, 1.15, 0.95)[i], "fewest_joules": (0.75, 1.2, 0.85, 1.1, 0.9)[i]}
        return True, f.get(arm, 1.0) * base_joules(repo, i)
    recs, manifest = build(["fewest_turns", "fewest_joules", "random"], fn)
    cfg = {**CFG, "h4_contrasts": [["headline", "random"], ["fewest_joules", "random"]]}
    h4 = analyze(recs + grammar_copy(recs, {}), manifest, EM, "fewest_turns", cfg)["h4"]
    raw = [c["gpu"]["p_one_sided"] for c in h4["contrasts"]]
    assert len(raw) == 2 and 0 < min(raw) < 0.5
    assert [c["holm_p"] for c in h4["contrasts"]] == pytest.approx(holm(raw))
    with pytest.raises(ValueError, match=r"h4_contrasts must include \[headline, random\]"):
        check_config({**CFG, "h4_contrasts": [["fewest_joules", "random"]]}, upper_bound=False)
    g = grammar_copy([r for r in recs if r["model"] != "random_seed0"], {})
    with pytest.raises(ValueError, match="H4 needs grammar rows of fewest_turns and random at Q4_K_M"):
        analyze(recs + g, manifest, EM, "fewest_turns", cfg)
