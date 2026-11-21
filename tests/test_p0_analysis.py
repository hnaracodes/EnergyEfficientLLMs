"""Tests for the P0 gate statistics on synthetic attempt records. Run:  python -m pytest tests/test_p0_analysis.py -q"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

from qtrace.p0_analysis import (
    clean,
    design_mismatches,
    flagged,
    flagged_share,
    p0a_factors,
    p0a_primary,
    p0a_secondary,
    p0a_verdict,
    p0b_report,
    p0c_breakage,
    paired_permutation,
    processed_tokens,
    profile_summary,
)

_ids = iter(range(10**9))


def rec(task, profile, resolved, out=100, unc=900, cached=4000, lst="heldout100", runtime="llama.cpp", repo="scrapy",
        end_reason=None, run_id="run-a", turns=10):
    end_reason = end_reason or ("submitted" if resolved else "turn_limit")
    return {"attempt_id": f"a{next(_ids)}", "run_id": run_id, "task_id": task, "profile": profile,
            "resolved": resolved, "list": lst, "runtime": runtime, "repo": repo, "end_reason": end_reason, "turns": turns,
            "grading": {"test_files_touched": False},
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached}}


def varied(n_tasks=40, inflate=1.0, q_rate=1.0, seed=0):
    """Tasks differ in difficulty and length, so the bootstrap is not degenerate."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_tasks):
        p, size = rng.uniform(0.2, 0.9), rng.uniform(0.5, 2.0)
        for _ in range(3):
            rows.append(rec(f"t{i}", "BF16", rng.random() < p, out=int(100 * size), unc=int(900 * size),
                            cached=int(4000 * size)))
            rows.append(rec(f"t{i}", "Q4_K_M", rng.random() < p * q_rate, out=int(100 * size * inflate),
                            unc=int(900 * size * inflate), cached=int(4000 * size * inflate)))
    return rows


def test_processed_tokens_counts_output_and_all_input():
    assert processed_tokens(rec("t", "BF16", True, out=10, unc=20, cached=30)) == 60


def test_verdict_boundaries():
    assert p0a_verdict(1.15, 1.0001, 2.0) == "pass"
    assert p0a_verdict(1.15, 1.0, 2.0) == "inconclusive"  # lower bound must exceed 1
    assert p0a_verdict(1.10, 0.9, 1.1499) == "fail"
    assert p0a_verdict(1.10, 0.9, 1.15) == "inconclusive"  # upper bound must be below 1.15
    assert p0a_verdict(None, None, None) == "inconclusive"


def test_identical_profiles_give_r_one_in_every_draw():
    # BF16 and Q4 have the same per-task data; if pairing broke under resampling, draws would vary.
    rows = []
    for r in varied(seed=1):
        if r["profile"] == "BF16":
            rows += [r, {**r, "profile": "Q4_K_M", "attempt_id": r["attempt_id"] + "q"}]
    res = p0a_primary(rows, n_boot=300, seed=0)
    assert abs(res["R"] - 1) < 1e-12 and abs(res["ci95"][0] - 1) < 1e-12 and abs(res["ci95"][1] - 1) < 1e-12
    assert res["verdict"] == "fail"


def test_bootstrap_has_spread_and_depends_on_seed():
    rows = varied(inflate=1.3, seed=2)
    a, b = p0a_primary(rows, n_boot=400, seed=3), p0a_primary(rows, n_boot=400, seed=4)
    assert a["ci95"][0] < a["R"] < a["ci95"][1]
    assert a["ci95"] != b["ci95"] and a == p0a_primary(rows, n_boot=400, seed=3)


def test_clear_inflation_passes_and_factors_compose():
    rows = varied(inflate=1.6, seed=5)
    res = p0a_primary(rows, n_boot=500, seed=0)
    assert res["verdict"] == "pass"
    f = p0a_factors(rows)
    assert abs(f["length_factor"] * f["completion_factor"] - res["R"]) < 1e-9


def test_fewer_resolves_at_equal_tokens_is_completion_not_length():
    rows = []
    for i in range(10):
        rows += [rec(f"t{i}", "BF16", True) for _ in range(3)]
        rows += [rec(f"t{i}", "Q4_K_M", a == 0) for a in range(3)]  # 1 of 3 resolved
    f = p0a_factors(rows)
    assert abs(f["length_factor"] - 1.0) < 1e-12 and abs(f["completion_factor"] - 3.0) < 1e-12
    assert abs(p0a_primary(rows, n_boot=50, seed=0)["R"] - 3.0) < 1e-12


def test_zero_resolve_cases_are_explicit_and_json_safe():
    only_q = [rec("t0", "BF16", False), rec("t0", "Q4_K_M", True)]
    assert p0a_primary(only_q, n_boot=50, seed=0)["R"] == 0.0
    only_b = [rec("t0", "BF16", True), rec("t0", "Q4_K_M", False)]
    res = p0a_primary(only_b, n_boot=50, seed=0)
    assert res["R"] == "inf"
    neither = [rec("t0", "BF16", False), rec("t0", "Q4_K_M", False)]
    assert p0a_primary(neither, n_boot=50, seed=0)["R"] is None
    json.dumps(res, allow_nan=False)


def test_errors_are_dropped_and_counted():
    rows = varied(seed=6)
    extra = [rec("t0", "Q4_K_M", False, end_reason="error", out=50) for _ in range(10)]
    kept, dropped = clean(rows + extra + [rows[0]])
    assert dropped == {"error": 10, "duplicate": 1} and len(kept) == len(rows)


def test_design_mismatches_flag_wrong_attempt_counts():
    rows = [rec("t0", "BF16", True) for _ in range(3)] + [rec("t1", "BF16", True) for _ in range(4)]
    assert design_mismatches(rows, 3) == [["t1", "BF16", 4]]


def test_p0b_report_criteria_and_per_task_counts():
    rows = []
    for i, k in enumerate([0, 2, 5, 8, 16, 4, 4, 1, 0, 12]):
        rows += [rec(f"p{i}", "BF16", a < k, lst="pool200", runtime="vllm", repo="pandas") for a in range(16)]
    for i in range(4):
        rows += [rec(f"h{i}", "BF16", a < 4, lst="heldout100", runtime="vllm") for a in range(16)]
    r = p0b_report(rows, inrepo_size=300)
    assert r["pool"]["n_ge4"] == 6 and r["per_task"]["p4"]["successes"] == 16
    assert abs(r["projected_pool"] - 0.6 * (3887 - 300)) < 1e-6
    assert abs(r["rl_share_mixed"] - 7 / 10) < 1e-12  # 1..15 of 16: 2, 5, 8, 4, 4, 1, 12
    assert r["pass"] == {"c1": True, "c2": True, "c3": True} and r["tasks_below_design_attempts"] == []
    no_held = p0b_report([x for x in rows if x["list"] == "pool200"][:-1], inrepo_size=300)
    assert no_held["pass"]["c3"] == "not computed" and no_held["tasks_below_design_attempts"] == ["p9"]


def test_p0c_bins_ci_and_verdict():
    counts, rows = {}, []
    import random
    rng = random.Random(7)
    for i in range(60):
        counts[f"c{i}"] = rng.choice([0, 2, 6, 12])
        base_p = 0.8 if counts[f"c{i}"] >= 4 else 0.3
        rows += [rec(f"c{i}", "BF16", rng.random() < base_p, lst="pool200") for _ in range(4)]
        rows += [rec(f"c{i}", "Q4_K_M", rng.random() < base_p * 0.6, lst="pool200") for _ in range(4)]
    rows += [rec("unbinned", "BF16", True, lst="pool200"), rec("unbinned", "Q4_K_M", False, lst="pool200")]
    r = p0c_breakage(rows, counts, "Q4_K_M", n_boot=400, seed=0)
    assert r["n_tasks"] == 60 and r["tasks_excluded"] == ["unbinned"]
    assert r["L_total_lo95"] > 0 and r["verdict"] == "pass" and r["share_4plus"] >= 0.4
    assert r["share_4plus_ci95"][0] <= r["share_4plus"] <= r["share_4plus_ci95"][1]


def test_cli_selects_runs_by_manifest(tmp_path):
    rows = []
    for i in range(20):
        rows += [rec(f"p{i}", "BF16", a < 6, lst="pool200", runtime="vllm", repo="pandas", run_id="p0b") for a in range(16)]
        rows += [rec(f"p{i}", "BF16", True, lst="pool200", runtime="vllm", repo="pandas", run_id="pilot") for _ in range(4)]
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "m.json").write_text(json.dumps({"P0b": ["p0b"]}))
    out = tmp_path / "g.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "p0_analyze.py"), str(path), "--manifest",
                    str(tmp_path / "m.json"), "--inrepo-size", "300", "--n-boot", "50", "--out", str(out)],
                   check=True, capture_output=True)
    g = json.loads(out.read_text())["P0b"]
    assert all(v["attempts"] == 16 and v["successes"] == 6 for v in g["per_task"].values())  # pilot excluded
    assert g["design_mismatches"] == []


def test_p0b_counts_only_unflagged_successes():
    rows = []
    for a in range(16):  # 6 resolved, but 3 of them touched test files
        r = rec("p0", "BF16", a < 6, lst="pool200", runtime="vllm", repo="pandas")
        r["grading"] = {"test_files_touched": a < 3}
        rows.append(r)
    rep = p0b_report(rows, inrepo_size=300)
    assert rep["per_task"]["p0"]["successes"] == 3 and rep["pool"]["n_ge4"] == 0
    assert rep["pool"]["flagged_successes_excluded"] == 3


def test_clean_rejects_resolved_without_submission():
    # Integration finding: a scored-budget hit or infra_timeout is a failure (04), whatever the grader said.
    for reason in ("turn_limit", "context_limit", "infra_timeout", "error"):
        with pytest.raises(ValueError, match="resolved"):
            clean([rec("t0", "BF16", True, end_reason=reason)])


def test_flagged_reads_the_tripwire_strictly():
    # Integration finding: a record without grading.test_files_touched must not read as "not flagged".
    r = rec("t0", "BF16", True)
    assert flagged(r) is False and flagged({**r, "grading": {"test_files_touched": True}}) is True
    with pytest.raises(KeyError):
        flagged({**r, "grading": {"fail_to_pass_ok": 1}})
    with pytest.raises(KeyError):
        flagged({k: v for k, v in r.items() if k != "grading"})
    with pytest.raises(TypeError):
        flagged({**r, "grading": None})
    with pytest.raises(TypeError, match="bool"):
        flagged({**r, "grading": {"test_files_touched": None}})


def test_flagged_share_and_heldout_excluding_flagged():
    rows = [rec("h0", "BF16", a < 8, lst="heldout100", runtime="vllm") for a in range(16)]
    for r in rows[:4]:  # 4 of the 8 successes touched test files
        r["grading"] = {"test_files_touched": True}
    rows.append(rec("h0", "Q4_K_M", False))
    assert flagged_share(rows) == {"BF16": 4 / 16, "Q4_K_M": 0.0}
    rep = p0b_report(rows, inrepo_size=300)
    assert rep["heldout_mean_pass1"] == 0.5 and rep["heldout_mean_pass1_excluding_flagged"] == 4 / 12


def _run(tmp_path, rows, manifest, *extra):
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "p0_analyze.py"), str(path), "--manifest",
                           str(tmp_path / "m.json"), "--inrepo-size", "300", "--n-boot", "50",
                           "--out", str(tmp_path / "g.json"), *extra], capture_output=True, text=True, check=False)


def _cli(tmp_path, rows, manifest, *extra):
    _run(tmp_path, rows, manifest, *extra).check_returncode()
    return json.loads((tmp_path / "g.json").read_text())


def test_cli_min_candidates_reaches_p0b(tmp_path):
    # Integration finding: the P0b fallback (min_candidates 2) must be settable without editing code.
    rows = [rec(f"p{i}", "BF16", a < 3, lst="pool200", runtime="vllm", repo="pandas", run_id="p0b")
            for i in range(5) for a in range(16)]
    assert _cli(tmp_path, rows, {"P0b": ["p0b"]})["P0b"]["pool"]["n_ge4"] == 0
    g = _cli(tmp_path, rows, {"P0b": ["p0b"]}, "--min-candidates", "2")["P0b"]
    assert g["pool"]["n_ge4"] == 5 and g["sensitivity_excluding_infra_timeout"]["pool"]["n_ge4"] == 5


def test_cli_reports_flagged_and_infra_timeout_variants(tmp_path):
    # Integration findings: #9's excluding-flagged reporting rule and 04's infra_timeout sensitivity
    # for every gate, not only P0a.
    import random
    rng = random.Random(11)
    rows = []
    for i in range(30):
        k = rng.choice([0, 2, 6, 12])
        rows += [rec(f"p{i}", "BF16", a < k, lst="pool200", runtime="vllm", run_id="p0b") for a in range(16)]
        rows += [rec(f"p{i}", prof, rng.random() < (0.8 if k >= 4 else 0.3) * (0.5 if prof != "BF16" else 1),
                     lst="pool200", run_id="p0c") for prof in ("BF16", "Q4_K_M") for _ in range(4)]
    for i in range(10):
        rows += [rec(f"h{i}", "BF16", a < 5, runtime="vllm", run_id="p0b") for a in range(16)]
        rows += [rec(f"h{i}", prof, rng.random() < 0.5, run_id="p0a") for prof in ("BF16", "Q4_K_M") for _ in range(3)]
    p0c_q4 = [r for r in rows if r["run_id"] == "p0c" and r["profile"] == "Q4_K_M" and r["resolved"]]
    for r in p0c_q4[:3]:
        r["grading"] = {"test_files_touched": True}
    rows.append(rec("h0", "BF16", False, runtime="vllm", run_id="p0b", end_reason="infra_timeout"))
    g = _cli(tmp_path, rows, {"P0b": ["p0b"], "P0a": ["p0a"], "P0c": ["p0c"]})
    assert g["P0a"]["primary_excluding_flagged"]["verdict"] and g["P0a"]["flagged_share"]["Q4_K_M"] == 0.0
    assert g["P0b"]["flagged_share"] == {"BF16": 0.0} and "heldout_mean_pass1_excluding_flagged" in g["P0b"]
    sens = g["P0b"]["sensitivity_excluding_infra_timeout"]
    assert sens["heldout_mean_pass1"] == 0.3125
    assert abs(g["P0b"]["heldout_mean_pass1"] - (9 * 5 / 16 + 5 / 17) / 10) < 1e-12  # the timeout charged as a failure
    assert g["P0c"]["flagged_share"]["Q4_K_M"] == 3 / 120 and g["P0c"]["flagged_share"]["BF16"] == 0.0
    q4 = g["P0c"]["Q4_K_M"]
    assert q4["excluding_flagged"]["L_total"] > q4["L_total"]  # dropping flagged Q4 successes widens the gap
    assert q4["sensitivity_excluding_infra_timeout"]["L_total"] == q4["L_total"]


def test_p0b_tables_per_repo_heldout_per_task_and_mixed_share_at_group_8():
    # Audit item "Complete the P0b report tables": docs/04 P0b report and docs/03 S2 gate input.
    rows = []
    for i, (k, repo) in enumerate([(0, "pandas"), (16, "pandas"), (1, "flask"), (8, "flask")]):
        rows += [rec(f"p{i}", "BF16", a < k, lst="pool200", runtime="vllm", repo=repo) for a in range(16)]
    for i, (k, repo) in enumerate([(4, "sympy"), (2, "sympy"), (8, "astropy")]):
        rows += [rec(f"h{i}", "BF16", a < k, lst="heldout100", runtime="vllm", repo=repo) for a in range(16)]
    r = p0b_report(rows, inrepo_size=300, n_sub=4000, seed=1)
    assert r["pool"]["by_repo"]["flask"] == {"n_tasks": 2, "n_ge4": 1, "successes": 9, "attempts": 32}
    assert r["heldout_by_repo"]["sympy"] == {"n_tasks": 2, "n_ge4": 1, "successes": 6, "attempts": 32}
    assert r["heldout_per_task"]["h2"] == {"successes": 8, "attempts": 16, "repo": "astropy"}
    g8 = r["rl_share_mixed_g8"]
    # P(mixed) per task when drawing 8 of 16: 0, 0, 1/2 (the one success drawn), 1 - 2 / C(16, 8).
    assert abs(g8["mean"] - (0.5 + 1 - 2 / 12870) / 4) < 0.01 and g8["group_size"] == 8
    assert g8["ci95"][0] <= g8["mean"] <= g8["ci95"][1] and g8["n_tasks"] == 4
    assert g8 == p0b_report(rows, inrepo_size=300, n_sub=4000, seed=1)["rl_share_mixed_g8"]
    flat = p0b_report([x for x in rows if x["task_id"] in ("p0", "p1")], inrepo_size=300, n_sub=100)
    assert flat["rl_share_mixed_g8"]["mean"] == 0.0 and flat["rl_share_mixed_g8"]["ci95"] == [0.0, 0.0]
    short = p0b_report([x for x in rows if x["list"] == "pool200"][:16 * 3 + 5], inrepo_size=300, n_sub=100)
    assert short["rl_share_mixed_g8"]["n_tasks"] == 3 and short["rl_share_mixed_g8"]["tasks_below_group_size"] == 1


def test_cli_p0b_tables_use_the_run_seed_and_draw_count(tmp_path):
    rows = [rec(f"{lst[0]}{i}", "BF16", a < 3, lst=lst, runtime="vllm", run_id="p0b")
            for lst in ("pool200", "heldout100") for i in range(3) for a in range(16)]
    g = _cli(tmp_path, rows, {"P0b": ["p0b"]}, "--seed", "7")["P0b"]
    assert g["rl_share_mixed_g8"]["n_sub"] == 50 and g["rl_share_mixed_g8"]["seed"] == 7
    assert set(g["heldout_per_task"]) == {"h0", "h1", "h2"}
    assert "heldout_per_task" not in g["sensitivity_excluding_infra_timeout"]
    assert g["sensitivity_excluding_infra_timeout"]["rl_share_mixed_g8"] == g["rl_share_mixed_g8"]


def test_profile_summary_turns_and_flagged_rates():
    # Audit item "Add P0a's secondary analyses": turns per resolved task, and every resolve rate also
    # excluding flagged attempts (04 tripwire rule).
    rows = [rec("t0", "Q8_0", True, turns=4), rec("t0", "Q8_0", True, turns=6), rec("t0", "Q8_0", False, turns=20),
            rec("t1", "Q8_0", False, turns=30)]
    rows[0]["grading"] = {"test_files_touched": True}
    s = profile_summary(rows, "Q8_0")
    assert s["turns_per_resolved"] == 60 / 2 and s["mean_turns_resolved"] == 5.0  # failed attempts charged
    assert s["resolve_rate"] == 0.5 and s["resolve_rate_excluding_flagged"] == 1 / 3 and s["flagged_share"] == 0.25
    assert s["n_attempts"] == 4 and s["n_tasks"] == 2
    none = profile_summary([rec("t0", "Q8_0", False)], "Q8_0")
    assert none["turns_per_resolved"] is None and none["mean_turns_resolved"] is None


def test_paired_permutation_on_per_task_pass_counts():
    same = []
    for r in varied(seed=8):
        if r["profile"] == "BF16":
            same += [r, {**r, "profile": "Q3_K_M", "attempt_id": r["attempt_id"] + "q"}]
    t = paired_permutation(same, "Q3_K_M", n_perm=500, seed=0)
    assert t["mean_rate_diff"] == 0.0 and t["p_value"] == 1.0 and t["n_tasks"] == 40
    gap = []
    for i in range(20):
        gap += [rec(f"t{i}", "BF16", True) for _ in range(3)] + [rec(f"t{i}", "Q3_K_M", a == 0) for a in range(3)]
    t = paired_permutation(gap, "Q3_K_M", n_perm=500, seed=0)
    assert abs(t["mean_rate_diff"] + 2 / 3) < 1e-12 and t["p_value"] == 1 / 501
    mixed = varied(q_rate=0.7, seed=9)
    assert paired_permutation(mixed, "Q4_K_M", n_perm=300, seed=1) == paired_permutation(mixed, "Q4_K_M", n_perm=300,
                                                                                            seed=1)
    assert 0 < paired_permutation(mixed, "Q4_K_M", n_perm=300, seed=1)["p_value"] <= 1


def test_p0a_secondary_covers_every_profile_and_both_lists():
    import random
    rng = random.Random(10)
    rows = []
    for lst in ("heldout100", "verified100"):
        for i in range(15):
            for prof, rate in (("BF16", 0.6), ("Q8_0", 0.6), ("Q4_K_M", 0.5), ("Q3_K_M", 0.3), ("Q4_K_M_gate8", 0.5)):
                rows += [rec(f"{lst[0]}{i}", prof, rng.random() < rate, lst=lst, turns=rng.randint(3, 40))
                         for _ in range(3)]
    for r in [r for r in rows if r["profile"] == "Q3_K_M" and r["resolved"]][:4]:
        r["grading"] = {"test_files_touched": True}
    sec = p0a_secondary(rows, n_boot=100, seed=2)
    assert set(sec) == {"heldout100", "verified100"}
    v = sec["verified100"]
    assert set(v["per_profile"]) == {"BF16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8"}
    assert set(v["vs_BF16"]) == {"Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8"}
    ver = [r for r in rows if r["list"] == "verified100"]
    assert v["vs_BF16"]["Q3_K_M"]["primary"] == p0a_primary(ver, quantized="Q3_K_M", n_boot=100, seed=2)
    q3 = sec["heldout100"]["vs_BF16"]["Q3_K_M"]
    assert q3["factors_excluding_flagged"]["resolve_rate"]["Q3_K_M"] < q3["factors"]["resolve_rate"]["Q3_K_M"]
    assert q3["permutation_excluding_flagged"]["mean_rate_diff"] < q3["permutation"]["mean_rate_diff"]
    json.dumps(sec, allow_nan=False)
    assert p0a_secondary([r for r in rows if r["profile"] != "BF16"], n_boot=10)["heldout100"]["vs_BF16"] == {}


def test_cli_p0a_reports_secondary_and_flagged_factors(tmp_path):
    rows = [rec(f"{lst[0]}{i}", prof, (i + a) % 3 == 0, lst=lst, run_id="p0a")
            for lst in ("heldout100", "verified100") for i in range(6) for prof in ("BF16", "Q4_K_M", "Q8_0")
            for a in range(3)]
    rows.append(rec("v0", "Q8_0", False, lst="verified100", run_id="p0a"))
    g = _cli(tmp_path, rows, {"P0a": ["p0a"]})["P0a"]
    assert set(g["secondary"]) == {"heldout100", "verified100"} and "Q8_0" in g["secondary"]["verified100"]["vs_BF16"]
    assert g["secondary"]["heldout100"]["vs_BF16"]["Q8_0"]["permutation"]["n_perm"] == 50
    assert "factors_excluding_flagged" in g and g["design_mismatches"] == [["v0", "Q8_0", 4]]


def _p0a_first_look(inflate=1.2, seed=0):
    """BF16 and Q4_K_M on heldout100 (3 attempts each) plus Q3_K_M, as run "p0a"."""
    rows = [{**r, "run_id": "p0a"} for r in varied(n_tasks=20, inflate=inflate, seed=seed)]
    return rows + [{**r, "profile": "Q3_K_M", "attempt_id": r["attempt_id"] + "q3"} for r in rows
                   if r["profile"] == "BF16"]


def _topup(rows, q4_scale=1.0, run_id="p0a-topup"):
    """Three fresh attempts per task at BF16 and Q4_K_M, Q4_K_M tokens scaled by q4_scale."""
    return [{**r, "run_id": run_id, "attempt_id": r["attempt_id"] + run_id,
             "tokens": {k: v * (q4_scale if r["profile"] == "Q4_K_M" else 1) for k, v in r["tokens"].items()}}
            for r in rows if r["profile"] in ("BF16", "Q4_K_M")]


def test_cli_p0a_inconclusive_needs_the_retest(tmp_path):
    # Audit item "Add the P0a inconclusive re-test path" (04, P0a Inconclusive).
    g = _cli(tmp_path, _p0a_first_look(), {"P0a": ["p0a"]}, "--seed", "0")["P0a"]
    assert g["primary"]["verdict"] == "inconclusive" and g["retest"] is None
    assert g["final_verdict"] is None and "P0a_topup" in g["final"]["note"] and g["final"]["look"] == 1


def test_cli_p0a_decisive_first_look_is_final(tmp_path):
    g = _cli(tmp_path, _p0a_first_look(inflate=2.0), {"P0a": ["p0a"]}, "--seed", "0")["P0a"]
    assert g["primary"]["verdict"] == "pass" and g["final_verdict"] == "pass" and g["retest"] is None
    assert g["final"] == {"look": 1, "b_continues": True, "stress_profile": None, "note": "B continues"}


def test_cli_p0a_retest_pools_six_attempts_and_is_final(tmp_path):
    first = _p0a_first_look()
    rows = first + _topup(first, q4_scale=3.0)
    proc = _run(tmp_path, rows, {"P0a": ["p0a"], "P0a_topup": ["p0a-topup"]}, "--seed", "0")
    proc.check_returncode()
    g = json.loads((tmp_path / "g.json").read_text())["P0a"]
    assert g["primary"]["verdict"] == "inconclusive"  # the first look is kept as it was
    assert g["design_mismatches"] == [] and g["retest"]["design_mismatches"] == []  # 3 per profile, then 6
    assert g["retest"]["primary"]["verdict"] == "pass" and g["final_verdict"] == "pass"
    assert g["final"]["look"] == 2 and '"final": "pass"' in proc.stdout
    assert set(g["retest"]["flagged_share"]) == {"BF16", "Q4_K_M"}
    short = _cli(tmp_path, rows[:-1], {"P0a": ["p0a"], "P0a_topup": ["p0a-topup"]}, "--seed", "0")["P0a"]
    assert short["retest"]["design_mismatches"] == [[rows[-1]["task_id"], rows[-1]["profile"], 5]]


def test_cli_p0a_still_inconclusive_keeps_b_with_q3_stress(tmp_path):
    first = _p0a_first_look()
    g = _cli(tmp_path, first + _topup(first), {"P0a": ["p0a"], "P0a_topup": ["p0a-topup"]}, "--seed", "0")["P0a"]
    assert g["retest"]["primary"]["verdict"] == "inconclusive" and g["final_verdict"] == "inconclusive"
    assert g["final"]["b_continues"] is True and g["final"]["stress_profile"] == "Q3_K_M" and g["final"]["look"] == 2


def test_cli_p0a_refuses_a_topup_outside_the_protocol(tmp_path):
    first = _p0a_first_look()
    m = {"P0a": ["p0a"], "P0a_topup": ["p0a-topup"]}
    decisive = _p0a_first_look(inflate=2.0)
    cases = [
        (decisive + _topup(decisive), m, "only an inconclusive"),
        (first + _topup(first) + _topup(first, run_id="p0a-topup2"), {**m, "P0a_topup": ["p0a-topup", "p0a-topup2"]},
         "third look"),
        (first + _topup(first) + [{**first[0], "run_id": "p0a-topup", "profile": "Q8_0", "attempt_id": "x"}], m,
         "BF16 and Q4_K_M on heldout100 only"),
        (first + _topup(first), {**m, "P0a_topup": ["p0a-topup", "p0a"]}, "both P0a and P0a_topup"),
    ]
    for rows, manifest, msg in cases:
        proc = _run(tmp_path, rows, manifest, "--seed", "0")
        assert proc.returncode != 0 and msg in proc.stderr, (msg, proc.stderr)
