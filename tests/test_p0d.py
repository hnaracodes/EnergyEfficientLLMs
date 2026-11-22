"""Tests for P0d: the score-choice rule, replay pairs, the DiD test, its power, and the pool scorer.

Everything runs on synthetic data with a planted answer, on CPU, with no models. The last test is
an optional real-GGUF run that skips unless QTRACE_KL_SCORE, QTRACE_GGUF_P and QTRACE_GGUF_Q are set
(the pattern of tests/test_kl_gguf.py). Run:  python -m pytest tests/test_p0d.py -q
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace.common import read_jsonl
from qtrace.p0d import (
    CANDIDATES,
    average_ranks,
    candidate_stats,
    choose_score,
    did_kl_mean,
    did_q4,
    did_test,
    join_runs,
    power,
    select_pairs,
    spearman,
    spearman_brown,
    within_task,
)

P0B = {"p0b"}  # the P0b run ids of the manifest; attempt() writes run_id "p0b"
MALFORMED = ("misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attempt(aid, task, turns=10, output=3000, malformed=0, lst="pool200", profile="BF16", resolved=True,
            touched=False, episode_path=None, run_id="p0b", end_reason="submitted"):
    """An attempt record in the agreed format, with only the fields P0d reads filled in realistically."""
    grading = {"fail_to_pass_ok": 1, "fail_to_pass_total": 1, "pass_to_pass_ok": 5, "pass_to_pass_total": 5,
               "test_files_touched": touched}
    return {"attempt_id": aid, "run_id": run_id, "task_id": task, "list": lst, "repo": "pandas", "profile": profile,
            "runtime": "vllm", "seed": 0, "resolved": resolved, "grading": grading, "end_reason": end_reason,
            "turns": turns, "tool_calls": turns, "repairs": 0,
            "malformed": {k: (malformed if k == "wrong_name" else 0) for k in MALFORMED},
            "tokens": {"output": output, "input_uncached": 1000, "input_cached": 50_000, "server_reported_cached": 0},
            "episode_path": episode_path or f"{aid}.npz"}


def score_row(aid, task, probe, vals, halves, entropy, entropy_decision=None):
    """A scores.jsonl row: ids, GGUF paths and every key of qtrace.kl.score_stats."""
    return {"attempt_id": aid, "task_id": task, "probe": probe, "gguf_p": "bf16.gguf", "gguf_q": f"{probe}.gguf",
            "n_scored": 1000, "n_decision": 100, "n_command": 50, **vals, "kl_sum_decision": 1.0,
            "kl_sum_all": 10.0, "flip_rate": 0.01, "entropy_mean": entropy,
            "entropy_decision": entropy if entropy_decision is None else entropy_decision,
            "nll_p_mean": 0.3, "nll_q_mean": 0.35,
            "halves": {h: {"n_scored": 500, **halves[h]} for h in ("even", "odd")}}


def planted(role, latent, sd, rng):
    """A positive score: exp of the role's latent value plus noise, or pure noise."""
    if role == "noise":
        return float(np.exp(rng.normal()))
    return float(np.exp(latent[role] + sd * rng.normal()))


def make_pool(n_tasks=40, runs=6, seed=0, roles=None, task_prefix="task"):
    """Attempts, scores and est_joules rows with a planted role per candidate.

    "entropy": the score tracks BF16 entropy (fails |rho| <= 0.8). "noise": independent in every half
    and probe (fails split-half and probe agreement). "trait": a stable per-run property plus small
    noise, unrelated to length, joules and entropy (passes).
    """
    roles = roles or {"kl_decision": "entropy", "kl_command": "noise", "kl_mean": "trait"}
    rng = np.random.default_rng(seed)
    attempts, scores, joules = [], [], []
    for t in range(n_tasks):
        task = f"{task_prefix}{t}"
        for k in range(runs):
            aid = f"{task}-r{k}"
            turns = int(rng.integers(5, 60))
            output = turns * 300 + int(rng.integers(0, 500))
            entropy, trait = float(rng.normal()), float(rng.normal())
            attempts.append(attempt(aid, task, turns=turns, output=output, malformed=int(rng.integers(0, 3))))
            joules.append({"attempt_id": aid, "est_joules_q4": 2.0 * output + 50.0 * turns})

            latent = {"entropy": entropy, "trait": trait}
            for probe in ("Q3_K_M", "Q4_K_M"):
                vals = {c: planted(roles[c], latent, 0.1, rng) for c in CANDIDATES}
                halves = {h: {c: planted(roles[c], latent, 0.3, rng) for c in CANDIDATES} for h in ("even", "odd")}
                scores.append(score_row(aid, task, probe, vals, halves, float(np.exp(entropy))))
    return attempts, scores, joules


# Ranks, Spearman and Spearman-Brown.

def test_average_ranks_split_ties():
    assert average_ranks([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert average_ranks([3, 3, 3]).tolist() == [2.0, 2.0, 2.0]


def test_spearman_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    x, y = [1, 2, 2, 3, 5, 5, 5], [2, 1, 4, 4, 3, 6, 6]
    assert abs(spearman(x, y) - stats.spearmanr(x, y).statistic) < 1e-12


def test_spearman_is_undefined_when_a_variable_is_constant():
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2, 3], [4, 4, 4]) is None


def test_spearman_brown_arithmetic():
    assert spearman_brown(0.0) == 0.0 and spearman_brown(1.0) == 1.0
    assert abs(spearman_brown(0.5) - 2 / 3) < 1e-12
    assert spearman_brown(-1.0) is None


def test_split_half_corrects_the_mean_of_per_task_rhos():
    # Task a: halves in the same order (rho 1). Task b: one swap (rho 0.5). Mean 0.75 -> 2 * 0.75 / 1.75.
    halves = {"a": ([1, 2, 3], [1, 2, 3]), "b": ([1, 2, 3], [1, 3, 2])}
    attempts, scores = [], []
    for task, (even, odd) in halves.items():
        for k in range(3):
            aid = f"{task}{k}"
            attempts.append(attempt(aid, task))
            h = {"even": {c: float(even[k]) for c in CANDIDATES}, "odd": {c: float(odd[k]) for c in CANDIDATES}}
            scores.append(score_row(aid, task, "Q3_K_M", {c: float(k) for c in CANDIDATES}, h, 1.0))
    runs, _ = join_runs(attempts, scores, run_ids=P0B)
    split = choose_score(runs, missing_q3=0)["candidates"]["kl_mean"]["split_half"]
    assert split["rho"]["mean"] == 0.75 and split["rho"]["n_tasks"] == 2
    assert abs(split["spearman_brown_mean"] - 1.5 / 1.75) < 1e-12


def test_within_task_minimum_runs_and_constant_tasks():
    groups = {"ok": [(1, 1), (2, 3), (3, 2)], "two": [(1, 1), (2, 2)], "flat": [(1, 5), (2, 5), (3, 5)],
              "gap": [(1, 1), (2, None), (3, 3)]}
    s = within_task(groups)
    assert s["n_tasks"] == 1 and s["mean"] == 0.5 and s["median"] == 0.5
    assert s["n_tasks_too_few_runs"] == 2 and s["n_tasks_constant"] == 1 and s["n_runs_missing_value"] == 1


# The score-choice rule.

def test_rule_chooses_first_passing_candidate():
    attempts, scores, joules = make_pool()
    runs, info = join_runs(attempts, scores, joules, run_ids=P0B)
    out = choose_score(runs, missing_q3=0)
    c = out["candidates"]
    assert c["kl_decision"]["failed"] == ["entropy_bf16"]  # tracks entropy, otherwise fine
    assert set(c["kl_command"]["failed"]) == {"split_half", "probe_agreement"}  # noise
    assert c["kl_mean"]["pass"] and not c["kl_decision"]["pass"] and not c["kl_command"]["pass"]
    assert out["chosen"] == "kl_mean" and out["decided"]
    assert c["kl_decision"]["against"]["entropy_bf16"]["mean"] > 0.8
    assert info["n_runs"] == 240 and info["n_tasks"] == 40


def test_rule_order_prefers_the_primary_score():
    attempts, scores, joules = make_pool(roles={c: "trait" for c in CANDIDATES})
    out = choose_score(join_runs(attempts, scores, joules, run_ids=P0B)[0], missing_q3=0)
    assert all(out["candidates"][c]["pass"] for c in CANDIDATES) and out["chosen"] == "kl_decision"


def test_no_passing_candidate_answers_h3_negatively():
    roles = {"kl_decision": "entropy", "kl_command": "noise", "kl_mean": "noise"}
    attempts, scores, joules = make_pool(roles=roles)
    out = choose_score(join_runs(attempts, scores, joules, run_ids=P0B)[0], missing_q3=0)
    assert out["chosen"] is None and out["decided"] and "H3" in out["verdict"]


def test_missing_joules_leaves_the_rule_undecided():
    attempts, scores, _ = make_pool(roles={c: "trait" for c in CANDIDATES})
    out = choose_score(join_runs(attempts, scores, run_ids=P0B)[0], have_joules=False, missing_q3=0)
    first = out["candidates"]["kl_decision"]
    assert first["checks"]["est_joules_q4"] is None and first["not_run"] == ["est_joules_q4"]
    assert first["pass"] is False and "not_run" in first["against"]["est_joules_q4"]
    assert out["chosen"] is None and not out["decided"] and "incomplete" in out["verdict"]


def test_a_definite_failure_needs_no_joules():
    attempts, scores, _ = make_pool()
    out = choose_score(join_runs(attempts, scores, run_ids=P0B)[0], have_joules=False, missing_q3=0)
    assert out["candidates"]["kl_decision"]["failed"] == ["entropy_bf16"]
    assert out["chosen"] is None and not out["decided"] and "kl_mean" in out["verdict"]  # kl_mean undecided


def test_partial_q4_or_joules_coverage_leaves_the_rule_undecided():
    # Mid-run state: the scorer writes every Q3_K_M row before any Q4_K_M row, so Q4 rows cover 3 of 40 tasks.
    attempts, scores, joules = make_pool()
    partial = [s for s in scores if s["probe"] == "Q3_K_M" or s["task_id"] in {"task0", "task1", "task2"}]
    runs, info = join_runs(attempts, partial, joules, run_ids=P0B)
    out = choose_score(runs, missing_q3=0)
    assert info["n_runs_without_q4"] == 222 and out["candidates"]["kl_mean"]["not_run"] == ["probe_agreement"]
    assert out["chosen"] is None and not out["decided"] and "probe_agreement" in out["verdict"]
    runs, info = join_runs(attempts, scores, [j for j in joules if j["attempt_id"].startswith(("task0-", "task1-"))],
                           run_ids=P0B)
    out = choose_score(runs, missing_q3=0)
    assert info["n_runs_without_est_joules"] == 228 and out["candidates"]["kl_mean"]["not_run"] == ["est_joules_q4"]
    assert out["chosen"] is None and not out["decided"]


def test_partial_halves_coverage_leaves_split_half_not_run():
    # Halves in 3 of 40 tasks must not decide split-half reliability for the pool.
    attempts, scores, joules = make_pool()
    keep = {"task0", "task1", "task2"}
    partial = [s if s["task_id"] in keep else {k: v for k, v in s.items() if k != "halves"} for s in scores]
    runs, info = join_runs(attempts, partial, joules, run_ids=P0B)
    out = choose_score(runs, missing_q3=0)
    assert info["n_runs_without_halves"] == 222
    assert out["candidates"]["kl_mean"]["checks"]["split_half"] is None
    assert "not_run" in out["candidates"]["kl_mean"]["split_half"]
    assert out["chosen"] is None and not out["decided"] and "split_half" in out["verdict"]


def test_missing_q3_rows_leave_the_rule_undecided():
    # 240 eligible attempts, Q3_K_M rows for 120 of them: the rule must not decide on the scored half.
    attempts, scores, joules = make_pool()
    scored = {f"task{t}" for t in range(20)}
    runs, info = join_runs(attempts, [s for s in scores if s["task_id"] in scored], joules, run_ids=P0B)
    assert info["n_eligible_without_q3"] == 120
    out = choose_score(runs, missing_q3=info["n_eligible_without_q3"])
    assert out["chosen"] is None and not out["decided"] and "120" in out["verdict"]
    assert choose_score(runs, missing_q3=0)["chosen"] == "kl_mean"  # the same runs decide once nothing is missing


def test_choose_score_requires_the_missing_q3_count():
    # Owner review of PR #12: a caller that leaves the count out must not get a decision by default.
    attempts, scores, joules = make_pool(n_tasks=4, runs=3)
    with pytest.raises(TypeError, match="missing_q3"):
        choose_score(join_runs(attempts, scores, joules, run_ids=P0B)[0])


def test_runs_on_other_lists_are_not_missing_and_never_scored():
    attempts, scores, joules = make_pool(n_tasks=4, runs=3)
    other = attempt("other", "task0", lst="pool_extra")
    _, info = join_runs(attempts + [other], scores, joules, run_ids=P0B)
    assert info["n_eligible_without_q3"] == 0
    _, info = join_runs(attempts + [other], scores, joules, lists=("pool200", "pool_extra"), run_ids=P0B)
    assert info["n_eligible_without_q3"] == 1
    row = score_row("other", "task0", "Q3_K_M", {c: 1.0 for c in CANDIDATES},
                    {"even": {c: 1.0 for c in CANDIDATES}, "odd": {c: 1.0 for c in CANDIDATES}}, 1.0)
    with pytest.raises(ValueError, match="pool lists"):
        join_runs(attempts + [other], scores + [row], joules, run_ids=P0B)


def test_join_keeps_only_the_p0b_runs_of_the_manifest():
    # Owner review of PR #12: P0c's BF16 attempts on pool200 are another draw and are told apart by run_id.
    attempts, scores, joules = make_pool(n_tasks=4, runs=3)
    p0c = [attempt("c0", "task0", run_id="p0c"), attempt("c1", "task1", run_id="p0c", touched=True)]
    _, info = join_runs(attempts + p0c, scores, joules, run_ids=P0B)
    assert info["n_runs"] == 12 and info["n_eligible_without_q3"] == 0 and info["flagged_excluded"] == {}
    row = score_row("c0", "task0", "Q3_K_M", {c: 1.0 for c in CANDIDATES},
                    {"even": {c: 1.0 for c in CANDIDATES}, "odd": {c: 1.0 for c in CANDIDATES}}, 1.0)
    with pytest.raises(ValueError, match="not a P0b run"):
        join_runs(attempts + p0c, scores + [row], joules, run_ids=P0B)


def boundary_pool(field, values):
    """One task with five runs. Every score ranks them 1..5, under both probes and in both halves,
    except `field` ("odd", "Q4_K_M" or "entropy"), which gives run k the value values[k]."""
    attempts, scores, joules = [], [], []
    for k, v in enumerate(values):
        aid = f"b{k}"
        rank, planted = {c: float(k + 1) for c in CANDIDATES}, {c: float(v) for c in CANDIDATES}
        attempts.append(attempt(aid, "t"))
        joules.append({"attempt_id": aid, "est_joules_q4": 1.0})
        entropy = float(v) if field == "entropy" else 1.0
        odd = planted if field == "odd" else rank
        scores.append(score_row(aid, "t", "Q3_K_M", rank, {"even": rank, "odd": odd}, entropy))
        scores.append(score_row(aid, "t", "Q4_K_M", planted if field == "Q4_K_M" else rank,
                                {"even": rank, "odd": rank}, entropy))
    return join_runs(attempts, scores, joules, run_ids=P0B)[0]


@pytest.mark.parametrize("field, values, check, rho, passes", [
    ("odd", [4, 1, 2, 3, 5], "split_half", 0.4, True),  # Spearman-Brown 0.57
    ("odd", [3, 1, 5, 2, 4], "split_half", 0.3, False),  # Spearman-Brown 0.46
    ("entropy", [4, 5, 3, 2, 1], "entropy_bf16", -0.9, False),
    ("entropy", [3, 1, 2, 4, 5], "entropy_bf16", 0.7, True),
    ("Q4_K_M", [4, 1, 2, 3, 5], "probe_agreement", 0.4, False),
    ("Q4_K_M", [2, 1, 4, 5, 3], "probe_agreement", 0.6, True),
])
def test_rule_thresholds_at_their_boundaries(field, values, check, rho, passes):
    c = choose_score(boundary_pool(field, values), missing_q3=0)["candidates"]["kl_mean"]
    stat = {"split_half": c["split_half"]["rho"], "entropy_bf16": c["against"]["entropy_bf16"],
            "probe_agreement": c["probe_agreement"]}[check]
    assert stat["mean"] == pytest.approx(rho) and c["checks"][check] is passes


def runs_in_orders(field, orders):
    """Runs whose scores rank task t's runs 1..n and whose `field` ("q4", "odd" or "turns") puts them in
    orders[t], so that task's rho is the Spearman of 1..n against orders[t]."""
    runs = []
    for t, order in enumerate(orders):
        for k, v in enumerate(order):
            rank = {c: float(k + 1) for c in CANDIDATES}
            run = {"attempt_id": f"t{t}-{k}", "task_id": f"t{t}", "turns": 1, "output_tokens": 1,
                   "est_joules_q4": 1.0, "entropy_bf16": 1.0, "score": rank, "q4": rank, "even": rank, "odd": rank}
            run[field] = v if field == "turns" else {c: float(v) for c in CANDIDATES}
            runs.append(run)
    return runs


@pytest.mark.parametrize("field, orders, check, mean", [
    # rhos -1/2, 1, 1: the float mean is 0.49999999999999994
    ("q4", [(2, 3, 1), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)], "probe_agreement", 0.5),
    # rhos -1/2, 1/2, 1: the float mean is 0.33333333333333326
    ("odd", [(2, 3, 1), (1, 3, 5, 2, 4), (1, 2, 3, 4, 5)], "split_half", 1 / 3),
    # rhos 0.7, 0.9, 0.8 and their negatives; the float mean is 0.7999999999999998, inside the bound either way
    ("turns", [(1, 2, 4, 5, 3), (1, 2, 3, 5, 4), (1, 2, 4, 3)], "turns", 0.8),
    ("turns", [(5, 4, 2, 1, 3), (5, 4, 3, 1, 2), (4, 3, 1, 2)], "turns", -0.8),
])
def test_rule_thresholds_hold_exactly_at_the_boundary(field, orders, check, mean):
    c = candidate_stats(runs_in_orders(field, orders), "kl_mean")
    stat = {"probe_agreement": c["probe_agreement"], "split_half": c["split_half"]["rho"],
            "turns": c["against"]["turns"]}[check]
    assert stat["n_tasks"] == 3 and stat["mean"] == pytest.approx(mean, abs=1e-12)
    assert c["checks"][check] is True


@pytest.mark.parametrize("mean", [0.8000000000000002, -0.8000000000000002])
def test_confound_threshold_tolerates_a_mean_a_few_ulps_past_0_8(monkeypatch, mean):
    # No 3- to 5-run rho lattice point found lands above 0.8 with this module's Spearman (the triple above
    # averages to 0.7999999999999998), so the mean is planted directly.
    from qtrace import p0d
    real = p0d.within_task
    monkeypatch.setattr(p0d, "within_task", lambda g, m: {**real(g, m), "mean": mean})
    assert candidate_stats(runs_in_orders("turns", [(1, 2, 3)] * 3), "kl_mean")["checks"]["turns"] is True


def test_tasks_below_three_runs_count_for_pairs_but_not_for_the_rule():
    a1, s1, j1 = make_pool(n_tasks=10, runs=6)
    a2, s2, j2 = make_pool(n_tasks=5, runs=2, seed=1, task_prefix="pair")
    runs, _ = join_runs(a1 + a2, s1 + s2, j1 + j2, run_ids=P0B)
    turns = choose_score(runs, missing_q3=0)["candidates"]["kl_mean"]["against"]["turns"]
    assert turns["n_tasks"] == 10 and turns["n_tasks_too_few_runs"] == 5
    assert select_pairs(runs, "kl_mean")["n_tasks"] == 15


# Joining attempts and scores (input validation and the test-file tripwire).

def test_flagged_runs_are_excluded_and_counted():
    attempts, scores, joules = make_pool(n_tasks=3, runs=3)
    attempts = [attempt(a["attempt_id"], a["task_id"], touched=True) if a["attempt_id"] == "task0-r0" else a
                for a in attempts]
    # The scorer writes no rows for flagged runs, so the count has to come from the attempt records.
    attempts += [attempt("unscored", "task1", touched=True), attempt("failed", "task1", touched=True, resolved=False),
                 attempt("held", "h0", lst="heldout100", touched=True)]
    runs, info = join_runs(attempts, scores, joules, run_ids=P0B)
    assert info["flagged_excluded"] == {"pool200": 2} and len(runs) == 8
    assert "task0-r0" not in {r["attempt_id"] for r in runs}


@pytest.mark.parametrize("change, message", [
    ({"list": "heldout100"}, "held-out"),
    ({"profile": "Q4_K_M"}, "BF16"),
    ({"resolved": False}, "resolved"),
])
def test_join_rejects_runs_that_cannot_be_pool_candidates(change, message):
    attempts, scores, _ = make_pool(n_tasks=2, runs=3)
    attempts = [{**attempts[0], **change}] + attempts[1:]
    with pytest.raises(ValueError, match=message):
        join_runs(attempts, scores, run_ids=P0B)


def test_join_rejects_unknown_attempts_duplicates_and_bad_numbers():
    attempts, scores, _ = make_pool(n_tasks=2, runs=3)
    with pytest.raises(ValueError, match="not in the attempts"):
        join_runs(attempts[1:], scores, run_ids=P0B)
    with pytest.raises(ValueError, match="duplicate"):
        join_runs(attempts, scores + scores[:1], run_ids=P0B)
    with pytest.raises(ValueError, match="kl_mean"):
        join_runs(attempts, [{**scores[0], "kl_mean": float("nan")}] + scores[1:], run_ids=P0B)
    with pytest.raises(ValueError, match="turns"):
        join_runs([{k: v for k, v in attempts[0].items() if k != "turns"}] + attempts[1:], scores, run_ids=P0B)
    with pytest.raises(ValueError, match="turns"):
        join_runs([{**attempts[0], "turns": None}] + attempts[1:], scores, run_ids=P0B)
    with pytest.raises(ValueError, match="entropy_mean"):
        join_runs(attempts, [{**scores[0], "entropy_mean": None}] + scores[1:], run_ids=P0B)
    with pytest.raises(ValueError, match="entropy_decision"):
        join_runs(attempts, [{**scores[0], "entropy_decision": None}] + scores[1:], run_ids=P0B)
    with pytest.raises(ValueError, match="entropy_decision"):
        join_runs(attempts, [{k: v for k, v in scores[0].items() if k != "entropy_decision"}] + scores[1:],
                  run_ids=P0B)
    empty = {**scores[0], "entropy_mean": None, "entropy_decision": None, **dict.fromkeys(CANDIDATES)}
    join_runs(attempts, [empty] + scores[1:], run_ids=P0B)  # an empty run nulls every mean


def test_join_skips_and_counts_infrastructure_errors():
    attempts, scores, joules = make_pool(n_tasks=3, runs=3)
    # A resolved BF16 attempt that ended in error, still in an older scores file: never a run, as in p1_select.
    attempts = [{**a, "end_reason": "error", "grading": None} if a["attempt_id"] == "task0-r0" else a
                for a in attempts]
    attempts.append(attempt("unscored-error", "task1", end_reason="error"))  # no score row and not "missing"
    runs, info = join_runs(attempts, scores, joules, run_ids=P0B)
    assert info["infra_error_excluded"] == {"pool200": 2} and len(runs) == 8
    assert "task0-r0" not in {r["attempt_id"] for r in runs} and info["n_eligible_without_q3"] == 0


def test_join_sums_malformed_over_the_fixed_categories():
    attempts, scores, _ = make_pool(n_tasks=2, runs=3)
    runs, _ = join_runs(attempts, scores, run_ids=P0B)
    assert {r["attempt_id"]: r["malformed"] for r in runs}["task0-r0"] == attempts[0]["malformed"]["wrong_name"]
    extra = {**attempts[0], "malformed": {**attempts[0]["malformed"], "repaired": 4}}
    short = {**attempts[0], "malformed": {k: 0 for k in MALFORMED[1:]}}
    for bad in (extra, short):
        with pytest.raises(ValueError, match="malformed has keys"):
            join_runs([bad] + attempts[1:], scores, run_ids=P0B)


def test_tripwire_field_is_read_strictly():
    attempts, scores, _ = make_pool(n_tasks=2, runs=3)
    grading = {k: v for k, v in attempts[0]["grading"].items() if k != "test_files_touched"}
    with pytest.raises(KeyError):
        join_runs([{**attempts[0], "grading": grading}] + attempts[1:], scores, run_ids=P0B)
    with pytest.raises(TypeError, match="must be a bool"):
        join_runs([{**attempts[0], "grading": {**grading, "test_files_touched": None}}] + attempts[1:], scores,
                  run_ids=P0B)


# Replay pairs.

def test_pairs_take_lowest_and_highest_with_attempt_id_tie_break():
    rows = {"x": [("x-b", 0.1), ("x-a", 0.1), ("x-c", 0.5), ("x-d", 0.3)], "y": [("y-a", 0.2), ("y-b", 0.2)],
            "z": [("z-a", 0.4)]}
    attempts, scores = [], []
    for task, runs in rows.items():
        for i, (aid, s) in enumerate(runs):
            attempts.append(attempt(aid, task, turns=10 + i, malformed=i))
            vals = {c: s for c in CANDIDATES}
            scores.append(score_row(aid, task, "Q3_K_M", vals, {"even": vals, "odd": vals}, 1.0 + i, 3.0 * i))
    p = select_pairs(join_runs(attempts, scores, run_ids=P0B)[0], "kl_decision")
    by_task = {t["task_id"]: t for t in p["tasks"]}
    assert p["n_tasks"] == 2 and p["n_tasks_one_run"] == 1
    assert (by_task["x"]["low"], by_task["x"]["high"]) == ("x-a", "x-c")
    # low (i=1) minus high (i=2); the entropy covariate is the decision-position entropy, not entropy_mean
    assert by_task["x"]["diff"] == {"entropy": -3.0, "turns": -1, "malformed": -1}
    assert (by_task["y"]["low"], by_task["y"]["high"]) == ("y-a", "y-b") and by_task["y"]["spread"] == 0.0
    assert (by_task["x"]["low_kl_mean"], by_task["x"]["high_kl_mean"]) == (0.1, 0.5)  # frozen for did_kl_mean
    assert p["spread"]["max"] == 0.4 and p["n_tasks_zero_spread"] == 1


# The DiD test.

def did_rows(n_tasks, seeds, q_low, q_high, b=0.7, seed=0, invalid=()):
    """Continuation rows with binomial successes: BF16 at rate b on both prefixes."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        for prefix, q in (("low", q_low), ("high", q_high)):
            for profile, rate in (("BF16", b), ("Q3_K_M", q)):
                rows.append({"task_id": f"t{t}", "prefix": prefix, "profile": profile,
                             "successes": int(rng.binomial(seeds, rate)), "attempts": seeds,
                             "replay_valid": f"t{t}" not in invalid,
                             "low_attempt_id": f"t{t}-lo", "high_attempt_id": f"t{t}-hi"})
    return rows


def frozen_pairs(n_tasks):
    """score_choice.json's pairs["tasks"] for did_rows' tasks."""
    cov = noise_covariates(n_tasks)
    return [{"task_id": f"t{t}", "low": f"t{t}-lo", "high": f"t{t}-hi", "diff": cov[f"t{t}"]} for t in range(n_tasks)]


def noise_covariates(n_tasks, seed=1):
    rng = np.random.default_rng(seed)
    return {f"t{t}": {"entropy": float(rng.normal()), "turns": int(rng.integers(-10, 11)),
                      "malformed": int(rng.integers(-2, 3))} for t in range(n_tasks)}


def test_did_detects_a_planted_effect():
    out = did_test(did_rows(80, 8, q_low=0.65, q_high=0.4), noise_covariates(80))
    assert out["p_one_sided"] < 0.05 and abs(out["did_mean"] - 0.25) < 0.06
    assert out["decision"] == "pass" and out["adjusted"]["estimate"] > 0.5 * out["did_mean"]


def test_did_null_is_not_detected_and_fails_the_minimum_effect():
    out = did_test(did_rows(200, 8, q_low=0.5, q_high=0.5), noise_covariates(200))
    assert out["p_one_sided"] >= 0.05 and out["upper95_one_sided"] < 0.10
    assert out["decision"] == "fail"


def exact_rows(did_by_task, b=0.6, q_high=0.5, n=1000):
    """Rows whose success rates are exact to 1/n, so the per-task DiD is the planted value."""
    rows = []
    for t, d in did_by_task.items():
        for prefix, q in (("low", q_high + d), ("high", q_high)):
            for profile, rate in (("BF16", b), ("Q3_K_M", q)):
                rows.append({"task_id": t, "prefix": prefix, "profile": profile, "successes": round(rate * n),
                             "attempts": n, "replay_valid": True})
    return rows


def test_adjustment_removes_a_planted_confound():
    rng = np.random.default_rng(3)
    turns = {f"t{i}": float(rng.uniform(0.5, 1.5)) for i in range(60)}
    cov = {t: {"entropy": float(rng.normal()), "turns": x, "malformed": 0} for t, x in turns.items()}
    confounded = did_test(exact_rows({t: 0.15 * x for t, x in turns.items()}), cov)
    assert confounded["did_mean"] > 0.12 and confounded["p_one_sided"] < 0.05
    assert abs(confounded["adjusted"]["estimate"]) < 0.01
    assert list(confounded["adjusted"]["dropped"]) == ["malformed"]
    assert confounded["adjusted"]["note"] is None  # all zero: no pair's runs differ in it, so nothing to separate
    assert confounded["decision"] == "inconclusive"
    real = did_test(exact_rows({t: 0.2 + 0.1 * x for t, x in turns.items()}), cov)
    assert abs(real["adjusted"]["estimate"] - 0.2) < 0.01 and real["decision"] == "pass"


def test_adjustment_drops_constant_and_collinear_covariates():
    # A real effect at p = 0 with turns constant at 2 and malformed a multiple of entropy: both are dropped and
    # named, and the estimate exists, so the decision is not "inconclusive" for want of an adjustment.
    rng = np.random.default_rng(4)
    tasks = [f"t{i}" for i in range(30)]
    cov = {}
    for t in tasks:
        e = float(rng.normal())
        cov[t] = {"entropy": e, "turns": 2, "malformed": 3 * e}
    out = did_test(exact_rows({t: 0.3 for t in tasks}), cov)
    adj = out["adjusted"]
    assert set(adj["dropped"]) == {"turns", "malformed"} and "constant" in adj["dropped"]["turns"]
    assert "collinear" in adj["dropped"]["malformed"] and set(adj["coef"]) == {"entropy"}
    # Owner review of PR #12: a non-zero constant is absorbed by the intercept, and the result says so.
    assert "not separated from the effect of turns" in adj["note"] and "malformed" not in adj["note"]
    assert abs(adj["estimate"] - 0.3) < 1e-9 and out["p_one_sided"] == 0 and out["decision"] == "pass"
    few = did_test(exact_rows({t: 0.3 for t in tasks[:4]}), {t: noise_covariates(4)[f"t{i}"] for i, t in
                                                             enumerate(tasks[:4])})
    assert few["adjusted"]["estimate"] is None and "degrees of freedom" in few["adjusted"]["reason"]


def test_did_reports_the_liberal_bootstrap_below_50_tasks():
    assert "liberal" in did_test(did_rows(20, 4, 0.6, 0.4))["small_sample_note"]
    assert did_test(did_rows(50, 4, 0.6, 0.4))["small_sample_note"] is None


def test_did_drops_invalid_replays_and_validates_rows():
    out = did_test(did_rows(20, 4, 0.6, 0.4, invalid={"t0", "t1", "t2", "t3", "t4"}), noise_covariates(20))
    assert out["pair_drop_rate"] == 0.25 and out["n_tasks"] == 15 and out["n_pairs_dropped"] == 5
    rows = did_rows(10, 4, 0.6, 0.4)
    with pytest.raises(ValueError, match="no Q3_K_M attempts"):
        did_test([r for r in rows if not (r["task_id"] == "t0" and r["profile"] == "Q3_K_M" and r["prefix"] == "low")])
    with pytest.raises(ValueError, match="successes"):
        did_test([{**rows[0], "successes": 5}] + rows[1:])
    with pytest.raises(ValueError, match="prefix"):
        did_test([{**rows[0], "prefix": "mid"}] + rows[1:])
    with pytest.raises(ValueError, match="covariates"):
        did_test(rows, {"t0": {"entropy": 0, "turns": 0, "malformed": 0}})
    cov = noise_covariates(10)
    with pytest.raises(ValueError, match="covariates t0 entropy"):
        did_test(rows, {**cov, "t0": {**cov["t0"], "entropy": None}})
    assert did_test(rows)["adjusted"] is None and did_test(rows)["decision"] != "pass"


def test_did_counts_bootstrap_means_of_exactly_zero():
    # The reviewer's 20 tasks at 4 seeds: every DiD is a multiple of 0.25, so many bootstrap means are exactly 0.
    # Float weights left some at +1e-17, outside "<= 0", and gave p = 0.0466 instead of 0.0502 at seed 0.
    did = [-.25, .25, .5, 0, .5, 0, 0, .25, .25, .5, .5, -.5, 0, .5, -.25, -.5, 0, .25, .25, .25]
    rows = []
    for i, d in enumerate(did):
        for prefix, q in (("low", 2 + round(4 * d)), ("high", 2)):
            for profile, s in (("BF16", 2), ("Q3_K_M", q)):
                rows.append({"task_id": f"t{i:02d}", "prefix": prefix, "profile": profile, "successes": s,
                             "attempts": 4, "replay_valid": True})
    assert did_test(rows, seed=0)["p_one_sided"] == 0.0502


def test_pair_drop_rate_is_over_every_selected_pair():
    pairs, rows = frozen_pairs(30), did_rows(30, 4, 0.6, 0.4)
    first10 = {f"t{i}" for i in range(10)}
    done = [r for r in rows if r["task_id"] not in first10]
    # The reviewer's case: 10 of 30 pairs produced no rows, which read as a drop rate of 0.
    out = did_test(done, pairs=pairs)
    assert out["n_pairs_selected"] == 30 and out["pair_tasks_without_continuations"] == sorted(first10)
    assert out["n_tasks"] == 20 and out["decision"] == "incomplete"
    # The harness writes an explicit replay_valid false row, without results, for each failed replay.
    failed = [{"task_id": t, "prefix": "low", "profile": "BF16", "replay_valid": False,
               "low_attempt_id": f"{t}-lo", "high_attempt_id": f"{t}-hi"} for t in sorted(first10)]
    out = did_test(failed + done, pairs=pairs)
    assert out["pair_drop_rate"] == 10 / 30 and out["n_pairs_dropped"] == 10 and out["n_tasks"] == 20
    assert out["pair_tasks_without_continuations"] == [] and out["decision"] != "incomplete"
    # A task with only Q4_K_M rows is listed, not dropped without a trace.
    q4 = [{**r, "profile": "Q4_K_M"} for r in rows if r["task_id"] == "t0"]
    out = did_test(failed[1:] + q4 + done, pairs=pairs)
    assert out["pair_tasks_only_other_profiles"] == ["t0"] and out["decision"] == "incomplete"


def test_other_profile_replay_drops_leave_the_primary_test_unchanged():
    # The reviewer's case: a secondary Q4_K_M run with failed replays of its own on t0-t5.
    pairs, rows = frozen_pairs(30), did_rows(30, 4, 0.6, 0.4)
    q4 = [{**r, "profile": "Q4_K_M", "replay_valid": r["task_id"] not in {f"t{i}" for i in range(6)}} for r in rows]
    alone, merged = did_test(rows, pairs=pairs), did_test(rows + q4, pairs=pairs)
    assert merged["rows_ignored_by_profile"] == {"Q4_K_M": len(q4)}
    assert {**merged, "rows_ignored_by_profile": {}} == alone and alone["n_tasks"] == 30


def test_q4_secondary_is_incomplete_until_every_frozen_pair_has_q4_rows():
    pairs, rows = frozen_pairs(30), did_rows(30, 4, 0.6, 0.4)
    assert did_q4(rows, pairs) is None  # no Q4_K_M continuations: no secondary result
    all_q4 = [{**r, "profile": "Q4_K_M"} for r in did_rows(30, 4, 0.7, 0.3, seed=5) if r["profile"] == "Q3_K_M"]
    q4 = [r for r in all_q4 if int(r["task_id"][1:]) < 20]  # 20 of the 30 pairs done so far
    out = did_q4(rows + q4, pairs, n_boot=500)
    assert out["profile"] == "Q4_K_M" and out["secondary"] and out["n_tasks"] == 20
    assert (out["n_pairs_frozen"], out["n_pairs_with_q4_rows"]) == (30, 20) and out["decision"] == "incomplete"
    assert out["pair_tasks_without_continuations"] == sorted(f"t{i}" for i in range(20, 30))
    # A pair whose BF16 replay failed is a drop, not a missing pair, with or without Q4_K_M rows.
    failed = [{**r, "replay_valid": False} for r in rows if r["task_id"] == "t29" and r["profile"] == "BF16"]
    done = did_q4([r for r in rows if r["task_id"] != "t29"] + failed + all_q4, pairs, n_boot=500)
    assert done["n_pairs_dropped"] == 1 and done["pair_tasks_without_continuations"] == []
    assert done["n_tasks"] == 29 and done["decision"] != "incomplete"
    # The same numbers as the primary test run on those rows relabelled Q3_K_M.
    as_q3 = [{**r, "profile": "Q3_K_M"} for r in q4] + [r for r in rows if r["profile"] == "BF16"
                                                       and int(r["task_id"][1:]) < 20]
    want = did_test(as_q3, pairs=pairs[:20], n_boot=500)
    assert out["per_task"] == want["per_task"] and out["p_one_sided"] == want["p_one_sided"]


def test_kl_mean_secondary_reorders_pairs_by_kl_mean():
    pairs, rows = frozen_pairs(30), did_rows(30, 4, 0.6, 0.4)
    assert did_kl_mean(rows, pairs) is None  # pairs frozen before low_kl_mean was recorded
    same = [{**p, "low_kl_mean": 0.1, "high_kl_mean": 0.2} for p in pairs]
    primary = did_test(rows, pairs=pairs, n_boot=500)
    out = did_kl_mean(rows, same, n_boot=500)
    assert out["n_pairs_reordered"] == 0 and out["per_task"] == primary["per_task"] and out["order"] == "kl_mean"
    # kl_mean ranks t0-t9 the other way round; t10 ties, so it drops out and is counted (no attempt_id tie-break).
    swapped = [{**p, "low_kl_mean": 0.3 if i < 10 else 0.1, "high_kl_mean": 0.1 if i <= 10 else 0.2}
               for i, p in enumerate(same)]
    out = did_kl_mean(rows, swapped, n_boot=500)
    assert out["pairs_reordered"] == sorted(f"t{i}" for i in range(10)) and out["n_pairs_reordered"] == 10
    assert (out["pairs_tied"], out["n_pairs_tied"], out["n_tasks"], out["n_pairs_frozen"]) == (["t10"], 1, 29, 30)
    assert out["secondary"] and out["n_pairs_selected"] == 29 and out["decision"] != "incomplete"
    flipped = set(out["pairs_reordered"])
    want = {d["task_id"]: -d["did"] if d["task_id"] in flipped else d["did"] for d in primary["per_task"]
            if d["task_id"] != "t10"}
    assert {d["task_id"]: d["did"] for d in out["per_task"]} == want
    # Per-pair kl_mean values of every frozen pair, under its frozen labels.
    assert len(out["per_pair"]) == 30
    assert out["per_pair"][0] == {"task_id": "t0", "low": "t0-lo", "high": "t0-hi", "low_kl_mean": 0.3,
                                  "high_kl_mean": 0.1, "kl_mean_order": "flipped"}
    assert [p["kl_mean_order"] for p in out["per_pair"][9:12]] == ["flipped", "tied", "kept"]
    assert out["per_pair"][11]["high_kl_mean"] == 0.2
    with pytest.raises(ValueError, match="every frozen pair is tied"):
        did_kl_mean(rows, [{**p, "low_kl_mean": 0.1, "high_kl_mean": 0.1} for p in pairs])
    # Covariates are low minus high, so they flip with the pair: the adjusted fit sees the negated diffs.
    diff = {p["task_id"]: p["diff"] for p in pairs}
    cov = {t: {k: -v for k, v in d.items()} if t in flipped else d for t, d in diff.items()}
    relabelled = [{**r, "prefix": {"low": "high", "high": "low"}[r["prefix"]]} if r["task_id"] in flipped else r
                  for r in rows if r["task_id"] != "t10"]
    assert out["adjusted"] == did_test(relabelled, covariates=cov, n_boot=500)["adjusted"]


def test_continuations_must_name_the_frozen_pair():
    pairs, rows = frozen_pairs(10), did_rows(10, 4, 0.6, 0.4)
    assert did_test(rows, pairs=pairs)["adjusted"]["estimate"] is not None  # covariates come from the pairs
    with pytest.raises(ValueError, match="not the frozen pair"):
        did_test([{**rows[0], "low_attempt_id": "t0-other"}] + rows[1:], pairs=pairs)
    with pytest.raises(ValueError, match="high_attempt_id"):
        did_test([{k: v for k, v in rows[0].items() if k != "high_attempt_id"}] + rows[1:], pairs=pairs)
    with pytest.raises(ValueError, match="not a frozen replay pair"):
        did_test(rows, pairs=pairs[1:])


def test_power_rises_with_the_effect():
    out = power(0.7, 0.5, n_tasks=200, seeds=8, effects=(0.0, 0.10), n_sim=200, n_boot=500)
    by_effect = {e["effect"]: e["power"] for e in out["power"]}
    assert by_effect[0.0] < 0.15 and by_effect[0.10] > 0.9 and out["mde_80"] == 0.10
    paired = power([0.9, 0.5], [0.8, 0.3], n_tasks=10, seeds=4, effects=(0.10,), n_sim=200, n_boot=500)
    assert paired["power"][0]["power"] < 0.5 and paired["mde_80"] is None


def test_power_counts_bootstrap_means_of_exactly_zero():
    # At 4 seeds every DiD is a multiple of 1/4, so many bootstrap means are exactly 0. Float weights left some
    # at +1e-17, outside "<= 0", and gave a null rate of 0.0525 instead of 0.0425 at this seed.
    out = power(0.6, 0.5, n_tasks=20, seeds=4, effects=(0.0,), n_sim=400, n_boot=500, seed=0)
    assert out["power"][0]["power"] == 0.0425


def test_power_plants_the_full_effect_at_the_rate_bounds():
    # Shifted inside [0, 1], a gap of 0.10 at Q3 rate 1.0 or 0.98 is the rates 1.0 and 0.9, and at 0.0 or 0.02 it
    # is 0.1 and 0.0. Clipping each side instead would plant a smaller gap the nearer the rate is to the bound.
    kw = {"n_tasks": 30, "seeds": 4, "effects": (0.10,), "n_sim": 200, "n_boot": 300}
    assert power(0.5, 1.0, **kw)["power"] == power(0.5, 0.98, **kw)["power"]
    assert power(0.5, 0.0, **kw)["power"] == power(0.5, 0.02, **kw)["power"]


def test_power_counts_tasks_after_replay_drops():
    kw = {"seeds": 4, "effects": (0.10,), "n_sim": 200, "n_boot": 300}
    dropped = power(0.7, 0.5, n_tasks=50, drop_rate=0.2, **kw)
    assert dropped["n_tasks_after_drops"] == 40 and dropped["power"] == power(0.7, 0.5, n_tasks=40, **kw)["power"]
    with pytest.raises(ValueError, match="drop_rate"):
        power(0.7, 0.5, n_tasks=50, drop_rate=1.0, **kw)


def test_power_applies_the_adjusted_estimate_condition():
    kw = {"n_tasks": 50, "seeds": 4, "effects": (0.10, 0.30), "n_sim": 200, "n_boot": 300}
    plain = power(0.7, 0.5, **kw)
    assert all(e["power"] == e["power_p_only"] for e in plain["power"]) and not plain["adjusted_condition"]
    noise = power(0.7, 0.5, covariates=list(noise_covariates(50).values()), **kw)
    assert noise["adjusted_condition"] and all(e["power"] <= e["power_p_only"] for e in noise["power"])
    assert noise["power"][1]["power"] > 0.9 * noise["power"][1]["power_p_only"] > 0.5  # noise rarely costs a pass
    # Covariate differences far from 0 make the intercept an extrapolation, so it often loses half the DiD.
    far = [{**c, "turns": 100 + c["turns"] / 10} for c in noise_covariates(50).values()]
    strong = power(0.7, 0.5, covariates=far, **kw)["power"][1]
    assert strong["power"] < strong["power_p_only"] - 0.2


# The pool scorer (scripts/p0d_score_pool.py), with the llama.cpp call replaced.

def write_episode(path, n=60):
    mask = np.zeros(n, dtype=bool)
    mask[10:25] = mask[35:55] = True
    np.savez(path, input_ids=np.arange(n, dtype=np.int32), model_mask=mask,
             tool_call_positions=np.array([12, 37], dtype=np.int32),
             command_spans=np.array([[15, 20], [40, 44]], dtype=np.int32),
             turn_spans=np.array([[10, 25], [35, 55]], dtype=np.int32),
             tool_response_spans=np.array([[25, 35]], dtype=np.int32))


@pytest.fixture
def pool_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run_scorer names the helper and GGUFs relative to here; the scorer hashes them
    for name in ("kl_score", "bf16.gguf", "q3.gguf", "q4.gguf", "other_q3.gguf"):
        (tmp_path / name).write_bytes(name.encode())
    d = tmp_path / "attempts"
    d.mkdir()
    recs = [attempt(f"ok{i}", f"t{i % 3}") for i in range(7)]
    recs += [attempt("flag", "t0", touched=True), attempt("fail", "t0", resolved=False),
             attempt("q4", "t0", profile="Q4_K_M"), attempt("held", "h0", lst="heldout100"),
             attempt("p0c", "t0", run_id="p0c")]  # a P0c BF16 run on pool200: not in the manifest's P0b
    for r in recs:
        write_episode(d / f"{r['attempt_id']}.npz")
    with open(d / "run.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    return d


def fake_scorer(calls, fail_on=None):
    def score(binary, gguf_p, gguf_q, episodes, n_batch=512, entropy_threshold=None, positions=False):
        calls.append((os.path.basename(gguf_q), len(episodes)))
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("helper crashed")
        return [{"n_scored": int(e["model_mask"].sum()), "kl_mean": 0.1, "kl_decision": 0.2, "kl_command": 0.3,
                 "turn_spans": e["turn_spans"], "runtime": {"devices": ["CPU"]},
                 **({"positions": {"idx": np.arange(3), "kl": np.ones(3)}} if positions else {})}
                for e in episodes]
    return score


def run_scorer(mod, d, out, batch=3, **kw):
    return mod.score_pool([str(d / "run.jsonl")], "kl_score", "bf16.gguf",
                          {"Q3_K_M": "q3.gguf", "Q4_K_M": "q4.gguf"}, ["pool200"], str(out), batch=batch,
                          run_ids=P0B, **kw)


def test_scorer_filters_batches_and_resumes(pool_dir, tmp_path, monkeypatch):
    mod = load_script("p0d_score_pool")
    out = tmp_path / "p0d" / "scores.jsonl"
    calls = []
    monkeypatch.setattr(mod, "score_trajectories_gguf", fake_scorer(calls, fail_on=3))
    with pytest.raises(RuntimeError, match="helper crashed"):
        run_scorer(mod, pool_dir, out)
    assert calls == [("q3.gguf", 3), ("q3.gguf", 3), ("q3.gguf", 1)]
    assert len(read_jsonl(out)) == 6  # two batches made it to disk before the crash

    calls.clear()
    monkeypatch.setattr(mod, "score_trajectories_gguf", fake_scorer(calls))
    report = run_scorer(mod, pool_dir, out)
    assert calls == [("q3.gguf", 1), ("q4.gguf", 3), ("q4.gguf", 3), ("q4.gguf", 1)]
    assert report["kept"] == 7 and report["flagged_excluded"] == 1 and report["already_scored"] == 6
    rows = read_jsonl(out)
    assert sorted((r["attempt_id"], r["probe"]) for r in rows) == \
        sorted((f"ok{i}", p) for i in range(7) for p in ("Q3_K_M", "Q4_K_M"))
    r = rows[0]
    assert (r["task_id"], r["gguf_p"], r["n_scored"]) == ("t0", "bf16.gguf", 35)
    assert r["turn_spans"] == [[10, 25], [35, 55]]  # spans reach the scorer as plain lists

    calls.clear()
    assert run_scorer(mod, pool_dir, out)["scored"] == 0 and calls == []


def test_scorer_logs_each_batch_and_writes_positions(pool_dir, tmp_path, monkeypatch):
    import hashlib

    mod = load_script("p0d_score_pool")
    out, pos = tmp_path / "p0d" / "scores.jsonl", tmp_path / "positions"
    monkeypatch.setattr(mod, "score_trajectories_gguf", fake_scorer([]))
    report = run_scorer(mod, pool_dir, out, batch=5, positions_dir=str(pos), llama_commit="b1b770d9",
                        machine="mac-48gb")
    log = read_jsonl(tmp_path / "p0d" / "scores.batches.jsonl")
    assert [(b["probe"], len(b["attempt_ids"])) for b in log] == [("Q3_K_M", 5), ("Q3_K_M", 2), ("Q4_K_M", 5),
                                                                  ("Q4_K_M", 2)]
    assert log[0]["attempt_ids"] == [f"ok{i}" for i in range(5)] and log[0]["n_tokens"] == 5 * 60
    assert log[0]["runtime"] == {"devices": ["CPU"]} and log[0]["llama_commit"] == "b1b770d9"
    assert {b["machine_label"] for b in log} == {"mac-48gb"}
    assert log[0]["n_batch"] == 512 and log[0]["platform"] and log[0]["joules"] is None  # no NVIDIA GPU here
    assert log[2]["sha256"] == {k: hashlib.sha256(name.encode()).hexdigest()
                                for k, name in (("kl_score", "kl_score"), ("gguf_p", "bf16.gguf"), ("gguf_q", "q4.gguf"))}
    assert report["seconds"] == pytest.approx(sum(b["seconds"] for b in log)) and report["joules"] is None
    assert sorted(os.listdir(pos)) == sorted(f"ok{i}.{p}.npz" for i in range(7) for p in ("Q3_K_M", "Q4_K_M"))
    assert np.load(pos / "ok0.Q3_K_M.npz")["kl"].tolist() == [1, 1, 1]
    assert all("positions" not in r for r in read_jsonl(out))
    run_scorer(mod, pool_dir, out)  # nothing left to score: no batch is logged
    assert len(read_jsonl(tmp_path / "p0d" / "scores.batches.jsonl")) == 4


def test_scorer_skips_and_counts_infrastructure_errors(tmp_path):
    mod = load_script("p0d_score_pool")
    recs = [attempt("ok", "t0"), attempt("err", "t0", end_reason="error"),
            attempt("err-failed", "t0", resolved=False, end_reason="error"),
            attempt("err-p0c", "t0", run_id="p0c", end_reason="error")]  # not a P0b run: not counted
    recs[1]["grading"] = recs[2]["grading"] = None  # an infrastructure fault may never reach the grader
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    runs, n_flagged, n_error = mod.pool_runs([str(path)], ["pool200"], P0B)
    assert [r["attempt_id"] for r, _ in runs] == ["ok"] and (n_flagged, n_error) == (0, 2)


def test_scorer_cuts_a_torn_line_and_refuses_other_ggufs(pool_dir, tmp_path, monkeypatch):
    mod = load_script("p0d_score_pool")
    out = tmp_path / "scores.jsonl"
    monkeypatch.setattr(mod, "score_trajectories_gguf", fake_scorer([]))
    run_scorer(mod, pool_dir, out, batch=25)
    out.write_text(out.read_text()[:-1])  # a crash between the last row and its newline: the row is kept
    report = run_scorer(mod, pool_dir, out)
    assert report["scored"] == 0 and len(read_jsonl(out)) == 14 and out.read_text().endswith("}\n")
    with open(out, "a") as f:
        f.write('{"attempt_id": "ok0", "pro')
    report = run_scorer(mod, pool_dir, out)
    assert report["scored"] == 0 and len(read_jsonl(out)) == 14
    with pytest.raises(ValueError, match="different GGUF"):
        mod.score_pool([str(pool_dir / "run.jsonl")], "kl_score", "bf16.gguf",
                       {"Q3_K_M": "other_q3.gguf", "Q4_K_M": "q4.gguf"}, ["pool200"], str(out), run_ids=P0B)


def test_scorer_refuses_an_out_file_that_is_not_a_scores_file(tmp_path):
    mod = load_script("p0d_score_pool")
    path = tmp_path / "attempts.jsonl"
    text = json.dumps({"attempt_id": "a1", "task_id": "t"}) + "\n" + json.dumps({"attempt_id": "a2", "task_id": "t"})
    path.write_text(text)
    with pytest.raises(ValueError, match="not a P0d scores file"):
        mod.resume(str(path), "bf16.gguf", {"Q3_K_M": "q3.gguf"})
    assert path.read_text() == text  # nothing cut


def test_scorer_rejects_held_out_lists(pool_dir, tmp_path):
    mod = load_script("p0d_score_pool")
    with pytest.raises(ValueError, match="held-out"):
        mod.score_pool([str(pool_dir / "run.jsonl")], "kl_score", "bf16.gguf", {"Q3_K_M": "q3.gguf"},
                       ["pool200", "heldout100"], str(tmp_path / "s.jsonl"), run_ids=P0B)


def test_scorer_rejects_malformed_episodes(tmp_path):
    mod = load_script("p0d_score_pool")
    path = tmp_path / "bad.npz"
    write_episode(path)
    good = dict(np.load(path))
    for change, message in (({"model_mask": good["model_mask"][:-1]}, "model_mask"),
                            ({"turn_spans": np.array([[10, 99]], dtype=np.int32)}, "turn_spans"),
                            ({"tool_call_positions": np.array([60], dtype=np.int32)}, "tool_call_positions"),
                            ({"tool_call_positions": np.array([30], dtype=np.int32)}, "outside model_mask")):
        np.savez(path, **{**good, **change})
        with pytest.raises(ValueError, match=message):
            mod.load_episode(path)


def test_scorer_feeds_the_real_score_stats(pool_dir, tmp_path, monkeypatch):
    """compare_gguf replaced, so score_trajectories_gguf and score_stats run on the loaded episodes."""
    from qtrace import kl_gguf

    def fake_compare(binary, gguf_p, gguf_q, runs, n_batch=512, cpu=False):
        return [{**{k: torch.linspace(0.1, 1.0, len(idx)) for k in kl_gguf.FIELDS}, "runtime": {"devices": ["CPU"]}}
                for _, idx in runs]

    mod = load_script("p0d_score_pool")
    monkeypatch.setattr(kl_gguf, "compare_gguf", fake_compare)
    out = tmp_path / "scores.jsonl"
    run_scorer(mod, pool_dir, out, positions_dir=str(tmp_path / "pos"))
    r = read_jsonl(out)[0]
    assert r["n_scored"] == 35 and r["n_command"] == 9 and set(r["halves"]) == {"even", "odd"}
    with np.load(tmp_path / "pos" / "ok0.Q3_K_M.npz") as z:
        assert sorted(z.files) == sorted(("idx",) + kl_gguf.FIELDS) and z["idx"].dtype == np.int32
        assert z["idx"].tolist() == list(range(9, 24)) + list(range(34, 54))  # logit t - 1 of each written t
        assert np.allclose(z["entropy_p"], np.linspace(0.1, 1.0, 35))
    assert r["halves"]["even"]["n_scored"] == 15 and r["kl_decision"] is not None
    assert r["runtime"] == {"devices": ["CPU"]}  # the helper's settings are kept with the scores


# The analysis script end to end.

def write_inputs(tmp_path, attempts, scores, joules):
    """The analysis inputs as files, with a manifest in p0_analyze.py's format naming the P0b run."""
    paths = {"manifest": tmp_path / "manifest.json"}
    paths["manifest"].write_text(json.dumps({"P0b": sorted(P0B), "P0c": ["p0c"]}))
    for name, rows in (("attempts", attempts), ("scores", scores), ("joules", joules)):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    return paths


def test_analyze_writes_score_choice_and_did(tmp_path, capsys):
    attempts, scores, joules = make_pool(n_tasks=30)
    attempts.append(attempt("flagged", "task0", touched=True))  # no score row, as the scorer skips it
    attempts.append(attempt("p0c-run", "task0", run_id="p0c"))  # P0c's draw: never scored, never missing
    paths = write_inputs(tmp_path, attempts, scores, joules)
    out_dir = tmp_path / "p0d"
    mod = load_script("p0d_analyze")
    mod.main([str(paths["attempts"]), "--manifest", str(paths["manifest"]), "--scores", str(paths["scores"]),
              "--est-joules", str(paths["joules"]),
              "--out-dir", str(out_dir), "--power-rates", "0.9,0.5", "0.8,0.3", "--replay-drop-rate", "0.1",
              "--n-boot", "300"])
    choice = json.loads((out_dir / "score_choice.json").read_text())
    assert choice["chosen"] == "kl_mean" and set(choice["candidates"]) == set(CANDIDATES)
    assert all(isinstance(c["pass"], bool) for c in choice["candidates"].values())
    assert choice["pairs"]["n_tasks"] == 30 and {p["seeds"] for p in choice["power"]} == {4, 8}
    assert all(p["p_bf16"] == [0.9, 0.5] and p["n_tasks_after_drops"] == 27 and p["adjusted_condition"]
               for p in choice["power"])
    assert choice["inputs"]["flagged_excluded"] == {"pool200": 1}

    cont = tmp_path / "continuations.jsonl"
    pair = {p["task_id"]: p for p in choice["pairs"]["tasks"]}
    rows = did_rows(30, 4, 0.6, 0.4)
    for r in rows:
        r["task_id"] = r["task_id"].replace("t", "task")
        r["low_attempt_id"], r["high_attempt_id"] = pair[r["task_id"]]["low"], pair[r["task_id"]]["high"]
    cont.write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths["scores"].unlink()  # the DiD step reads the frozen pairs, not the inputs they came from
    mod.main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    did = json.loads((out_dir / "did.json").read_text())
    assert did["n_tasks"] == 30 and did["decision"] in ("pass", "fail", "inconclusive")
    assert did["pair_tasks_without_continuations"] == [] and did["adjusted"]["estimate"] is not None
    assert did["score"] == "kl_mean" and did["n_pairs_selected"] == 30
    # The chosen score is kl_mean here, so the kl_mean-ordered secondary reorders nothing.
    sec = json.loads((out_dir / "did_kl_mean.json").read_text())
    assert sec["secondary"] and sec["n_pairs_reordered"] == 0 and sec["did_mean"] == did["did_mean"]
    assert not (out_dir / "did_q4.json").exists()
    q4 = [{**r, "profile": "Q4_K_M"} for r in rows if r["profile"] == "Q3_K_M"]
    cont.write_text("".join(json.dumps(r) + "\n" for r in rows + q4))
    mod.main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    q4_did = json.loads((out_dir / "did_q4.json").read_text())
    assert q4_did["profile"] == "Q4_K_M" and q4_did["did_mean"] == did["did_mean"] and q4_did["pairs_score"] == "kl_mean"
    assert json.loads((out_dir / "did.json").read_text())["did_mean"] == did["did_mean"]  # the primary is unchanged
    # A later file with one more Q3_K_M seed while the Q4_K_M run is part-way: task0 has Q4_K_M rows on the
    # low prefix only, so did_q4 raises. did.json and did_kl_mean.json are rewritten and no did_q4.json is left.
    extra = {**rows[1], "successes": 4 - rows[1]["successes"]}  # task0, low prefix, Q3_K_M
    assert extra["task_id"] == "task0" and extra["profile"] == "Q3_K_M" and extra["prefix"] == "low"
    partial = [r for r in q4 if not (r["task_id"] == "task0" and r["prefix"] == "high")]
    cont.write_text("".join(json.dumps(r) + "\n" for r in rows + [extra] + partial))
    mod.main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    new = json.loads((out_dir / "did.json").read_text())
    assert new["did_mean"] != did["did_mean"] and not (out_dir / "did_q4.json").exists()
    assert json.loads((out_dir / "did_kl_mean.json").read_text())["did_mean"] == new["did_mean"]
    assert "no Q4_K_M attempts on the high prefix" in capsys.readouterr().out
    # Back to complete Q4_K_M rows, then a file without them: the did_q4.json they produced is removed.
    cont.write_text("".join(json.dumps(r) + "\n" for r in rows + q4))
    mod.main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    assert (out_dir / "did_q4.json").exists()
    cont.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mod.main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    assert not (out_dir / "did_q4.json").exists() and (out_dir / "did_kl_mean.json").exists()


def test_did_step_says_a_constant_covariate_is_not_separated(tmp_path, capsys):
    out_dir = tmp_path / "p0d"
    out_dir.mkdir()
    pairs = [{**p, "diff": {**p["diff"], "turns": 2}} for p in frozen_pairs(20)]
    (out_dir / "score_choice.json").write_text(json.dumps({"chosen": "kl_mean", "decided": True,
                                                           "verdict": "chosen: kl_mean", "pairs": {"tasks": pairs}}))
    cont = tmp_path / "continuations.jsonl"
    cont.write_text("".join(json.dumps(r) + "\n" for r in did_rows(20, 4, 0.6, 0.4)))
    load_script("p0d_analyze").main(["--out-dir", str(out_dir), "--continuations", str(cont), "--n-boot", "300"])
    did = json.loads((out_dir / "did.json").read_text())
    assert "turns" in did["adjusted"]["dropped"] and "not separated from the effect of turns" in did["adjusted"]["note"]
    assert did["adjusted"]["note"] in capsys.readouterr().out


def test_analyze_rerun_removes_a_stale_did_and_says_why_there_are_no_pairs(tmp_path):
    out_dir = tmp_path / "p0d"
    out_dir.mkdir()
    for name in ("did.json", "did_q4.json", "did_kl_mean.json"):
        (out_dir / name).write_text("{}")
    mod = load_script("p0d_analyze")

    def run(roles, with_joules):
        paths = write_inputs(tmp_path, *make_pool(n_tasks=10, roles=roles))
        mod.main([str(paths["attempts"]), "--manifest", str(paths["manifest"]), "--scores", str(paths["scores"]),
                  "--out-dir", str(out_dir)] +
                 (["--est-joules", str(paths["joules"])] if with_joules else []))
        with pytest.raises(SystemExit) as stop:
            mod.main(["--out-dir", str(out_dir), "--continuations", str(paths["attempts"])])
        return str(stop.value)

    assert "undecided" in run({c: "trait" for c in CANDIDATES}, with_joules=False)
    assert not any((out_dir / name).exists() for name in ("did.json", "did_q4.json", "did_kl_mean.json"))
    assert "H3 was answered negatively" in run({c: "noise" for c in CANDIDATES}, with_joules=True)


# Optional: the scorer against real GGUF files (about 18.5 GB peak for the 4B pair).

BIN, P, Q = (os.environ.get(k) for k in ("QTRACE_KL_SCORE", "QTRACE_GGUF_P", "QTRACE_GGUF_Q"))


@pytest.mark.skipif(not (BIN and P and Q), reason="QTRACE_KL_SCORE, QTRACE_GGUF_P and QTRACE_GGUF_Q not set")
def test_scorer_with_real_gguf(pool_dir, tmp_path):
    mod = load_script("p0d_score_pool")
    out = tmp_path / "scores.jsonl"
    mod.score_pool([str(pool_dir / "run.jsonl")], BIN, P, {"Q3_K_M": Q, "Q4_K_M": Q}, ["pool200"], str(out),
                   batch=25, run_ids=P0B)
    rows = read_jsonl(out)
    assert len(rows) == 14 and all(r["kl_mean"] > 0 and r["n_scored"] == 35 for r in rows)
    q3 = {r["attempt_id"]: r["kl_mean"] for r in rows if r["probe"] == "Q3_K_M"}
    q4 = {r["attempt_id"]: r["kl_mean"] for r in rows if r["probe"] == "Q4_K_M"}
    assert all(abs(q3[a] - q4[a]) < 1e-3 for a in q3)  # same file under both probe names
