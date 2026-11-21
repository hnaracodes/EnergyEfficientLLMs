"""Tests for P1 selection on synthetic pool data. Run:  python -m pytest tests/test_p1_select.py -q"""

import json
import os
import random
import subprocess
import sys
from fractions import Fraction

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace.p1_select import ARMS, STABILITY_ARMS, avg_ranks, halves, overlap, pick, read_attempts, select

MAL = ("misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer")
RUN = {"task": "t", "turns": 5, "out": 100, "kl": 0.1, "ent": 1.0, "j": 1000.0, "mal": None, "resolved": True,
       "profile": "BF16", "lst": "pool200", "touched": False, "end": "submitted",
       "unc": None, "cac": None}
_ids = iter(range(10**9))


def att(r, aid):
    """One attempt record in the format agreed in issue #1."""
    grading = {"fail_to_pass_ok": 2, "fail_to_pass_total": 2, "pass_to_pass_ok": 9, "pass_to_pass_total": 9}
    if r["touched"] is not None:
        grading["test_files_touched"] = r["touched"]
    out = r["out"]
    return {"attempt_id": aid, "run_id": "run-p0b", "task_id": r["task"], "list": r["lst"], "repo": "pandas",
            "profile": r["profile"], "runtime": "vllm", "runtime_commit": "abc", "model": "Qwen/Qwen3.5-4B",
            "seed": 1, "sampler": {"temperature": 0.6}, "concurrency": 16, "resolved": r["resolved"],
            "grading": grading, "end_reason": r["end"], "turns": r["turns"], "tool_calls": r["turns"],
            "repairs": 0, "malformed": {**dict.fromkeys(MAL, 0), **(r["mal"] or {})},
            "tokens": {"output": out, "input_uncached": 10 * out if r["unc"] is None else r["unc"],
                       "input_cached": 50 * out if r["cac"] is None else r["cac"],
                       "server_reported_cached": 50 * out},
            "per_turn": [], "gen_seconds": 1.0, "tool_seconds": 1.0, "wall_seconds": 2.0,
            "container_cpu_seconds": 1.0,
            "energy": {"gpu_j": None, "attributable": False, "counter_start": 0, "counter_end": 1, "t_start": 0,
                       "t_end": 1, "idle_w_ref": 30.0},
            "episode_path": f"/data/episodes/{aid}.npz"}


def srow(a, kl, ent, probe="Q3_K_M", **extra):
    """One P0d score row: every key qtrace.kl.score_stats returns, plus the join keys."""
    half = {"n_scored": 50, "kl_mean": kl, "kl_decision": kl, "kl_command": kl}
    return {"attempt_id": a["attempt_id"], "task_id": a["task_id"], "probe": probe, "gguf_p": "bf16.gguf",
            "gguf_q": "q3.gguf", "n_scored": 100, "n_decision": 20, "n_command": 16, "kl_mean": kl,
            "kl_decision": kl, "kl_command": kl, "kl_sum_decision": 20 * kl, "kl_sum_all": 100 * kl,
            "flip_rate": 0.05, "entropy_mean": ent, "nll_p_mean": 0.3, "nll_q_mean": 0.35,
            "halves": {"even": half, "odd": half}, **extra}


def make(runs):
    """(attempts, score rows, est-joules rows); attempt ids sort in creation order."""
    attempts, scores, joules = [], [], []
    for spec in runs:
        r = {**RUN, **spec}
        a = att(r, f"{r['task']}-{next(_ids):06d}")
        attempts.append(a)
        if r["kl"] is not None:
            scores.append(srow(a, r["kl"], r["ent"]))
        if r["j"] is not None:
            joules.append({"attempt_id": a["attempt_id"], "est_joules_q4": r["j"]})
    return attempts, scores, joules


def go(runs, chosen="kl_decision", fallback=False, energy=True, **cfg):
    attempts, scores, joules = make(runs)
    model = {"profiles": {"Q4_K_M": {}}, "fewest_joules_fallback": fallback} if energy else None
    return select(attempts, scores, {"chosen": chosen, "candidates": {}}, model, joules if energy else None,
                  **{"min_candidates": 2, "seeds": (0,), **cfg})


def picked(picks, arm, seed=0):
    return {r["task_id"]: r["attempt_id"] for r in picks[arm][seed]}


def cand(aid, turns=5, out=100, score=0.1, ent=1.0, j=1000.0, mal=0):
    return {"task_id": "t", "attempt_id": aid, "episode_path": f"/e/{aid}.npz", "turns": turns, "output_tokens": out,
            "est_joules_q4": j, "score": score, "entropy_mean": ent, "malformed_total": mal}


def drawn(cands, rule, n_seeds=40, **kw):
    return {pick(cands, rule, s, "t", **kw)["attempt_id"] for s in range(n_seeds)}


def random_pool(tasks, n_runs=6, seed=0):
    """Small integer ranges, so every tie-break path is exercised."""
    rng = random.Random(seed)
    return [{"task": t, "turns": rng.randint(3, 5), "out": rng.choice([100, 200]), "kl": rng.choice([0.1, 0.2, 0.3]),
             "ent": rng.choice([1.0, 2.0]), "j": rng.choice([10.0, 20.0]), "mal": {"wrong_name": rng.randint(0, 1)}}
            for t in tasks for _ in range(n_runs)]


# Ranks and single-task rules.

def test_avg_ranks_average_ties_and_normalize_to_unit_interval():
    assert avg_ranks([5, 3, 3, 9]) == [Fraction(2, 3), Fraction(1, 6), Fraction(1, 6), Fraction(1)]
    assert avg_ranks([7, 7]) == [Fraction(1, 2)] * 2
    assert avg_ranks([4]) == [0]


def test_fewest_turns_breaks_ties_by_output_tokens_then_seeded_draw():
    assert drawn([cand("a", 3, 200), cand("b", 3, 100), cand("c", 4, 50)], "fewest_turns") == {"b"}
    tied = [cand("a", 3, 100), cand("b", 3, 100), cand("c", 5, 10)]
    assert drawn(tied, "fewest_turns") == {"a", "b"}
    assert pick(tied, "fewest_turns", 7, "t") == pick(list(reversed(tied)), "fewest_turns", 7, "t")


def test_fewest_joules_breaks_ties_by_turns_then_seeded_draw():
    assert drawn([cand("a", 9, j=5.0), cand("b", 2, j=6.0)], "fewest_joules") == {"a"}
    assert drawn([cand("a", 4, j=5.0), cand("b", 3, j=5.0), cand("c", 1, j=7.0)], "fewest_joules") == {"b"}
    assert drawn([cand("a", 3, j=5.0), cand("b", 3, j=5.0), cand("c", 1, j=9.0)], "fewest_joules") == {"a", "b"}


def test_stable_low_entropy_and_clean_format_take_the_minimum():
    cs = [cand("a", score=0.3, ent=0.5, mal=4), cand("b", score=0.1, ent=0.9, mal=2), cand("c", score=0.2, ent=0.7)]
    assert pick(cs, "stable", 0, "t")["attempt_id"] == "b"
    assert pick(cs, "low_entropy", 0, "t")["attempt_id"] == "a"
    assert pick(cs, "clean_format", 0, "t")["attempt_id"] == "c"
    assert drawn([cand("a", score=0.1), cand("b", score=0.1), cand("c", score=0.2)], "stable") == {"a", "b"}
    assert drawn([cand("a", mal=0), cand("b", mal=0), cand("c", mal=1)], "clean_format") == {"a", "b"}


def test_clean_format_sums_all_five_malformed_categories():
    runs = [{"task": "t", "mal": {"misnested_tag": 3}},
            {"task": "t", "mal": dict.fromkeys(("wrong_name", "text_after_call", "truncated_call",
                                                "no_call_no_answer"), 1)}]
    picks, _ = go(runs)
    assert picks["clean_format"][0][0]["malformed_total"] == 3  # the other run has 4, but 0 misnested tags


def test_random_is_seeded_and_reaches_every_candidate():
    cs = [cand(x) for x in "abcd"]
    assert drawn(cs, "random") == set("abcd")
    assert pick(cs, "random", 3, "t") == pick(cs, "random", 3, "t")


def test_qtrace_uses_average_ranks_not_input_order():
    # Turns D=1, the rest tie at 2: average cost rank 5/8 for the tie, so D (mix 1/4) beats A (mix 5/16).
    # Ordinal or min ranks would give A a cost rank of 1/4 or less and pick A.
    cs = [cand("A", 2, score=0.1), cand("B", 2, score=0.2), cand("C", 2, score=0.4), cand("D", 1, score=0.3),
          cand("E", 2, score=0.5)]
    assert pick(cs, "qtrace", 0, "t")["attempt_id"] == "D"
    assert pick(cs, "stable", 0, "t")["attempt_id"] == "A"


def test_qtrace_alpha_weights_and_seeded_tie_break_in_the_mix():
    cs = [cand("a", 1, score=0.9), cand("b", 2, score=0.1)]
    assert drawn(cs, "qtrace") == {"a", "b"}  # mix 1/2 each
    assert pick(cs, "qtrace", 0, "t", alpha=1.0)["attempt_id"] == "a"
    assert pick(cs, "qtrace", 0, "t", alpha=0.0)["attempt_id"] == "b"


def test_qtrace_cost_joules_ranks_estimated_joules():
    cs = [cand("a", 1, j=90.0, score=0.5), cand("b", 3, j=10.0, score=0.5), cand("c", 2, j=50.0, score=0.5)]
    assert pick(cs, "qtrace", 0, "t")["attempt_id"] == "a"
    assert pick(cs, "qtrace", 0, "t", cost="joules")["attempt_id"] == "b"


def test_halves_split_floor_n_over_2_with_output_token_tie_break():
    cs = [cand("a", 1), cand("b", 2, 300), cand("c", 2, 100), cand("d", 2, 200), cand("e", 3)]
    short, long_ = halves(cs, "turns", 0, "t")
    assert {c["attempt_id"] for c in short} == {"a", "c"} and {c["attempt_id"] for c in long_} == {"b", "d", "e"}
    assert [len(h) for h in halves(cs[:4], "turns", 0, "t")] == [2, 2]


def test_halves_full_cost_ties_use_a_seeded_shuffle():
    cs = [cand("a", 1), cand("b", 2), cand("c", 2), cand("d", 2), cand("e", 3)]
    second = set()
    for s in range(40):
        short, _ = halves(cs, "turns", s, "t")
        assert short[0]["attempt_id"] == "a" and len(short) == 2
        second.add(short[1]["attempt_id"])
        assert short == halves(list(reversed(cs)), "turns", s, "t")[0]
    assert second == {"b", "c", "d"}


def test_halves_split_on_joules_when_cost_is_joules():
    cs = [cand("a", 1, j=90.0), cand("b", 3, j=10.0), cand("c", 2, j=50.0), cand("d", 4, j=70.0)]
    assert {c["attempt_id"] for c in halves(cs, "est_joules_q4", 0, "t")[0]} == {"b", "c"}


def test_2x2_arms_share_one_split_and_take_the_score_extremes():
    cs = [cand(x, 2, score=s) for x, s in zip("abcdefg", (0.5, 0.1, 0.7, 0.3, 0.9, 0.2, 0.4))]
    for seed in range(20):
        short, long_ = halves(cs, "turns", seed, "t")
        got = {arm: pick(cs, arm, seed, "t")["attempt_id"] for arm in
               ("short_stable", "short_unstable", "long_stable", "long_unstable")}
        assert got["short_stable"] == min(short, key=lambda c: c["score"])["attempt_id"]
        assert got["short_unstable"] == max(short, key=lambda c: c["score"])["attempt_id"]
        assert got["long_stable"] == min(long_, key=lambda c: c["score"])["attempt_id"]
        assert got["long_unstable"] == max(long_, key=lambda c: c["score"])["attempt_id"]


def test_2x2_ties_inside_a_half_are_a_seeded_draw():
    cs = [cand("a", 1, score=0.2), cand("b", 1, out=50, score=0.2), cand("c", 9, score=0.5), cand("d", 9, score=0.5)]
    assert drawn(cs, "short_stable") == {"a", "b"} and drawn(cs, "long_unstable") == {"c", "d"}


# Candidates, flags and lists.

def test_flagged_success_is_never_a_candidate_and_is_counted_per_list():
    best = {"turns": 1, "out": 1, "kl": 0.0, "ent": 0.0, "j": 1.0}
    runs = [{"task": "t1", **best, "touched": True}, {"task": "t1", "touched": False}, {"task": "t1"},
            {"task": "t2", "lst": "pool_x"}, {"task": "t2", "lst": "pool_x"},
            {"task": "t2", "lst": "pool_x", **best, "touched": True},
            {"task": "t2", "lst": "pool_x", **best, "touched": True}]
    picks, summary = go(runs, pool_lists=["pool200", "pool_x"], seeds=(0, 1, 2))
    assert summary["pool"]["flagged_excluded"] == {"pool200": 1, "pool_x": 2}
    assert summary["pool"]["n_candidates"] == 4
    # The flagged runs are the only ones with 1 turn and would win every arm if they were candidates.
    for arm in picks:
        for seed in picks[arm]:
            assert all(r["turns"] != 1 for r in picks[arm][seed]), arm


@pytest.mark.parametrize("touched", [None, 0, "false"])
def test_test_files_touched_is_read_strictly(touched):
    # An absent or non-bool flag must not read as "not flagged": that would switch the tripwire off.
    with pytest.raises(ValueError, match="test_files_touched"):
        go([{"task": "t"}, {"task": "t", "touched": touched}])


def test_unresolved_other_profiles_and_unlisted_lists_are_not_candidates():
    runs = [{"task": "t"}, {"task": "t"}, {"task": "t", "resolved": False, "turns": 1},
            {"task": "t", "profile": "Q4_K_M", "turns": 1}, {"task": "h", "lst": "heldout100"},
            {"task": "h", "lst": "heldout100"}]
    picks, summary = go(runs)
    assert summary["pool"]["n_tasks"] == 1 and summary["pool"]["n_candidates"] == 2
    assert all(r["turns"] == 5 for r in picks["fewest_turns"][0])


def test_infrastructure_errors_are_never_candidates_and_are_counted_per_list():
    # docs/04 reruns infrastructure faults as fresh attempts, and P0a writes no joule row for them. The
    # resolved error run has the fewest turns, and the second one carries no resolved verdict at all.
    runs = ([{"task": "t", "turns": 1, "j": None, "end": "error"}, {"task": "t", "end": "error", "resolved": None}]
            + [{"task": "t"}] * 3)
    picks, summary = go(runs, min_candidates=3)
    assert summary["pool"]["infra_error_excluded"] == {"pool200": 2} and summary["pool"]["n_candidates"] == 3
    assert picks["fewest_turns"][0][0]["turns"] == 5
    assert go(runs + [{"task": "u"}] * 4, min_candidates=4)[1]["pool"]["dropped_tasks"] == {"t": 3}


@pytest.mark.parametrize("end", ["turn_limit", "context_limit", "infra_timeout"])
def test_resolved_without_submitting_is_an_error(end):
    # docs/04 charges a scored budget hit and an infra_timeout as failures, so resolved true there is a bad record.
    picks, _ = go([{"task": "t"}, {"task": "t"}, {"task": "t", "end": end, "resolved": False}])
    assert len(picks["random"][0]) == 1
    with pytest.raises(ValueError, match=f"end_reason is '{end}'"):
        go([{"task": "t"}, {"task": "t"}, {"task": "t", "end": end}])


@pytest.mark.parametrize("lst", ["heldout100", "verified100", "inrepo_test"])
def test_held_out_lists_are_refused_as_pool_lists(lst):
    with pytest.raises(ValueError, match=lst):
        go([{"task": "t"}, {"task": "t"}], pool_lists=["pool200", lst])


def test_min_candidates_default_four_and_dropped_tasks_reported():
    runs = ([{"task": "a"}] * 4 + [{"task": "b"}] * 3
            + [{"task": "b", "touched": True}, {"task": "b", "resolved": False}, {"task": "c", "resolved": False}])
    attempts, scores, joules = make(runs)
    model = {"fewest_joules_fallback": False}
    picks, summary = select(attempts, scores, {"chosen": "kl_decision"}, model, joules, seeds=(0,))
    assert set(picked(picks, "random")) == {"a"}
    assert summary["pool"]["dropped_tasks"] == {"b": 3, "c": 0}
    picks, _ = select(attempts, scores, {"chosen": "kl_decision"}, model, joules, seeds=(0,), min_candidates=3)
    assert set(picked(picks, "random")) == {"a", "b"}
    with pytest.raises(ValueError, match="min_candidates"):
        select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=1)


def test_no_surviving_task_is_an_error():
    with pytest.raises(ValueError, match="no task"):
        go([{"task": "t"}], min_candidates=2)


# Seeding.

def test_selection_is_reproducible_and_independent_of_input_order():
    runs = random_pool([f"t{i}" for i in range(30)])
    attempts, scores, joules = make(runs)
    model, choice = {"fewest_joules_fallback": False}, {"chosen": "kl_decision"}
    a = select(attempts, scores, choice, model, joules)
    shuffled = attempts[:]
    random.Random(1).shuffle(shuffled)
    assert a == select(attempts, scores, choice, model, joules)
    assert a[0] == select(shuffled, scores[::-1], choice, model, joules[::-1])[0]
    assert set(a[0]["random"]) == {0, 1, 2}
    assert picked(a[0], "random", 0) != picked(a[0], "random", 1)


def test_tie_draws_are_independent_across_tasks_and_arms():
    # Every run ties on everything, so each pick is a pure draw. One shared stream would repeat the same
    # position in every task and make random and stable pick the same run everywhere.
    picks, summary = go([{"task": f"t{i:02d}"} for i in range(30) for _ in range(4)])
    position = {t: int(a.split("-")[1]) % 4 for t, a in picked(picks, "random").items()}
    assert len(set(position.values())) > 1
    pairs = {tuple(p["arms"]): p["share_same"] for p in summary["overlap"]["by_seed"]["0"]}
    assert pairs[("random", "stable")] < 0.6 and pairs[("fewest_turns", "clean_format")] < 0.6


def test_adding_a_task_leaves_every_other_pick_unchanged():
    a1, s1, j1 = make(random_pool([f"t{i}" for i in range(20)], seed=3))
    a2, s2, j2 = make(random_pool(["a_first", "z_last"], seed=4))
    choice, model = {"chosen": "kl_decision"}, {"fewest_joules_fallback": False}
    small, _ = select(a1, s1, choice, model, j1, min_candidates=2)
    big, _ = select(a2[:6] + a1 + a2[6:], s1 + s2, choice, model, j1 + j2, min_candidates=2)
    assert set(small) == set(big)
    for arm in small:
        for seed in (0, 1, 2):
            new = picked(big, arm, seed)
            assert {t: new[t] for t in picked(small, arm, seed)} == picked(small, arm, seed), (arm, seed)


# Joins, fallbacks and missing data.

def test_fewest_joules_ranks_joules_when_the_energy_model_passes():
    picks, summary = go([{"task": "t", "turns": 3, "j": 900.0}, {"task": "t", "turns": 8, "j": 100.0}], cost="joules")
    assert picks["fewest_joules"][0][0]["turns"] == 8 and picks["fewest_turns"][0][0]["turns"] == 3
    assert summary["fewest_joules"]["picks_as"] == "fewest_joules"


@pytest.mark.parametrize("energy, fallback, reason", [(True, True, "fallback"), (False, False, "no est_joules")])
def test_energy_fallback_picks_exactly_as_fewest_turns(energy, fallback, reason):
    runs = random_pool([f"t{i}" for i in range(15)], seed=5)
    picks, summary = go(runs, fallback=fallback, energy=energy, seeds=(0, 1, 2))
    for seed in (0, 1, 2):
        assert picked(picks, "fewest_joules", seed) == picked(picks, "fewest_turns", seed)
    assert summary["fewest_joules"]["picks_as"] == "fewest_turns" and reason in summary["fewest_joules"]["reason"]
    pairs = {tuple(p["arms"]): p for p in summary["overlap"]["by_seed"]["0"]}
    assert pairs[("fewest_turns", "fewest_joules")]["share_same"] == 1.0
    assert pairs[("fewest_turns", "fewest_joules")]["not_separable"] is True
    with pytest.raises(ValueError, match="cost: joules"):
        go(runs, fallback=fallback, energy=energy, cost="joules")


def test_joules_without_the_energy_model_is_an_error():
    attempts, scores, joules = make([{"task": "t"}, {"task": "t"}])
    with pytest.raises(ValueError, match="energy_model"):
        select(attempts, scores, {"chosen": "kl_decision"}, None, joules, min_candidates=2)


def test_chosen_null_drops_the_stability_arms_and_says_h3_is_negative():
    runs = random_pool(["t1", "t2"])
    picks, summary = go(runs, chosen=None)
    assert set(picks) == set(ARMS) - STABILITY_ARMS - {"fewest_joules"}  # merged by P0b's overlap rule
    assert "answered negatively" in summary["h3"]
    assert all(r["score_name"] is None and r["score"] is None for r in picks["random"][0])
    assert picks["low_entropy"][0][0]["entropy_mean"] is not None
    picks, summary = go(runs, chosen=None, score_override="kl_mean")
    assert set(picks) == set(ARMS) - {"fewest_joules"}
    assert summary["score"] == {"file_present": True, "file_chosen": None, "file_decided": True, "override": "kl_mean",
                                "used": "kl_mean"}


def test_undecided_null_choice_is_an_error_not_a_negative_h3():
    # P0d writes chosen null with decided false when a check could not run (here: no est_joules.jsonl yet).
    attempts, scores, joules = make(random_pool(["t1", "t2"]))
    model = {"fewest_joules_fallback": False}
    undecided = {"chosen": None, "decided": False, "candidates": {},
                 "verdict": "incomplete: kl_decision fails no check but est_joules_q4 did not run"}

    def run(choice, **kw):
        return select(attempts, scores, choice, model, joules, min_candidates=2, seeds=(0,), **kw)

    with pytest.raises(ValueError, match="undecided.*est_joules_q4 did not run"):
        run(undecided)
    picks, summary = run(undecided, score_override="kl_mean")
    assert set(picks) == set(ARMS) - {"fewest_joules"} and summary["score"]["file_decided"] is False
    assert "not answered" in summary["h3"] and "answered negatively" not in summary["h3"]
    picks, summary = run(undecided, arms=sorted(set(ARMS) - STABILITY_ARMS))
    assert "answered negatively" not in summary["h3"]
    picks, summary = run({**undecided, "decided": True, "verdict": "none passed"})
    assert set(picks) == set(ARMS) - STABILITY_ARMS - {"fewest_joules"} and "answered negatively" in summary["h3"]
    with pytest.raises(ValueError, match="decided"):
        run({**undecided, "decided": "no"})


def test_score_override_switches_the_ranked_field_and_q4_probe_rows_are_ignored():
    attempts, scores, joules = make([{"task": "t"}, {"task": "t"}])
    x, y = attempts
    scores = [srow(x, 0.1, 1.0, kl_mean=0.9), srow(y, 0.5, 1.0, kl_mean=0.2),
              srow(x, 0.9, 1.0, probe="Q4_K_M"), srow(y, 0.0, 1.0, probe="Q4_K_M")]
    model = {"fewest_joules_fallback": False}
    picks, summary = select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=2, seeds=(0,))
    assert picks["stable"][0][0]["attempt_id"] == x["attempt_id"] and picks["stable"][0][0]["score"] == 0.1
    assert summary["score"]["used"] == "kl_decision" and summary["score"]["override"] is None
    picks, summary = select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=2, seeds=(0,),
                            score_override="kl_mean")
    assert picks["stable"][0][0]["attempt_id"] == y["attempt_id"]
    assert picks["random"][0][0]["score_name"] == "kl_mean"
    assert summary["score"] == {"file_present": True, "file_chosen": "kl_decision", "file_decided": True,
                                "override": "kl_mean", "used": "kl_mean"}
    with pytest.raises(ValueError, match="score_override"):
        select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=2, score_override="flip_rate")
    with pytest.raises(ValueError, match="chosen"):
        select(attempts, scores, {"chosen": "flip_rate"}, model, joules, min_candidates=2)


def test_a_missing_join_fails_only_for_the_arms_that_need_it():
    runs = [{"task": "t"}, {"task": "t", "kl": None}, {"task": "u"}, {"task": "u"}]  # one run has no score row
    with pytest.raises(ValueError, match="stable needs score.*low_entropy needs entropy_mean"):
        go(runs)
    assert set(go(runs, arms=["random", "fewest_turns", "fewest_joules", "clean_format"])[0]) == {
        "random", "fewest_turns", "clean_format"}  # equal joules: P0b's overlap rule merges fewest_joules
    attempts, scores, joules = make([{"task": "t"}, {"task": "t"}])
    scores[1]["kl_decision"] = None  # the row exists, but the chosen score is null
    model = {"fewest_joules_fallback": False}
    with pytest.raises(ValueError, match="stable needs score"):
        select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=2)
    picks, _ = select(attempts, scores, {"chosen": "kl_decision"}, model, joules, min_candidates=2,
                      arms=sorted(set(ARMS) - STABILITY_ARMS))
    assert set(picks) == set(ARMS) - STABILITY_ARMS - {"fewest_joules"}
    runs = [{"task": "t"}, {"task": "t", "j": None}]
    with pytest.raises(ValueError, match="fewest_joules.*est_joules_q4"):
        go(runs)
    assert "fewest_joules" in go(runs, fallback=True)[0]
    with pytest.raises(ValueError, match="low_entropy.*entropy_mean"):
        go([{"task": "t"}, {"task": "t", "ent": None}])


def test_cost_joules_splits_the_2x2_and_mixes_qtrace_on_joules():
    # Turn order a<c<b<d, joule order b<c<d<a, score order a<c<d<b. Runs are named by their turns.
    runs = [{"task": "t", "turns": t, "j": j, "kl": k} for t, j, k in ((1, 90.0, 0.1), (3, 10.0, 0.4),
                                                                     (2, 50.0, 0.2), (4, 70.0, 0.3))]

    def turns_of(picks):
        return {arm: picks[arm][0][0]["turns"] for arm in ("qtrace", "short_stable", "short_unstable", "long_stable",
                                                         "long_unstable")}

    assert turns_of(go(runs, cost="joules")[0]) == {"qtrace": 2, "short_stable": 2, "short_unstable": 3,
                                                    "long_stable": 1, "long_unstable": 4}
    # cost turns is only allowed when P0b's rule merges the arms or the energy model fails (check_cost).
    assert turns_of(go(runs, fallback=True)[0]) == {"qtrace": 1, "short_stable": 1, "short_unstable": 2, "long_stable": 4,
                                     "long_unstable": 3}


def test_missing_joules_under_cost_joules_fails_for_qtrace_and_the_2x2():
    runs = [{"task": "t"}, {"task": "t", "j": None}]
    arms = [a for a in ARMS if a != "fewest_joules"]
    with pytest.raises(ValueError, match="missing joins") as err:
        go(runs, cost="joules", arms=arms)
    for arm in ("qtrace", "short_stable", "short_unstable", "long_stable", "long_unstable"):
        assert f"{arm} needs est_joules_q4" in str(err.value)
    assert set(go(runs, arms=arms)[0]) == set(arms)  # cost: turns needs no joules here


def test_arm_mean_is_null_when_a_pick_lacks_the_value():
    runs = [{"task": "t", "turns": 1, "j": None}, {"task": "t", "turns": 2}, {"task": "u"}, {"task": "u"}]
    _, summary = go(runs, fallback=True)
    assert summary["arms"]["fewest_turns"]["0"]["mean_est_joules_q4"] is None
    assert summary["arms"]["fewest_turns"]["0"]["mean_turns"] == 3.0
    assert summary["pool"]["joins_missing"]["est_joules_q4"] == 1


def test_missing_score_choice_is_an_error_when_stability_arms_are_asked_for():
    attempts, scores, joules = make([{"task": "t"}, {"task": "t"}])
    model = {"fewest_joules_fallback": False}
    with pytest.raises(ValueError, match="score_choice"):
        select(attempts, scores, None, model, joules, min_candidates=2)
    picks, summary = select(attempts, scores, None, model, joules, min_candidates=2,
                            arms=["random", "fewest_turns", "low_entropy"])
    assert set(picks) == {"random", "fewest_turns", "low_entropy"} and summary["score"]["file_present"] is False


def test_bad_input_fails_with_a_message_naming_the_problem():
    def broken(edit, match, **kw):
        attempts, scores, joules = make([{"task": "t"}, {"task": "t"}])
        edit(attempts, scores, joules)
        with pytest.raises(ValueError, match=match):
            select(attempts, scores, {"chosen": "kl_decision"}, {"fewest_joules_fallback": False}, joules,
                   min_candidates=2, **kw)

    broken(lambda a, s, j: a.append(dict(a[0])), "duplicate attempt_id")
    broken(lambda a, s, j: a[0]["tokens"].pop("output"), "tokens.output")
    broken(lambda a, s, j: a[0]["malformed"].pop("wrong_name"), "malformed.wrong_name")
    broken(lambda a, s, j: a[0].pop("grading"), "grading")
    broken(lambda a, s, j: a[0].update(resolved="true"), "resolved")
    broken(lambda a, s, j: a[0].update(episode_path="ep.npz"), "absolute")
    broken(lambda a, s, j: s[0].update(kl_decision=float("nan")), "finite")
    broken(lambda a, s, j: s[0].update(task_id="other"), "task_id")
    broken(lambda a, s, j: s.append(dict(s[0])), "two Q3_K_M rows")
    broken(lambda a, s, j: j.append(dict(j[0])), "two est_joules_q4 rows")
    broken(lambda a, s, j: None, "unknown arm", arms=["random", "shortest"])
    broken(lambda a, s, j: None, "cost", cost="tokens")
    broken(lambda a, s, j: None, "alpha", alpha=1.5)


# Summary and overlap.

def test_overlap_table_flags_pairs_above_85_percent():
    def rows(ids):
        return [{"task_id": f"t{i}", "attempt_id": x} for i, x in enumerate(ids)]

    a = rows(["x"] * 20)
    b = rows(["x"] * 17 + ["y"] * 3)
    c = rows(["x"] * 18 + ["y"] * 2)
    pairs = {tuple(p["arms"]): p for p in overlap({"a": a, "b": b, "c": c})}
    assert pairs[("a", "b")] == {"arms": ["a", "b"], "share_same": 0.85, "not_separable": False}  # "above 85%"
    assert pairs[("a", "c")]["share_same"] == 0.9 and pairs[("a", "c")]["not_separable"] is True
    assert overlap({"a": a, "d": rows(["x"] * 16 + ["y"] * 4)})[0]["not_separable"] is False
    assert pairs[("b", "c")]["share_same"] == 0.95


def test_arm_overlap_rule_ranks_turns_tokens_joules_and_both_hero_costs():
    # t1: turns picks r1 (300 J), output tokens r3 (200 J), joules r2 (100 J); HERO total tokens
    # (6100, 4100, 12050) picks r2 and HERO API price (1900, 4400, 4100) picks r1. t2 agrees everywhere.
    runs = [{"task": "t1", "turns": 2, "out": 100, "unc": 1000, "cac": 5000, "j": 300.0},
            {"task": "t1", "turns": 5, "out": 100, "unc": 4000, "cac": 0, "j": 100.0},
            {"task": "t1", "turns": 6, "out": 50, "unc": 3000, "cac": 9000, "j": 200.0},
            {"task": "t2", "turns": 1, "out": 10, "unc": 10, "cac": 10, "j": 50.0},
            {"task": "t2", "turns": 9, "out": 90, "unc": 90, "cac": 90, "j": 90.0}]
    picks, summary = go(runs, cost="joules")
    rule = summary["arm_overlap_rule"]
    s0 = rule["by_seed"]["0"]
    assert s0["median_joule_regret"] == {"turns": 1.0, "output_tokens": 0.5, "joules": 0.0, "hero_total_tokens": 0.0,
                                         "hero_api_price": 1.0}
    same = {tuple(p["arms"]): p["share_same"] for p in s0["pairs"]}
    assert same[("turns", "joules")] == 0.5 and same[("turns", "hero_api_price")] == 1.0
    assert same[("joules", "hero_total_tokens")] == 1.0 and same[("joules", "hero_api_price")] == 0.5
    assert s0["merge_fewest_joules"] is False and s0["calibrated_token_mix"] is False
    assert rule["merged"] is False and rule["note"].startswith("separate") and "fewest_joules" in picks
    assert rule["threshold"] == 0.85 and rule["regret_threshold"] == 0.05


def test_arm_overlap_rule_merges_fewest_joules_on_low_regret_for_every_seed():
    # fewest_turns never picks the cheapest run, but costs only 3% more: below docs/04's 5%, so merged.
    runs = [r for i in range(4) for r in ({"task": f"t{i}", "turns": 2, "j": 103.0},
                                          {"task": f"t{i}", "turns": 4, "j": 100.0})]
    picks, summary = go(runs, seeds=(0, 1))
    rule = summary["arm_overlap_rule"]
    for s in ("0", "1"):
        same = {tuple(p["arms"]): p["share_same"] for p in rule["by_seed"][s]["pairs"]}
        assert same[("turns", "joules")] == 0.0
        assert rule["by_seed"][s]["median_joule_regret"]["turns"] == pytest.approx(0.03)
        assert rule["by_seed"][s]["merge_fewest_joules"] is True
    assert rule["merged"] is True and "not built for any seed" in rule["note"]
    assert "fewest_joules" not in picks and "fewest_joules" not in summary["arms_built"]
    assert "fewest_turns" in picks
    assert summary["fewest_joules"]["picks_as"] == "fewest_turns" and summary["fewest_joules"]["fallback"] is False
    assert "not built" in summary["fewest_joules"]["reason"] and "arm_overlap_rule" in summary["fewest_joules"]["reason"]
    # At 6% the arms stay separate.
    picks, summary = go([{**r, "j": 106.0} if r["j"] == 103.0 else r for r in runs], cost="joules")
    assert summary["arm_overlap_rule"]["merged"] is False and "fewest_joules" in picks
    assert summary["fewest_joules"] == {"fallback": False, "picks_as": "fewest_joules", "reason": None}


def test_one_merging_seed_drops_fewest_joules_for_every_seed():
    # Equal turns and output tokens: fewest_turns is a seeded draw between the 100 J and the 200 J run, so
    # seed 0 picks the cheap run (merge) and seed 1 the dear one (regret 100%, share 0%).
    picks, summary = go([{"task": "t", "j": 100.0}, {"task": "t", "j": 200.0}], seeds=(0, 1))
    by_seed = summary["arm_overlap_rule"]["by_seed"]
    assert by_seed["0"]["merge_fewest_joules"] is True and by_seed["1"]["merge_fewest_joules"] is False
    assert summary["arm_overlap_rule"]["merged"] is True and "fewest_joules" not in picks


def test_calibrated_token_mix_at_exactly_85_percent_agreement():
    # API price is 19 x output tokens here. Joules follow it in 17 of 20 tasks and invert it in 3.
    runs = [r for i in range(20) for r in ({"task": f"t{i:02d}", "out": 100, "j": 100.0 if i < 17 else 200.0},
                                           {"task": f"t{i:02d}", "out": 200, "j": 200.0 if i < 17 else 100.0})]
    s0 = go(runs)[1]["arm_overlap_rule"]["by_seed"]["0"]
    assert {tuple(p["arms"]): p["share_same"] for p in s0["pairs"]}[("joules", "hero_api_price")] == 0.85
    assert s0["calibrated_token_mix"] is True
    more = [{"task": "t20", "out": 100, "j": 200.0}, {"task": "t20", "out": 200, "j": 100.0}]
    assert go(runs + more)[1]["arm_overlap_rule"]["by_seed"]["0"]["calibrated_token_mix"] is False  # 17 of 21


def test_cost_must_match_the_p0b_verdict():
    # docs/03 Method step 3: qtrace and the 2x2 split rank by joules when P0b keeps fewest_joules and
    # fewest_turns separate, and by turns when it merges them.
    separate = [r for i in range(4) for r in ({"task": f"t{i}", "turns": 2, "j": 300.0, "kl": 0.3},
                                              {"task": f"t{i}", "turns": 6, "j": 100.0, "kl": 0.3},
                                              {"task": f"t{i}", "turns": 4, "j": 200.0, "kl": 0.35})]
    with pytest.raises(ValueError, match="keeps fewest_joules and fewest_turns separate.*set select.cost: joules"):
        go(separate)
    picks, summary = go(separate, cost="joules")
    assert summary["arm_overlap_rule"]["note"].startswith("separate")
    assert [r["turns"] for r in picks["qtrace"][0]] == [6, 6, 6, 6]
    merged = [r for i in range(4) for r in ({"task": f"t{i}", "turns": 2, "j": 103.0},
                                            {"task": f"t{i}", "turns": 4, "j": 100.0})]
    with pytest.raises(ValueError, match="merges fewest_joules into fewest_turns.*set select.cost: turns"):
        go(merged, cost="joules")
    # No built arm ranks by cost, so any cost is accepted.
    assert set(go(separate, arms=["random", "fewest_turns", "fewest_joules", "stable"])[0]) == {
        "random", "fewest_turns", "fewest_joules", "stable"}


def test_arm_overlap_rule_is_not_applied_under_the_energy_fallback_or_without_joules():
    runs = [{"task": "t"}, {"task": "t"}]  # equal joules: the rule would merge
    picks, summary = go(runs, fallback=True)
    assert summary["arm_overlap_rule"]["merged"] is False and "fewest_joules" in picks
    assert summary["arm_overlap_rule"]["note"].startswith("not applied")
    assert summary["arm_overlap_rule"]["by_seed"]["0"]["merge_fewest_joules"] is True
    picks, summary = go(runs, energy=False)
    assert summary["arm_overlap_rule"]["by_seed"] is None and "fewest_joules" in picks
    assert summary["arm_overlap_rule"]["note"] == "not computed: 2 candidate(s) lack est_joules_q4"
    with pytest.raises(ValueError, match="minimum est_joules_q4 is 0.0"):
        go([{"task": "t", "j": 0.0}, {"task": "t"}])


def test_summary_reports_chance_rate_means_and_every_arm_pair():
    runs = [{"task": "a", "turns": 2, "out": 10, "kl": 0.4, "j": 8.0},
            {"task": "a", "turns": 6, "out": 30, "kl": 0.2, "j": 4.0}]
    runs += [{"task": "b", "turns": t, "out": 20, "kl": k, "j": 2.0 * t}
             for t, k in ((4, 0.3), (5, 0.1), (7, 0.5), (8, 0.6))]
    _, summary = go(runs, seeds=(0, 1), cost="joules")
    assert summary["overlap"]["chance_rate"] == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert summary["overlap"]["threshold"] == 0.85
    assert len(summary["overlap"]["by_seed"]["1"]) == len(ARMS) * (len(ARMS) - 1) // 2
    ft = summary["arms"]["fewest_turns"]["0"]
    assert ft == {"n_tasks": 2, "mean_turns": 3.0, "mean_output_tokens": 15.0, "mean_est_joules_q4": 8.0,
                  "mean_score": pytest.approx(0.35)}
    assert summary["arms"]["stable"]["1"]["mean_score"] == pytest.approx(0.15)
    assert summary["pool"]["n_tasks"] == 2 and summary["pool"]["n_candidates"] == 6
    assert summary["arms_built"] == list(ARMS)
    json.dumps(summary, allow_nan=False)


def test_rows_have_the_agreed_keys():
    picks, _ = go([{"task": "t"}, {"task": "t"}])
    assert set(picks["qtrace"][0][0]) == {"arm", "seed", "task_id", "attempt_id", "episode_path", "turns",
                                          "output_tokens", "input_uncached", "input_cached", "est_joules_q4",
                                          "score_name", "score", "entropy_mean",
                                          "malformed_total"}
    assert picks["qtrace"][0][0]["arm"] == "qtrace" and picks["qtrace"][0][0]["seed"] == 0


# Files and the script.

def test_read_attempts_resolves_relative_episode_paths(tmp_path):
    a, b = make([{"task": "t"}, {"task": "t"}])[0]
    a["episode_path"] = "episodes/x.npz"
    path = tmp_path / "attempts" / "run-p0b.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    rows = read_attempts(str(path))
    assert rows[0]["episode_path"] == str(tmp_path / "attempts" / "episodes" / "x.npz")
    assert rows[1]["episode_path"] == b["episode_path"]
    path.write_text(json.dumps({**a, "run_id": "run-other"}) + "\n")
    with pytest.raises(ValueError, match="run_id"):
        read_attempts(str(path))


def run_script(cfg_path):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "p1_select.py"), "--config", str(cfg_path)],
                          check=False, capture_output=True, text=True, cwd=ROOT)


def test_script_writes_arm_files_and_summary(tmp_path):
    attempts, scores, _ = make(random_pool(["t1", "t2", "t3"]))  # no P0a files: fewest_joules falls back
    for a in attempts:
        a["episode_path"] = f"episodes/{a['attempt_id']}.npz"
    (tmp_path / "attempts").mkdir()
    (tmp_path / "attempts" / "run-p0b.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))
    (tmp_path / "p0d").mkdir()
    (tmp_path / "p0d" / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    (tmp_path / "p0d" / "score_choice.json").write_text(json.dumps({"chosen": "kl_command", "candidates": {}}))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stable_seed7.jsonl").write_text("{}\n")
    (tmp_path / "out" / "notes.txt").write_text("kept")
    cfg = {"pool_runs": ["run-p0b"],
           "paths": {"attempts_dir": str(tmp_path / "attempts"), "scores": str(tmp_path / "p0d" / "scores.jsonl"),
                     "score_choice": str(tmp_path / "p0d" / "score_choice.json"),
                     "energy_model": str(tmp_path / "p0a" / "energy_model.json"),
                     "est_joules": str(tmp_path / "p0a" / "est_joules.jsonl"), "out_dir": str(tmp_path / "out")},
           "select": {"pool_lists": ["pool200"], "min_candidates": 4, "alpha": 0.5, "cost": "turns",
                      "score_override": None, "seeds": [0, 1], "arms": list(ARMS)}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    res = run_script(tmp_path / "cfg.yaml")
    assert res.returncode == 0, res.stderr
    names = sorted(os.listdir(tmp_path / "out"))
    assert names == sorted([f"{a}_seed{s}.jsonl" for a in ARMS for s in (0, 1)] + ["notes.txt", "summary.json"])
    rows = [json.loads(x) for x in (tmp_path / "out" / "qtrace_seed1.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["t1", "t2", "t3"] and all(r["score_name"] == "kl_command" for r in rows)
    assert all(r["episode_path"].startswith(str(tmp_path / "attempts" / "episodes")) for r in rows)
    assert all(r["est_joules_q4"] is None for r in rows)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["fewest_joules"]["picks_as"] == "fewest_turns"
    assert summary["inputs"]["est_joules"]["present"] is False and summary["pool_runs"] == ["run-p0b"]
    assert "removed stale stable_seed7.jsonl" in res.stdout and "fewest_turns" in res.stdout
    assert "P0b fewest_joules/fewest_turns rule: not computed" in res.stdout
    cfg["pool_runs"] = []
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    res = run_script(tmp_path / "cfg.yaml")
    assert res.returncode != 0 and "pool_runs" in res.stderr
