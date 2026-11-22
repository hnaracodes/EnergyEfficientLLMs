"""Tests for P2 energy in the batched regime (qtrace/p2_batched.py), on test_p2_analysis's synthetic records.

Run:  python -m pytest tests/test_p2_batched.py -q
"""

import copy
import json
import math
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_p2_analysis import CFG, EM, Q4, build, run, scaled, write_jsonl

from qtrace.p2_analysis import analyze

# P0a's batched vs concurrency-1 picks agree on 60% of tasks, below docs/04's 85%.
BATCHED = {
    **EM,
    "batched": {
        "coef": {"intercept_w": 50.0, "prefill": 0.5, "decode": 2.0},
        "agreement_with_c1": {"share_same_pick": 0.6},
        "below_85pct": True,
    },
}
BLOCK_J = {"fewest_turns": 4000.0, "random": 5000.0}


def with_counters(recs, block_j=BLOCK_J):
    """Each seed model x profile is one block of run "p2", its attempts 1 s apart; blocks are 1000 s and 1e6 J apart
    on the GPU counter, and a block spans block_j[arm] J of it."""
    out, blocks, n = [], {}, {}
    for r in recs:
        key = (r["model"], r["profile"])
        k = blocks.setdefault(key, len(blocks))
        j = n[key] = n.get(key, -1) + 1
        energy = {
            **r["energy"],
            "t_start": 1000.0 * k + j,
            "t_end": 1000.0 * k + j + 1,
            "counter_start": 1e6 * k + j,
            "counter_end": 1e6 * k + block_j[r["model"].split("_seed")[0]] - j,
        }
        out.append({**r, "run_id": "p2", "energy": energy})
    return out


def data():
    recs, manifest = build(
        ["fewest_turns", "random"], scaled({"fewest_turns": 0.8}), seeds=(0, 1)
    )
    return with_counters(recs), manifest


def test_batched_regime_primary_and_run_level_totals():
    recs, manifest = data()
    out = analyze(recs, manifest, BATCHED, "fewest_turns", CFG)
    b, c1 = out["batched_regime"], out["primary"]
    assert (
        out["energy"]["batched_below_85pct"] is True
        and out["energy"]["batched_share_same_pick"] == 0.6
    )
    # No uncached input, so the batched marginal is 2 J per output token: twice the concurrency-1 estimate.
    assert b["primary"]["T"] == pytest.approx(c1["T"]) == pytest.approx(math.log(0.8))
    assert b["primary"]["jpr"]["arm"] == pytest.approx(2 * c1["jpr"]["arm"])
    assert b["primary"]["n_tasks"] == 20 and b["primary"]["guard"]["pass"]
    run = {(r["arm"], r["profile"]): r for r in b["run_level"]}
    # Seed 0 resolves 4 of 5 tasks per repo and seed 1 resolves 2 of 5, over 4 repos: 24 resolves in 2 blocks.
    assert run["fewest_turns", Q4] == {
        "arm": "fewest_turns",
        "profile": Q4,
        "n_blocks": 2,
        "n_attempts": 40,
        "joules": 8000.0,
        "n_resolved": 24,
        "jpr": 8000.0 / 24,
    }
    assert run["random", Q4]["joules"] == 10000.0 and len(run) == 4
    assert analyze(recs, manifest, EM, "fewest_turns", CFG)["batched_regime"] is None


def test_run_level_totals_need_separate_blocks_and_counter_fields():
    recs, manifest = data()
    cases = [
        (lambda r: r[0]["energy"].update(t_end=5000.0), "overlap in time"),
        (
            lambda r: r[0]["energy"].pop("counter_start"),
            "missing field energy.counter_start",
        ),
        (
            lambda r: r[0]["energy"].update(counter_start=1e9),
            "counter_end below its counter_start",
        ),
        (lambda r: r[0].pop("run_id"), "missing field run_id"),
    ]
    for mutate, match in cases:
        bad = copy.deepcopy(recs)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            analyze(bad, manifest, BATCHED, "fewest_turns", CFG)
    no_coef = {**BATCHED, "batched": {"below_85pct": True}}
    with pytest.raises(ValueError, match="batched.coef"):
        analyze(recs, manifest, no_coef, "fewest_turns", CFG)


def test_grammar_attempts_join_the_overlap_check_not_the_totals():
    recs, manifest = data()
    plain = analyze(recs, manifest, BATCHED, "fewest_turns", CFG)["batched_regime"]
    # H4's grammar reruns of the Q4_K_M rows, as their own blocks 1e5 s after the main ones.
    g = [
        {
            **r,
            "attempt_id": "g-" + r["attempt_id"],
            "grammar": True,
            "energy": {
                **r["energy"],
                "t_start": r["energy"]["t_start"] + 1e5,
                "t_end": r["energy"]["t_end"] + 1e5,
            },
        }
        for r in recs
        if r["profile"] == Q4
    ]
    mixed = analyze(recs + g, manifest, BATCHED, "fewest_turns", CFG)
    assert mixed["batched_regime"]["run_level"] == plain["run_level"]
    # One grammar attempt of random that ran during fewest_turns_seed0's main block.
    t0 = min(
        r["energy"]["t_start"]
        for r in recs
        if r["model"] == "fewest_turns_seed0" and r["profile"] == Q4
    )
    hit = next(i for i, r in enumerate(g) if r["model"] == "random_seed0")
    g[hit] = {
        **g[hit],
        "energy": {**g[hit]["energy"], "t_start": t0 + 0.2, "t_end": t0 + 0.4},
    }
    with pytest.raises(ValueError, match="overlap in time"):
        analyze(recs + g, manifest, BATCHED, "fewest_turns", CFG)


def test_error_attempts_count_in_block_joules_not_in_resolves():
    recs, manifest = data()
    err = next(
        r
        for r in recs
        if r["model"] == "fewest_turns_seed0" and r["profile"] == Q4 and r["resolved"]
    )
    recs = [{**r, "end_reason": "error"} if r is err else r for r in recs]
    run = {
        (r["arm"], r["profile"]): r
        for r in analyze(recs, manifest, BATCHED, "fewest_turns", CFG)[
            "batched_regime"
        ]["run_level"]
    }
    assert (
        run["fewest_turns", Q4]["n_attempts"] == 40
        and run["fewest_turns", Q4]["joules"] == 8000.0
    )
    assert run["fewest_turns", Q4]["n_resolved"] == 23


def test_script_writes_the_batched_regime(tmp_path):
    recs, manifest = data()
    write_jsonl(tmp_path / "p2.jsonl", recs)
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "em.json").write_text(json.dumps(BATCHED))
    (tmp_path / "head.json").write_text(json.dumps({"headline": "fewest_turns"}))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(CFG))
    res = run(
        "p2_analyze.py",
        tmp_path / "p2.jsonl",
        "--models",
        tmp_path / "m.json",
        "--energy-model",
        tmp_path / "em.json",
        "--headline",
        tmp_path / "head.json",
        "--config",
        tmp_path / "cfg.yaml",
        "--out-dir",
        tmp_path / "out",
    )
    assert res.returncode == 0, res.stderr
    md = (tmp_path / "out" / "analysis.md").read_text()
    assert "Batched regime (P0a batched and concurrency-1 picks agree on 0.600" in md
    assert "| fewest_turns | Q4_K_M | 2 | 40 | 8000 | 24 | 333 |" in md
