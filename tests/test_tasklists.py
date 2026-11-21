"""Tests for the frozen P0 task lists. Run:  python -m pytest tests/test_tasklists.py -q"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.tasklists import HELDOUT_REPOS, POOL_REPOS, allocate, build_lists, stratified_sample

LISTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "task_lists")


def fake_rows(per_repo=60):
    return [(f"img/{repo}:{i:04d}", repo) for repo in POOL_REPOS + HELDOUT_REPOS for i in range(per_repo)]


def test_allocate_is_proportional_and_exact():
    counts = {"a": 500, "b": 300, "c": 200}
    assert allocate(counts, 10) == {"a": 5, "b": 3, "c": 2}
    assert sum(allocate({"a": 7, "b": 5, "c": 1}, 6).values()) == 6


def test_allocate_never_exceeds_group_size():
    alloc = allocate({"big": 1000, "tiny": 1}, 50)
    assert alloc["tiny"] <= 1 and sum(alloc.values()) == 50


def test_stratified_sample_is_deterministic_and_order_free():
    groups = {"a": [f"a{i}" for i in range(20)], "b": [f"b{i}" for i in range(10)]}
    shuffled = {"a": list(reversed(groups["a"])), "b": groups["b"]}
    assert stratified_sample(groups, 6, seed=1) == stratified_sample(shuffled, 6, seed=1)
    assert stratified_sample(groups, 6, seed=1) != stratified_sample(groups, 6, seed=2)


def test_build_lists_are_disjoint_and_sized():
    verified = [f"repo__x-{i}" for i in range(500)]
    lists = build_lists(fake_rows(), verified, seed=7, inrepo_size=30, pool_size=20, heldout_size=10, verified_size=5,
                        pilot_size=1, pilot_verified_size=1, subrun_size=1)
    inrepo, pool, held, ver = (set(lists[k]["ids"]) for k in ("inrepo_test", "pool200", "heldout100", "verified100"))
    assert (len(inrepo), len(pool), len(held), len(ver)) == (30, 20, 10, 5)
    assert not inrepo & pool
    pool_repos = {tid.split("/")[1].split(":")[0] for tid in inrepo | pool}
    held_repos = {tid.split("/")[1].split(":")[0] for tid in held}
    assert pool_repos <= set(POOL_REPOS) and held_repos <= set(HELDOUT_REPOS)


def test_build_lists_rejects_duplicate_task_ids():
    rows = fake_rows() + [fake_rows()[0]]
    try:
        build_lists(rows, ["v-1"], seed=0, inrepo_size=1, pool_size=1, heldout_size=1, verified_size=1)
    except ValueError as err:
        assert "duplicate" in str(err)
    else:
        raise AssertionError("duplicate task ids were accepted")


def test_heldout691_is_every_heldout_task():
    rows = fake_rows()
    lists = build_lists(rows, ["v-1"], seed=7, inrepo_size=1, pool_size=2, heldout_size=10, verified_size=1,
                        pilot_size=1, pilot_verified_size=1, subrun_size=1)
    every = lists["heldout691"]
    assert every["ids"] == sorted(tid for tid, repo in rows if repo in HELDOUT_REPOS)
    assert every["size"] == 60 * len(HELDOUT_REPOS)
    assert set(lists["heldout100"]["ids"]) <= set(every["ids"])


def test_frozen_heldout691_matches_04_and_contains_heldout100():
    with open(os.path.join(LISTS, "heldout691.json")) as f:
        every = json.load(f)
    with open(os.path.join(LISTS, "heldout100.json")) as f:
        held = json.load(f)
    assert every["size"] == len(every["ids"]) == len(set(every["ids"])) == 691
    assert set(every["by_repo"]) == set(HELDOUT_REPOS)
    assert set(held["ids"]) <= set(every["ids"])
    assert every["sources"] == held["sources"]


def test_pilot_and_subrun_lists_are_seeded_subsets():
    verified = [f"repo__x-{i}" for i in range(500)]
    kw = {"inrepo_size": 30, "pool_size": 60, "heldout_size": 40, "verified_size": 20,
          "pilot_size": 12, "pilot_verified_size": 3, "subrun_size": 12}
    lists = build_lists(fake_rows(), verified, seed=7, **kw)
    pairs = {"pilot20": "pool200", "pilot_verified3": "verified100", "energy_subrun15": "pool200"}
    for sub, parent in pairs.items():
        assert set(lists[sub]["ids"]) <= set(lists[parent]["ids"])
    assert [lists[k]["size"] for k in pairs] == [12, 3, 12]
    assert not set(lists["energy_subrun15"]["ids"]) & set(lists["pilot20"]["ids"])
    # Stratified by repo: 60 pool tasks over 6 equal repos give 10 each, so 12 picks give 2 each.
    assert set(lists["pilot20"]["by_repo"].values()) == {2}
    assert set(lists["energy_subrun15"]["by_repo"].values()) == {2}
    assert lists == build_lists(fake_rows(), verified, seed=7, **kw)
    assert lists["pilot20"]["ids"] != build_lists(fake_rows(), verified, seed=8, **kw)["pilot20"]["ids"]


def test_frozen_pilot_and_subrun_lists():
    def load(name):
        with open(os.path.join(LISTS, f"{name}.json")) as f:
            return json.load(f)
    for sub, parent, size in [("pilot20", "pool200", 20), ("pilot_verified3", "verified100", 3),
                              ("energy_subrun15", "pool200", 15)]:
        child, par = load(sub), load(parent)
        assert child["size"] == len(child["ids"]) == len(set(child["ids"])) == size
        assert set(child["ids"]) <= set(par["ids"])
        assert child["seed"] == par["seed"] and child["sources"] == par["sources"]
    assert not set(load("energy_subrun15")["ids"]) & set(load("pilot20")["ids"])
