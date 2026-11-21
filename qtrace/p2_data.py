"""Input checks and per-task arrays for the P1 headline pick and the P2 analysis (qtrace/p2_analysis.py).

Attempt records are in the issue #1 format. A models manifest maps each record's `model` value to
{arm, seed}: seed is the training seed, arm "base" is the untouched model and "base_nothink" the same model
with thinking off (docs/03 Baselines). A base entry may give seed null, and then its replicates are the
attempts' sampling seeds, because the base model has no training seed but P2 runs it 3 times like the arms.
Every record carries a required bool `grammar`: true only for H4's grammar-constrained runs (docs/03
Baselines). load() indexes one side of it, so grammar rows enter H4 only and every other statistic uses
grammar false rows.
Joules come from energy_model.json (#13): per profile, the raw generation-window model plus that profile's
idle_w x tool_seconds, the provisional whole-attempt quantity its ranking block names.
Everything is checked here, once, and an input that would make a statistic quietly wrong fails with a message
naming the attempt, field or cell.
"""

import math
from collections import defaultdict

import numpy as np

from qtrace.energy_model import HELDOUT_REPOS, estimate_attempt_joules

# qtrace/p1_select.py (#14), in docs/03 Method step 3 order; base and random_fq are the extra P2 models.
ARMS = ("random", "fewest_turns", "fewest_joules", "stable", "qtrace", "low_entropy", "clean_format",
        "short_stable", "short_unstable", "long_stable", "long_unstable")
BASE_ARMS = ("base", "base_nothink")  # untrained, so no training seed
ARM_NAMES = frozenset(ARMS) | set(BASE_ARMS) | {"random_fq"}
POOL_REPOS = ("pandas", "numpy", "pillow", "orange3", "aiohttp", "tornado")  # qtrace/tasklists.py (#3)
# #3's frozen lists that hold no task of the 4 held-out repos. A P2 row is on heldout691 (all 691 held-out
# tasks), its subset heldout100, or another list of the held-out repos (a secondary-tier subset), so it is
# recognized by its repo and any of those list names.
NOT_HELDOUT_LISTS = frozenset({"pool200", "inrepo_test", "verified100"})
# The estimator check's 30 tasks, drawn from heldout691 minus heldout100 (owner's procedure, frozen before P1).
# Those concurrency-1 runs are a separate input, so the main P2 input refuses this list.
ESTIMATOR_LIST = "estimator_check30"
END_REASONS = ("submitted", "turn_limit", "context_limit", "infra_timeout", "error")
MALFORMED = ("misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer")  # issue #1
TURN_FIELDS = ("output_tokens", "input_uncached", "input_cached", "gen_s", "tool_s")
PRIMARY, REFERENCE = "Q4_K_M", "BF16"
WHOLE_ATTEMPT = "whole_attempt_raw_joules"  # energy_model.json ranking.quantity (#13, provisional endpoint)


def need(rec, *keys):
    """rec[k1][k2]..., or a ValueError naming the attempt and the field. A null value counts as missing."""
    x = rec
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
        if x is None:
            raise ValueError(f"attempt {rec.get('attempt_id')!r}: missing field {'.'.join(keys)}")
    return x


def number(x, what, positive=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (positive and x == 0):
        raise ValueError(f"{what} must be a finite {'positive' if positive else 'non-negative'} number, got {x!r}")
    return float(x)


def js(x):
    """JSON-safe number: infinities become "inf" / "-inf" and NaN (an undefined ratio) becomes None."""
    if x is None or isinstance(x, (bool, int)):
        return x
    x = float(x)
    if math.isnan(x):
        return None
    return x if math.isfinite(x) else ("inf" if x > 0 else "-inf")


def check_config(cfg, upper_bound):
    """The analysis settings, checked, as a new dict with infra_error_cap (0.05), families ({}),
    h4_contrasts ([[headline, random]]) and upper_bound_required ([]) filled in. guard_margin has no default:
    docs/04 sets it from P0b's measured SE. h4_contrasts may name "headline" (the P1 pick) and must include
    [headline, random], the H4 contrast docs/03 Cost option b runs."""
    if cfg.get("guard_margin") is None:
        raise ValueError("guard_margin is required (docs/04: set from P0b's measured standard error, frozen before P1)")
    margin = number(cfg["guard_margin"], "guard_margin", positive=True)
    if margin >= 1:
        raise ValueError(f"guard_margin is a resolve-rate difference in (0, 1), got {margin}")
    for key, least in (("n_boot", 100), ("seed", 0)):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), int) or cfg[key] < least:
            raise ValueError(f"{key} must be an integer >= {least}, got {cfg.get(key)!r}")
    if not 0 < number(cfg.get("alpha"), "alpha") < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg['alpha']}")
    cap = number(cfg.get("infra_error_cap", 0.05), "infra_error_cap")
    if cap >= 1:
        raise ValueError(f"infra_error_cap must be in [0, 1), got {cap}")
    if upper_bound:
        for key in ("host_cpu_tdp_w", "host_vcpus"):
            if cfg.get(key) is None:
                raise ValueError(f"{key} is required for the energy upper bound (docs/04)")
            number(cfg[key], key, positive=True)
    families = cfg.get("families") or {}
    for fam, pairs in families.items():
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or not set(pair) <= ARM_NAMES
                    or pair[0] == pair[1]):
                raise ValueError(f"family {fam}: each contrast is [arm, comparator] of two known arms, got {pair!r}")
    h4 = [list(p) for p in cfg.get("h4_contrasts") or [["headline", "random"]]]
    for pair in h4:
        if len(pair) != 2 or not set(pair) <= ARM_NAMES | {"headline"} or pair[0] == pair[1]:
            raise ValueError(f"h4_contrasts: each contrast is [arm, comparator] of two known arms or headline, "
                             f"got {pair!r}")
    if ["headline", "random"] not in h4:
        raise ValueError(f"h4_contrasts must include [headline, random] (docs/03 H4), got {h4!r}")
    ub_required = list(cfg.get("upper_bound_required") or ())
    unknown = set(ub_required) - set(families) - {"H4"}
    if unknown:
        raise ValueError(f"upper_bound_required names families that are not configured: {sorted(unknown)}")
    return {**cfg, "infra_error_cap": cap, "families": families, "h4_contrasts": h4,
            "upper_bound_required": ub_required}


def check_manifest(manifest):
    """model value -> {arm, seed}. Two models for one (arm, seed) fail: either could be the one analyzed."""
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("the models manifest must be a non-empty {model: {arm, seed}} mapping")
    seen = {}
    for model, m in manifest.items():
        arm, seed = (m or {}).get("arm"), (m or {}).get("seed")
        if arm not in ARM_NAMES:
            raise ValueError(f"model {model!r}: unknown arm {arm!r}; the arms are {sorted(ARM_NAMES)}")
        if (seed is None and arm not in BASE_ARMS
                or seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))):
            raise ValueError(f"model {model!r}: seed must be an integer training seed (null only for base), "
                             f"got {seed!r}")
        if (arm, seed) in seen:
            raise ValueError(f"models {seen[arm, seed]!r} and {model!r} are both {arm} seed {seed}")
        seen[arm, seed] = model


def energy_settings(energy_model):
    """The ranking block of energy_model.json, checked: the estimate must be #13's whole-attempt raw joules."""
    em = energy_model if isinstance(energy_model, dict) else {}
    ranking, profiles = em.get("ranking") or {}, em.get("profiles") or {}
    if ranking.get("quantity") != WHOLE_ATTEMPT:
        raise ValueError(f"energy_model.json ranking.quantity is {ranking.get('quantity')!r}, expected "
                         f"{WHOLE_ATTEMPT!r}; refit with the current scripts/p0a_energy_model.py")
    if type(ranking.get("provisional")) is not bool:
        raise ValueError("energy_model.json ranking.provisional must be true or false")
    return {"quantity": WHOLE_ATTEMPT, "provisional": ranking["provisional"],
            "idle_w": {p: v.get("idle_w") for p, v in sorted(profiles.items())},
            "idle_subtracted_fitted": {p: "coef" in (v.get("idle_subtracted") or {})
                                       for p, v in sorted(profiles.items())},
            "rule": "raw generation-window model + the profile's idle_w x tool_seconds (estimate_attempt_joules)"}


def profile_model(energy_model, profile, aid):
    """energy_model.json's entry for the profile of attempt aid, or a ValueError: a failed or missing raw fit, or no
    valid idle_w, cannot estimate the whole-attempt joules of its rows."""
    entry = (energy_model.get("profiles") or {}).get(profile) or {}
    raw = entry.get("raw") or {}
    if "error" in raw:
        raise ValueError(f"attempt {aid!r}: the P0a raw fit for profile {profile!r} failed ({raw['error']}), "
                         "so its rows cannot be analyzed")
    if "coef" not in raw:
        raise ValueError(f"attempt {aid!r}: energy_model.json has no raw model for profile {profile!r}")
    number(entry.get("idle_w"), f"energy_model.json profiles.{profile}.idle_w (W per tool second)")
    return entry


def _check_phase(rec, phase, lst, repo):
    if phase == "p1" and (lst != "inrepo_test" or repo not in POOL_REPOS):
        raise ValueError(f"attempt {rec['attempt_id']!r}: P1 rows are inrepo_test tasks of the pool repos, "
                         f"got list {lst!r}, repo {repo!r}")
    if phase in ("p2", "estimator") and (repo not in HELDOUT_REPOS or lst in NOT_HELDOUT_LISTS):
        raise ValueError(f"attempt {rec['attempt_id']!r}: P2 rows are tasks of the held-out repos {HELDOUT_REPOS}, "
                         f"got list {lst!r}, repo {repo!r}")
    if phase == "p2" and lst == ESTIMATOR_LIST:
        raise ValueError(f"attempt {rec['attempt_id']!r}: {ESTIMATOR_LIST} rows go in the --estimator-check input, "
                         "not the P2 attempts")


def summarize(rec, entry):
    """The per-attempt values the analysis uses, from a non-error record, with every field it reads checked.

    entry: the profile's energy_model.json entry (profile_model). e is the whole-attempt estimate, e_gen its
    generation-window part, e_idle the idle-subtracted fit's estimate (None when that fit failed): with tool
    windows charged at idle_w, nothing is left of them after idle subtraction. tool_seconds must equal the
    per-turn tool_s sum, so a turn-capped attempt (the frontier) adds up to the same estimate.
    """
    where = f"attempt {rec['attempt_id']!r}"
    turns, per_turn = need(rec, "turns"), need(rec, "per_turn")
    if not isinstance(per_turn, list) or not per_turn or turns != len(per_turn):
        raise ValueError(f"{where}: per_turn must be a non-empty list with one entry per turn ({turns!r} turns)")
    for i, t in enumerate(per_turn):
        missing = [k for k in (*TURN_FIELDS, "gen_energy_j", "tool_energy_j") if not isinstance(t, dict) or k not in t]
        if missing:
            raise ValueError(f"{where} turn {i}: missing {missing}")
    tool_s = sum(number(t["tool_s"], f"{where} turn {i} tool_s") for i, t in enumerate(per_turn))
    tool_seconds = number(need(rec, "tool_seconds"), f"{where} tool_seconds")
    # The harness writes tool_seconds as sum(per_turn[].tool_s) in turn order, the sum taken here, so the two are
    # equal; 1e-9 is the owner's suggested tolerance.
    if not math.isclose(tool_s, tool_seconds, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{where}: tool_seconds {tool_seconds} is not the per-turn tool_s sum {tool_s}")
    # docs/04 Test-file tripwire; a required field (format pin), so a missing flag is an error, never "unflagged".
    if type(need(rec, "grading").get("test_files_touched")) is not bool:
        raise ValueError(f"{where}: grading.test_files_touched must be true or false")
    if type(rec.get("resolved")) is not bool:
        raise ValueError(f"{where}: resolved must be true or false, got {rec.get('resolved')!r}")
    if type(need(rec, "energy", "attributable")) is not bool:
        raise ValueError(f"{where}: energy.attributable must be true or false")
    need(rec, "concurrency")
    malformed = {k: number(need(rec, "malformed", k), f"{where} malformed.{k}") for k in MALFORMED}
    # Processed tokens as in qtrace/p0_analysis.py (#6): output plus every input token, cache hits and misses.
    processed = sum(number(need(rec, "tokens", k), f"{where} tokens.{k}") for k in ("output", "input_uncached",
                                                                                  "input_cached"))
    # docs/04 Runtimes: hitting a budget (turns, context, the wall-clock cap) is a failure, so only a submitted
    # attempt can count as resolved, whatever a harness that grades the final repo state wrote.
    resolved = rec["resolved"] and rec["end_reason"] == "submitted"
    sub = entry.get("idle_subtracted") or {}
    return {"attempt_id": rec["attempt_id"], "r": float(resolved),
            "e": estimate_attempt_joules(rec, entry["raw"], idle_w=entry["idle_w"]),
            "e_gen": estimate_attempt_joules(rec, entry["raw"]),
            "e_idle": estimate_attempt_joules(rec, sub) if "coef" in sub else None,
            "flagged": float(rec["grading"]["test_files_touched"]),
            "p": processed, "cpu_s": number(need(rec, "container_cpu_seconds"), f"{where} container_cpu_seconds"),
            "timeout": rec["end_reason"] == "infra_timeout", "budget_hit_resolved": rec["resolved"] and not resolved,
            "malformed": malformed, "rec": rec}


def load(records, manifest, energy_model, phase, every_model=True, grammar=False):
    """Check attempt rows and index them as cells[(arm, rep, profile)][task_id] -> summarize() output.

    phase "p1" accepts only inrepo_test rows of the pool repos, "p2" only rows of the held-out repos and not the
    estimator check's list, "estimator" (the --estimator-check input) held-out rows including that list. Every
    record must have a bool grammar; only rows whose grammar equals the argument are indexed, so the main
    analysis (grammar false) and H4 (grammar true) never share a row. Attempt ids are unique over all records. A
    cell
    entry is None when every attempt there ended in an infrastructure error; an error attempt next to a
    finished one is the rerun docs/04 asks for, and the finished one is used. Fails on duplicate attempt
    ids, unknown models, a profile without an energy model, more than one finished attempt per task per seed
    model per profile, and an incomplete design: a manifest model with no rows (unless every_model is false,
    for a run of only some models), a cell missing a task that another cell ran under the same task list, or
    an arm with different seed models at different profiles. Cells may run different lists (P2's secondary
    tier runs a held-out subset), and arrays() uses the tasks the groups share. The task lists themselves are
    not an input, so a task no cell of its list ran is not caught.
    """
    if phase not in ("p1", "p2", "estimator"):
        raise ValueError(f"phase must be 'p1', 'p2' or 'estimator', got {phase!r}")
    check_manifest(manifest)
    energy = energy_settings(energy_model)
    seen, repo_of, rows, models, n_other = set(), {}, defaultdict(list), set(), 0
    lists, list_tasks = defaultdict(set), defaultdict(set)  # task lists per cell, tasks per task list
    for rec in records:
        aid = need(rec, "attempt_id")
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r}")
        seen.add(aid)
        task, lst, repo, profile, model, end = (need(rec, k) for k in ("task_id", "list", "repo", "profile", "model",
                                                                      "end_reason"))
        need(rec, "seed")
        if type(rec.get("grammar")) is not bool:
            raise ValueError(f"attempt {aid!r}: grammar must be true or false, got {rec.get('grammar')!r}")
        if end not in END_REASONS:
            raise ValueError(f"attempt {aid!r}: end_reason {end!r} is not one of {END_REASONS}")
        _check_phase(rec, phase, lst, repo)
        if repo_of.setdefault(task, repo) != repo:
            raise ValueError(f"task {task!r} is in repo {repo_of[task]!r} and {repo!r}")
        if model not in manifest:
            raise ValueError(f"attempt {aid!r}: model {model!r} is not in the models manifest")
        profile_model(energy_model, profile, aid)
        if rec["grammar"] is not grammar:
            n_other += 1
            continue
        m = manifest[model]
        rep = m["seed"] if m["seed"] is not None else f"sampling{rec['seed']}"
        rows[m["arm"], rep, profile, task].append(rec)
        models.add(model)
        lists[m["arm"], rep, profile].add(lst)
        list_tasks[lst].add(task)
    unused = sorted(set(manifest) - models)
    if every_model and unused:
        raise ValueError(f"incomplete design: manifest models {unused} have no attempts (a missing attempt file?)")

    cells, n_reruns, n_budget = defaultdict(dict), 0, 0
    for (arm, rep, profile, task), recs in rows.items():
        done = [r for r in recs if r["end_reason"] != "error"]
        if len(done) > 1:
            raise ValueError(f"{arm} seed {rep} at {profile}: {len(done)} attempts on task {task!r} "
                             f"({[r['attempt_id'] for r in done]}); the design is 1 per task per seed model")
        n_reruns += bool(done) and len(recs) > 1
        cells[arm, rep, profile][task] = s = summarize(done[0], energy_model["profiles"][profile]) if done else None
        n_budget += bool(s and s["budget_hit_resolved"])

    reps = defaultdict(dict)
    for (arm, rep, profile), cell in cells.items():
        missing = sorted(set().union(*(list_tasks[lst] for lst in lists[arm, rep, profile])) - set(cell))
        if missing:
            raise ValueError(f"incomplete design: {arm} seed {rep} at {profile} has no attempt on {len(missing)} "
                             f"task(s), e.g. {missing[:3]}")
        reps[arm].setdefault(profile, set()).add(rep)
    for arm, by_profile in reps.items():
        if len({frozenset(r) for r in by_profile.values()}) > 1:
            seen_reps = {p: sorted(map(str, r)) for p, r in by_profile.items()}
            raise ValueError(f"incomplete design: {arm} has different seed models per profile: {seen_reps}")
    return {"cells": dict(cells), "repo": repo_of, "n_rows": len(seen) - n_other, "n_rows_other_grammar": n_other,
            "n_reruns_used": n_reruns,
            "n_resolved_after_budget_hit": n_budget, "energy": energy}


def present(data):
    """(arm, profile) pairs with attempts."""
    return {(arm, profile) for arm, _, profile in data["cells"]}


def reps_of(data, arm, profile):
    # Training seeds are ints and base sampling replicates strings; never both within one arm.
    return sorted((rep for a, rep, p in data["cells"] if (a, p) == (arm, profile)),
                  key=lambda r: (isinstance(r, str), r))


def arrays(data, groups, cap, cpu_w=None, drop_timeouts=False, only=None):
    """Seed-level arrays [S, n] per (arm, profile) group, on the tasks every group ran and can use.

    Keys: E (estimated whole-attempt GPU joules), G (its generation-window part), I (idle-subtracted, None when a
    profile's idle-subtracted fit failed), U (E plus container CPU-seconds x cpu_w, when cpu_w is given), R
    (resolved), F (test-file tripwire flag), P (processed tokens), reps (seed models, in row order). A task is
    dropped for every group when any seed model of any group has only an infrastructure-error attempt on it; more
    than `cap` of the tasks dropped is an error. drop_timeouts also drops tasks with an infra_timeout attempt
    (docs/04's sensitivity analysis; everywhere else they are charged as failures). only, when given, limits the
    tasks to that set (H4 compares grammar on and off on the same tasks).
    """
    cells = {}
    for g in groups:
        reps = reps_of(data, *g)
        if not reps:
            raise ValueError(f"no attempts for arm {g[0]!r} at {g[1]}")
        cells[g] = (reps, [data["cells"][g[0], r, g[1]] for r in reps])
    every = set.intersection(*(set(c) for _, cs in cells.values() for c in cs))
    every = sorted(every if only is None else every & set(only))
    errors = {t for _, cs in cells.values() for c in cs for t in every if c[t] is None}
    if len(errors) > cap * len(every):
        raise ValueError(f"{len(errors)} of {len(every)} tasks have an unrerun infrastructure error in "
                         f"{sorted(cells)}, above infra_error_cap {cap}")
    timeouts = {t for _, cs in cells.values() for c in cs for t in every
                if c[t] is not None and c[t]["timeout"]} - errors if drop_timeouts else set()
    tasks = [t for t in every if t not in errors and t not in timeouts]
    if not tasks:
        raise ValueError(f"no task is left in {sorted(cells)} after infrastructure drops")
    out = {}
    for g, (reps, cs) in cells.items():
        m = {k: np.array([[c[t][k] for t in tasks] for c in cs], dtype=float)
             for k in ("e", "e_gen", "r", "p", "cpu_s", "flagged")}
        idle = [[c[t]["e_idle"] for t in tasks] for c in cs]
        out[g] = {"E": m["e"], "G": m["e_gen"], "R": m["r"], "P": m["p"], "F": m["flagged"], "reps": reps,
                  "I": None if any(v is None for row in idle for v in row) else np.array(idle, dtype=float),
                  "U": None if cpu_w is None else m["e"] + m["cpu_s"] * cpu_w}
    return {"tasks": tasks, "repos": np.array([data["repo"][t] for t in tasks]), "groups": out,
            "n_dropped_error": len(errors), "n_dropped_timeout": len(timeouts), "n_tasks_input": len(every)}


def strat_weights(repos, n_boot, rng):
    """Bootstrap task counts [n_boot, n]: tasks drawn with replacement within each repo, so every draw keeps each
    repo's task count. qtrace/p0d.py (#12) _boot_weights applied per repo, as counts rather than shares (the
    ratios do not change, and resolve rates divide by the fixed n)."""
    W = np.zeros((n_boot, len(repos)))
    for repo in sorted(set(repos)):
        idx = np.flatnonzero(repos == repo)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
    return W
