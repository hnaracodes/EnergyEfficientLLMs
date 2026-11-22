"""P0d: the score-choice rule, the replay pairs, the continuation DiD test and its power (docs/04, P0d).

Pure functions over the formats agreed in issue #1: attempt records, outputs/p0d/scores.jsonl rows
(one per attempt and probe, holding every key of qtrace.kl.score_stats), est_joules.jsonl rows and
continuation results. scripts/p0d_analyze.py does the file I/O.

Within-task statistics are Spearman correlations with average ranks for ties, computed per task and
summarized as the mean over tasks (the rule uses the mean) with the median alongside. A task enters
a statistic only with at least MIN_RUNS runs that have both values; a task where either variable is
constant has no rho and is counted, not scored as 0.

Spearman-Brown is applied to the mean (and the median) of the per-task even-vs-odd rhos, not per
task before averaging. With 3 runs a per-task rho is one of -1, -0.5, 0.5 and 1, and 2r / (1 + r) maps
-0.5 to -2 and -1 to minus infinity, so correcting per task would let a few tasks dominate the mean.
The correction is monotone, so the median is the same either way.
"""

import math
from collections import defaultdict

import numpy as np

CANDIDATES = ("kl_decision", "kl_command", "kl_mean")  # frozen order, docs/04 (2026-11-20)
PROBE, AGREEMENT_PROBE = "Q3_K_M", "Q4_K_M"
HELD_OUT_LISTS = ("heldout100", "verified100", "inrepo_test")  # never part of any pool (docs/04, Task sets)
POOL_LISTS = ("pool200",)  # the default of p0d_score_pool.py --lists
CONFOUNDS = ("turns", "output_tokens", "est_joules_q4", "entropy_bf16")
COVARIATES = ("entropy", "turns", "malformed")
# The issue #1 malformed categories, the same tuple as qtrace/exit_report.py (#7) and qtrace/p1_select.py (#14).
MALFORMED = ("misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer")
MAX_ABS_RHO, MIN_SPLIT_HALF, MIN_AGREEMENT, MIN_RUNS = 0.8, 0.5, 0.5, 3
MIN_EFFECT = 0.10
SMALL_SAMPLE = 50  # docs/04's "fewer than about 50 qualify"
SMALL_SAMPLE_NOTE = ("fewer than 50 tasks: the task bootstrap is liberal here (a null false-positive rate of about "
                     "0.066-0.071 at 10 tasks in the owner's review of PR #12), and the upper bound shares this")
# Means of 3- to 5-run task rhos sit on a lattice that contains 0.8, 0.5 and 1/3, and float sums land
# them a few ulps to either side (rhos -1/2, 1, 1 average to 0.49999999999999994). The lattice spacing
# is far above this, so the tolerance moves only those float errors.
THRESHOLD_TOL = 1e-12
UPPER_BOUND_DEFINITION = ("95th percentile (inverted_cdf order statistic) of the task-bootstrap means: the "
                          "one-sided 95% interval is (-inf, upper]")
COVARIATE_CONVENTION = ("each covariate is the low-score run minus the high-score run of the task's pair, in raw "
                        "units (BF16 entropy_decision, the mean entropy at decision positions, in nats; turns; "
                        "malformed calls), uncentered, so the intercept "
                        "is the DiD of a pair that does not differ in any covariate")


def flagged(rec):
    """The grader's test-file tripwire (docs/04, "Test-file tripwire").

    grading.test_files_touched is a required bool of every attempt record, so a missing key raises
    instead of silently turning the tripwire off.
    """
    v = rec["grading"]["test_files_touched"]
    if not isinstance(v, bool):
        raise TypeError(f"attempt {rec.get('attempt_id')!r}: grading.test_files_touched must be a bool, got {v!r}")
    return v


def malformed_total(rec, what):
    """Malformed calls summed over MALFORMED; a missing or unknown category raises."""
    m = _field(rec, "malformed", what)
    if set(m) != set(MALFORMED):
        raise ValueError(f"{what}: malformed has keys {sorted(m)}, expected {list(MALFORMED)}")
    return sum(_number(m[k], f"{what} malformed.{k}", optional=False) for k in MALFORMED)


def average_ranks(x):
    """1-based ranks, tied values sharing the mean of their positions."""
    _, inverse, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(x, y):
    """Spearman's rho with average ranks; None when either variable is constant (no rho exists)."""
    rx, ry = average_ranks(x), average_ranks(y)
    if len(rx) < 2 or rx.min() == rx.max() or ry.min() == ry.max():
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_brown(r):
    """Full-length reliability 2r / (1 + r) from a half-length correlation r; None at r = -1."""
    return None if r is None or r <= -1 else 2 * r / (1 + r)


def within_task(groups, min_runs=MIN_RUNS):
    """Per-task Spearman over {task: [(x, y), ...]}, summarized over tasks.

    Pairs with a None on either side are dropped and counted (n_runs_missing_value). Tasks left with
    fewer than min_runs pairs are counted in n_tasks_too_few_runs, constant ones in n_tasks_constant.
    """
    rhos, constant, too_few, missing = [], 0, 0, 0
    for task in sorted(groups):
        pts = [(x, y) for x, y in groups[task] if x is not None and y is not None]
        missing += len(groups[task]) - len(pts)
        if len(pts) < min_runs:
            too_few += 1
            continue
        r = spearman(*zip(*pts))
        if r is None:
            constant += 1
        else:
            rhos.append(r)
    return {"mean": float(np.mean(rhos)) if rhos else None, "median": float(np.median(rhos)) if rhos else None,
            "n_tasks": len(rhos), "n_tasks_constant": constant, "n_tasks_too_few_runs": too_few,
            "n_runs_missing_value": missing}


def _number(value, what, optional=True):
    """A finite number, or None if optional; anything else (NaN, inf, a string) is a broken input."""
    if (value is None and optional) or \
            (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
        return value
    raise ValueError(f"{what}: expected a finite number{' or null' if optional else ''}, got {value!r}")


def _field(rec, key, what):
    if key not in rec:
        raise ValueError(f"{what}: missing field {key!r}")
    return rec[key]


def _scores(row, what):
    return {c: _number(_field(row, c, what), f"{what} {c}") for c in CANDIDATES}


def join_runs(attempts, scores, est_joules=None, lists=POOL_LISTS, *, run_ids):
    """One record per scored pool run, joining attempt fields, both probes' scores and estimated joules.

    run_ids: the manifest's P0b run ids (scripts/p0_analyze.py's --manifest). Only attempts of those
    runs are considered, for the runs and for every count below: P0c's BF16 attempts on pool200 are
    another draw, and a score row for an attempt outside the set raises ValueError. Runs are the
    attempts with a Q3_K_M score row. Each must be a resolved BF16 attempt on one of
    `lists` (the scorer's --lists); anything else (an unknown attempt, another profile, a failure,
    another list, a duplicate row, a non-finite number) raises ValueError, since it means the scorer
    read the wrong files. A BF16 attempt on `lists` that ended in end_reason error is an infrastructure
    fault, rerun as a fresh attempt (docs/04), so it is never a run, as in qtrace.p1_select.candidates;
    info's infra_error_excluded counts them per list. A run the test-file tripwire flagged can never be
    a selection candidate and is excluded. info's flagged_excluded counts, per list, the resolved BF16
    attempts it flagged, from the attempt records, because the scorer writes no rows for them. info's n_eligible_without_q3
    counts the resolved, unflagged BF16 attempts on `lists` with no Q3_K_M row (the scorer still
    running or a row lost); choose_score must not decide while it is non-zero. est_joules:
    est_joules.jsonl rows, or None if P0a's file does not exist yet. Runs without a Q4_K_M row or an
    est_joules value are counted in info, and candidate_stats does not run the check they feed.
    Returns (runs, info).
    """
    bad = sorted(set(lists) & set(HELD_OUT_LISTS))
    if bad:
        raise ValueError(f"{bad} are held-out lists and never enter a pool")
    by_id, flagged_by_list, errors_by_list, eligible, other_runs = {}, defaultdict(int), defaultdict(int), set(), {}
    for a in attempts:
        aid = _field(a, "attempt_id", "attempt")
        if aid in by_id or aid in other_runs:
            raise ValueError(f"duplicate attempt_id {aid!r} in the attempts files")
        what = f"attempt {aid}"
        if _field(a, "run_id", what) not in run_ids:
            other_runs[aid] = a["run_id"]
            continue
        by_id[aid] = a
        if _field(a, "profile", what) == "BF16" and _field(a, "list", what) in lists and \
                _field(a, "end_reason", what) == "error":
            errors_by_list[a["list"]] += 1
            continue
        if _field(a, "profile", what) == "BF16" and _field(a, "resolved", what) and _field(a, "list", what) in lists:
            if flagged(a):
                flagged_by_list[a["list"]] += 1
            else:
                eligible.add(aid)
    rows = {}
    for s in scores:
        key = (_field(s, "attempt_id", "score row"), _field(s, "probe", "score row"))
        if key[1] not in (PROBE, AGREEMENT_PROBE):
            raise ValueError(f"score row {key}: unknown probe")
        if key in rows:
            raise ValueError(f"duplicate score row for {key}")
        rows[key] = s
    joules = {}
    for j in est_joules or ():
        aid = _field(j, "attempt_id", "est_joules row")
        if aid in joules:
            raise ValueError(f"duplicate est_joules row for {aid!r}")
        joules[aid] = _number(_field(j, "est_joules_q4", f"est_joules {aid}"), f"est_joules {aid}")

    runs = []
    for (aid, probe), s in sorted(rows.items()):
        if probe != PROBE:
            continue
        what = f"attempt {aid}"
        if aid in other_runs:
            raise ValueError(f"{what}: scored, but its run {other_runs[aid]!r} is not a P0b run of the manifest "
                             f"{sorted(run_ids)}")
        if aid not in by_id:
            raise ValueError(f"{what}: scored but not in the attempts files")
        a = by_id[aid]
        if _field(a, "list", what) in HELD_OUT_LISTS:
            raise ValueError(f"{what}: list {a['list']!r} is held-out and never enters a pool")
        if a["list"] not in lists:
            raise ValueError(f"{what}: list {a['list']!r} is not one of the pool lists {list(lists)}")
        if _field(a, "profile", what) == "BF16" and _field(a, "end_reason", what) == "error":
            continue  # counted in infra_error_excluded; an older scores file may still hold its row
        if _field(a, "profile", what) != "BF16" or not _field(a, "resolved", what):
            raise ValueError(f"{what}: pool runs are resolved BF16 attempts (profile {a['profile']!r}, "
                             f"resolved {a['resolved']!r})")
        if _field(s, "task_id", what) != _field(a, "task_id", what):
            raise ValueError(f"{what}: score row task {s['task_id']!r} differs from the attempt's {a['task_id']!r}")
        if flagged(a):
            continue
        q4 = rows.get((aid, AGREEMENT_PROBE))
        halves = s.get("halves") or {}
        score = _scores(s, what)
        entropy = _number(_field(s, "entropy_mean", what), f"{what} entropy_mean")
        if entropy is None and any(v is not None for v in score.values()):
            raise ValueError(f"{what}: entropy_mean is null but the row has KL scores")  # kl.py nulls both together
        entropy_dec = _number(_field(s, "entropy_decision", what), f"{what} entropy_decision")
        if (entropy_dec is None) != (score["kl_decision"] is None):
            raise ValueError(f"{what}: entropy_decision and kl_decision must be null together")  # same mask in kl.py
        runs.append({
            "attempt_id": aid, "task_id": a["task_id"],
            "turns": _number(_field(a, "turns", what), f"{what} turns", optional=False),
            "output_tokens": _number(_field(_field(a, "tokens", what), "output", what), f"{what} tokens.output",
                                     optional=False),
            "malformed": malformed_total(a, what),
            "est_joules_q4": joules.get(aid),
            "entropy_bf16": entropy,
            "entropy_decision_bf16": entropy_dec,
            "score": score,
            "q4": _scores(q4, f"{what} {AGREEMENT_PROBE}") if q4 else None,
            "even": _scores(halves["even"], f"{what} even half") if "even" in halves else None,
            "odd": _scores(halves["odd"], f"{what} odd half") if "odd" in halves else None,
        })
    q3_ids = {aid for aid, p in rows if p == PROBE}
    info = {"n_runs": len(runs), "n_tasks": len({r["task_id"] for r in runs}),
            "flagged_excluded": dict(sorted(flagged_by_list.items())),
            "infra_error_excluded": dict(sorted(errors_by_list.items())),
            "q4_rows_without_q3": sum(1 for aid, p in rows if p == AGREEMENT_PROBE and aid not in q3_ids),
            "lists": list(lists), "run_ids": sorted(run_ids), "n_eligible_without_q3": len(eligible - q3_ids),
            "n_runs_without_q4": sum(1 for r in runs if r["q4"] is None),
            "n_runs_without_halves": sum(1 for r in runs if r["even"] is None or r["odd"] is None),
            "est_joules_given": est_joules is not None,
            "n_runs_without_est_joules": sum(1 for r in runs if r["est_joules_q4"] is None)}
    return runs, info


def _grouped(runs, pair):
    out = defaultdict(list)
    for r in runs:
        out[r["task_id"]].append(pair(r))
    return out


def _check(value, ok):
    """True or False, or None when the statistic could not be computed (not run)."""
    return None if value is None else bool(ok(value))


def candidate_stats(runs, name, have_joules=True, min_runs=MIN_RUNS):
    """Every number docs/04 asks for one candidate, the six checks and whether it passes.

    The joules, probe-agreement and split-half checks do not run while any run lacks its est_joules
    value, its Q4_K_M row or either turn-parity half (P0a or the scorer still running): on the few
    tasks that happen to be complete they could pass a candidate the full pool fails.
    """
    no_joules = sum(1 for r in runs if r["est_joules_q4"] is None)
    no_q4 = sum(1 for r in runs if r["q4"] is None)
    no_halves = sum(1 for r in runs if r["even"] is None or r["odd"] is None)
    against = {}
    for k in CONFOUNDS:
        if k == "est_joules_q4" and not have_joules:
            against[k] = {"not_run": "no est_joules.jsonl given (P0a energy model not fitted yet)"}
        elif k == "est_joules_q4" and no_joules:
            against[k] = {"not_run": f"{no_joules} of {len(runs)} runs have no est_joules_q4"}
        else:
            against[k] = within_task(_grouped(runs, lambda r, k=k: (r["score"][name], r[k])), min_runs)
    agreement = ({"not_run": f"{no_q4} of {len(runs)} runs have no {AGREEMENT_PROBE} row"} if no_q4 else
                 within_task(_grouped(runs, lambda r: (r["score"][name], r["q4"][name])), min_runs))
    if no_halves:
        rho, split = {}, {"not_run": f"{no_halves} of {len(runs)} runs lack the even or odd half"}
    else:
        rho = within_task(_grouped(runs, lambda r: (r["even"][name], r["odd"][name])), min_runs)
        split = {"rho": rho, "spearman_brown_mean": spearman_brown(rho["mean"]),
                 "spearman_brown_median": spearman_brown(rho["median"])}
    checks = {k: _check(against[k].get("mean"), lambda m: abs(m) <= MAX_ABS_RHO + THRESHOLD_TOL) for k in CONFOUNDS}
    # SB(r) >= 0.5 exactly when r >= 1/3; testing the raw mean keeps r = -1 a failure, not "not run".
    checks["split_half"] = _check(rho.get("mean"), lambda m: m >= MIN_SPLIT_HALF / (2 - MIN_SPLIT_HALF) - THRESHOLD_TOL)
    checks["probe_agreement"] = _check(agreement.get("mean"), lambda m: m >= MIN_AGREEMENT - THRESHOLD_TOL)
    failed = [k for k, v in checks.items() if v is False]
    not_run = [k for k, v in checks.items() if v is None]
    return {"against": against, "probe_agreement": agreement, "split_half": split, "checks": checks,
            "failed": failed, "not_run": not_run, "pass": not failed and not not_run}


def choose_score(runs, have_joules=True, min_runs=MIN_RUNS, *, missing_q3):
    """docs/04's rule: the first candidate, in the frozen order, that passes every check.

    A candidate that fails a check is skipped. One that fails none but has a check that could not run
    (an est_joules value or a Q4_K_M row missing for any run, no task with enough runs) stops the walk:
    chosen stays None and decided is False, because a later candidate may only be chosen once the
    earlier one has failed. missing_q3 (join_runs' n_eligible_without_q3) above 0 leaves the rule
    undecided whatever the candidates show, since every check ran on part of the pool. It has no
    default, so a caller cannot skip that check by leaving it out.
    decided with chosen None answers H3 negatively and drops the stability arms before P1.
    """
    candidates = {c: candidate_stats(runs, c, have_joules, min_runs) for c in CANDIDATES}
    chosen, decided = None, True
    verdict = "none passed: H3 answered negatively, stability arms dropped before P1"
    for c in CANDIDATES:
        if candidates[c]["pass"]:
            chosen, verdict = c, f"chosen: {c}"
            break
        if not candidates[c]["failed"]:
            decided = False
            verdict = f"incomplete: {c} fails no check but {', '.join(candidates[c]['not_run'])} did not run"
            break
    if missing_q3:
        chosen, decided, verdict = None, False, f"incomplete: {missing_q3} eligible pool attempts have no {PROBE} score row"
    return {"chosen": chosen, "decided": decided, "verdict": verdict, "order": list(CANDIDATES), "probe": PROBE,
            "agreement_probe": AGREEMENT_PROBE,
            "rule": {"max_abs_rho": MAX_ABS_RHO, "min_split_half": MIN_SPLIT_HALF, "min_probe_agreement": MIN_AGREEMENT,
                     "min_runs_per_task": min_runs, "statistic": "mean over tasks of the within-task Spearman",
                     "spearman_brown": "applied to the mean (and median) of the per-task even-vs-odd rhos"},
            "candidates": candidates}


def select_pairs(runs, score):
    """The lowest- and highest-scoring run of every task with at least 2 scored runs, under `score`.

    Runs are ordered by (score, attempt_id): low is the first, high the last, so ties go to the
    smaller attempt_id for low and the larger for high, and the two are always different runs. diff
    holds the DiD covariates (COVARIATE_CONVENTION). low_kl_mean and high_kl_mean are frozen with the
    pair for the secondary kl_mean-ordered DiD (did_kl_mean).
    """
    by_task, unscored = defaultdict(list), 0
    for r in runs:
        if r["score"][score] is None:
            unscored += 1
        else:
            by_task[r["task_id"]].append(r)
    pairs = []
    for task in sorted(by_task):
        rs = sorted(by_task[task], key=lambda r: (r["score"][score], r["attempt_id"]))
        if len(rs) < 2:
            continue
        lo, hi = rs[0], rs[-1]
        pairs.append({"task_id": task, "low": lo["attempt_id"], "high": hi["attempt_id"],
                      "low_score": lo["score"][score], "high_score": hi["score"][score],
                      "low_kl_mean": lo["score"]["kl_mean"], "high_kl_mean": hi["score"]["kl_mean"],
                      "spread": hi["score"][score] - lo["score"][score], "n_runs": len(rs),
                      "diff": {"entropy": lo["entropy_decision_bf16"] - hi["entropy_decision_bf16"],
                               "turns": lo["turns"] - hi["turns"],
                               "malformed": lo["malformed"] - hi["malformed"]}})
    spreads = [p["spread"] for p in pairs]
    summary = ({"mean": float(np.mean(spreads)), "median": float(np.median(spreads)), "min": min(spreads),
                "max": max(spreads)} if spreads else None)
    return {"score": score, "probe": PROBE, "n_tasks": len(pairs), "fewer_than_50": len(pairs) < 50,
            "n_tasks_one_run": sum(1 for rs in by_task.values() if len(rs) < 2), "n_runs_unscored": unscored,
            "n_tasks_zero_spread": sum(1 for s in spreads if s == 0), "spread": summary,
            "tie_break": "order by (score, attempt_id); low = first, high = last",
            "covariates": COVARIATE_CONVENTION, "tasks": pairs}


def _boot_counts(n, n_boot, rng):
    """Task-bootstrap draws: row b counts how often each task is drawn in draw b.

    Callers multiply the counts by integer DiDs (the DiD times a common denominator), so a resampled
    mean of exactly 0 is the integer 0. Float weights left some at +1e-17, outside "<= 0", which only
    ever lowered p.
    """
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)


def _continuation_cells(rows, profile, pairs=None):
    """(task, prefix, profile) -> [successes, attempts], the tasks with rows of BF16 or `profile`, the
    tasks with an invalid replay on a row of BF16 or `profile`, rows of other profiles ignored (valid or not).

    pairs: {task_id: frozen pair}; each row must then name the pair's runs in low_attempt_id and
    high_attempt_id, so a continuation can only be scored against the pair it replayed.
    """
    cells, tasks, invalid, ignored = defaultdict(lambda: [0, 0]), set(), set(), defaultdict(int)
    for r in rows:
        what = f"continuation row {r!r}"
        task, prefix, prof, valid = (_field(r, k, what) for k in ("task_id", "prefix", "profile", "replay_valid"))
        if prefix not in ("low", "high") or not isinstance(valid, bool):
            raise ValueError(f"{what}: prefix must be 'low' or 'high' and replay_valid a boolean")
        if pairs is not None:
            if task not in pairs:
                raise ValueError(f"{what}: task {task!r} is not a frozen replay pair")
            ids = (_field(r, "low_attempt_id", what), _field(r, "high_attempt_id", what))
            if ids != (pairs[task]["low"], pairs[task]["high"]):
                raise ValueError(f"{what}: replayed runs {ids} are not the frozen pair "
                                 f"({pairs[task]['low']!r}, {pairs[task]['high']!r})")
        if prof not in ("BF16", profile):
            ignored[prof] += 1  # a secondary run's own replay drops must not change the primary sample
            continue
        if not valid:
            invalid.add(task)  # the harness writes these without results, so successes are not read
            continue
        s, n = (_field(r, k, what) for k in ("successes", "attempts"))
        if not (isinstance(s, int) and isinstance(n, int) and 0 <= s <= n):
            raise ValueError(f"{what}: need integer 0 <= successes <= attempts")
        tasks.add(task)
        cells[task, prefix, prof][0] += s
        cells[task, prefix, prof][1] += n
    return cells, tasks, invalid, dict(ignored)


def _adjusted(did, cov):
    """OLS of per-task DiD on the covariate differences; the adjusted estimate is the intercept.

    Covariates enter in COVARIATES order, and one that does not raise the design's rank is dropped and
    named in `dropped` with its reason: constant over the pairs (all zero, or a non-zero constant the
    intercept absorbs) or collinear with the columns before it. The estimate is then not adjusted for
    it. A non-zero constant is also named in `note`: the intercept then holds that covariate's effect
    too, so the estimate is not separated from it (an all-zero one leaves nothing to separate). With no
    more tasks than coefficients the fit has no residual and the estimate is None.
    """
    x = np.array([[cov_t[k] for k in COVARIATES] for cov_t in cov], dtype=float)
    design, kept, dropped, absorbed = np.ones((len(did), 1)), [], {}, []
    for i, k in enumerate(COVARIATES):
        trial = np.column_stack([design, x[:, i]])
        if np.linalg.matrix_rank(trial) > design.shape[1]:
            design, kept = trial, kept + [k]
        elif np.ptp(x[:, i]) == 0:
            dropped[k] = f"constant ({x[0, i]:g}) over the pairs, so the intercept absorbs it"
            if x[0, i] != 0:
                absorbed.append(k)
        else:
            dropped[k] = "collinear with the intercept and the covariates before it"
    note = (f"the adjusted estimate is not separated from the effect of {', '.join(absorbed)}: constant and "
            "non-zero over the pairs, it is absorbed by the intercept" if absorbed else None)
    out = {"dropped": dropped, "note": note, "convention": COVARIATE_CONVENTION}
    if len(did) <= design.shape[1]:
        return {**out, "estimate": None,
                "reason": f"{len(did)} tasks leave no residual degrees of freedom for {design.shape[1]} coefficients"}
    beta = np.linalg.lstsq(design, did, rcond=None)[0]
    return {**out, "estimate": float(beta[0]), "coef": {k: float(b) for k, b in zip(kept, beta[1:])}}


def _keeps(mean, adj):
    """docs/04's pass condition on the adjusted estimate: it keeps the DiD's sign and at least half its size."""
    return bool(adj is not None and mean != 0 and np.sign(adj) == np.sign(mean) and abs(adj) >= 0.5 * abs(mean))


def did_test(rows, covariates=None, profile=PROBE, n_boot=5000, seed=0, alpha=0.05, min_effect=MIN_EFFECT,
             pairs=None):
    """docs/04's P0d test on continuation results.

    rows: {task_id, prefix "low"|"high", profile, replay_valid, successes, attempts}, plus
    low_attempt_id and high_attempt_id when `pairs` is given; rows of one cell are summed (one row per
    cell or per seed both work). Rows for profiles other than BF16 and `profile` are ignored and
    counted, replay_valid false or not (Q4_K_M continuations are a separate, secondary run). A pair is
    dropped if any of its BF16 or `profile` rows has replay_valid false; those rows need no successes
    or attempts. Per task DiD = (Q - BF16 success rate on the low prefix) - (Q - BF16 on the high
    prefix). p = share of task-bootstrap means <= 0.
    covariates: {task_id: {entropy, turns, malformed}} per COVARIATE_CONVENTION, for the adjusted
    estimate.

    pairs: the frozen pair list (select_pairs' "tasks", read back from score_choice.json). Every row
    must belong to one of them and name its runs, the covariates default to the pairs' diff, and the
    pair drop rate is over all of them. A pair with neither a dropped replay nor BF16 and `profile`
    rows (none at all, or only rows of other profiles) is listed in pair_tasks_without_continuations
    and the decision is "incomplete": the test would otherwise run on whichever pairs finished first.
    """
    frozen = None if pairs is None else {p["task_id"]: p for p in pairs}
    cells, tasks, invalid, ignored = _continuation_cells(rows, profile, frozen)
    if frozen is not None and covariates is None:
        covariates = {t: p["diff"] for t, p in frozen.items()}
    selected = set(frozen) if frozen is not None else tasks | invalid
    without = sorted(selected - tasks - invalid)
    only_other = sorted(t for t in without if any(r["task_id"] == t for r in rows))
    kept = sorted(tasks - invalid)
    if not kept:
        raise ValueError("no task has a valid replayed pair")
    for t in kept:
        for prefix in ("low", "high"):
            for prof in ("BF16", profile):
                if cells[t, prefix, prof][1] == 0:
                    raise ValueError(f"task {t}: no {prof} attempts on the {prefix} prefix")
    # Exact arithmetic: each DiD times the common denominator of every cell's attempts is an integer.
    denom = math.lcm(*(cells[t, prefix, prof][1] for t in kept for prefix in ("low", "high")
                       for prof in ("BF16", profile)))
    sign = {("low", profile): 1, ("low", "BF16"): -1, ("high", profile): -1, ("high", "BF16"): 1}
    did_int = np.array([sum(k * cells[t, prefix, prof][0] * (denom // cells[t, prefix, prof][1])
                            for (prefix, prof), k in sign.items()) for t in kept], dtype=np.int64)
    did = did_int / denom
    sums = _boot_counts(len(did), n_boot, np.random.default_rng(seed)) @ did_int
    p = float(np.mean(sums <= 0))
    upper = float(np.percentile(sums, 95, method="inverted_cdf") / (len(did) * denom))
    adjusted = None
    if covariates is not None:
        missing = [t for t in kept if t not in covariates]
        if missing:
            raise ValueError(f"no covariates for tasks {missing[:5]}")
        adjusted = _adjusted(did, [{k: _number(_field(covariates[t], k, f"covariates {t}"), f"covariates {t} {k}",
                                               optional=False) for k in COVARIATES} for t in kept])
    mean = float(did_int.sum() / (len(did) * denom))  # exact sign, so "mean != 0" in _keeps is too
    pass_rule, fail_rule = p < alpha and _keeps(mean, adjusted and adjusted["estimate"]), upper < min_effect
    # docs/04 lists pass before fail; both can hold for a significant effect below min_effect, so both are reported.
    decision = "incomplete" if without else "pass" if pass_rule else "fail" if fail_rule else "inconclusive"
    return {"profile": profile, "n_tasks": len(kept), "n_pairs_selected": len(selected),
            "n_pairs_dropped": len(invalid), "pair_drop_rate": len(invalid) / len(selected),
            "pair_tasks_without_continuations": without, "pair_tasks_only_other_profiles": only_other,
            "rows_ignored_by_profile": ignored,
            "did_mean": mean, "p_one_sided": p, "upper95_one_sided": upper,
            "upper95_definition": UPPER_BOUND_DEFINITION, "adjusted": adjusted, "alpha": alpha,
            "small_sample_note": SMALL_SAMPLE_NOTE if len(kept) < SMALL_SAMPLE else None,
            "min_effect": min_effect, "pass_rule": pass_rule, "fail_rule": fail_rule, "decision": decision,
            "n_boot": n_boot, "seed": seed, "per_task": [{"task_id": t, "did": float(d)} for t, d in zip(kept, did)]}


def did_q4(rows, pairs, **kw):
    """docs/04's secondary Q4_K_M result: did_test with profile Q4_K_M on every frozen pair.

    As in the primary test, a frozen pair with neither Q4_K_M rows nor a dropped replay makes the
    decision "incomplete", so a test run while the Q4_K_M continuations are part-way never reads as
    a result on whichever pairs finished first. The numbers are still those of the pairs with rows.
    BF16 rows of a pair without Q4_K_M rows are left out, so that pair counts as missing; its
    replay_valid false rows are kept, so a failed replay counts as a drop. None without Q4_K_M rows.
    kw goes to did_test.
    """
    tasks = {r.get("task_id") for r in rows if r.get("profile") == AGREEMENT_PROBE}
    if not tasks:
        return None
    kept = [r for r in rows if r.get("task_id") in tasks or r.get("replay_valid") is False]
    out = did_test(kept, pairs=pairs, profile=AGREEMENT_PROBE, **kw)
    return {**out, "secondary": True, "n_pairs_frozen": len(pairs),
            "n_pairs_with_q4_rows": sum(1 for p in pairs if p["task_id"] in tasks)}


def did_kl_mean(rows, pairs, **kw):
    """docs/04's secondary kl_mean result on the same replays: the Q3_K_M DiD with each pair's low and high
    prefix set by kl_mean instead of the chosen score (owner answer on #12: relabel, report the flips
    and the per-pair values, drop and count ties).

    A pair whose high run has the lower kl_mean is flipped: it swaps its runs, its rows' prefix labels
    and attempt ids, and the sign of its covariate diff (COVARIATE_CONVENTION is low minus high). A
    pair tied on kl_mean has no kl_mean order, so it and its rows leave the test and are counted; no
    attempt_id tie-break. per_pair gives both kl_mean values of every frozen pair, under its frozen
    labels. None for pairs frozen without low_kl_mean (before this field existed). kw goes to did_test.
    """
    if any("low_kl_mean" not in p for p in pairs):
        return None
    order = {p["task_id"]: "tied" if p["low_kl_mean"] == p["high_kl_mean"]
             else "flipped" if p["high_kl_mean"] < p["low_kl_mean"] else "kept" for p in pairs}
    flipped = {t for t, o in order.items() if o == "flipped"}
    tied = sorted(t for t, o in order.items() if o == "tied")
    if len(tied) == len(pairs):
        raise ValueError("every frozen pair is tied on kl_mean")
    swap = {"low": "high", "high": "low"}
    new_pairs = [{**p, "low": p["high"], "high": p["low"], "low_kl_mean": p["high_kl_mean"],
                  "high_kl_mean": p["low_kl_mean"], "diff": {k: -v for k, v in p["diff"].items()}}
                 if p["task_id"] in flipped else p for p in pairs if p["task_id"] not in tied]
    new_rows = [{**r, "prefix": swap.get(r.get("prefix"), r.get("prefix")), "low_attempt_id": r.get("high_attempt_id"),
                 "high_attempt_id": r.get("low_attempt_id")} if r.get("task_id") in flipped else r
                for r in rows if r.get("task_id") not in tied]
    out = did_test(new_rows, pairs=new_pairs, **kw)
    return {**out, "secondary": True, "order": "kl_mean", "n_pairs_frozen": len(pairs),
            "n_pairs_reordered": len(flipped), "pairs_reordered": sorted(flipped),
            "n_pairs_tied": len(tied), "pairs_tied": tied,
            "per_pair": [{"task_id": p["task_id"], "low": p["low"], "high": p["high"],
                          "low_kl_mean": p["low_kl_mean"], "high_kl_mean": p["high_kl_mean"],
                          "kl_mean_order": order[p["task_id"]]} for p in pairs]}


def power(p_bf16, p_q3, n_tasks, seeds, effects=(MIN_EFFECT,), n_sim=1000, n_boot=2000, alpha=0.05, seed=0,
          drop_rate=0.0, covariates=None):
    """Simulated power of did_test's pass decision at each planted effect.

    p_bf16, p_q3: continuation success rates, scalars or paired per-task arrays (pilot or P0c), drawn
    with replacement for each simulated task. The Q3 rates after the low- and high-score prefixes are
    q + effect/2 and q - effect/2, shifted together back inside [0, 1] where they would leave it, so
    every task carries a gap of exactly `effect` (clipping each side would plant less near 0 and 1,
    and P0c rates of x/4 sit there often). BF16 has the same rate after both. Each of the four cells
    gets `seeds` attempts. drop_rate: the expected pair drop rate, so a study tests the
    round(n_tasks * (1 - drop_rate)) pairs left after replay drops.

    covariates: the pairs' covariate differences (select_pairs' diff). Given, each simulated study
    draws one per task, independent of its DiD, and passes only when p < alpha and the adjusted
    estimate keeps its sign and half its size, as in did_test; "power_p_only" is p < alpha alone.
    Without them the adjusted condition is not simulated and both are the same.
    """
    b, q = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (p_bf16, p_q3))
    if b.shape != q.shape or ((b < 0) | (b > 1) | (q < 0) | (q > 1)).any():
        raise ValueError("p_bf16 and p_q3 must be rates in [0, 1] of the same length")
    if not 0 <= drop_rate < 1:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    n_planned, n_tasks = n_tasks, round(n_tasks * (1 - drop_rate))
    if n_tasks < 2 or seeds < 1:
        raise ValueError("need at least 2 tasks after replay drops and 1 seed per prefix")
    cov = None if covariates is None else np.array([[c[k] for k in COVARIATES] for c in covariates], dtype=float)
    if any(not 0 <= e <= 1 for e in effects):
        raise ValueError(f"effects must be success-rate gaps in [0, 1], got {effects}")
    rng = np.random.default_rng(seed)
    # One set of bootstrap weights for every simulated study; each study's test stays valid
    # and power is unbiased, the shared draws only add a little Monte Carlo correlation.
    counts = _boot_counts(n_tasks, n_boot, rng)

    def successes(p):
        return rng.binomial(seeds, np.clip(p, 0, 1))

    out = []
    for e in effects:
        j = rng.integers(0, len(b), (n_sim, n_tasks))
        low = np.clip(q[j] + e / 2, e, 1)
        did_int = (successes(low) - successes(b[j])) - (successes(low - e) - successes(b[j]))  # DiD x seeds
        p = ((counts @ did_int.T) <= 0).mean(axis=0)
        significant = p < alpha
        passed = significant.copy()
        if cov is not None:
            for i in np.flatnonzero(significant):
                draw = cov[rng.integers(0, len(cov), n_tasks)]
                adj = _adjusted(did_int[i] / seeds, [dict(zip(COVARIATES, row)) for row in draw])["estimate"]
                passed[i] = _keeps(did_int[i].mean() / seeds, adj)
        out.append({"effect": float(e), "power": float(passed.mean()), "power_p_only": float(significant.mean())})
    passing = [o["effect"] for o in out if o["power"] >= 0.8]
    return {"n_tasks": n_planned, "drop_rate": drop_rate, "n_tasks_after_drops": n_tasks, "seeds": seeds,
            "p_bf16": b.tolist(), "p_q3": q.tolist(), "alpha": alpha, "adjusted_condition": cov is not None,
            "n_sim": n_sim, "n_boot": n_boot, "power": out, "mde_80": min(passing) if passing else None}
