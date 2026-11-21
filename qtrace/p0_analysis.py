"""P0 gate statistics (docs/04: P0b report, P0a primary test, P0c breakage).

Pure functions over per-attempt records in the format agreed in issue #1. Callers pass only the
records of the gate's own runs (selected by run_id, see scripts/p0_analyze.py). Records with
end_reason "error" are infrastructure faults that 04 reruns as fresh attempts, so they are dropped
and counted; infra_timeout stays in as a failure, with a sensitivity analysis that excludes it.
Bootstraps resample tasks, keep every attempt and profile of a task together, and are seeded.
grading.test_files_touched is a required bool in every record and is read strictly.
"""

import math
from collections import Counter, defaultdict

import numpy as np

BINS = [("0", 0, 0), ("1-3", 1, 3), ("4-8", 4, 8), ("9-16", 9, 16)]


def processed_tokens(rec):
    """Output tokens plus every input token per turn, cache hits and misses (04, P0a)."""
    t = rec["tokens"]
    return t["output"] + t["input_uncached"] + t["input_cached"]


def clean(rows):
    """Drop infrastructure errors (rerun as fresh attempts) and duplicate attempt ids; count both.

    Raises on a resolved record whose end_reason is not "submitted": 04 charges a budget hit or an
    infra_timeout as a failure, so such a record means the harness graded a run it should not have.
    """
    seen, kept, dropped = set(), [], Counter()
    for r in rows:
        if r["resolved"] and r["end_reason"] != "submitted":
            raise ValueError(f"attempt {r.get('attempt_id')!r} is resolved with end_reason {r['end_reason']!r}; "
                             "resolved requires end_reason 'submitted'")
        if r.get("end_reason") == "error":
            dropped["error"] += 1
            continue
        aid = r.get("attempt_id")
        if aid is not None and aid in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(aid)
        kept.append(r)
    return kept, dict(dropped)


def design_mismatches(rows, expected):
    """(task, profile) pairs whose attempt count differs from the design (16, 3 or 4)."""
    n = Counter((r["task_id"], r["profile"]) for r in rows)
    return sorted([t, p, k] for (t, p), k in n.items() if k != expected)


def finite_or_str(x):
    """JSON-safe number: inf becomes "inf", NaN never appears."""
    if x is None or (isinstance(x, float) and math.isfinite(x)) or isinstance(x, int):
        return x
    return "inf" if x > 0 else "-inf"


def _per_task(rows, profile):
    """task -> [processed tokens summed, resolved count, attempts]."""
    out = defaultdict(lambda: [0.0, 0, 0])
    for r in rows:
        if r["profile"] == profile:
            acc = out[r["task_id"]]
            acc[0] += processed_tokens(r)
            acc[1] += int(bool(r["resolved"]))
            acc[2] += 1
    return out


def _ratio(pq, rq, pb, rb):
    """(pq / rq) / (pb / rb). Undefined (None) if neither resolves; 0 if only `quantized` resolves."""
    if rq == 0 and rb == 0:
        return None
    if rb == 0:
        return 0.0
    if rq == 0:
        return math.inf
    return (pq / rq) / (pb / rb)


def p0a_verdict(R, lo, hi, threshold=1.15):
    """04: pass if R >= threshold and lower > 1; fail if upper < threshold; else inconclusive."""
    if R is None or lo is None or hi is None:
        return "inconclusive"
    if R >= threshold and lo > 1:
        return "pass"
    if hi < threshold:
        return "fail"
    return "inconclusive"


def p0a_primary(rows, quantized="Q4_K_M", base="BF16", n_boot=5000, seed=0, threshold=1.15):
    """Pooled R = (processed tokens / resolved attempts) at `quantized` over the same at `base`."""
    q, b = _per_task(rows, quantized), _per_task(rows, base)
    tasks = sorted(set(q) & set(b))
    dropped = sorted(set(q) ^ set(b))
    arr = np.array([[q[t][0], q[t][1], b[t][0], b[t][1]] for t in tasks], dtype=float)
    R = _ratio(*arr.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = [_ratio(*arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)) for _ in range(n_boot)]
    defined = np.array([d for d in draws if d is not None], dtype=float)
    lo = hi = None
    if len(defined):
        # Order statistics, not interpolation, so an infinite draw never turns a bound into NaN.
        lo, hi = (float(x) for x in np.percentile(defined, [2.5, 97.5], method="inverted_cdf"))
    return {"R": finite_or_str(R), "ci95": [finite_or_str(lo), finite_or_str(hi)],
            "verdict": p0a_verdict(R, lo, hi, threshold), "n_tasks": len(tasks),
            "tasks_missing_a_profile": dropped, "n_boot": n_boot, "n_boot_undefined": n_boot - len(defined),
            "seed": seed}


def p0a_factors(rows, quantized="Q4_K_M", base="BF16"):
    """2610.32417's split, composing exactly: R = length_factor x completion_factor.

    length_factor = processed tokens per attempt, quantized over base.
    completion_factor = base resolve rate over quantized resolve rate (above 1: quantized resolves less).
    Also the geometric mean, over tasks resolved under both profiles, of per-task tokens per resolved
    attempt (a within-task view that excludes tasks either profile never solves).
    """
    q, b = _per_task(rows, quantized), _per_task(rows, base)
    tasks = set(q) & set(b)
    sq = [sum(q[t][i] for t in tasks) for i in range(3)]
    sb = [sum(b[t][i] for t in tasks) for i in range(3)]
    length = (sq[0] / sq[2]) / (sb[0] / sb[2]) if sq[2] and sb[2] and sb[0] else None
    rate_q, rate_b = (sq[1] / sq[2] if sq[2] else None), (sb[1] / sb[2] if sb[2] else None)
    completion = rate_b / rate_q if rate_q else None
    both = [t for t in tasks if q[t][1] and b[t][1]]
    logs = [math.log((q[t][0] / q[t][1]) / (b[t][0] / b[t][1])) for t in both]
    return {"length_factor": length, "completion_factor": finite_or_str(completion),
            "resolve_rate": {quantized: rate_q, base: rate_b},
            "within_task_tokens_per_resolved_geomean": math.exp(sum(logs) / len(logs)) if logs else None,
            "n_tasks_resolved_both": len(both)}


def flagged(rec):
    """The grader's test-file tripwire (04, "Test-file tripwire"). A missing key raises, never reads as False."""
    v = rec["grading"]["test_files_touched"]
    if not isinstance(v, bool):
        raise TypeError(f"attempt {rec.get('attempt_id')!r}: grading.test_files_touched must be a bool, got {v!r}")
    return v


def flagged_share(rows):
    """profile -> share of its attempts that tripped the test-file tripwire (#9 reporting rule)."""
    n, k = Counter(), Counter()
    for r in rows:
        n[r["profile"]] += 1
        k[r["profile"]] += flagged(r)
    return {p: k[p] / n[p] for p in sorted(n)}


def profile_summary(rows, profile):
    """One profile's resolve rate (overall and excluding flagged attempts), flagged share and turns.

    turns_per_resolved charges failed attempts, like R: turns summed over all attempts / resolved
    attempts. mean_turns_resolved is the mean over resolved attempts only. Both are None with no resolve.
    """
    sel = [r for r in rows if r["profile"] == profile]
    un = [r for r in sel if not flagged(r)]
    res = [r["turns"] for r in sel if r["resolved"]]
    return {"n_attempts": len(sel), "n_tasks": len({r["task_id"] for r in sel}),
            "resolve_rate": len(res) / len(sel) if sel else None,
            "resolve_rate_excluding_flagged": sum(bool(r["resolved"]) for r in un) / len(un) if un else None,
            "flagged_share": (len(sel) - len(un)) / len(sel) if sel else None,
            "turns_per_resolved": sum(r["turns"] for r in sel) / len(res) if res else None,
            "mean_turns_resolved": float(np.mean(res)) if res else None}


def paired_permutation(rows, quantized, base="BF16", n_perm=5000, seed=0):
    """Two-sided sign-flip test on per-task resolve-rate differences, quantized minus base (04, P0a secondary).

    04 states it on per-task pass counts (0-3). With the design's 3 attempts per task the rate is the
    count / 3, so the p-value is the same; rates also stay comparable when flagged attempts are dropped.
    p = (1 + draws at least as extreme) / (1 + n_perm), seeded.
    """
    q, b = _per_task(rows, quantized), _per_task(rows, base)
    tasks = sorted(set(q) & set(b))
    d = np.array([q[t][1] / q[t][2] - b[t][1] / b[t][2] for t in tasks])
    out = {"mean_rate_diff": None, "p_value": None, "n_tasks": len(tasks), "n_perm": n_perm, "seed": seed}
    if not len(d):
        return out
    obs = float(d.mean())
    perm = (np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d))) * d).mean(axis=1)
    extreme = int(np.sum(np.abs(perm) >= abs(obs) - 1e-12))  # tolerance: the same mean summed in another order
    return {**out, "mean_rate_diff": obs, "p_value": (1 + extreme) / (1 + n_perm)}


SECONDARY_LISTS = ("heldout100", "verified100")


def p0a_secondary(rows, n_boot=5000, seed=0, base="BF16"):
    """04's P0a secondaries (reported, not used for the gate): every profile on both lists.

    Per list: each profile's summary, and each non-base profile against `base` with R, the factors
    (also excluding flagged attempts) and the paired permutation test (also excluding flagged attempts).
    """
    out = {}
    for lst in SECONDARY_LISTS:
        sel = [r for r in rows if r["list"] == lst]
        if not sel:
            continue
        un = [r for r in sel if not flagged(r)]
        profiles = sorted({r["profile"] for r in sel})
        vs = {p: {"primary": p0a_primary(sel, quantized=p, base=base, n_boot=n_boot, seed=seed),
                  "factors": p0a_factors(sel, quantized=p, base=base),
                  "factors_excluding_flagged": p0a_factors(un, quantized=p, base=base),
                  "permutation": paired_permutation(sel, p, base, n_perm=n_boot, seed=seed),
                  "permutation_excluding_flagged": paired_permutation(un, p, base, n_perm=n_boot, seed=seed)}
              for p in profiles if p != base and base in profiles}
        out[lst] = {"per_profile": {p: profile_summary(sel, p) for p in profiles}, "vs_BF16": vs}
    return out


def _by_repo(tasks, min_candidates):
    """repo -> task count, tasks with at least `min_candidates` successes, successes and attempts."""
    out = defaultdict(lambda: {"n_tasks": 0, "n_ge4": 0, "successes": 0, "attempts": 0})
    for s, k, repo in tasks.values():
        d = out[repo]
        d["n_tasks"] += 1
        d["n_ge4"] += int(s >= min_candidates)
        d["successes"] += s
        d["attempts"] += k
    return dict(out)


def mixed_share_subsampled(counts, group=8, n_sub=5000, seed=0):
    """Share of tasks whose group of `group` attempts mixes successes and failures (docs/03 S2 gate).

    counts: [(successes, attempts)] per task. Each draw takes `group` of a task's attempts without
    replacement, for every task at once; the mean and 95% interval are over draws. Tasks with fewer
    attempts than `group` are left out and counted.
    """
    usable = [(s, k) for s, k in counts if k >= group]
    out = {"group_size": group, "n_tasks": len(usable), "tasks_below_group_size": len(counts) - len(usable),
           "n_sub": n_sub, "seed": seed, "mean": None, "ci95": None}
    if not usable:
        return out
    s, k = np.array(usable).T
    x = np.random.default_rng(seed).hypergeometric(s, k - s, group, size=(n_sub, len(usable)))
    share = ((x > 0) & (x < group)).mean(axis=1)
    return {**out, "mean": float(share.mean()), "ci95": [float(v) for v in np.percentile(share, [2.5, 97.5])]}


def p0b_report(rows, inrepo_size, pool_total=3887, heldout_total=691, min_candidates=4, design=16, group=8,
               n_sub=5000, seed=0):
    """Pool size, RL-usable share and the three P0b criteria, from the P0b run's BF16 attempts.

    Pool successes count unflagged runs only, because flagged runs can never be selection candidates.
    """
    def successes(lst, skip_flagged=False):
        acc = defaultdict(lambda: [0, 0, None])
        for r in rows:
            if r["list"] == lst and r["profile"] == "BF16" and not (skip_flagged and flagged(r)):
                a = acc[r["task_id"]]
                a[0] += int(bool(r["resolved"]) and not (lst == "pool200" and flagged(r)))
                a[1] += 1
                a[2] = r.get("repo")
        return acc

    pool, held = successes("pool200"), successes("heldout100")
    n = len(pool)
    ge1 = sum(1 for s, _, _ in pool.values() if s >= 1)
    ge4 = sum(1 for s, _, _ in pool.values() if s >= min_candidates)
    mixed = sum(1 for s, k, _ in pool.values() if 0 < s < k)
    share = ge4 / ge1 if ge1 else 0.0
    projected = (ge4 / n) * (pool_total - inrepo_size) if n else 0.0
    held_pass1 = float(np.mean([s / k for s, k, _ in held.values()])) if held else None
    held_ex = successes("heldout100", skip_flagged=True)  # flagged attempts leave the denominator too
    held_pass1_ex = float(np.mean([s / k for s, k, _ in held_ex.values()])) if held_ex else None
    short = sorted(t for t, (_, k, _) in {**pool, **held}.items() if k < design)
    return {
        "pool": {"n_tasks": n, "n_ge1": ge1, "n_ge4": ge4, "by_repo": _by_repo(pool, min_candidates),
                 "flagged_successes_excluded": sum(1 for r in rows if r["list"] == "pool200"
                                                   and r["profile"] == "BF16" and r["resolved"] and flagged(r))},
        "per_task": {t: {"successes": s, "attempts": k, "repo": repo} for t, (s, k, repo) in sorted(pool.items())},
        "heldout_per_task": {t: {"successes": s, "attempts": k, "repo": repo} for t, (s, k, repo) in sorted(held.items())},
        "heldout_by_repo": _by_repo(held, min_candidates),
        "tasks_below_design_attempts": short,
        "criteria": {"share_ge4_among_solvable": share},
        "projected_pool": projected,
        "heldout_mean_pass1": held_pass1,
        "heldout_mean_pass1_excluding_flagged": held_pass1_ex,
        "heldout_expected_resolves": None if held_pass1 is None else held_pass1 * heldout_total,
        "rl_share_mixed": mixed / n if n else 0.0,
        "rl_share_mixed_g8": mixed_share_subsampled([(s, k) for s, k, _ in pool.values()], group, n_sub, seed),
        "pass": {"c1": share > 0.5, "c2": projected >= 500,
                 "c3": "not computed" if held_pass1 is None else held_pass1 * heldout_total >= 100},
        "mde_note": "simulated minimum detectable effect not computed here; added with the P2 analysis",
    }


def _bin(count):
    for name, lo, hi in BINS:
        if lo <= count <= hi:
            return name
    raise ValueError(f"success count {count} outside 0-16")


def p0c_breakage(rows, vllm_counts, profile, base="BF16", n_boot=5000, seed=0, share_min=0.4):
    """L_b = sum over tasks in bin b of (BF16 pass rate - quantized pass rate), same runtime.

    vllm_counts: task -> BF16 successes out of 16 from P0b's per_task, used only to bin tasks.
    """
    q, b = _per_task(rows, profile), _per_task(rows, base)
    tasks = sorted(t for t in set(q) & set(b) if t in vllm_counts)
    excluded = sorted((set(q) | set(b)) - set(tasks))
    diff = np.array([b[t][1] / b[t][2] - q[t][1] / q[t][2] for t in tasks])
    bins = np.array([_bin(vllm_counts[t]) for t in tasks])
    four_plus = np.isin(bins, ["4-8", "9-16"])
    total = float(diff.sum())
    share = float(diff[four_plus].sum() / total) if total > 0 else None
    rng = np.random.default_rng(seed)
    tot_b, share_b, margin_b, nonpos = [], [], [], 0
    for _ in range(n_boot):
        i = rng.integers(0, len(tasks), len(tasks))
        t_sum, l4 = diff[i].sum(), diff[i][four_plus[i]].sum()
        tot_b.append(t_sum)
        margin_b.append(l4 - share_min * t_sum)  # no division, stable near L_total = 0
        if t_sum > 0:
            share_b.append(l4 / t_sum)
        else:
            nonpos += 1
    lo_total = float(np.percentile(tot_b, 2.5))
    share_ci = [float(x) for x in np.percentile(share_b, [2.5, 97.5])] if share is not None and share_b else None
    if lo_total <= 0:
        verdict = "no detectable breakage"
    elif share >= share_min:
        verdict = "pass"
    else:
        verdict = "below 0.4: use the P0b fallbacks or claim generalization only"
    return {"profile": profile, "L_by_bin": {name: float(diff[bins == name].sum()) for name, _, _ in BINS},
            "L_total": total, "L_total_lo95": lo_total, "share_4plus": share, "share_4plus_ci95": share_ci,
            "share_draws_dropped_nonpositive_total": nonpos,
            "margin_L4plus_minus_0p4_Ltotal_ci95": [float(x) for x in np.percentile(margin_b, [2.5, 97.5])],
            "verdict": verdict, "n_tasks": len(tasks), "tasks_excluded": excluded}
