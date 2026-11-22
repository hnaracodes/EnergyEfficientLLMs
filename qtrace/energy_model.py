"""P0a per-turn energy model (docs/04: P0a energy sub-run, per-turn energy model, batched coefficients).

Fitted per profile on the generation windows of concurrency-1 sub-run attempts, one row per turn with
a measured gen_energy_j. Tool-window energy stays out. Terms (TERMS) are output tokens, uncached
prefill tokens, the context position at turn start (input_cached under full prefix reuse, which
replaces a separate cached-token term), uncached x context, output x context and a per-turn
intercept. Coefficients are non-negative.

Two targets per profile, both fitted and reported: "raw" (the counter's gen_energy_j) and
"idle_subtracted" (gen_energy_j - energy.idle_w_ref x gen_s). Runs are ranked with the raw Q4_K_M
fit: the P2 endpoint, the estimator check and the energy upper bound all compare against measured
GPU joules, which include the idle draw, and at concurrency 1 the device is held for the whole
generation window. Idle subtraction also adds the error of a separately measured idle reference.

The intercept (the fixed per-turn overhead docs/04 names) is always in the model. The token terms
are the subset of the five, the empty one included, with the lowest leave-one-attempt-out (LOAO) CV
error: the median absolute relative error on attempt joules (an attempt's summed predicted turn
joules against its summed measured ones). Ties go to fewer terms, then the lower bitmask. The
pre-registered fewest_joules fallback reads the nested LOAO error, where the term choice is redone
inside each outer fold, because the plain LOAO error of a chosen term set is optimistic.

Whole-attempt joules (provisional, pending the owner's answer to our question 2 in issue #1): the
P2 endpoint and the fewest_joules ranking use raw GPU joules over the whole attempt, since a
single-user local agent keeps the GPU powered while tools run. They are estimated as the
generation-window model plus idle_w x tool_seconds (estimate_attempt_joules with idle_w), so a run
that waits on a slow test suite is charged for it. idle_w is per profile: the median of the fitted
sub-run attempts' energy.idle_w_ref, robust to one bad idle reading. The fit itself stays on raw
generation windows, where the intercept and the token terms absorb the idle draw while generating.

NNLS is exact without scipy: with k <= 6 terms, the NNLS solution is the least-squares fit on its
active set, so fitting every column subset and keeping the best one with no negative coefficient
finds it. Everything runs on per-attempt Gram matrices, so a fold or bootstrap draw is a sum.
"""

import itertools
import math
from collections import Counter, defaultdict

import numpy as np

TERMS = ("output", "uncached", "context", "uncached_x_context", "output_x_context", "intercept")
CANDIDATES = range(1 << TERMS.index("intercept"), 2 ** len(TERMS))  # term-set bitmasks holding the intercept
TARGETS = ("raw", "idle_subtracted")
RANK_PROFILE, RANK_TARGET = "Q4_K_M", "raw"
FALLBACK_MAX_REL_ERR = 0.10  # 04: fewest_joules falls back to fewest_turns above this
AGREEMENT_MIN = 0.85  # 04: batched vs concurrency-1 picks below this name their regime
MAX_SHARE_WITHOUT_FIRST_TOKEN = 0.01  # owner review of PR #13: flag a block above about 1% of such turns
MIN_CANDIDATES = 4  # 03 Method step 1: pool tasks have at least 4 successful BF16 runs; the P0b fallback is 2
HELDOUT_LISTS = ("heldout100", "verified100", "inrepo_test")
POOL_LISTS = ("pool200",)  # the default of p0a_energy_model.py --lists, as in p0d_score_pool.py
HELDOUT_REPOS = ("scrapy", "pyramid", "datalad", "coveragepy")
CONTEXT_BINS = (16384, 32768, 49152)  # residual summary bins over the 64K context
# 04 "Energy boundary on EC2": NVML reads GPU joules only, so every energy table says so and reports container
# CPU-seconds next to it.
ENERGY_BOUNDARY = "GPU only"
PROFILES = ("BF16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8")  # the agreed record format (issue #1)
# 04 P0a: idle W per profile with no process and with llama-server loaded, before and after the sub-run, in the
# order they are taken. The harness writes them to outputs/energy/idle_<run_id>.json (owner's agent, PR #13).
IDLE_READINGS = ("none_before_w", "loaded_before_w", "loaded_after_w", "none_after_w")
IDLE_FIELDS = (*IDLE_READINGS, "window_s", "settle_s", "t")


def _get(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"{where}: missing field {key!r}")
    return d[key]


def _num(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise ValueError(f"{where} must be a finite non-negative number, got {x!r}")
    return float(x)


def _turns(rec):
    turns = rec.get("per_turn")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"attempt {rec.get('attempt_id')!r}: per_turn must be a non-empty list")
    return turns


def turn_features(turn, where="turn"):
    """The TERMS values of one per_turn entry, validated."""
    o, u, c = (_num(_get(turn, k, where), f"{where}: {k}") for k in ("output_tokens", "input_uncached", "input_cached"))
    return [o, u, c, u * c, o * c, 1.0]


def attempt_arrays(rec):
    """Feature matrix and both targets over the turns of `rec` with a measured gen_energy_j.

    Returns {attempt_id, task_id, profile, X [n, 6], raw [n], idle_subtracted [n] or None, idle_error,
    n_unmeasured, server_cached_differs (attempt totals; None when the server count is null),
    turn_cached_differs (one bool per turn with a non-null per-turn server_reported_cached), n_turns,
    container_cpu_seconds (required, finite, >= 0)}. The raw target needs
    only gen_energy_j; a missing or invalid energy.idle_w_ref or gen_s sets idle_subtracted to None and
    idle_error to the reason, so only the idle-subtracted fit fails. Idle-subtracted turn joules may be
    negative on tiny turns; only attempt sums must be positive (checked in fit_profile).
    """
    aid = _get(rec, "attempt_id", "attempt")
    where = f"attempt {aid!r}"
    X, raw, gen_s = [], [], []
    turns = _turns(rec)
    for i, t in enumerate(turns):
        w = f"{where} turn {i}"
        e = _get(t, "gen_energy_j", w)
        if e is None:
            continue
        X.append(turn_features(t, w))
        raw.append(_num(e, f"{w}: gen_energy_j"))
        gen_s.append((t.get("gen_s"), f"{w}: gen_s"))
    idle, idle_error, idle_w = None, None, None
    try:
        idle_w = _num(_get(rec, "energy", where).get("idle_w_ref"), f"{where}: energy.idle_w_ref")
        idle = np.array(raw) - idle_w * np.array([_num(v, w) for v, w in gen_s])
    except ValueError as err:
        idle_error = str(err)
    tokens = _get(rec, "tokens", where)
    server = tokens.get("server_reported_cached")
    return {"attempt_id": aid, "task_id": _get(rec, "task_id", where), "profile": _get(rec, "profile", where),
            "X": np.array(X, dtype=float).reshape(-1, len(TERMS)), "raw": np.array(raw),
            "idle_subtracted": idle, "idle_error": idle_error, "idle_w_ref": idle_w,
            "n_unmeasured": len(turns) - len(raw),
            "server_cached_differs": None if server is None else server != _get(tokens, "input_cached", where),
            "turn_cached_differs": [_num(t["server_reported_cached"], f"{where} turn {i}: server_reported_cached")
                                    != t["input_cached"] for i, t in enumerate(turns)
                                    if t.get("server_reported_cached") is not None],
            "n_turns": len(turns),
            "container_cpu_seconds": _num(_get(rec, "container_cpu_seconds", where), f"{where}: container_cpu_seconds")}


def fit_attempts(records):
    """Sub-run attempts usable for the fit, as {profile: [attempt_arrays]}, plus exclusion counts.

    Kept: concurrency 1, energy.attributable true, end_reason not "error" (an infrastructure fault,
    rerun as a fresh attempt), and at least one measured turn. Duplicate attempt ids fail, and so do
    profiles whose concurrency-1 attempts (every end_reason) ran different task_ids: 04 runs the
    same seeded IDs for every profile.
    """
    seen, by_profile, excluded, ran = set(), defaultdict(list), Counter(), defaultdict(set)
    for r in records:
        aid = _get(r, "attempt_id", "sub-run record")
        where = f"attempt {aid!r}"
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r} in the sub-run files")
        seen.add(aid)
        if _get(r, "concurrency", where) != 1:
            excluded["concurrency_not_1"] += 1
            continue
        ran[_get(r, "profile", where)].add(_get(r, "task_id", where))
        if _get(_get(r, "energy", where), "attributable", where) is not True:
            excluded["not_attributable"] += 1
        elif _get(r, "end_reason", where) == "error":
            excluded["error"] += 1
        else:
            a = attempt_arrays(r)
            if len(a["raw"]):
                by_profile[a["profile"]].append(a)
            else:
                excluded["no_measured_turns"] += 1
    every = set().union(*ran.values())
    lacking = [f"{p} lacks {sorted(every - t)}" for p, t in sorted(ran.items()) if every - t]
    if lacking:
        raise ValueError("the energy sub-run must use the same task_ids for every profile (04 P0a): "
                         + "; ".join(lacking))
    return dict(by_profile), dict(excluded)


def idle_readings(d):
    """Validated --idle readings, the harness's outputs/energy/idle_<run_id>.json.

    {profile: {none_before_w, loaded_before_w, loaded_after_w, none_after_w (W), window_s (the averaging
    window), settle_s (45 in 04), t: {reading: epoch seconds}}}, every key present and nothing else. The t
    values must be non-decreasing in IDLE_READINGS order, the order the readings are taken.
    """
    if not isinstance(d, dict):
        raise TypeError(f"idle readings must be an object of profiles, got {type(d).__name__}")
    for p, r in d.items():
        if p not in PROFILES:
            raise ValueError(f"idle readings: unknown profile {p!r}; expected one of {PROFILES}")
        if not isinstance(r, dict) or set(r) != set(IDLE_FIELDS):
            got = sorted(r) if isinstance(r, dict) else r
            raise ValueError(f"idle readings for {p}: need exactly {IDLE_FIELDS}, got {got!r}")
        for k in IDLE_READINGS:
            _num(r[k], f"idle readings for {p}: {k}")
        for k in ("window_s", "settle_s"):
            if _num(r[k], f"idle readings for {p}: {k}") == 0:
                raise ValueError(f"idle readings for {p}: {k} must be positive")
        t = r["t"]
        if not isinstance(t, dict) or set(t) != set(IDLE_READINGS):
            got = sorted(t) if isinstance(t, dict) else t
            raise ValueError(f"idle readings for {p}: t needs exactly {IDLE_READINGS}, got {got!r}")
        ts = [_num(t[k], f"idle readings for {p}: t.{k}") for k in IDLE_READINGS]
        if ts != sorted(ts):
            raise ValueError(f"idle readings for {p}: t must run {IDLE_READINGS} in that order, got {ts}")
    return d


def _scale(X):
    """Column RMS (1 for an all-zero column). NNLS predictions do not depend on positive column scaling."""
    s = np.sqrt((X ** 2).mean(axis=0))
    return np.where(s > 0, s, 1.0)


def _nnls_all(G, g, yy):
    """Exact NNLS restricted to every column subset at once, from the Gram matrix G = X'X, g = X'y, yy = y'y.

    Returns B [2^k, k]: row m is the NNLS solution using only the columns in bitmask m (row 0 is 0).
    Each subset's least-squares fit is computed once; the NNLS solution on m is the feasible subset
    of m (no negative coefficient) with the least residual sum of squares.
    """
    k = len(g)
    masks = np.arange(2 ** k)
    inn = ((masks[:, None] >> np.arange(k)) & 1).astype(float)
    # Columns outside a subset get an identity row and a zero target, so one batched pinv solves all.
    A = G * inn[:, :, None] * inn[:, None, :] + np.eye(k) * (1 - inn)[:, None, :]
    # Zero the excluded columns exactly: pinv leaves ~1e-15 there, and a tiny negative one would mark a
    # feasible subset infeasible.
    beta = np.einsum("mij,mj->mi", np.linalg.pinv(A), g * inn) * inn
    rss = yy - 2 * beta @ g + np.einsum("mi,ij,mj->m", beta, G, beta)
    rss = np.where((beta >= 0).all(axis=1), rss, np.inf)
    sub = (masks[None, :] & ~masks[:, None]) == 0  # [m, t]: t is a subset of m
    return beta[np.argmin(np.where(sub, rss[None, :], np.inf), axis=1)]


def nnls(X, y):
    """Exact non-negative least squares on all columns of X (k small; every subset is tried)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    s = _scale(X)
    Xs = X / s
    return _nnls_all(Xs.T @ Xs, Xs.T @ y, y @ y)[-1] / s


def _loao_preds(G, g, yy, S, members):
    """Held-out attempt joules [len(members), 2^k] for every term set, each fold fitted on members minus one."""
    tG, tg, tyy = G[members].sum(0), g[members].sum(0), yy[members].sum()
    return np.array([_nnls_all(tG - G[i], tg - g[i], tyy - yy[i]) @ S[i] for i in members])


def _choose(err):
    """Term set from held-out errors [n, 2^k]: the CANDIDATES set with the lowest median error.

    Ties go to fewer terms, then the lower bitmask. The lowest error, not the fewest terms within one
    bootstrap SE of it: in the owner's planted-model probe (15 x 2 attempts, 5% per-turn and 3%
    per-attempt noise, 40 repeats) the one-SE rule found the planted set 0/40 times and its fresh-data
    error was 1.7x this rule's, because it dropped real terms.
    """
    cv = np.median(err, axis=0)
    return min(CANDIDATES, key=lambda m: (cv[m], m.bit_count(), m)), cv


def _terms(mask):
    return [t for j, t in enumerate(TERMS) if mask >> j & 1]


def pair_ordering(task_ids, attempt_ids, measured, predicted, n_unmeasured=None):
    """Share of measured within-task pairs whose predicted joules are ordered like the measured ones.

    A pair where either attempt has unmeasured turns (n_unmeasured, per attempt) compares partial sums:
    it is listed with "correct": None and counted in n_pairs_unmeasured, not in n_pairs.
    """
    unm = [0] * len(task_ids) if n_unmeasured is None else list(n_unmeasured)
    by_task = defaultdict(list)
    for i, t in enumerate(task_ids):
        by_task[t].append(i)
    pairs = []
    for t, idx in sorted(by_task.items()):
        for i, j in itertools.combinations(idx, 2):
            pairs.append({"task_id": t, "attempts": [attempt_ids[i], attempt_ids[j]],
                          "measured_j": [float(measured[i]), float(measured[j])],
                          "predicted_j": [float(predicted[i]), float(predicted[j])],
                          "n_unmeasured": [unm[i], unm[j]],
                          "correct": None if unm[i] or unm[j] else
                          bool((predicted[i] - predicted[j]) * (measured[i] - measured[j]) > 0)})
    scored = [p["correct"] for p in pairs if p["correct"] is not None]
    n_ok = sum(scored)
    return {"n_pairs": len(scored), "n_correct": n_ok, "share_correct": n_ok / len(scored) if scored else None,
            "n_pairs_unmeasured": len(pairs) - len(scored), "pairs": pairs}


def _residuals(attempts, target, coef, rel_loao):
    """In-sample turn residuals (J, overall and by context bin) and LOAO relative errors on attempt joules."""
    X = np.vstack([a["X"] for a in attempts])
    res = np.concatenate([a[target] for a in attempts]) - X @ coef
    bins = np.digitize(X[:, TERMS.index("context")], CONTEXT_BINS)
    names = ["<16K", "16-32K", "32-48K", ">=48K"]
    return {"turn_rmse_j": float(np.sqrt((res ** 2).mean())), "turn_median_abs_j": float(np.median(np.abs(res))),
            "turn_mean_j_by_context": {n: {"n": int((bins == b).sum()),
                                           "mean_j": float(res[bins == b].mean()) if (bins == b).any() else None}
                                       for b, n in enumerate(names)},
            "attempt_loao_rel": {"median_abs": float(np.median(np.abs(rel_loao))),
                                 "p90_abs": float(np.percentile(np.abs(rel_loao), 90)),
                                 "max_abs": float(np.abs(rel_loao).max()), "mean_signed": float(rel_loao.mean())}}


def fit_profile(attempts, target="raw", n_boot=2000, seed=0):
    """Choose terms, fit, and report one profile's model for one target (docs/04 P0a reporting list).

    attempts: one profile's attempt_arrays (from fit_attempts). Returns {terms, coef (J per unit of
    each chosen term), coef_ci95 (percentile bootstrap over attempts, terms held fixed, seeded),
    cv {loao, nested_loao},
    fallback (nested_loao above FALLBACK_MAX_REL_ERR), residuals, pairs (LOAO predictions), n_attempts,
    n_turns}. Raises if there are fewer than 3 attempts (nested LOAO needs them), an attempt has no
    value for the target (idle_error), or an attempt's measured sum is <= 0.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    missing = [a["idle_error"] for a in attempts if a[target] is None]
    if missing:
        raise ValueError(f"no {target} joules for {len(missing)} attempts: " + "; ".join(missing))
    n = len(attempts)
    if n < 3:
        raise ValueError(f"nested LOAO needs at least 3 attempts per profile, got {n} "
                         f"for {sorted({a['profile'] for a in attempts})}")
    scale = _scale(np.vstack([a["X"] for a in attempts]))
    Xs = [a["X"] / scale for a in attempts]
    ys = [a[target] for a in attempts]
    G = np.stack([x.T @ x for x in Xs])
    g = np.stack([x.T @ y for x, y in zip(Xs, ys)])
    yy = np.array([y @ y for y in ys])
    S = np.stack([x.sum(0) for x in Xs])
    Y = np.array([y.sum() for y in ys])
    bad = [a["attempt_id"] for a, v in zip(attempts, Y) if v <= 0]
    if bad:
        raise ValueError(f"non-positive measured {target} joules for attempts {bad}; check energy.idle_w_ref")

    everyone = np.arange(n)
    P = _loao_preds(G, g, yy, S, everyone)
    m, cv = _choose(np.abs(P - Y[:, None]) / Y[:, None])

    nested = []
    for i in everyone:
        rest = everyone[everyone != i]
        mi = _choose(np.abs(_loao_preds(G, g, yy, S, rest) - Y[rest, None]) / Y[rest, None])[0]
        pred = _nnls_all(G[rest].sum(0), g[rest].sum(0), yy[rest].sum())[mi] @ S[i]
        nested.append(abs(pred - Y[i]) / Y[i])
    nested = float(np.median(nested))

    coef = _nnls_all(G.sum(0), g.sum(0), yy.sum())[m] / scale
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(_nnls_all(G[idx].sum(0), g[idx].sum(0), yy[idx].sum())[m] / scale)
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0) if draws else (coef, coef)
    terms = _terms(m)
    return {"terms": terms, "coef": {t: float(coef[TERMS.index(t)]) for t in terms},
            "coef_ci95": {t: [float(lo[TERMS.index(t)]), float(hi[TERMS.index(t)])] for t in terms},
            "cv": {"loao": float(cv[m]), "nested_loao": nested},
            "fallback": nested > FALLBACK_MAX_REL_ERR,
            "residuals": _residuals(attempts, target, coef, (P[:, m] - Y) / Y),
            "pairs": pair_ordering([a["task_id"] for a in attempts], [a["attempt_id"] for a in attempts], Y, P[:, m],
                                   [a["n_unmeasured"] for a in attempts]),
            "n_attempts": n, "n_turns": int(sum(len(y) for y in ys)), "n_boot": n_boot, "seed": seed}


def _fit_or_error(attempts, target, n_boot, seed):
    try:
        return fit_profile(attempts, target=target, n_boot=n_boot, seed=seed)
    except ValueError as err:
        return {"error": str(err)}


def _range_values(X):
    """Per-turn values of the calibration-range keys for feature rows X.

    output_share is output / (output + uncached), the decoded share of the tokens a turn processes; the
    cached context has its own range. A turn with neither counts as share 0. The denominator is our
    choice (docs/04 S2 says "output share" only), flagged for the owner.
    """
    o, u = X[:, TERMS.index("output")], X[:, TERMS.index("uncached")]
    return {"output": o, "uncached": u, "context": X[:, TERMS.index("context")],
            "output_share": o / np.maximum(o + u, 1.0)}


def profile_report(attempts, n_boot=2000, seed=0):
    """One profile's entry in energy_model.json: both targets, the calibration range and the cache-count check.

    A target whose fit fails (too few attempts, missing idle reference, non-positive sums) is reported
    as {"error": message}, so one broken fit never takes the other target or profile with it.
    """
    X = np.vstack([a["X"] for a in attempts])
    flags = [a["server_cached_differs"] for a in attempts]
    known = [f for f in flags if f is not None]
    rng_of = {t: [float(v.min()), float(v.max())] for t, v in _range_values(X).items()}
    # Every turn, measured or not: check_range and estimate_attempt_joules count them all.
    n_turns = [a["n_turns"] for a in attempts]
    per_turn = [f for a in attempts for f in a["turn_cached_differs"]]
    idle_refs = [a["idle_w_ref"] for a in attempts if a["idle_w_ref"] is not None]
    cpu = [a["container_cpu_seconds"] for a in attempts]
    return {"n_attempts": len(attempts), "n_turns": len(X),
            "n_turns_unmeasured": sum(a["n_unmeasured"] for a in attempts),
            "calibration_range": {**rng_of, "turns_per_attempt": [min(n_turns), max(n_turns)]},
            # Provisional whole-attempt estimate: W charged per tool second (module docstring).
            "idle_w": float(np.median(idle_refs)) if idle_refs else None,
            "idle_w_ref": {"n": len(idle_refs), "min": min(idle_refs, default=None),
                           "max": max(idle_refs, default=None)},
            # Next to the GPU-only joules (ENERGY_BOUNDARY), per fitted sub-run attempt.
            "container_cpu_seconds": {"n": len(cpu), "median": float(np.median(cpu)), "min": min(cpu), "max": max(cpu)},
            # 04: how often llama.cpp's per-request cached-prompt counts differ from the ideal split. The
            # attempt-total comparison stays for records without per-turn server counts.
            "server_cache_mismatch": {"n_attempts": len(flags), "n_differ": sum(known),
                                      "n_missing": len(flags) - len(known),
                                      "share_differ": sum(known) / len(known) if known else None,
                                      "per_request": {"n_requests": len(per_turn), "n_differ": sum(per_turn),
                                                      "share_differ": sum(per_turn) / len(per_turn) if per_turn
                                                      else None,
                                                      "n_missing": sum(a["n_turns"] for a in attempts)
                                                      - len(per_turn)}},
            **{t: _fit_or_error(attempts, t, n_boot, seed) for t in TARGETS}}


def check_range(records, report):
    """Share of `records`' turns outside each bound of a profile_report's calibration_range (docs/04 S2).

    records: attempt records with per_turn (S2 rollouts); every turn counts, measured or not. Returns
    {n_turns, n_attempts, outside: {key: {below, above}}, any_turn}: per-turn keys give shares of turns,
    turns_per_attempt a share of attempts, and any_turn the share of turns outside at least one per-turn
    bound. What share calls for a re-measurement on llama.cpp Q4_K_M and a refit is the owner's.
    """
    if not records:
        raise ValueError("check_range needs at least one record")
    rng = _get(report, "calibration_range", "report")
    X = np.array([turn_features(t, f"attempt {r.get('attempt_id')!r} turn {i}")
                  for r in records for i, t in enumerate(_turns(r))])
    per_turn = _range_values(X)
    values = {**per_turn, "turns_per_attempt": np.array([len(_turns(r)) for r in records])}
    bounds = {k: _get(rng, k, "calibration_range") for k in values}
    beyond = {k: (v < bounds[k][0], v > bounds[k][1]) for k, v in values.items()}
    return {"n_turns": len(X), "n_attempts": len(records),
            "outside": {k: {"below": float(lo.mean()), "above": float(hi.mean())} for k, (lo, hi) in beyond.items()},
            "any_turn": float(np.any([lo | hi for k, (lo, hi) in beyond.items() if k in per_turn], axis=0).mean())}


def estimate_attempt_joules(rec, model, idle_w=None):
    """Estimated joules of an attempt: predicted turn joules summed over rec["per_turn"].

    model: a fit_profile report or any {"coef": {term: J per unit}}; terms left out count as 0. Rank
    runs with energy_model.json["profiles"]["Q4_K_M"]["raw"]. With idle_w None (the default) this is
    generation windows only; with idle_w (W, the profile's "idle_w") it adds idle_w x rec["tool_seconds"],
    the provisional whole-attempt estimate.
    """
    unknown = set(model["coef"]) - set(TERMS)
    if unknown:
        raise ValueError(f"unknown energy-model terms {sorted(unknown)}; expected a subset of {TERMS}")
    w = np.array([model["coef"].get(t, 0.0) for t in TERMS])
    aid = rec.get("attempt_id")
    gen = float(sum(w @ turn_features(t, f"attempt {aid!r} turn {i}") for i, t in enumerate(_turns(rec))))
    if idle_w is None:
        return gen
    where = f"attempt {aid!r}"
    return gen + _num(idle_w, "idle_w") * _num(_get(rec, "tool_seconds", where), f"{where}: tool_seconds")


def pool_records(records, lists=POOL_LISTS):
    """Pool attempts to estimate, with other lists skipped and infrastructure errors dropped, both counted.

    The P0b run file also holds the heldout100 attempts (04: one vLLM block covers both lists), so rows
    whose list is not one of `lists` are skipped and counted per list. Fails on an empty pool, duplicate
    attempt ids, missing fields, a held-out list among `lists`, and a held-out repo under a pool list
    (held-out tasks never enter a pool). Returns (kept records, number of end_reason "error" dropped,
    {list: rows skipped}).
    """
    bad = sorted(set(lists) & set(HELDOUT_LISTS))
    if bad:
        raise ValueError(f"{bad} are held-out lists and never enter a pool")
    if not records:
        raise ValueError("the pool is empty")
    kept, seen, n_error, skipped = [], set(), 0, defaultdict(int)
    for r in records:
        aid = _get(r, "attempt_id", "pool record")
        where = f"pool attempt {aid!r}"
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r} in the pool files")
        seen.add(aid)
        lst = _get(r, "list", where)
        if lst not in lists:
            skipped[lst] += 1
            continue
        repo = _get(r, "repo", where)
        if repo in HELDOUT_REPOS:
            raise ValueError(f"{where} is from list {lst!r}, repo {repo!r}: held-out tasks never enter a pool")
        for k in ("task_id", "profile", "turns", "end_reason"):
            _get(r, k, where)
        if _get(r, "resolved", where) is not True and r["resolved"] is not False:
            raise ValueError(f"{where}: resolved must be true or false, got {r['resolved']!r}")
        if r["end_reason"] == "error":
            n_error += 1
        else:
            kept.append(r)
    if not kept and not n_error:
        raise ValueError(f"no pool attempt on {list(lists)}; skipped {dict(skipped)}")
    return kept, n_error, dict(sorted(skipped.items()))


def _avg_ranks(x):
    _, inv, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (start + (counts - 1) / 2)[inv]


def _spearman(a, b):
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _cheapest(cands, est):
    return min(range(len(cands)), key=lambda i: (est[i], cands[i]["turns"], cands[i]["attempt_id"]))


def flagged(rec):
    """The grader's test-file tripwire (docs/04), read strictly as in qtrace/p0d.py.

    grading.test_files_touched is a required bool of every attempt record: a missing key raises KeyError and
    a null or 0/1 raises TypeError, instead of silently counting the run as unflagged.
    """
    v = rec["grading"]["test_files_touched"]
    if not isinstance(v, bool):
        raise TypeError(f"attempt {rec.get('attempt_id')!r}: grading.test_files_touched must be a bool, got {v!r}")
    return v


def pick_agreement(records, est_a, est_b, min_candidates=MIN_CANDIDATES):
    """How often two energy estimators pick the same cheapest run per task, and their mean within-task Spearman.

    Candidates are the runs the P1 arms pick from: BF16 runs (records from pool_records) that resolved
    and whose grader did not flag test-file edits (grading.test_files_touched, read through
    flagged), grouped by task. Tasks with
    fewer than min_candidates are skipped; fewer candidates agree more often by chance, so a looser
    pool would bias the 85% check toward agreement. A pick breaks ties by fewer turns, then by
    attempt id (deterministic here; the arms break remaining ties with their seed). Spearman uses
    average ranks and is skipped for a task where either estimator gives every run the same value.
    """
    if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < 2:
        raise ValueError(f"min_candidates must be an integer >= 2, got {min_candidates!r}")
    by_task = defaultdict(list)
    for r in records:
        if r["profile"] == "BF16" and r["resolved"] and not flagged(r):
            by_task[r["task_id"]].append(r)
    same, rhos = [], []
    for c in by_task.values():
        if len(c) < min_candidates:
            continue
        a, b = [est_a(r) for r in c], [est_b(r) for r in c]
        same.append(_cheapest(c, a) == _cheapest(c, b))
        rho = _spearman(a, b)
        if rho is not None:
            rhos.append(rho)
    return {"min_candidates": min_candidates, "n_tasks": len(same),
            "share_same_pick": float(np.mean(same)) if same else None,
            "mean_spearman": float(np.mean(rhos)) if rhos else None, "n_tasks_spearman": len(rhos)}


SAMPLE_FIELDS = ("t", "counter_j", "prompt_tokens_total", "predicted_tokens_total")


def _metrics_columns(samples):
    """The 1 s metrics log as validated columns: t strictly increasing, every counter non-decreasing."""
    if len(samples) < 2:
        raise ValueError(f"batched fit needs at least two metrics samples, got {len(samples)}")
    cols = np.array([[_num(_get(s, k, f"sample {i}"), f"sample {i}: {k}") for k in SAMPLE_FIELDS]
                     for i, s in enumerate(samples)])
    for j, k in enumerate(SAMPLE_FIELDS):
        d = np.diff(cols[:, j])
        if (d <= 0).any() if k == "t" else (d < 0).any():
            i = int(np.flatnonzero(d <= 0 if k == "t" else d < 0)[0])
            raise ValueError(f"metrics samples {i} -> {i + 1}: {k} goes from {cols[i, j]} to {cols[i + 1, j]}; "
                             "t must increase and the counters are cumulative (a reset or per-window values?)")
    return cols


def _harness_window_tokens(edges, records):
    """Prefill and decode tokens per window [edges[i], edges[i+1]) from the harness per-turn timestamps.

    A turn's input_uncached is spread uniformly over [gen_t_start, gen_t_first_token) and its
    output_tokens over [gen_t_first_token, gen_t_end); a zero-length span goes to the window holding it.
    A turn with gen_t_first_token null (no token arrived) has no prefill/decode split, so its tokens
    are left out and counted; a faked timestamp would misplace them. Returns (tokens, n_turns, n_skipped).
    """
    lo, hi = edges[:-1], edges[1:]
    tokens, n_turns, n_skipped = np.zeros((2, len(lo))), 0, 0

    def spread(row, a, b, n):
        if b > a:
            tokens[row] += n * np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None) / (b - a)
        elif lo[0] <= a < hi[-1]:
            tokens[row, np.searchsorted(hi, a, side="right")] += n

    for rec in records:
        aid = rec.get("attempt_id")
        for i, t in enumerate(_turns(rec)):
            w = f"attempt {aid!r} turn {i}"
            n_turns += 1
            t0, t2 = (_num(_get(t, k, w), f"{w}: {k}") for k in ("gen_t_start", "gen_t_end"))
            if _get(t, "gen_t_first_token", w) is None:
                if not t0 <= t2:
                    raise ValueError(f"{w}: need gen_t_start <= gen_t_end, got {t0}, {t2}")
                n_skipped += 1
                continue
            t1 = _num(t["gen_t_first_token"], f"{w}: gen_t_first_token")
            if not t0 <= t1 <= t2:
                raise ValueError(f"{w}: need gen_t_start <= gen_t_first_token <= gen_t_end, got {t0}, {t1}, {t2}")
            spread(0, t0, t1, _num(_get(t, "input_uncached", w), f"{w}: input_uncached"))
            spread(1, t1, t2, _num(_get(t, "output_tokens", w), f"{w}: output_tokens"))
    return tokens, n_turns, n_skipped


def fit_batched(samples, records):
    """Batched coefficients: window joules ~ intercept_w x seconds + prefill x tokens + decode x tokens, >= 0.

    samples: the block's llama-server metrics log, [{t (epoch s), counter_j, prompt_tokens_total,
    predicted_tokens_total}], cumulative values about every 1 s. Windows are consecutive samples, so
    they are contiguous deltas; a missed sample makes one longer window, which the per-second
    intercept absorbs. records: the block's attempts, every turn with gen_t_start, gen_t_first_token and
    gen_t_end; turns with gen_t_first_token null are skipped and counted in n_turns_without_first_token,
    and turns_without_first_token_over_1pct flags a block where they are over 1% of n_turns. The
    regression uses the harness-side tokens per window (_harness_window_tokens), not the server
    counters: at the llama.cpp pin the predicted-token counter credits a whole request when it ends. A
    fit on the counter deltas is reported only as a cross-check. Raw joules only: idle
    subtraction would shift the intercept and leave the token slopes unchanged.
    """
    cols = _metrics_columns(samples)
    seconds, joules, server_prefill, server_decode = np.diff(cols, axis=0).T
    (prefill, decode), n_turns, n_skipped = _harness_window_tokens(cols[:, 0], records)
    share = n_skipped / n_turns if n_turns else 0.0

    def fit(p, d):
        b = nnls(np.column_stack([seconds, p, d]), joules)
        return {"intercept_w": float(b[0]), "prefill": float(b[1]), "decode": float(b[2])}

    return {"coef": fit(prefill, decode), "n_windows": len(joules), "seconds": float(seconds.sum()),
            "n_turns": n_turns, "n_turns_without_first_token": n_skipped, "share_turns_without_first_token": share,
            "turns_without_first_token_over_1pct": share > MAX_SHARE_WITHOUT_FIRST_TOKEN,
            "server_counter_fit": {"coef": fit(server_prefill, server_decode)},
            "tokens": {"harness": {"prefill": float(prefill.sum()), "decode": float(decode.sum())},
                       "server": {"prefill": float(server_prefill.sum()), "decode": float(server_decode.sum())}}}


def estimate_batched_joules(rec, batched):
    """An attempt's marginal joules under the batched fit: prefill x uncached tokens + decode x output tokens.

    The intercept (W) is static power shared by every concurrent attempt, so it is left out.
    """
    aid = rec.get("attempt_id")
    feats = np.array([turn_features(t, f"attempt {aid!r} turn {i}") for i, t in enumerate(_turns(rec))])
    return float(batched["coef"]["prefill"] * feats[:, 1].sum() + batched["coef"]["decode"] * feats[:, 0].sum())
