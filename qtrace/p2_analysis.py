"""P1 headline pick and P2 analysis, as frozen before P1 (docs/04, "Analysis plan for P1 and P2", without Stage 2).

Also from docs/03: the P1 headline rule (Phases, P1 row), the B metrics and the Eq. 2-3 breakdown (Metrics) and
the post hoc turn-cap frontier (Baselines). Pure functions; input checks and per-task arrays are in
qtrace/p2_data.py, file I/O in scripts/p1_pick_headline.py and scripts/p2_analyze.py.

Endpoint: joules per resolved task, JPR = sum over tasks of E_t / sum over tasks of R_t, where E_t and R_t are the
means over an arm's seed models of the task's estimated joules and resolved flag (docs/04: the primary analysis
averages each task over the 3 seed models). This is 2610.32417's E_op = O / S, "the operational energy per
verified completion", which "charges every attempt, failed or not, to the completions"; its Eq. 4 splits it by
outcome as E+ + E- (1 - p) / p. E here is the provisional whole-attempt GPU estimate energy_model.json's ranking
block names (#13): the profile's raw P0a model over the generation windows plus the profile's idle_w x
tool_seconds, since a single-user local agent keeps the GPU powered while tools run. Every contrast, guard and
table uses it, and the output carries its provisional flag. The generation-window-only and idle-subtracted JPRs
are reported alongside as point estimates. The upper bound adds container CPU-seconds x the per-vCPU share of
the host CPU's TDP to the whole-attempt estimate.

Statistic: T = log(JPR_arm / JPR_comparator), H: T < 0, from a paired task bootstrap stratified by repo (one draw
shared by both arms). p and bounds follow qtrace/p0d.py did_test (#12): p is the share of draws on the null
side (T >= 0), a one-sided 95% bound is the 95th (or, for the guard, the 5th) inverted_cdf order statistic of
the draws and a two-sided 95% interval the 2.5th and 97.5th. A draw where an arm resolves nothing gives that
arm JPR = inf; a draw where both do is undefined, counted, and put on the null side.

Grammar: every attempt record has a bool grammar, true only for H4's grammar-constrained runs. The primary, the
families, the arm table, the B metrics, the estimator check, the frontier and the P1 pick use grammar false rows
only; grammar true rows enter only h4() (H4_READING).

Batched regime: when P0a's batched agreement is below 85%, docs/04 requires P2 energy in both regimes, so analyze()
adds batched_regime (qtrace/p2_batched.py): the primary on the batched marginal estimate and run-level measured
totals. Every claim stays on the concurrency-1 estimate, the proposed claim regime (docs/03 Metrics). analyze()
refuses to run when energy_model.json does not record that agreement check.

Not here: the per-task length covariate docs/04 (Selection) asks for in the H2/H3 analysis, because that length
difference is between runs selected on pool tasks, and the contrasts are over held-out tasks, so it has no
per-task value here (each arm's mean turns and processed tokens are in the arm table); the training-cost
break-even of docs/03 Metrics, whose training joules no input here carries.
"""

import math

import numpy as np

from qtrace.energy_model import estimate_attempt_joules
from qtrace.p2_batched import batched_arrays, batched_fit, run_level_totals
from qtrace.p2_data import (
    MALFORMED,
    PRIMARY,
    REFERENCE,
    TOKMATCH,
    arrays,
    check_config,
    js,
    load,
    present,
    reps_of,
    strat_weights,
)

HEADLINE_ARMS = ("fewest_turns", "fewest_joules", "stable", "qtrace")  # docs/03 Phases, P1 row, in tie-break order
P0D_ARMS = ("stable", "qtrace")  # the eligible arms that rank by the P0d score (the stability arms)
P0A_VERDICTS = ("pass", "fail", "inconclusive")
SIGN_MIN_REPOS = 3  # docs/04: the sign should hold in at least 3 of the 4 held-out repos
ESTIMATOR_BAND = (0.95, 1.05)  # docs/04 estimator check: the exponentiated 95% CI within +/-5%
TOKEN_RATIO_FAMILIES = ("H1a", "H2")  # docs/03: their secondary is a smaller Q4_K_M/BF16 tokens-per-resolved ratio
NOT_SEPARABLE_WHY = "P0b arm overlap: not separable from another arm, so not trained (docs/04 P0b)"
H4_READING = (
    "H4 (docs/03: the differences between arms survive grammar-constrained tool calls) is read per configured "
    "contrast and profile as: on the grammar true rows the contrast wins (one-sided test of T < 0 with Holm within "
    "H4 below alpha, and the resolve-rate guard against random under grammar passes; the upper bound too when "
    "upper_bound_required names H4), and T keeps the sign it has without grammar on the same tasks. The ratio of "
    "the effects, T with grammar / T without, is reported and not tested: no threshold for how much of a gain may "
    "be formatting is pre-registered.")


def _q(x, q):
    return js(np.percentile(x, q, method="inverted_cdf"))


def _sums(W, X, C=None):
    """Per draw, the sum over tasks (counts W [k, n]) of the seed-mean of X [S, n]; seed counts C [k, S] resample
    seed models (two-stage bootstrap), None keeps them all."""
    per_seed = W @ X.T
    return per_seed.mean(1) if C is None else (per_seed * C).sum(1) / X.shape[0]


def _jpr(e, r):
    return np.divide(e, r, out=np.full(np.shape(e), np.inf), where=np.asarray(r) > 0)


def _log_ratio(ja, jb):
    with np.errstate(invalid="ignore"):
        return np.log(ja) - np.log(jb)  # inf - inf (neither arm resolves anything) is NaN


def _point(A, B, energy="E"):
    one = np.ones((1, A["R"].shape[1]))
    ea, ra, eb, rb = (float(_sums(one, X)[0]) for X in (A[energy], A["R"], B[energy], B["R"]))
    ja, jb = float(_jpr(ea, ra)), float(_jpr(eb, rb))
    t = float(_log_ratio(ja, jb))
    n = one.shape[1]
    return {"jpr": {"arm": js(ja), "comparator": js(jb)}, "resolve_rate": {"arm": ra / n, "comparator": rb / n},
            "T": js(t), "ratio": None if math.isnan(t) else js(math.exp(t)), "rr_diff": (ra - rb) / n}


def _test(W, A, B, energy="E", CA=None, CB=None):
    """One-sided bootstrap test of T < 0 and the one-sided 95% lower bound of RR_A - RR_B, on the same draws."""
    ra, rb = _sums(W, A["R"], CA), _sums(W, B["R"], CB)
    t = _log_ratio(_jpr(_sums(W, A[energy], CA), ra), _jpr(_sums(W, B[energy], CB), rb))
    undefined = np.isnan(t)
    t = np.where(undefined, np.inf, t)  # no resolves in either arm is no evidence for T < 0
    d = (ra - rb) / A["R"].shape[1]
    return {"p_one_sided": float(np.mean(t >= 0)), "upper95_one_sided": _q(t, 95),
            "n_draws_undefined": int(undefined.sum()), "rr_diff_lower95_one_sided": _q(d, 5)}


def _alongside(A, B):
    """The same JPR contrast on the generation-window-only and idle-subtracted estimates, reported only."""
    keep = ("jpr", "T", "ratio")
    gen = {k: v for k, v in _point(A, B, "G").items() if k in keep}
    idle = None if A["I"] is None or B["I"] is None else {k: v for k, v in _point(A, B, "I").items() if k in keep}
    return {"generation_window": gen, "idle_subtracted": idle}


def _tripwire(X):
    """docs/04 Test-file tripwire: resolve rate over all attempts, over unflagged ones, and the flagged share."""
    keep = X["F"] == 0
    return {"resolve_rate": float(X["R"].mean()), "flagged_share": float(X["F"].mean()),
            "resolve_rate_unflagged": float(X["R"][keep].mean()) if keep.any() else None}


def _sub(X, idx):
    return {k: (v[:, idx] if isinstance(v, np.ndarray) else v) for k, v in X.items()}


def contrast(arr, a, b, cfg, detail=False):
    """T and the guard for group a vs group b (each (arm, profile)) on arrays() output; the same under the upper
    bound when arrays() was given cpu_w.

    Guard: the one-sided 95% lower bound of RR_a - RR_random from the same draws is >= -guard_margin, so arr must
    hold ("random", a's profile). docs/04 states the guard against random and applies it to every hypothesis,
    so a secondary contrast is guarded against random, not its comparator; at a profile other than Q4_K_M,
    random at that profile stands in for the stated Q4_K_M rates. detail, for the primary (b is random), adds
    the two-stage bootstrap (seed models resampled with replacement independently per arm, then tasks within
    repo, paired across arms), per-seed estimates (the s-th seed model of each arm, in seed order) and per-repo
    effects on the repo's block of the same draws.
    """
    A, B, G = arr["groups"][a], arr["groups"][b], arr["groups"].get(("random", a[1]))
    if G is None:
        raise ValueError(f"the resolve-rate guard needs random at {a[1]} in the contrast's groups")
    rng = np.random.default_rng(cfg["seed"])
    W = strat_weights(arr["repos"], cfg["n_boot"], rng)
    gpu = _test(W, A, B)
    gpu.pop("rr_diff_lower95_one_sided")
    rr_d = (_sums(W, A["R"]) - _sums(W, G["R"])) / A["R"].shape[1]
    guard = {"vs": "random", "rr_diff": _point(A, G)["rr_diff"], "lower95_one_sided": _q(rr_d, 5),
             "margin": cfg["guard_margin"]}
    guard["pass"] = guard["lower95_one_sided"] >= -cfg["guard_margin"]
    out = {"arm": a[0], "comparator": b[0], "profile": a[1], "n_tasks": len(arr["tasks"]),
           "n_tasks_dropped_infra_error": arr["n_dropped_error"],
           "n_tasks_dropped_infra_timeout": arr["n_dropped_timeout"],
           **_point(A, B), "gpu": gpu, "guard": guard, "alongside": _alongside(A, B)}
    if A["U"] is not None:
        ub = _test(W, A, B, "U")
        ub.pop("rr_diff_lower95_one_sided")
        out["upper_bound"] = {**_point(A, B, "U"), **ub}
    if detail:
        if B is not G or len(A["reps"]) != len(B["reps"]):
            raise ValueError(f"the primary detail pairs {a} with random seed model by seed model; got {b} with "
                             f"{len(A['reps'])} and {len(B['reps'])} seed models")
        CA, CB = (rng.multinomial(X["R"].shape[0], np.full(X["R"].shape[0], 1 / X["R"].shape[0]), size=cfg["n_boot"])
                  for X in (A, B))
        two = _test(W, A, B, "E", CA, CB)
        out["two_stage"] = {**two, "guard_pass": two["rr_diff_lower95_one_sided"] >= -cfg["guard_margin"]}
        out["per_seed"] = []
        for s, (sa, sb) in enumerate(zip(A["reps"], B["reps"])):
            As, Bs = ({k: (v[s:s + 1] if isinstance(v, np.ndarray) else v) for k, v in X.items()} for X in (A, B))
            out["per_seed"].append({"seeds": [sa, sb], **_point(As, Bs), **_test(W, As, Bs)})
        out["per_repo"] = {}
        for repo in sorted(set(arr["repos"])):
            idx = np.flatnonzero(arr["repos"] == repo)
            Ar, Br = _sub(A, idx), _sub(B, idx)
            out["per_repo"][repo] = {"n_tasks": len(idx), **_point(Ar, Br), **_test(W[:, idx], Ar, Br)}
        # float() reads js()'s "inf" / "-inf" back, so a repo where only one arm resolves still has a sign.
        sign = [0 if x is None else np.sign(float(x)) for x in [out["T"]] + [r["T"] for r in out["per_repo"].values()]]
        same = sum(1 for s in sign[1:] if s == sign[0] != 0)
        out["sign_holds"] = {"n_repos": len(sign) - 1, "n_same_sign": same, "min_repos": SIGN_MIN_REPOS,
                             "holds": same >= SIGN_MIN_REPOS}
    return out


def holm(pvals):
    """Holm step-down adjusted p-values, in input order (reject where adjusted < alpha)."""
    m, adj, run = len(pvals), [None] * len(pvals), 0.0
    for i, j in enumerate(sorted(range(m), key=lambda j: pvals[j])):
        run = max(run, min(1.0, (m - i) * pvals[j]))
        adj[j] = run
    return adj


def families(data, cfg, cpu_w, headline):
    """Secondary contrasts by hypothesis family (cfg families), each at every profile where both arms and random ran.

    "headline" in a pair is the P1 pick. Holm runs within a family over its tested contrasts, once on the GPU
    p-values and once on the upper-bound ones. A contrast wins when its Holm p < alpha and it passes the guard
    (against random, see contrast()); a family in upper_bound_required is supported only where the win also holds
    under the upper bound. A contrast whose arm, comparator or random did not run at a profile, or whose arm is
    its comparator once headline is filled in, is listed as untested. Each contrast is on the tasks its arm,
    comparator and random share, so a secondary-tier contrast is on its held-out subset.
    """
    have, profiles = present(data), sorted({p for _, p in present(data)})
    out = {}
    for fam, pairs in cfg["families"].items():
        tested, untested = [], []
        for pair in pairs:
            a, b = (headline if x == "headline" else x for x in pair)
            for prof in profiles:
                arms = list(dict.fromkeys((a, b, "random")))
                missing = [arm for arm in arms if (arm, prof) not in have]
                if a == b or missing:
                    untested.append({"arm": a, "comparator": b, "profile": prof, "missing_arms": missing})
                    continue
                arr = arrays(data, [(arm, prof) for arm in arms], cfg["infra_error_cap"], cpu_w)
                tested.append(contrast(arr, (a, prof), (b, prof), cfg))
        needs_ub = fam in cfg["upper_bound_required"]
        judged = _judge(tested, needs_ub, cfg)
        out[fam] = {"upper_bound_required": needs_ub, "n_tested": len(judged), "contrasts": judged,
                    "untested": untested}
    return out


def _judge(tested, needs_ub, cfg):
    """Holm within one family, on the GPU and the upper-bound p-values, and the win and support of each contrast."""
    judged = []
    for c, p, pu in zip(tested, holm([c["gpu"]["p_one_sided"] for c in tested]),
                        holm([c["upper_bound"]["p_one_sided"] for c in tested])):
        wins, holds = p < cfg["alpha"] and c["guard"]["pass"], pu < cfg["alpha"] and c["guard"]["pass"]
        judged.append({**c, "holm_p": p, "holm_p_upper_bound": pu, "wins": wins, "holds_under_upper_bound": holds,
                       "supported": wins and (holds or not needs_ub)})
    return judged


def _sign(t):
    return 0 if t is None else np.sign(float(t))  # float() reads js()'s "inf" / "-inf"


def h4(data, gdata, headline, cfg, cpu_w):
    """docs/03 H4 on the grammar true rows (gdata) against the grammar false rows (data), read as H4_READING.

    Each h4_contrasts pair ("headline" is the P1 pick) runs at every profile where both arms and random have
    grammar rows, on the tasks those groups ran both with and without grammar, so the two effects are on the same
    tasks. Holm runs within H4. supported is the H4_READING verdict for the contrast. Fails when headline and
    random have no grammar rows at Q4_K_M, the contrast docs/03 Cost option b names.
    """
    pairs = dict.fromkeys(tuple(headline if x == "headline" else x for x in p) for p in cfg["h4_contrasts"])
    have, main = present(gdata), present(data)
    if any((arm, PRIMARY) not in have for arm in (headline, "random")):
        raise ValueError(f"H4 needs grammar rows of {headline} and random at {PRIMARY}")
    tested, untested = [], []
    for a, b in pairs:
        for prof in sorted({p for _, p in have}):
            arms = list(dict.fromkeys((a, b, "random")))
            missing = [arm for arm in arms if (arm, prof) not in have or (arm, prof) not in main]
            if a == b or missing:
                untested.append({"arm": a, "comparator": b, "profile": prof, "missing_arms": missing})
                continue
            groups, cap = [(arm, prof) for arm in arms], cfg["infra_error_cap"]
            on = arrays(gdata, groups, cap, cpu_w)
            off = arrays(data, groups, cap, cpu_w, only=on["tasks"])
            if len(off["tasks"]) < len(on["tasks"]):  # a task with an infrastructure error without grammar
                on = arrays(gdata, groups, cap, cpu_w, only=off["tasks"])
            c_on, c_off = (contrast(x, (a, prof), (b, prof), cfg) for x in (on, off))
            t_on, t_off = (None if c["T"] is None else float(c["T"]) for c in (c_on, c_off))
            finite = t_on is not None and t_off is not None and math.isfinite(t_on) and math.isfinite(t_off)
            tested.append({**c_on, "without_grammar": {k: c_off[k] for k in ("n_tasks", "jpr", "T", "ratio", "gpu",
                                                                              "guard")},
                           "keeps_sign": bool(_sign(t_off) != 0 and _sign(t_on) == _sign(t_off)),
                           "effect_ratio": js(t_on / t_off) if finite and t_off != 0 else None})
    judged = [{**c, "supported": c["supported"] and c["keeps_sign"]}
              for c in _judge(tested, "H4" in cfg["upper_bound_required"], cfg)]
    return {"reading": H4_READING, "n_rows": gdata["n_rows"], "n_tested": len(judged), "contrasts": judged,
            "untested": untested}


def _ratio_draws(W, A, B, X):
    """Per draw, (sum X_A / sum R_A) / (sum X_B / sum R_B); undefined draws go to +inf like _test's."""
    r = _log_ratio(_jpr(_sums(W, A[X]), _sums(W, A["R"])), _jpr(_sums(W, B[X]), _sums(W, B["R"])))
    return np.exp(np.where(np.isnan(r), np.inf, r))


def breakdown(tot_q, tot_b):
    """docs/03 Eq. 2-3 breakdown (2610.32417): log(JPR_Q / JPR_B) = log ratio of joules per processed token +
    log ratio of processed tokens per attempt + log ratio of attempts per resolved task.

    tot_*: {J, P, A, R} totals over the same tasks (seed-means summed, so A is the task count). The three
    terms are the per-token cost, the extra tokens and the extra failures, and they sum to the total exactly.
    """
    def lr(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.log(np.float64(tot_q[num]) / tot_q[den]) - np.log(np.float64(tot_b[num]) / tot_b[den]))
    terms = {"joules_per_token": lr("J", "P"), "tokens_per_attempt": lr("P", "A"),
             "attempts_per_resolved": lr("A", "R")}
    total = lr("J", "R")
    return {"log_total": js(total), **{k: js(v) for k, v in terms.items()},
            "sum_of_terms": js(sum(terms.values()))}


def b_metrics(data, arm, cfg):
    """docs/03 "B, always next to it" for one arm: Q4_K_M and BF16 resolve rate, JPR and processed tokens per
    resolved task (each profile's own raw energy model), their Q4_K_M/BF16 ratios with 95% intervals, the
    quantization gap (BF16 minus Q4_K_M resolve rate), the Eq. 2-3 breakdown, and the arm's BF16 resolve-rate
    difference from random with its two-sided 95% interval, all on the tasks the groups share."""
    groups = [(arm, PRIMARY), (arm, REFERENCE)] + ([("random", REFERENCE)] if arm != "random" else [])
    arr = arrays(data, groups, cfg["infra_error_cap"])
    Q, F = arr["groups"][groups[0]], arr["groups"][groups[1]]
    n = len(arr["tasks"])
    W = strat_weights(arr["repos"], cfg["n_boot"], np.random.default_rng(cfg["seed"]))
    tot = {p: {"J": X["E"].mean(0).sum(), "P": X["P"].mean(0).sum(), "A": n, "R": X["R"].mean(0).sum()}
           for p, X in ((PRIMARY, Q), (REFERENCE, F))}
    out = {"arm": arm, "n_tasks": n, "n_tasks_dropped_infra_error": arr["n_dropped_error"],
           "resolve_rate": {p: t["R"] / n for p, t in tot.items()},
           "jpr": {p: js(_jpr(t["J"], t["R"])) for p, t in tot.items()},
           "processed_tokens_per_resolved": {p: js(_jpr(t["P"], t["R"])) for p, t in tot.items()},
           "quant_gap": (tot[REFERENCE]["R"] - tot[PRIMARY]["R"]) / n}
    for name, X, key in (("jpr_ratio", "E", "J"), ("processed_tokens_ratio", "P", "P")):
        log_ratio = _log_ratio(_jpr(tot[PRIMARY][key], tot[PRIMARY]["R"]),
                               _jpr(tot[REFERENCE][key], tot[REFERENCE]["R"]))
        draws = _ratio_draws(W, Q, F, X)
        out[name] = {"value": js(np.exp(log_ratio)), "ci95": [_q(draws, 2.5), _q(draws, 97.5)]}
    out["breakdown"] = breakdown(tot[PRIMARY], tot[REFERENCE])
    if arm != "random":
        Rn = arr["groups"][("random", REFERENCE)]
        d = (_sums(W, F["R"]) - _sums(W, Rn["R"])) / n
        out["bf16_rr_diff_vs_random"] = {"value": (tot[REFERENCE]["R"] - Rn["R"].mean(0).sum()) / n,
                                         "ci95": [_q(d, 2.5), _q(d, 97.5)]}
    return out


def tokens_ratio_contrasts(data, cfg, headline):
    """docs/03 secondary endpoints of H1a and H2: the arm has a smaller Q4_K_M/BF16 ratio of processed tokens per
    resolved task than its comparator. Per configured pair of those families ("headline" is the P1 pick, as in
    families()) whose arms differ and both ran at both profiles (others are left out), the arm's ratio over the comparator's on the tasks the four groups share,
    with a two-sided 95% interval from stratified task draws shared by both arms. A draw where both ratios are
    undefined goes to +inf, as in _ratio_draws. Reported only: no test, Holm or guard."""
    have, out = present(data), []
    for fam in TOKEN_RATIO_FAMILIES:
        for pair in cfg["families"].get(fam, []):
            a, b = (headline if x == "headline" else x for x in pair)
            groups = [(x, p) for x in (a, b) for p in (PRIMARY, REFERENCE)]
            if a == b or any(g not in have for g in groups):
                continue
            arr = arrays(data, groups, cfg["infra_error_cap"])
            G, one = arr["groups"], np.ones((1, len(arr["tasks"])))
            W = strat_weights(arr["repos"], cfg["n_boot"], np.random.default_rng(cfg["seed"]))
            with np.errstate(invalid="ignore"):
                ra, rb, da, db = (_ratio_draws(w, G[x, PRIMARY], G[x, REFERENCE], "P") for w in (one, W) for x in (a, b))
                point, draws = ra / rb, np.nan_to_num(da / db, nan=np.inf)
            out.append({"family": fam, "arm": a, "comparator": b, "n_tasks": len(arr["tasks"]),
                        "arm_ratio": js(ra[0]), "comparator_ratio": js(rb[0]), "ratio_of_ratios": js(point[0]),
                        "ci95": [_q(draws, 2.5), _q(draws, 97.5)]})
    return out


def arm_table(data, cfg, cpu_w):
    """Per (arm, profile), on its own tasks: resolve rate, JPR (whole attempt, upper bound, and alongside the
    generation-window-only and idle-subtracted ones), the tripwire numbers, processed tokens per resolved
    task (the estimator-free check docs/04 co-reports for every arm), mean turns and mean malformed calls per
    attempt by type (docs/03 Metrics)."""
    rows = []
    for arm, prof in sorted(present(data)):
        arr = arrays(data, [(arm, prof)], cfg["infra_error_cap"], cpu_w)
        X = arr["groups"][arm, prof]
        att = [data["cells"][arm, r, prof][t] for r in X["reps"] for t in arr["tasks"]]
        turns = np.mean([s["rec"]["turns"] for s in att])
        r = X["R"].mean(0).sum()
        rows.append({"arm": arm, "profile": prof, "n_seed_models": len(X["reps"]), "n_tasks": len(arr["tasks"]),
                     "resolve_rate": r / len(arr["tasks"]), "jpr": js(_jpr(X["E"].mean(0).sum(), r)),
                     "jpr_upper_bound": js(_jpr(X["U"].mean(0).sum(), r)),
                     "jpr_generation_window": js(_jpr(X["G"].mean(0).sum(), r)),
                     "jpr_idle_subtracted": None if X["I"] is None else js(_jpr(X["I"].mean(0).sum(), r)),
                     "tripwire": _tripwire(X),
                     "processed_tokens_per_resolved": js(_jpr(X["P"].mean(0).sum(), r)), "mean_turns": float(turns),
                     "malformed_per_attempt": {k: float(np.mean([s["malformed"][k] for s in att])) for k in MALFORMED}})
    return rows


def _usable(data, arm):
    """estimator_check's attempts of one arm: {task: summarize() output} of the usable ones and the excluded
    counts by reason."""
    reps = reps_of(data, arm, PRIMARY)
    if len(reps) != 1:
        raise ValueError(f"estimator check: {arm} needs exactly one seed model at {PRIMARY}, got {reps}")
    used, excluded = {}, {"infra_error": 0, "concurrency": 0, "not_attributable": 0, "unmeasured_turn": 0}
    for task, s in data["cells"][arm, reps[0], PRIMARY].items():
        reason = ("infra_error" if s is None else "concurrency" if s["rec"]["concurrency"] != 1
                  else "not_attributable" if not s["rec"]["energy"]["attributable"]
                  else "unmeasured_turn" if any(t["gen_energy_j"] is None for t in s["rec"]["per_turn"]) else None)
        if reason:
            excluded[reason] += 1
            continue
        used[task] = s
    return used, excluded


def _measured(s, scope):
    """An attempt's measured joules in estimator_check's scope: generation windows, plus tool windows when
    scope is "whole_attempt"."""
    return sum(t["gen_energy_j"] + (t["tool_energy_j"] if scope == "whole_attempt" else 0) for t in s["rec"]["per_turn"])


def estimator_check(data, headline, cfg):
    """docs/04 Estimator check on load()-ed estimator-check rows (Q4_K_M, concurrency 1, one seed model per arm).

    An attempt is used when it is Q4_K_M at concurrency 1, energy.attributable is true and every turn has a
    measured gen_energy_j; r = log(measured / estimated), like with like. When every used attempt also has every
    turn's tool_energy_j, scope is "whole_attempt": measured = the sum of gen_energy_j + tool_energy_j and the
    estimate is the whole-attempt one. Otherwise scope is "generation_window" on both sides: the sum of
    gen_energy_j against the generation-window estimate. One scope for the whole check keeps the paired
    differences comparable. Per pair of arms, tasks with a usable attempt in both give the paired difference
    in r; the estimate stands if the exponentiated two-sided 95% interval of its mean (task bootstrap, not
    stratified: docs/04 stratifies only the primary test, and a repo with one paired task has no spread within
    repo, so stratified draws shrink the interval, to a point when every repo has one) lies within ESTIMATOR_BAND for each pair claim A compares on estimates: headline vs random (the
    primary) and headline vs base (the turn-cap frontier). docs/04 says "the between-arm difference" for three
    arms; random vs base feeds no claim and is reported only.
    """
    used, excluded = {}, {}
    for arm in (headline, "random", "base"):
        used[arm], excluded[arm] = _usable(data, arm)
    no_tool = sum(any(t["tool_energy_j"] is None for t in s["rec"]["per_turn"])
                  for by_task in used.values() for s in by_task.values())
    scope = "generation_window" if no_tool else "whole_attempt"
    r = {}
    for arm, by_task in used.items():
        r[arm] = {}
        for task, s in by_task.items():
            measured = _measured(s, scope)
            est = s["e"] if scope == "whole_attempt" else s["e_gen"]
            if measured <= 0 or est <= 0:
                raise ValueError(f"estimator check: attempt {s['attempt_id']!r} has measured {measured} J, "
                                 f"estimated {est} J ({scope}); log(measured / estimated) needs both positive")
            r[arm][task] = math.log(measured / est)
    pairs = {}
    for a, b in ((headline, "random"), (headline, "base"), ("random", "base")):
        tasks = sorted(set(r[a]) & set(r[b]))
        if len(tasks) < 2:
            raise ValueError(f"estimator check: {a} vs {b} has {len(tasks)} paired tasks")
        d = np.array([r[a][t] - r[b][t] for t in tasks])
        W = strat_weights(np.zeros(len(tasks)), cfg["n_boot"], np.random.default_rng(cfg["seed"]))  # one stratum
        means = W @ d / len(d)
        lo, hi = (math.exp(_q(means, q)) for q in (2.5, 97.5))
        pairs[f"{a}_vs_{b}"] = {"n_tasks": len(tasks), "mean_diff_log": float(d.mean()), "ratio": math.exp(d.mean()),
                                "ratio_ci95": [lo, hi],
                                "within_band": ESTIMATOR_BAND[0] <= lo and hi <= ESTIMATOR_BAND[1]}
    decides = [f"{headline}_vs_random", f"{headline}_vs_base"]
    return {"scope": scope, "n_attempts_without_tool_energy": no_tool, "band": list(ESTIMATOR_BAND), "decides": decides,
            "stands": all(pairs[k]["within_band"] for k in decides), "pairs": pairs,
            "mean_log_measured_over_estimated": {a: float(np.mean(list(v.values()))) if v else None
                                                 for a, v in r.items()},
            "excluded": excluded}


def measured_subset(edata, headline, scope, cfg, cpu_w):
    """docs/04 Estimator check, "otherwise only the measured subset is claimed": the primary contrast (headline vs
    random at Q4_K_M, contrast() with its guard) with measured joules in place of the estimate, on the
    estimator-check tasks where both arms have a usable attempt, in the scope estimator_check chose. The upper
    bound adds the same container CPU share to the measured joules. One seed model per arm, so no detail."""
    used = {arm: _usable(edata, arm)[0] for arm in (headline, "random")}
    groups = [(headline, PRIMARY), ("random", PRIMARY)]
    arr = arrays(edata, groups, cfg["infra_error_cap"], cpu_w, only=set(used[headline]) & set(used["random"]))
    meas = {g: np.array([[_measured(used[g[0]][t], scope) for t in arr["tasks"]]]) for g in groups}
    arr = {**arr, "groups": {g: {**X, "E": meas[g], "U": X["U"] - X["E"] + meas[g], "G": meas[g], "I": None}
                             for g, X in arr["groups"].items()}}
    c = contrast(arr, *groups, cfg)
    return {"scope": scope, **{k: c[k] for k in ("arm", "comparator", "profile", "n_tasks", "jpr", "resolve_rate",
                                                 "T", "ratio", "gpu", "guard", "upper_bound")},
            "wins": c["gpu"]["p_one_sided"] < cfg["alpha"] and c["guard"]["pass"],
            "holds_under_upper_bound": c["upper_bound"]["p_one_sided"] < cfg["alpha"] and c["guard"]["pass"]}


def _curve(data, arm, tasks, entry):
    """Turns, resolved flags (as charged by summarize()) and cumulative estimated whole-attempt joules by turn (one
    array per attempt) of an arm's attempts. Turn i is charged its generation window and idle_w x its own tool_s,
    so a cap charges tool time for the kept turns only."""
    att = [data["cells"][arm, rep, PRIMARY][t] for rep in reps_of(data, arm, PRIMARY) for t in tasks]
    recs = [s["rec"] for s in att]
    # Per-turn estimates through the same estimator; summarize() checked tool_seconds = sum of tool_s, so the
    # uncapped point equals the attempts' estimates.
    cums = [np.cumsum([estimate_attempt_joules({"attempt_id": r["attempt_id"], "per_turn": [t], "tool_seconds":
                                                t["tool_s"]}, entry["raw"], idle_w=entry["idle_w"])
                       for t in r["per_turn"]]) for r in recs]
    return np.array([r["turns"] for r in recs]), np.array([s["r"] > 0 for s in att]), cums


def turn_cap_frontier(data, headline, energy_model, cfg, base="base"):
    """docs/03 Baselines: the post hoc turn-cap curve of base and headline at Q4_K_M.

    At cap c an attempt counts as resolved only if it resolved within c turns, and its joules are the whole-attempt
    estimate over its first min(turns, c) turns, tool windows of those turns included; each point is (resolve
    rate, mean joules per attempt, JPR). The uncapped headline point must lie below base's whole frontier on goal
    A's endpoint: it fails if any cap gives base a resolve rate at least the headline's and a JPR at most the
    headline's. JPR is not monotone in the cap, so every cap is checked; a base point that dominates in joules per
    attempt also dominates in JPR.
    """
    arr = arrays(data, [(base, PRIMARY), (headline, PRIMARY)], cfg["infra_error_cap"])
    entry = energy_model["profiles"][PRIMARY]
    curves = {arm: _curve(data, arm, arr["tasks"], entry) for arm in (base, headline)}
    max_cap = int(max(turns.max() for turns, _, _ in curves.values()))
    pts = {}
    for arm, (turns, res, cums) in curves.items():
        J = np.array([np.pad(c, (0, max_cap - len(c)), mode="edge") for c in cums])  # a finished attempt stops spending
        pts[arm] = []
        for c in range(1, max_cap + 1):
            r, m = float(np.mean(res & (turns <= c))), float(J[:, c - 1].mean())
            pts[arm].append({"cap": c, "resolve_rate": r, "mean_joules": m, "jpr": js(_jpr(m, r))})
    top = pts[headline][-1]
    # float() reads js()'s "inf" back: a headline that resolves nothing is dominated by every base point.
    dom = [p for p in pts[base] if p["resolve_rate"] >= top["resolve_rate"] and float(p["jpr"]) <= float(top["jpr"])]
    return {"n_tasks": len(arr["tasks"]), "max_cap": max_cap, "headline_uncapped": top,
            "base_points_dominating": dom, "pass": not dom, "points": pts}


def p0d_dropped(score_choice, did):
    """Eligible arms P0d leaves out of the headline, with the reason.

    score_choice.json as in qtrace/p1_select.py score_info (#14): chosen null with decided true (or absent)
    answers H3 negatively and drops the stability arms; decided false is an error. With a chosen score, the DiD
    decision (did.json) applies: pass keeps them, fail drops them (kill rule), inconclusive leaves them
    exploratory and out of the headline (docs/04 P0d).
    """
    if score_choice is None or "chosen" not in score_choice:
        raise ValueError("score_choice.json (P0d) with a 'chosen' key is required to know which arms are eligible")
    chosen, decided = score_choice["chosen"], score_choice.get("decided", True)
    if type(decided) is not bool:
        raise ValueError(f"score_choice.json decided must be true or false, got {decided!r}")
    if chosen is None:
        if not decided:
            raise ValueError(f"score_choice.json is undecided ({score_choice.get('verdict')}), so H3 is not answered; "
                             "rerun P0d once that check can run")
        return {a: "P0d score choice answered H3 negatively; stability arms dropped before P1" for a in P0D_ARMS}
    if did is None:
        raise ValueError(f"P0d chose {chosen!r} but did.json is missing; its decision sets the stability arms' "
                         "eligibility")
    if did.get("score") != chosen:
        raise ValueError(f"did.json tests score {did.get('score')!r}, but score_choice.json chose {chosen!r}")
    decision = did.get("decision")
    if decision == "pass":
        return {}
    if decision in ("fail", "inconclusive"):
        why = ("failed: the kill rule drops the score" if decision == "fail"
               else "inconclusive: exploratory, out of the headline")
        return {a: f"P0d DiD {why}" for a in P0D_ARMS}
    raise ValueError(f"did.json decision must be pass, fail or inconclusive, got {decision!r}")


def token_match(headline, train_meta):
    """docs/04 Training: "If the headline arm saw more supervised tokens than random, one token-matched rerun of
    that contrast is added." train_meta: the P1 runs' train_meta.json contents (scripts/p1_train_lora.py);
    token-matched reruns among them (match_supervised_tokens set) are skipped. The rerun trains the headline arm
    with --match-supervised-tokens set to random's supervised_tokens_seen."""
    seen = {}
    for m in train_meta:
        if m.get("match_supervised_tokens") is None and m.get("arm") in (headline, "random"):
            if m["arm"] in seen:
                raise ValueError(f"two train_meta.json for {m['arm']} without match_supervised_tokens; P1 trains one")
            n = m.get("supervised_tokens_seen")
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise ValueError(f"train_meta.json of {m['arm']}: supervised_tokens_seen must be a positive integer, "
                                 f"got {n!r}")
            seen[m["arm"]] = n
    for arm in (headline, "random"):
        if arm not in seen:
            raise ValueError(f"no train_meta.json for {arm} among the P1 runs given")
    return {"arm": headline, "arm_supervised_tokens": seen[headline], "random_supervised_tokens": seen["random"],
            "rerun_required": seen[headline] > seen["random"], "match_supervised_tokens": seen["random"]}


def _rr_diff_vs_base(data, arm, cfg):
    """docs/04 Training drift, its resolve-rate half: the arm's BF16 resolve rate minus base's, on the tasks the two
    share. Its own arrays() call, so base's infrastructure drops never change the tasks the pick is made on."""
    arr = arrays(data, [(arm, REFERENCE), ("base", REFERENCE)], cfg["infra_error_cap"])
    return float((arr["groups"][arm, REFERENCE]["R"].sum() - arr["groups"]["base", REFERENCE]["R"].sum())
                 / len(arr["tasks"]))


def pick_headline(records, manifest, energy_model, score_choice, did, cfg, p0a_verdict, not_separable=(),
                  train_meta=None):
    """docs/03 P1 rule: among HEADLINE_ARMS minus those P0a or P0d dropped, the lowest estimated Q4_K_M JPR whose
    BF16 resolve rate is not clearly below random's (clearly below: the two-sided 95% upper bound of RR_arm -
    RR_random at BF16 is below 0). Controls are never eligible; an arm that resolves nothing at Q4_K_M cannot be
    picked. All arms are compared on the tasks they share. Exact JPR ties go to the earlier arm in HEADLINE_ARMS.

    JPR is the whole-attempt estimate (module docstring); the generation-window JPR and, per (arm, profile), the
    docs/04 tripwire numbers (resolve rate overall and without flagged attempts, flagged share) are reported.
    p0a_verdict is P0a's final gate verdict. On "fail" the docs/03 kill rule drops the robustness half, the
    stability arms included, so P0d's inputs are not read and may be None. not_separable names the eligible arms
    P0b's arm-overlap rule flagged as not separable and P1 did not train (docs/04 P0b); they are dropped, and
    they must have no P1 rows. Any other eligible arm without rows is an error. With train_meta (the P1 runs'
    train_meta.json contents), token_match says whether the headline needs the token-matched rerun (token_match());
    without it, or with no headline, token_match is None. When base ran at BF16 in P1, each arm row carries
    bf16_rr_diff_vs_base (docs/04 Training: drift from the base model, BF16 resolve-rate change); it is reported,
    not used by the rule. Base is P1's own inrepo_test rows: P0b's BF16 runs are on pool200 and heldout100, which
    share no task with inrepo_test.

    The tripwire numbers cover every (arm, profile) with rows, controls and dropped arms included, each on its
    own tasks (docs/04 Test-file tripwire: every P1 resolve rate)."""
    if p0a_verdict not in P0A_VERDICTS:
        raise ValueError(f"p0a_verdict must be one of {P0A_VERDICTS} (P0a's final gate verdict), got {p0a_verdict!r}")
    cfg = check_config(cfg, upper_bound=False)
    data = load(records, manifest, energy_model, "p1")
    multi = {a: reps_of(data, a, p) for a, p in present(data) if len(reps_of(data, a, p)) != 1}
    if multi:
        raise ValueError(f"P1 is a one-seed pilot; these arms have other than one seed model: {multi}")
    bad = [a for a in not_separable if a not in HEADLINE_ARMS or any(arm == a for arm, _ in present(data))]
    if bad:
        raise ValueError(f"not_separable arms must be untrained arms of {HEADLINE_ARMS}; got {bad}")
    dropped = ({a: "P0a failed: the kill rule drops the stability arms" for a in P0D_ARMS} if p0a_verdict == "fail"
               else p0d_dropped(score_choice, did))
    dropped = {a: NOT_SEPARABLE_WHY for a in not_separable} | dropped
    arms = [a for a in HEADLINE_ARMS if a not in dropped]
    groups = [(a, p) for a in arms for p in (PRIMARY, REFERENCE)] + [("random", REFERENCE)]
    arr = arrays(data, groups, cfg["infra_error_cap"])
    W = strat_weights(arr["repos"], cfg["n_boot"], np.random.default_rng(cfg["seed"]))
    n, Rn = len(arr["tasks"]), arr["groups"]["random", REFERENCE]["R"]
    has_base, rows = ("base", REFERENCE) in present(data), []
    for a in arms:
        Q, F = arr["groups"][a, PRIMARY], arr["groups"][a, REFERENCE]
        jpr = float(_jpr(Q["E"].sum(), Q["R"].sum()))
        d = (_sums(W, F["R"]) - _sums(W, Rn)) / n
        lo, hi = _q(d, 2.5), _q(d, 97.5)
        rows.append({"arm": a, "jpr_q4": js(jpr), "resolve_rate_q4": Q["R"].sum() / n,
                     "resolve_rate_bf16": F["R"].sum() / n,
                     "bf16_rr_diff_vs_random": (F["R"].sum() - Rn.sum()) / n, "bf16_rr_diff_ci95": [lo, hi],
                     "clearly_below_random": hi < 0, "eligible": hi >= 0 and math.isfinite(jpr),
                     "jpr_q4_generation_window": js(_jpr(Q["G"].sum(), Q["R"].sum())),
                     "bf16_rr_diff_vs_base": _rr_diff_vs_base(data, a, cfg) if has_base else None})
    # Eligible JPRs are finite, so jpr_q4 holds the exact float and ties compare exactly.
    ok = [r for r in rows if r["eligible"]]
    best = min(ok, key=lambda r: (r["jpr_q4"], HEADLINE_ARMS.index(r["arm"]))) if ok else None
    ties = [r["arm"] for r in ok if best and r["jpr_q4"] == best["jpr_q4"] and r is not best]
    return {"headline": best and best["arm"], "tied_with": ties, "n_tasks": n, "n_tasks_dropped_infra_error":
            arr["n_dropped_error"], "arms": rows, "dropped": dropped, "rule": "lowest estimated Q4_K_M joules per "
            "resolved task among eligible arms; ties by docs/03 arm order", "guard_margin": cfg["guard_margin"],
            "n_boot": cfg["n_boot"], "seed": cfg["seed"], "p0a_verdict": p0a_verdict, "energy": data["energy"],
            "tripwire": {f"{a}/{p}": _tripwire(arrays(data, [(a, p)], cfg["infra_error_cap"])["groups"][a, p])
                         for a, p in sorted(present(data))},
            "reason": None if best else "no eligible arm",
            "drift_base": "base BF16 rows of P1 (inrepo_test), same tasks" if has_base else None,
            "token_match": None if train_meta is None or not best else token_match(best["arm"], train_meta)}


def within_repo(records, manifest, energy_model, headline, cfg, cpu_w):
    """docs/03 Setup: the within-repo split (inrepo_test) reported next to the primary, to show how much is repo
    memorization. From the P1 records (load(phase="p1"), one seed model per arm): the arm table and headline vs
    random at Q4_K_M through contrast(). Labelled exploratory (docs/04 Seeds, repos and multiplicity): one seed,
    no Holm, no win."""
    data = load(records, manifest, energy_model, "p1")
    groups = [(headline, PRIMARY), ("random", PRIMARY)]
    return {"exploratory": True, "list": "inrepo_test", "n_rows": data["n_rows"], "arms": arm_table(data, cfg, cpu_w),
            "headline_vs_random": contrast(arrays(data, groups, cfg["infra_error_cap"], cpu_w), *groups, cfg)}


def analyze(records, manifest, energy_model, headline, cfg, estimator_rows=None, p1_records=None, p1_manifest=None):
    """The P2 analysis: the primary test (headline vs random at Q4_K_M, detail and the infra_timeout sensitivity),
    the token-matched rerun of it (token_matched, when a token_matched model of the headline arm ran at Q4_K_M,
    else None), the hypothesis families with Holm, the arm table, B metrics per arm at Q4_K_M and BF16, the H1a/H2 tokens-ratio
    contrasts, the estimator check (when its rows are given) and, when the estimate does not stand, the primary on
    the measured subset (measured_subset), the turn-cap frontier (when base ran), the claim-A summary, all on
    grammar false rows, H4 on the grammar true rows (None when there are none), and the exploratory within-repo
    split from the P1 records and manifest (within_repo(), None when they are not given). Claim A on
    estimates also needs the primary's sign in at least 3 of the 4 held-out repos (docs/04). batched_regime is
    None unless P0a's batched agreement is below 85% (module docstring)."""
    cfg = check_config(cfg, upper_bound=True)
    if headline is None:
        raise ValueError("P1 picked no headline arm, so there is no primary test")
    if headline not in HEADLINE_ARMS:
        raise ValueError(f"headline {headline!r} is not one of the arms the P1 rule picks from, {HEADLINE_ARMS}")
    if (p1_records is None) != (p1_manifest is None):
        raise ValueError("the within-repo split needs both p1_records and p1_manifest")
    data = load(records, manifest, energy_model, "p2")
    gdata = load(records, manifest, energy_model, "p2", every_model=False, grammar=True)
    batched = energy_model.get("batched") or {}
    if type(batched.get("below_85pct")) is not bool:
        raise ValueError("energy_model.json batched.below_85pct must be true or false: docs/04 (P0a, batched "
                         "coefficients) needs the batched vs concurrency-1 pick agreement before P2 reports energy; "
                         "fit P0a with --windows")
    energy = {**data["energy"], "batched_below_85pct": batched["below_85pct"],
              "batched_share_same_pick": (batched.get("agreement_with_c1") or {}).get("share_same_pick")}
    cpu_w = cfg["host_cpu_tdp_w"] / cfg["host_vcpus"]
    groups = [(headline, PRIMARY), ("random", PRIMARY)]
    c = contrast(arrays(data, groups, cfg["infra_error_cap"], cpu_w), *groups, cfg, detail=True)
    both = None
    if batched["below_85pct"]:
        cb = contrast(batched_arrays(data, groups, batched_fit(energy_model), cfg["infra_error_cap"]), *groups, cfg)
        both = {"primary": {**{k: cb[k] for k in ("arm", "comparator", "profile", "n_tasks", "jpr", "resolve_rate", "T",
                                                  "ratio", "gpu", "guard")},
                            "wins": cb["gpu"]["p_one_sided"] < cfg["alpha"] and cb["guard"]["pass"]},
                "run_level": run_level_totals(records, manifest, data),
                "estimate": "batched marginal joules (estimate_batched_joules), intercept and tool windows left out"}
    sens = contrast(arrays(data, groups, cfg["infra_error_cap"], cpu_w, drop_timeouts=True), *groups, cfg)
    primary = {**c, "wins": c["gpu"]["p_one_sided"] < cfg["alpha"] and c["guard"]["pass"],
               "holds_under_upper_bound": c["upper_bound"]["p_one_sided"] < cfg["alpha"] and c["guard"]["pass"],
               "sensitivity_without_infra_timeout": {k: sens[k] for k in ("n_tasks", "n_tasks_dropped_infra_timeout",
                                                                          "T", "ratio", "gpu", "guard")}}
    have, tm = present(data), (headline + TOKMATCH, PRIMARY)
    token_matched = None
    if tm in have:  # docs/04 Training: the token-matched rerun of the primary contrast, when P1 required one
        c = contrast(arrays(data, [tm, groups[1]], cfg["infra_error_cap"], cpu_w), tm, groups[1], cfg)
        token_matched = {**c, "wins": c["gpu"]["p_one_sided"] < cfg["alpha"] and c["guard"]["pass"],
                         "holds_under_upper_bound": c["upper_bound"]["p_one_sided"] < cfg["alpha"]
                         and c["guard"]["pass"]}
    b = {arm: b_metrics(data, arm, cfg) for arm in sorted({a for a, _ in have})
         if (arm, PRIMARY) in have and (arm, REFERENCE) in have and ("random", REFERENCE) in have}
    est = measured = None
    if estimator_rows is not None:
        edata = load(estimator_rows, manifest, energy_model, "estimator", every_model=False)
        est = estimator_check(edata, headline, cfg)
        if not est["stands"]:
            measured = measured_subset(edata, headline, est["scope"], cfg, cpu_w)
    frontier = turn_cap_frontier(data, headline, energy_model, cfg) if ("base", PRIMARY) in have else None
    claim = {"primary_wins": primary["wins"], "holds_under_upper_bound": primary["holds_under_upper_bound"],
             "sign_holds": primary["sign_holds"]["holds"],
             "frontier_pass": None if frontier is None else frontier["pass"],
             "estimator_stands": None if est is None else est["stands"]}
    claim["claimed_on_estimates"] = all(v is True for v in claim.values())
    # The fallback claim meets the same upper-bound standard as claimed_on_estimates.
    claim["measured_subset_wins"] = (None if measured is None
                                     else measured["wins"] and measured["holds_under_upper_bound"])
    return {"headline": headline, "energy": energy, "n_rows": data["n_rows"], "n_rows_grammar": gdata["n_rows"],
            "n_reruns_used": data["n_reruns_used"],
            "n_resolved_after_budget_hit": data["n_resolved_after_budget_hit"],
            "primary": primary, "batched_regime": both, "token_matched": token_matched, "families": families(data, cfg, cpu_w, headline), "arms": arm_table(data, cfg, cpu_w),
            "b_metrics": b, "tokens_ratio_contrasts": tokens_ratio_contrasts(data, cfg, headline), "estimator_check": est,
            "measured_subset": measured, "turn_cap_frontier": frontier, "claim_A": claim,
            "h4": h4(data, gdata, headline, cfg, cpu_w) if gdata["n_rows"] else None,
            "within_repo": None if p1_records is None else within_repo(p1_records, p1_manifest, energy_model,
                                                                       headline, cfg, cpu_w),
            "settings": {k: cfg[k] for k in ("guard_margin", "infra_error_cap", "n_boot", "seed", "alpha",
                                             "host_cpu_tdp_w", "host_vcpus")}}


# P2's task count per held-out repo: #3's data/task_lists/heldout691.json by_repo.
HELDOUT691_BY_REPO = {"coveragepy": 108, "datalad": 179, "pyramid": 189, "scrapy": 215}
# Owner's agent on PR #17 (answer 4): 0.50 to 1.00 in steps of 0.025, so the MDE stays on the grid at low rates.
MDE_RATIOS = tuple(round(0.5 + 0.025 * i, 3) for i in range(21))
MDE_POWER = 0.8


def _pass_rates(tasks, n_by_repo, rate_scale=1.0):
    """Per held-out repo, the tasks' pass rates p = rho s / k and unbiased binomial variances
    rho s / k - rho^2 s (s - 1) / (k (k - 1)) for rho = rate_scale (s (k - s) / (k (k - 1)) at rho = 1). Above 1 a
    scaled rate is capped at 1 and its variance at 0."""
    if not (isinstance(rate_scale, (int, float)) and math.isfinite(rate_scale) and rate_scale > 0):
        raise ValueError(f"rate_scale must be a positive finite number, got {rate_scale!r}")
    by_repo = {}
    for t in tasks:
        s, k = t["successes"], t["attempts"]
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (s, k)) or k < 2 or not 0 <= s <= k:
            raise ValueError(f"task {t!r}: successes and attempts must be integers with 0 <= s <= k and k >= 2")
        if rate_scale == 1:
            by_repo.setdefault(t["repo"], []).append((s / k, s * (k - s) / (k * (k - 1))))
            continue
        v = rate_scale * s / k - rate_scale ** 2 * s * (s - 1) / (k * (k - 1))
        by_repo.setdefault(t["repo"], []).append((min(rate_scale * s / k, 1.0), max(v, 0.0)))
    missing = sorted(set(n_by_repo) - set(by_repo))
    if missing:
        raise ValueError(f"no measured task in held-out repos {missing}; each repo's tasks are resampled from its own")
    return {r: np.array(by_repo[r]) for r in sorted(n_by_repo)}


def guard_se(tasks, n_by_repo=HELDOUT691_BY_REPO, seeds=3, rate_scale=1.0):
    """Standard errors of P2's Q4_K_M resolve rates, from measured per-task pass counts (P0b heldout100).

    tasks: [{"repo", "successes", "attempts"}] with attempts >= 2. v_t = s_t (k_t - s_t) / (k_t (k_t - 1)) is
    unbiased for the task's p_t (1 - p_t); v is the mean of v_t within each held-out repo, weighted by the repo's
    share of the n = sum(n_by_repo) P2 tasks. With S = seeds seed models and 1 attempt per task per seed:
      se_rr = sqrt(v / (S n)), one arm's resolve rate with each task's p_t fixed (attempt noise only);
      se_rr_diff = sqrt(2 v / (S n)), RR_arm - RR_random when both arms have each task's p_t, the paired
        difference the guard bounds (the task-level spread of p_t cancels in the shared draws).
    The guard's one-sided 95% lower bound sits about 1.645 se_rr_diff below the observed difference, so a margin
    of m se_rr_diff passes an arm with no true difference with probability about Phi(m - 1.645). margin_se is
    se_rr_diff, the SE the margin is set from (owner's agent on PR #17, answer 1); the multiple m is the owner's.
    rate_scale scales every p_t (see _pass_rates), for the variant at P0a's pooled Q4_K_M/BF16 resolve ratio.
    """
    pools = _pass_rates(tasks, n_by_repo, rate_scale)
    n = sum(n_by_repo.values())
    v = sum(n_by_repo[r] * pools[r][:, 1].mean() for r in pools) / n
    return {"n_tasks": n, "seeds": seeds, "rate_scale": rate_scale, "mean_binomial_variance": float(v),
            "se_rr": math.sqrt(v / (seeds * n)), "se_rr_diff": math.sqrt(2 * v / (seeds * n)),
            "margin_se": math.sqrt(2 * v / (seeds * n)), "margin_se_from": "se_rr_diff",
            "formula": "se_rr = sqrt(v / (S n)), se_rr_diff = sqrt(2 v / (S n)), v = repo-weighted mean of "
                       "s (k - s) / (k (k - 1)) over the measured tasks"}


def _cost_table(cost_tasks, matched):
    """Per outcome (True resolved, False failed), flat costs with each pool task's start and count, for drawing
    one cost per attempt from a given pool task's own attempts. A pool task with attempts of one outcome only
    supplies its costs for both. matched=False pools every attempt into one task: the independent draw."""
    for j, ct in enumerate(cost_tasks):
        c = [np.asarray(ct[o], dtype=float) for o in ("resolved", "failed")]
        if any(x.ndim != 1 for x in c) or not sum(map(len, c)):
            raise ValueError(f"pool task {j} has no attempts; each pool task needs at least one cost")
        if any(not np.isfinite(x).all() or (x <= 0).any() for x in c):
            raise ValueError(f"pool task {j}: costs must be positive finite per-attempt values")
    allc = {o: np.concatenate([np.asarray(ct[o], dtype=float) for ct in cost_tasks]) for o in ("resolved", "failed")}
    for o, c in allc.items():
        if not len(c):
            raise ValueError(f"no {o} pool attempt: the costs need at least one resolved and one failed attempt")
    tasks = cost_tasks if matched else [allc]
    table = {}
    for outcome, own, other in ((True, "resolved", "failed"), (False, "failed", "resolved")):
        lists = [np.asarray(ct[own] if len(ct[own]) else ct[other], dtype=float) for ct in tasks]
        counts = np.array([len(x) for x in lists])
        table[outcome] = (np.concatenate(lists), np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    summary = {"draw": "matched" if matched else "independent", "n_pool_tasks": len(cost_tasks),
               "n_pool_tasks_one_outcome": sum(not len(ct["resolved"]) or not len(ct["failed"]) for ct in cost_tasks),
               "n_resolved": len(allc["resolved"]), "n_failed": len(allc["failed"]),
               "mean_resolved": float(allc["resolved"].mean()), "mean_failed": float(allc["failed"].mean())}
    return table, summary


def _draw_costs(table, u, R, rng):
    """One cost per attempt in R [S, n], each simulated task j drawing from pool task u[j]'s own attempts."""
    def draw(outcome):
        vals, start, counts = table[outcome]
        return vals[start[u] + (rng.random(R.shape) * counts[u]).astype(int)]
    return np.where(R > 0, draw(True), draw(False))


def mde(tasks, cost_tasks, ratios=MDE_RATIOS, n_by_repo=HELDOUT691_BY_REPO, seeds=3, n_sim=500, n_boot=1000,
        alpha=0.05, seed=0, guard_margin=None, matched=True, rate_scale=1.0):
    """Simulated power of the frozen primary test at each true JPR ratio, and the minimum detectable effect.

    The test is contrast()'s: _test on strat_weights draws (tasks resampled within repo, shared by both arms),
    each task averaged over the seed models, p < alpha on T = log(JPR_arm / JPR_random). Each simulated study
    draws n_by_repo[repo] tasks per held-out repo with replacement from that repo's measured tasks (tasks as in
    guard_se: P0b heldout100 BF16 at k = 16) and gives each arm seeds x 1 attempt per task, resolving with the
    task's s / k (times rate_scale, see _pass_rates), the same rate for both arms. cost_tasks is one
    {"resolved": [...], "failed": [...]} per pool task (per attempt estimated joules or processed tokens, by
    outcome). With matched (the primary, owner's agent
    on PR #17, answer 3), each simulated task is paired with one pool task drawn at random, and both arms' attempts
    on it draw their costs from that pool task's own attempts of the same outcome, so the test's pairing by task
    carries the task-level cost spread as the real design does. matched=False draws every cost from all pool
    attempts of the outcome, independently of the task: the conservative sensitivity. The headline arm's costs
    are times the ratio, so the true JPR ratio is the ratio and the true resolve-rate difference 0.

    Simplification, stated in the output: s / k is used as the true rate, which understates p (1 - p) by a
    factor (k - 1) / k (slightly optimistic, more so for k = 3 than 16).

    One set of bootstrap weights serves every study and every ratio reuses a study's draws, as in qtrace/p0d.py
    power (#12), so the result is deterministic in seed and the curve differs only by the ratio. power counts
    p < alpha; with guard_margin, power_with_guard also needs the guard (lower bound of RR_arm - RR_random >=
    -guard_margin), which is analyze()'s win. mde_ratio_80 is the largest ratio with power >= 0.8 (the smallest
    detectable reduction, 1 - ratio), None if none. With guard_margin, mde_ratio_80_with_guard is the same on
    power_with_guard, and guard_pass_rate is the share of studies whose guard passes. The guard's bound uses only
    the resolves, so it does not depend on the ratio, and guard_pass_rate caps power_with_guard at every ratio.
    """
    if seeds < 1 or n_sim < 1 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError(f"need seeds >= 1, n_sim >= 1, n_boot >= 100, 0 < alpha < 1; got {seeds}, {n_sim}, "
                         f"{n_boot}, {alpha}")
    if not ratios or any(not 0 < r for r in ratios):
        raise ValueError(f"ratios are true JPR ratios arm / random, each > 0; got {ratios}")
    pools = _pass_rates(tasks, n_by_repo, rate_scale)
    table, cost = _cost_table(cost_tasks, matched)
    n_pool = len(table[True][1])
    repos = np.repeat(list(pools), [n_by_repo[r] for r in pools])
    n = len(repos)
    rng = np.random.default_rng(seed)
    W = strat_weights(repos, n_boot, rng)
    sig, won, rr_diffs, guard_ok = np.zeros(len(ratios)), np.zeros(len(ratios)), [], 0
    for _ in range(n_sim):
        p = np.concatenate([rng.choice(pools[r][:, 0], n_by_repo[r]) for r in pools])
        R = [(rng.random((seeds, n)) < p).astype(float) for _ in range(2)]
        u = rng.integers(n_pool, size=n)  # the pool task each simulated task takes its costs from, for both arms
        E = [_draw_costs(table, u, r, rng) for r in R]
        rr_diffs.append(float((R[0].mean(0) - R[1].mean(0)).sum() / n))
        for i, ratio in enumerate(ratios):
            t = _test(W, {"R": R[0], "E": E[0] * ratio}, {"R": R[1], "E": E[1]})
            s = t["p_one_sided"] < alpha
            g = guard_margin is None or t["rr_diff_lower95_one_sided"] >= -guard_margin
            sig[i] += s
            won[i] += s and g
        guard_ok += g  # the same at every ratio: the bound reads R and W only
    power = [{"ratio": float(r), "power": float(a / n_sim),
              **({} if guard_margin is None else {"power_with_guard": float(b / n_sim)})}
             for r, a, b in zip(ratios, sig, won)]
    passing = [x["ratio"] for x in power if x["power"] >= MDE_POWER]
    guarded = {}
    if guard_margin is not None:
        won80 = [x["ratio"] for x in power if x["power_with_guard"] >= MDE_POWER]
        guarded = {"mde_ratio_80_with_guard": max(won80) if won80 else None, "guard_pass_rate": guard_ok / n_sim}
    resolves = sum(n_by_repo[r] * pools[r][:, 0].mean() for r in pools)
    se = guard_se(tasks, n_by_repo, seeds, rate_scale)
    se["simulated_sd_rr_diff"] = float(np.std(rr_diffs, ddof=1)) if n_sim > 1 else None
    k = [t["attempts"] for t in tasks]
    return {"n_tasks": n, "n_by_repo": dict(n_by_repo), "seeds": seeds, "n_measured_tasks": len(tasks),
            "attempts_per_task": {"min": min(k), "max": max(k)}, "rate_scale": rate_scale,
            "mean_pass1": float(resolves / n), "expected_resolves_per_arm_seed": float(resolves),
            "cost": cost,
            "alpha": alpha, "n_sim": n_sim, "n_boot": n_boot, "seed": seed, "guard_margin": guard_margin,
            "power": power, "mde_ratio_80": max(passing) if passing else None, **guarded, "guard_se": se,
            "simplifications": [("costs matched to tasks: each simulated task takes one pool task's own attempts"
                                 if matched else "costs drawn independently of the task (conservative)"),
                                "measured s / k used as each task's true rate (slightly optimistic)"]}
