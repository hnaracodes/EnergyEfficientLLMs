"""P1 selection for multi-turn runs: one training run per task for each arm (docs/03, Method step 3).

Rules from docs/04, "Selection and training rules for P1" > "Selection", and its test-file tripwire.
The GSM8K version in qtrace/select.py stays as the docs/02 archive.

Candidates are the pool runs' BF16 attempts that resolved without the test-file flag or an infrastructure
error; tasks with fewer than min_candidates of them are dropped. Every arm picks exactly one candidate per
task, so arms differ only in which run. Exact ties go to a draw from random.Random(f"{seed}:{rule}:{task_id}")
over the candidates in attempt_id order, so a pick depends only on that task's candidates and the seed,
never on input order or on which other tasks are in the pool. Joined values (P0d scores, P0a joule
estimates) are checked only for the arms that rank by them; a gap there is an error, never a silent skip.

summary.json also carries docs/04 P0b's arm-overlap rule (arm_overlap_rule), and select applies it: when
fewest_joules and fewest_turns pick the same run in at least 85% of tasks, or fewest_turns' median joule
regret is below 5%, in any seed, fewest_joules is not built for any seed.

The docs/04 kill rules run before any arm is built (kill_rules in summary.json): a P0a fail
(gates.json P0a.final_verdict) or a P0d DiD fail (did.json decision) drops the stability arms, and with
them the KL term; a DiD inconclusive builds them as exploratory, out of the headline. Every arm pair
flagged not separable is listed under "untestable" with the H1a-H5 contrasts that compare it; both arms
are still built, since which one to skip is an owner rule (PR #14).
"""

import itertools
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from fractions import Fraction

from qtrace.common import read_jsonl

ARMS = ("random", "fewest_turns", "fewest_joules", "stable", "qtrace", "low_entropy", "clean_format",
        "short_stable", "short_unstable", "long_stable", "long_unstable")
STABILITY_ARMS = frozenset({"stable", "qtrace", "short_stable", "short_unstable", "long_stable", "long_unstable"})
SCORES = ("kl_decision", "kl_command", "kl_mean")  # the three candidates frozen in docs/04 P0d
HELDOUT_LISTS = frozenset({"heldout100", "verified100", "inrepo_test"})
MALFORMED = ("misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer")
PROBE = "Q3_K_M"
COST_FIELD = {"turns": "turns", "joules": "est_joules_q4"}
NOT_SEPARABLE = 0.85  # docs/04: the two P0b rules say "at least 85%", the arm-pair table "above 85%"
REGRET_MERGE = 0.05  # strict: fewest_turns' median joule regret "below 5%" merges it with fewest_joules
# docs/04 P0b's arm-overlap rankings. turns and joules are exactly the fewest_turns and fewest_joules rules; the
# HERO ones are its reward cost (total tokens) and its API price (uncached + 0.1 x cached + 4 x output tokens).
RANKINGS = ("turns", "output_tokens", "joules", "hero_total_tokens", "hero_api_price")
RANK_KEY = {"output_tokens": lambda c: c["output_tokens"],
            "hero_total_tokens": lambda c: c["input_uncached"] + c["input_cached"] + c["output_tokens"],
            "hero_api_price": lambda c: c["input_uncached"] + Fraction(c["input_cached"]) / 10 + 4 * c["output_tokens"]}

P0A_VERDICTS = ("pass", "fail", "inconclusive")  # gates.json P0a.final_verdict, after the re-test if one ran
DID_DECISIONS = ("pass", "fail", "inconclusive", "incomplete")  # did.json decision (qtrace/p0d.py did_test)
# The secondary contrasts by hypothesis, as in configs/p2_analysis.yaml "families" (feat/p2-analysis); keep the two
# in step. H4 reruns the headline on grammar-constrained rows, so no arm pair maps to it.
HYPOTHESES = {"H1a": (("fewest_turns", "random"),), "H1b": (("fewest_joules", "random"),),
              "H2": (("qtrace", "fewest_turns"),),
              "H3": (("stable", "low_entropy"), ("stable", "clean_format"), ("short_stable", "short_unstable"),
                     ("long_stable", "long_unstable")),
              "H5": (("qtrace", "fewest_joules"),)}


def need(rec, *keys):
    """rec[k1][k2]..., or a ValueError naming the attempt and the field. A null value counts as missing."""
    x = rec
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
        if x is None:
            raise ValueError(f"attempt {rec.get('attempt_id')!r}: missing field {'.'.join(keys)}")
    return x


def number(x, what):
    """x if it is None or a finite int or float; anything else (NaN, a string, a bool) is an error."""
    if x is None or (not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)):
        return x
    raise ValueError(f"{what}: expected a finite number, got {x!r}")


def read_attempts(path):
    """Attempt records of one run file, outputs/attempts/<run_id>.jsonl, with absolute episode paths.

    A relative episode_path is resolved against the attempts file's directory (issue #1). Every record
    must carry the file's run id, so a renamed or concatenated file cannot slip another run into the pool.
    """
    run_id = os.path.splitext(os.path.basename(path))[0]
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    for r in read_jsonl(path):
        if r.get("run_id") != run_id:
            raise ValueError(f"{path}: attempt {r.get('attempt_id')!r} has run_id {r.get('run_id')!r}, not {run_id!r}")
        rows.append({**r, "episode_path": os.path.abspath(os.path.join(base, need(r, "episode_path")))})
    return rows


def by_attempt(rows, what):
    """attempt_id -> row. Two rows for one attempt are an error, since either could be the one joined."""
    out = {}
    for r in rows:
        aid = need(r, "attempt_id")
        if aid in out:
            raise ValueError(f"two {what} rows for attempt {aid!r}")
        out[aid] = r
    return out


def check_config(pool_lists, min_candidates, alpha, cost, score_override, seeds, arms):
    """Fail fast on a config that would build the wrong training sets."""
    if isinstance(pool_lists, str) or not pool_lists:
        raise ValueError(f"pool_lists must be a non-empty list of list names, got {pool_lists!r}")
    held = sorted(HELDOUT_LISTS & set(pool_lists))
    if held:
        raise ValueError(f"pool_lists include held-out lists {held}; their tasks never enter a pool "
                         "(docs/04, Task sets)")
    if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < 2:
        raise ValueError(f"min_candidates must be an integer >= 2 (the 2x2 split needs a short and a long half), "
                         f"got {min_candidates!r}")
    if number(alpha, "alpha") is None or not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    if cost not in COST_FIELD:
        raise ValueError(f"cost must be 'turns' or 'joules', got {cost!r}")
    if score_override is not None and score_override not in SCORES:
        raise ValueError(f"score_override must be null or one of {SCORES}, got {score_override!r}")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError(f"seeds must be distinct integers, got {seeds!r}")
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; the arms are {ARMS}")


def score_info(score_choice, override, arms):
    """Which P0d score the stability arms rank by, and where that came from.

    A null choice answers H3 negatively only if P0d decided it; an absent "decided" counts as decided, as
    in the issue #1 format. P0d writes decided false when a check could not run, and that is an error
    unless score_override is set or no stability arm is asked for.
    """
    chosen, decided = None, None
    wants_file = override is None and STABILITY_ARMS & set(arms)
    if score_choice is not None:
        if "chosen" not in score_choice:
            raise ValueError("score_choice.json has no 'chosen' key")
        chosen, decided = score_choice["chosen"], score_choice.get("decided", True)
        if chosen is not None and chosen not in SCORES:
            raise ValueError(f"score_choice.json chosen must be null or one of {SCORES}, got {chosen!r}")
        if not isinstance(decided, bool):
            raise ValueError(f"score_choice.json decided must be true or false, got {decided!r}")
        if chosen is None and not decided and wants_file:
            raise ValueError(f"score_choice.json is undecided ({score_choice.get('verdict')}), so H3 is not "
                             "answered; rerun P0d once that check can run, set select.score_override, or drop the "
                             "stability arms from select.arms")
    elif wants_file:
        raise ValueError("score_choice.json is missing (P0d writes it); set select.score_override or drop the "
                         "stability arms from select.arms")
    return {"file_present": score_choice is not None, "file_chosen": chosen, "file_decided": decided,
            "override": override, "used": override or chosen}


def p0a_verdict(gates):
    """gates.json P0a.final_verdict (the re-test's verdict when one ran); "unchecked" without a P0a entry."""
    p0a = (gates or {}).get("P0a")
    if p0a is None:
        return "unchecked"
    v = p0a.get("final_verdict", "absent")
    if v is not None and v not in P0A_VERDICTS:
        raise ValueError(f"gates.json P0a.final_verdict must be null or one of {P0A_VERDICTS}, got {v!r}; "
                         "rerun scripts/p0_analyze.py")
    return v


def p0a_rule(v, score, arms):
    """docs/04 P0a kill rule, from p0a_verdict's v.

    fail drops the robustness half: the stability arms and with them the KL term. pass and a final
    inconclusive (B continues, Q3_K_M is the stress profile) keep them. A null final_verdict means the
    required re-test has not run, an error while a stability arm would be built (asked for, with a score
    to rank by, as in did_rule). Without a P0a entry (or without gates.json) the rule is not checked, and
    the summary says so.
    """
    if v == "unchecked":
        return {"verdict": None, "note": "not checked: no P0a verdict (gates.json absent or without P0a)"}
    if v is None and score["used"] is not None and STABILITY_ARMS & set(arms):
        raise ValueError("gates.json P0a.final_verdict is null: the inconclusive first look needs its re-test "
                         "(P0a_topup) before P1, or drop the stability arms from select.arms")
    note = {None: "not applied: P0a's re-test is pending, but no stability arm is built",
            "pass": "passed: the stability arms are kept",
            "inconclusive": "inconclusive after the re-test: B continues (Q3_K_M is the stress profile), so the "
                            "stability arms are kept",
            "fail": "failed: the kill rule drops the stability arms and the KL term"}[v]
    return {"verdict": v, "note": note}


def did_rule(did, score, arms):
    """docs/04 P0d kill rule, from did.json's decision on the score that score_choice.json chose.

    fail drops KL from selection, so the stability arms are not built; inconclusive builds them as
    exploratory, out of the headline and the RL default; pass keeps them. incomplete (pairs without
    continuations) is an error while a stability arm would be built. Without did.json the rule is not
    checked, and the summary says so.
    """
    if did is None:
        return {"decision": None, "note": "not checked: no did.json"}
    decision = did.get("decision")
    if decision not in DID_DECISIONS:
        raise ValueError(f"did.json decision must be one of {DID_DECISIONS}, got {decision!r}")
    if did.get("score") != score["file_chosen"]:
        raise ValueError(f"did.json tests score {did.get('score')!r}, but score_choice.json chose "
                         f"{score['file_chosen']!r}; rerun the DiD test on this score choice")
    if decision == "incomplete" and score["used"] is not None and STABILITY_ARMS & set(arms):
        raise ValueError("did.json decision is incomplete (pair tasks without continuations); finish P0d's "
                         "continuations before P1, or drop the stability arms from select.arms")
    note = {"incomplete": "not applied: the DiD test is incomplete, but no stability arm is built",
            "pass": "passed: the stability arms are kept",
            "inconclusive": "inconclusive: the stability arms are built as exploratory, out of the headline and "
                            "the RL default",
            "fail": "failed: the kill rule drops KL from selection, so the stability arms are not built"}[decision]
    return {"decision": decision, "note": note}


def energy_info(energy_model, est_joules, cost):
    """Whether fewest_joules ranks estimated joules or falls back to fewest_turns (docs/04, P0a)."""
    if est_joules is None:
        reason = "no est_joules file"
    elif energy_model is None:
        raise ValueError("est_joules is present but energy_model.json is missing; its fewest_joules_fallback "
                         "flag decides how fewest_joules picks")
    else:
        flag = energy_model.get("fewest_joules_fallback")
        if not isinstance(flag, bool):
            raise ValueError(f"energy_model.json fewest_joules_fallback must be true or false, got {flag!r}")
        reason = "energy model failed its pre-registered accuracy check (fewest_joules_fallback)" if flag else None
    if reason and cost == "joules":
        raise ValueError(f"cost: joules ranks by the energy model, but {reason}")
    return {"fallback": reason is not None, "picks_as": "fewest_turns" if reason else "fewest_joules",
            "reason": reason}


def candidate(r, score_row, est_joules, score_name):
    """One candidate run with its joined values; a value that was not joined is None."""
    aid, task = r["attempt_id"], r["task_id"]
    path = need(r, "episode_path")
    if not os.path.isabs(path):
        raise ValueError(f"attempt {aid!r}: episode_path {path!r} is not absolute (load attempts with read_attempts)")
    if score_row is not None and score_row.get("task_id") != task:
        raise ValueError(f"attempt {aid!r}: score row has task_id {score_row.get('task_id')!r}, not {task!r}")
    row = score_row or {}
    return {"task_id": task, "attempt_id": aid, "episode_path": path,
            "turns": number(need(r, "turns"), f"attempt {aid!r} turns"),
            "output_tokens": number(need(r, "tokens", "output"), f"attempt {aid!r} tokens.output"),
            "input_uncached": number(need(r, "tokens", "input_uncached"), f"attempt {aid!r} tokens.input_uncached"),
            "input_cached": number(need(r, "tokens", "input_cached"), f"attempt {aid!r} tokens.input_cached"),
            "est_joules_q4": number(est_joules, f"attempt {aid!r} est_joules_q4"),
            "score": number(row.get(score_name), f"attempt {aid!r} {score_name}") if score_name else None,
            "entropy_mean": number(row.get("entropy_mean"), f"attempt {aid!r} entropy_mean"),
            "malformed_total": sum(number(need(r, "malformed", k), f"attempt {aid!r} malformed.{k}")
                                   for k in MALFORMED)}


def candidates(attempts, pool_lists, min_candidates, scores, joules, score_name):
    """Group the pool's candidate runs by task, with their scores and joule estimates joined.

    A candidate is a BF16 attempt in a pool list that did not end in an infrastructure error, resolved
    (which must mean end_reason submitted; anything else raises), and has no test-file flag
    (grading.test_files_touched is a required bool; absent is an error). Returns (task -> candidates,
    report); the report counts infrastructure errors and flagged successes per pool list and gives each
    dropped task's candidate count (0 if none resolved).
    """
    by_task, seen, tasks, flagged, errors = defaultdict(list), set(), set(), Counter(), Counter()
    for r in attempts:
        if need(r, "profile") != "BF16" or need(r, "list") not in pool_lists:
            continue
        aid, task = need(r, "attempt_id"), need(r, "task_id")
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r}")
        seen.add(aid)
        tasks.add(task)
        # docs/04 reruns infrastructure faults as fresh attempts; P0b's counts and P0a's estimates drop them too.
        if need(r, "end_reason") == "error":
            errors[r["list"]] += 1
            continue
        resolved = r.get("resolved")
        if resolved is not True and resolved is not False:
            raise ValueError(f"attempt {aid!r}: resolved must be true or false, got {resolved!r}")
        if not resolved:
            continue
        if r["end_reason"] != "submitted":  # the harness sets resolved only on a submitted attempt that passes grading
            raise ValueError(f"attempt {aid!r}: resolved is true but end_reason is {r['end_reason']!r}; only a "
                             "submitted attempt can resolve (docs/04 charges budget hits and timeouts as failures)")
        touched = need(r, "grading", "test_files_touched")  # required in every record; absent never means unflagged
        if touched is not True and touched is not False:
            raise ValueError(f"attempt {aid!r}: grading.test_files_touched must be true or false, got {touched!r}")
        if touched:
            flagged[r["list"]] += 1
            continue
        by_task[task].append(candidate(r, scores.get(aid), joules.get(aid), score_name))
    kept = {t: c for t, c in by_task.items() if len(c) >= min_candidates}
    return kept, {"infra_error_excluded": {lst: errors[lst] for lst in pool_lists},
                  "flagged_excluded": {lst: flagged[lst] for lst in pool_lists},
                  "dropped_tasks": {t: len(by_task.get(t, ())) for t in sorted(tasks) if t not in kept}}


def avg_ranks(values):
    """Average ranks normalized to [0, 1] (0 = smallest), as exact fractions so tied mixes stay tied."""
    first, last = {}, {}
    for i, v in enumerate(sorted(values)):
        first.setdefault(v, i)
        last[v] = i
    scale = 2 * max(1, len(values) - 1)
    return [Fraction(first[v] + last[v], scale) for v in values]


def lowest(cands, key, rng):
    """The candidate with the smallest key; exact ties go to rng."""
    best = min(map(key, cands))
    return rng.choice([c for c in cands if key(c) == best])


def halves(cands, cost_field, seed, task_id):
    """(short, long): the floor(n/2) cheapest runs by cost_field, then the rest (docs/04, 2x2 controls).

    Cost ties are broken by output tokens, then by a shuffle seeded with (seed, task_id) only, so the four
    2x2 arms share one split of each task and a run is never short for one arm and long for another.
    """
    cands = sorted(cands, key=lambda c: c["attempt_id"])
    shuffled = cands[:]
    random.Random(f"{seed}:2x2_split:{task_id}").shuffle(shuffled)
    pos = {c["attempt_id"]: i for i, c in enumerate(shuffled)}
    ranked = sorted(cands, key=lambda c: (c[cost_field], c["output_tokens"], pos[c["attempt_id"]]))
    n = len(ranked) // 2
    return ranked[:n], ranked[n:]


def pick(cands, rule, seed, task_id, alpha=0.5, cost="turns"):
    """One rule's choice among one task's candidates (rules in docs/04, Selection).

    rule is an arm name. Remaining exact ties go to random.Random(f"{seed}:{rule}:{task_id}").
    """
    if rule not in ARMS:
        raise ValueError(f"unknown arm {rule!r}")
    cands = sorted(cands, key=lambda c: c["attempt_id"])
    rng = random.Random(f"{seed}:{rule}:{task_id}")
    if rule == "random":
        return rng.choice(cands)
    if rule == "fewest_turns":
        return lowest(cands, lambda c: (c["turns"], c["output_tokens"]), rng)
    if rule == "fewest_joules":
        return lowest(cands, lambda c: (c["est_joules_q4"], c["turns"]), rng)
    if rule == "stable":
        return lowest(cands, lambda c: c["score"], rng)
    if rule == "low_entropy":
        return lowest(cands, lambda c: c["entropy_mean"], rng)
    if rule == "clean_format":
        return lowest(cands, lambda c: c["malformed_total"], rng)
    if rule == "qtrace":
        # Ranks put the cost (turns or joules) and the KL score on one 0..1 scale.
        a = Fraction(str(alpha))
        cost_rank = avg_ranks([c[COST_FIELD[cost]] for c in cands])
        stab_rank = avg_ranks([c["score"] for c in cands])
        mix = {c["attempt_id"]: a * x + (1 - a) * y for c, x, y in zip(cands, cost_rank, stab_rank)}
        return lowest(cands, lambda c: mix[c["attempt_id"]], rng)
    short, long_ = halves(cands, COST_FIELD[cost], seed, task_id)
    sign = -1 if rule.endswith("unstable") else 1
    return lowest(short if rule.startswith("short") else long_, lambda c: sign * c["score"], rng)


def needs(arm, cost, joules_fallback):
    """The joined fields an arm ranks by, so a missing join fails only the arms that use it."""
    fields = set()
    if arm in STABILITY_ARMS:
        fields.add("score")
        if arm != "stable" and cost == "joules":
            fields.add("est_joules_q4")
    if arm == "fewest_joules" and not joules_fallback:
        fields.add("est_joules_q4")
    if arm == "low_entropy":
        fields.add("entropy_mean")
    return fields


def check_joins(by_task, arms, cost, joules_fallback):
    """Raise one error listing every arm that ranks by a value some candidate lacks."""
    problems = []
    for arm in arms:
        for field in sorted(needs(arm, cost, joules_fallback)):
            missing = sorted(c["attempt_id"] for cs in by_task.values() for c in cs if c[field] is None)
            if missing:
                problems.append(f"{arm} needs {field}, missing for {len(missing)} candidate(s): "
                                f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    if problems:
        raise ValueError("missing joins: " + "; ".join(problems) + ". Fix the inputs or drop those arms from "
                         "select.arms")


def arm_stats(rows):
    """Means over one arm's picks. A mean is None if any pick lacks the value (pool.joins_missing counts gaps)."""
    def mean(key):
        vals = [r[key] for r in rows]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)

    return {"n_tasks": len(rows), "mean_turns": mean("turns"), "mean_output_tokens": mean("output_tokens"),
            "mean_est_joules_q4": mean("est_joules_q4"), "mean_score": mean("score")}


def overlap(picks, threshold=NOT_SEPARABLE):
    """Share of tasks where two arms pick the same run, for every pair of arms built for one seed.

    picks: arm -> rows. A pair above threshold is "not separable" (docs/04, P0b): training both arms would
    test nothing, so their hypothesis is reported as untestable.
    """
    pairs = []
    for a, b in itertools.combinations(picks, 2):
        chosen = {r["task_id"]: r["attempt_id"] for r in picks[a]}
        same = sum(chosen[r["task_id"]] == r["attempt_id"] for r in picks[b]) / len(picks[b])
        pairs.append({"arms": [a, b], "share_same": same, "not_separable": same > threshold})
    return pairs


def untestable(by_seed):
    """Every arm pair flagged not separable in some seed, with the HYPOTHESES contrasts that compare exactly it.

    by_seed: seed -> overlap() pairs. Both arms stay built; which one to skip is an owner rule (PR #14).
    """
    hits = defaultdict(list)
    for s, pairs in by_seed.items():
        for p in pairs:
            if p["not_separable"]:
                hits[tuple(p["arms"])].append((s, p["share_same"]))
    return [{"arms": list(pair), "seeds": [s for s, _ in h], "max_share_same": max(x for _, x in h),
             "contrasts": [{"family": f, "arms": list(c)} for f, cs in HYPOTHESES.items() for c in cs
                           if set(c) == set(pair)]}
            for pair, h in hits.items()]


def rank_pick(cands, name, seed, task_id):
    """One RANKINGS entry's pick among one task's candidates; ties go to random.Random(f"{seed}:{name}:{task_id}")."""
    if name in ("turns", "joules"):
        return pick(cands, f"fewest_{name}", seed, task_id)
    rng = random.Random(f"{seed}:{name}:{task_id}")
    return lowest(sorted(cands, key=lambda c: c["attempt_id"]), RANK_KEY[name], rng)


def joule_regret(c, cands):
    """est_joules_q4 of the pick over the task's minimum, minus 1."""
    low = min(x["est_joules_q4"] for x in cands)
    if low <= 0:
        raise ValueError(f"task {c['task_id']!r}: minimum est_joules_q4 is {low}, so joule regret is undefined")
    return c["est_joules_q4"] / low - 1


def arm_overlap_rule(by_task, seed):
    """docs/04 P0b's arm-overlap table and fewest_joules/fewest_turns decision for one seed.

    Every candidate needs est_joules_q4. merge_fewest_joules is true when the turns and joules rankings pick
    the same run in at least NOT_SEPARABLE of tasks or the turns pick's median joule regret is below
    REGRET_MERGE. calibrated_token_mix is true when joules and HERO's API price agree in at least
    NOT_SEPARABLE of tasks: the selection cost is then described as a calibrated token-mix cost.
    """
    rows = {n: [rank_pick(by_task[t], n, seed, t) for t in sorted(by_task)] for n in RANKINGS}
    regret = {n: statistics.median([joule_regret(c, by_task[c["task_id"]]) for c in rs]) for n, rs in rows.items()}
    pairs = overlap(rows)
    same = {tuple(p["arms"]): p["share_same"] for p in pairs}
    return {"median_joule_regret": regret, "pairs": pairs,
            "merge_fewest_joules": same[("turns", "joules")] >= NOT_SEPARABLE or regret["turns"] < REGRET_MERGE,
            "calibrated_token_mix": same[("joules", "hero_api_price")] >= NOT_SEPARABLE}


def overlap_rule(by_task, seeds, arms, energy):
    """arm_overlap_rule per seed, and whether select drops fewest_joules because of it.

    Not computed when a candidate lacks est_joules_q4 (check_joins already refuses that if fewest_joules
    ranks joules). Not applied under the energy fallback, where fewest_joules already picks as fewest_turns.
    """
    missing = sum(c["est_joules_q4"] is None for cs in by_task.values() for c in cs)
    if missing:
        return {"by_seed": None, "merged": False,
                "note": f"not computed: {missing} candidate(s) lack est_joules_q4"}
    by_seed = {str(s): arm_overlap_rule(by_task, s) for s in seeds}
    hit = [s for s, r in by_seed.items() if r["merge_fewest_joules"]]
    if energy["fallback"]:
        note = f"not applied: fewest_joules already picks as fewest_turns ({energy['reason']})"
    elif "fewest_joules" not in arms or "fewest_turns" not in arms:
        note = "not applied: select.arms does not ask for both fewest_joules and fewest_turns"
    elif hit:
        note = (f"merged: in seed(s) {', '.join(hit)} fewest_joules and fewest_turns pick the same run in at least "
                f"{NOT_SEPARABLE:.0%} of tasks or fewest_turns' median joule regret is below {REGRET_MERGE:.0%}, so "
                "their hypothesis is untestable; fewest_joules is not built for any seed and fewest_turns stands "
                "for both")
    else:
        note = "separate: fewest_joules and fewest_turns are both built"
    return {"by_seed": by_seed, "merged": note.startswith("merged"), "note": note}


def check_cost(decision, energy, arms, cost):
    """Refuse a select.cost that contradicts P0b's verdict (docs/03 Method step 3, docs/04 Selection).

    qtrace and the 2x2 split rank by joules when the overlap rule keeps fewest_joules and fewest_turns
    separate in every seed, and by turns when it merges them in any seed. Not checked when the rule was not
    computed, under the energy fallback (energy_info already refuses cost joules there), or when no built arm
    ranks by cost.
    """
    if decision["by_seed"] is None or energy["fallback"] or not any(a in STABILITY_ARMS - {"stable"} for a in arms):
        return
    merges = any(r["merge_fewest_joules"] for r in decision["by_seed"].values())
    want = "turns" if merges else "joules"
    if cost != want:
        verdict = "merges fewest_joules into fewest_turns" if merges else "keeps fewest_joules and fewest_turns separate"
        raise ValueError(f"select.cost is {cost!r}, but P0b's overlap rule {verdict}, so qtrace and the 2x2 split "
                         f"rank by {want} (docs/03 Method step 3); set select.cost: {want}")


def h3_note(score):
    """What the summary says about H3 when score_choice.json chose null; None otherwise."""
    if not score["file_present"] or score["file_chosen"] is not None:
        return None
    if not score["file_decided"]:
        note = "not answered: score_choice.json chose null but is undecided (a P0d check did not run)"
        return note + (f"; score_override {score['override']} builds the stability arms" if score["override"] else "")
    if score["override"] is None:
        return "answered negatively: score_choice.json chose null, so stable, qtrace and the 2x2 arms are not built"
    return (f"score_choice.json chose null (H3 answered negatively), but score_override {score['override']} "
            "builds the stability arms anyway")


def select(attempts, scores=None, score_choice=None, energy_model=None, est_joules=None, *, gates=None, did=None,
           pool_lists=("pool200",), min_candidates=4, alpha=0.5, cost="turns", score_override=None, seeds=(0, 1, 2),
           arms=ARMS):
    """Build every configured arm for every seed. Returns (picks, summary).

    attempts are records with absolute episode paths (read_attempts). scores are the P0d rows (only probe
    Q3_K_M rows are used), score_choice and energy_model the parsed JSON files, est_joules the P0a rows;
    None stands for a file that does not exist. picks maps arm -> seed -> one row per task, sorted by
    task_id, in the agreed arm-file format; arms are built in ARMS order.

    Pre-registered fallbacks, both reported in the summary: a decided score_choice "chosen" null means H3
    was answered negatively and the stability arms are not built (score_override still builds them, and
    is recorded; an undecided null is an error, see score_info); with no joule estimates, or with
    fewest_joules_fallback set, fewest_joules picks exactly as fewest_turns (same rule and same RNG stream).
    cost "joules" is an error in both energy cases. Otherwise docs/04 P0b's overlap rule can drop
    fewest_joules for every seed (overlap_rule); summary["arm_overlap_rule"] says why. select.cost must
    match that verdict (check_cost).

    gates and did are the parsed gates.json (P0) and did.json (P0d), None when absent. Their kill rules
    run first (p0a_rule, did_rule); after a P0a fail P0d's inputs are not needed and did is not read.
    """
    check_config(pool_lists, min_candidates, alpha, cost, score_override, seeds, arms)
    v = p0a_verdict(gates)
    score = score_info(score_choice, score_override, [a for a in arms if not (v == "fail" and a in STABILITY_ARMS)])
    p0a = p0a_rule(v, score, arms)
    killed = p0a["verdict"] == "fail"
    p0d = ({"decision": None, "note": "not read: P0a failed, so the stability arms are already dropped"} if killed
           else did_rule(did, score, arms))
    killed = killed or p0d["decision"] == "fail"
    energy = energy_info(energy_model, est_joules, cost)
    q3 = by_attempt([r for r in scores or () if need(r, "probe") == PROBE], PROBE)
    joules = {aid: r.get("est_joules_q4") for aid, r in by_attempt(est_joules or (), "est_joules_q4").items()}
    by_task, pool = candidates(attempts, pool_lists, min_candidates, q3, joules, score["used"])
    if not by_task:
        raise ValueError(f"no task has at least {min_candidates} candidates; dropped tasks: {pool['dropped_tasks']}")
    built = [a for a in ARMS if a in arms and not (a in STABILITY_ARMS and (killed or score["used"] is None))]
    check_joins(by_task, built, cost, energy["fallback"])
    decision = overlap_rule(by_task, seeds, built, energy)
    check_cost(decision, energy, built, cost)
    built = [a for a in built if not (decision["merged"] and a == "fewest_joules")]
    if decision["merged"]:
        energy = {**energy, "picks_as": "fewest_turns",
                  "reason": "not built: P0b's overlap rule merged it into fewest_turns (see arm_overlap_rule)"}

    picks = {}
    for arm in built:
        rule = "fewest_turns" if arm == "fewest_joules" and energy["fallback"] else arm
        picks[arm] = {s: [{"arm": arm, "seed": s, **pick(by_task[t], rule, s, t, alpha, cost),
                           "score_name": score["used"]} for t in sorted(by_task)] for s in seeds}

    sizes = [len(c) for c in by_task.values()]
    by_seed = {str(s): overlap({a: picks[a][s] for a in built}) for s in seeds}
    summary = {
        "config": {"pool_lists": list(pool_lists), "min_candidates": min_candidates, "alpha": alpha, "cost": cost,
                   "seeds": list(seeds), "arms": list(arms), "probe": PROBE},
        "score": score,
        "h3": h3_note(score),
        "kill_rules": {"p0a": p0a, "p0d_did": p0d,
                       "dropped": [a for a in ARMS if a in arms and a in STABILITY_ARMS] if killed else [],
                       "exploratory": [a for a in built if a in STABILITY_ARMS] if p0d["decision"] == "inconclusive"
                       else []},
        "fewest_joules": energy,
        "pool": {"n_tasks": len(by_task), "n_candidates": sum(sizes), **pool,
                 "joins_missing": {f: sum(c[f] is None for cs in by_task.values() for c in cs)
                                   for f in ("score", "entropy_mean", "est_joules_q4")}},
        "arms_built": built,
        "arms": {a: {str(s): arm_stats(rows) for s, rows in picks[a].items()} for a in built},
        "overlap": {"chance_rate": sum(1 / n for n in sizes) / len(sizes), "threshold": NOT_SEPARABLE,
                    "by_seed": by_seed},
        "untestable": untestable(by_seed),
        "arm_overlap_rule": {"rankings": list(RANKINGS), "threshold": NOT_SEPARABLE, "regret_threshold": REGRET_MERGE,
                             **decision},
    }
    return picks, summary
