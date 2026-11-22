"""P2 energy in the batched regime, for when P0a's batched and concurrency-1 Q4_K_M picks agree on fewer than 85% of
tasks (docs/04, P0a batched coefficients: "P2 reports energy in both"). qtrace/p2_analysis.py analyze() calls these
and runs the batched arrays through the same contrast() as the primary.

Two numbers, neither of them the claim regime (docs/03 Metrics, claim regime: concurrency 1, owner to confirm):
- the estimated batched JPR: each attempt's marginal joules under energy_model.json's batched fit
  (energy_model.estimate_batched_joules: prefill x uncached input + decode x output; the per-second intercept is
  static power shared by every concurrent attempt and is left out, and so are tool windows);
- run-level measured totals: concurrent attempts share the GPU, so their measured joules exist only per block of
  runs (docs/04: "Concurrent runs report measured joules only as run-level totals").
"""

from collections import defaultdict
from itertools import pairwise

import numpy as np

from qtrace.energy_model import estimate_batched_joules
from qtrace.p2_data import PRIMARY, arm_rep, arrays, need, number

COUNTER_FIELDS = ("counter_start", "counter_end", "t_start", "t_end")


def batched_fit(energy_model):
    """energy_model.json's batched fit, checked to have the coefficients estimate_batched_joules reads."""
    coef = (energy_model.get("batched") or {}).get("coef") or {}
    for k in ("prefill", "decode"):
        number(coef.get(k), f"energy_model.json batched.coef.{k}")
    return energy_model["batched"]


def batched_arrays(data, groups, batched, cap):
    """arrays() for groups at Q4_K_M with E (and G, which contrast() reads alongside) the batched marginal estimate;
    no upper bound and no idle-subtracted estimate."""
    if any(p != PRIMARY for _, p in groups):
        raise ValueError(
            f"the batched fit is P0a's {PRIMARY} block; got groups {groups}"
        )
    arr = arrays(data, groups, cap)

    out = {}
    for (arm, prof), X in arr["groups"].items():
        E = np.array(
            [
                [
                    estimate_batched_joules(
                        data["cells"][arm, r, prof][t]["rec"], batched
                    )
                    for t in arr["tasks"]
                ]
                for r in X["reps"]
            ]
        )
        out[arm, prof] = {**X, "E": E, "G": E, "I": None, "U": None}
    return {**arr, "groups": out}


def run_level_totals(records, manifest, data):
    """Measured GPU joules per (arm, profile) as run-level totals, with the resolves of load()'s analyzed attempts.

    A block is one seed model at one profile within one run_id (one llama-server serving that model). Its joules
    are the GPU counter's span over the block, max energy.counter_end - min energy.counter_start, so attempts that
    ended in an infrastructure error count too: they drew power. Blocks must not overlap in time (energy.t_start to
    t_end), since energy cannot be split between concurrent attempts of different blocks. jpr is joules over the
    resolves of the arm's analyzed attempts (the cells, reruns used), so it charges every attempt as the primary
    does. Not paired by task and not tested: no per-task value exists. records: the rows load() accepted. Grammar
    true rows form blocks of their own, which join the overlap check (an H4 attempt running beside a main block
    would put its joules in that block's span) but are left out of the totals, like everywhere but H4.
    """
    blocks = defaultdict(list)
    for rec in records:
        aid = rec["attempt_id"]
        c0, c1, t0, t1 = (
            number(need(rec, "energy", k), f"attempt {aid!r} energy.{k}")
            for k in COUNTER_FIELDS
        )
        if c1 < c0 or t1 < t0:
            raise ValueError(
                f"attempt {aid!r}: energy.counter_end below its counter_start or t_end below its t_start"
            )
        blocks[
            (
                *arm_rep(manifest[rec["model"]], rec),
                rec["profile"],
                need(rec, "run_id"),
                rec["grammar"],
            )
        ].append((c0, c1, t0, t1))
    spans = sorted(
        (min(v[2] for v in vs), max(v[3] for v in vs), key)
        for key, vs in blocks.items()
    )
    for (_, end, a), (start, _, b) in pairwise(spans):
        if start < end:
            raise ValueError(
                f"run blocks {a} and {b} overlap in time, so their measured energy cannot be split"
            )
    tot = defaultdict(lambda: {"n_blocks": 0, "n_attempts": 0, "joules": 0.0})
    for (arm, _, prof, _, grammar), vs in blocks.items():
        if grammar:
            continue
        t = tot[arm, prof]
        t["n_blocks"] += 1
        t["n_attempts"] += len(vs)
        t["joules"] += max(v[1] for v in vs) - min(v[0] for v in vs)
    rows = []
    for (arm, prof), t in sorted(tot.items()):
        r = sum(
            s["r"]
            for (a, _, p), cell in data["cells"].items()
            if (a, p) == (arm, prof)
            for s in cell.values()
            if s is not None
        )
        rows.append(
            {
                "arm": arm,
                "profile": prof,
                **t,
                "n_resolved": int(r),
                "jpr": t["joules"] / r if r else "inf",
            }
        )
    return rows
