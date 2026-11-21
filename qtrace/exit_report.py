"""Pilot exit report (docs/04, P0b pilot): the numbers the owner reads to set the P0 spending ceiling.

attempt_stats and the markdown come from the attempt records agreed in issue #1. Numbers the attempt
records can't give (aggregate token rates from synthetic load, peak GPU memory, energy-counter checks,
idle watts, the replay match rate) come from an optional extras JSON; anything missing is listed under
"Not yet provided" rather than silently left out. Aggregate rates are also derived from the pilot records
(_pilot_rates) and printed as a cross-check; projections use synthetic-load rates only.
"""

import statistics
from collections import Counter, defaultdict

# Allowed record values. Runtimes from docs/04 (Serving); profile names from docs/04 (Profiles), with
# Q4_K_M_gate8 naming the Q4_K_M file whose ssm_alpha/ssm_beta are overridden to Q8_0. Issue #1 does not
# pin these strings yet, so an unknown value raises here instead of moving a step under "Not included".
RUNTIMES = ("vllm", "llama.cpp")
PROFILES = ("BF16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8")
MALFORMED = ["misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer"]
# Concurrent P0 blocks (docs/04): runtime, attempts per profile at the point and the upper count.
# P0a has a 600-attempt top-up (BF16 and Q4_K_M) if inconclusive; P0d can grow to about 1,600.
P0_STEPS = {
    "P0b": ("vllm", {"BF16": (4800, 4800)}),
    "P0a": ("llama.cpp", {"BF16": (600, 900), "Q8_0": (600, 600), "Q4_K_M": (600, 900), "Q3_K_M": (600, 600),
                          "Q4_K_M_gate8": (600, 600)}),
    "P0c": ("llama.cpp", {"BF16": (800, 800), "Q4_K_M": (800, 800), "Q3_K_M": (800, 800)}),
    "P0d": ("llama.cpp", {"BF16": (400, 800), "Q3_K_M": (400, 800)}),
}
# Profiles without their own measured rate borrow one: conservative for Q3_K_M (no faster than Q4_K_M).
RATE_FALLBACK = {"Q8_0": "BF16", "Q3_K_M": "Q4_K_M", "Q4_K_M_gate8": "Q4_K_M"}
SUB_RUN = {"attempts": 150, "settle_s": 45}  # energy sub-run: 5 profiles x 15 tasks x 2, concurrency 1
NOT_INCLUDED = ["KL scoring (projected separately below, on its own machine)",
                "per-block overhead: server restarts, GGUF loads and image pulls between serial blocks"]
HERO_PREFIX_HIT = 0.92
WALL_HIT_FLAG = 0.02  # 04: the owner revisits the wall-clock cap if any runtime/profile/concurrency cell exceeds it
WALL_CAP_FACTOR = 3  # 04: the infra cap is about 3x the pilot's 99th-percentile attempt time
# Owner's agent on PR #7 (answer 2): one cap per runtime, from its BF16 pilot attempts, for every profile on it,
# so a within-runtime profile comparison never runs under two caps.
CAP_PROFILE = "BF16"
# 04 (Model and decoding): pinned per request in both runtimes and logged; server defaults are never relied on.
# record.sampler holds exactly these keys, 04's names (owner's agent on PR #7, answer 6).
SAMPLER = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
           "repetition_penalty": 1.0}
# 04 asks for projections at 64K x 16 and 128K x 8: scenario -> (extras key of its measured rates, slots).
# 128K x 8 has no fallback: its rate must be measured, since a scaling from 64K would be a guess.
SCENARIOS = {"64K x 16": ("rates", 16), "128K x 8": ("rates_128k_x8", 8)}
# Owner's agent on PR #7 (answer 3): 64K x 16 shares tool time over extras concurrency.containers, the
# containers alive at once (16 in normal runs), assumed 16 until the harness writes it.
CONTAINERS_SCENARIO = "64K x 16"
# PR #11: one BF16 vs Q3_K_M scoring of a 16.5K-token run took about 43 s, both model loads included.
KL_SCORING = {"machine": "M5 Pro, llama.cpp b1b770d9 Metal (PR #11)", "seconds_per_run": 43.0,
              "tokens_per_run": 16500}
# Owner's agent on PR #7 (answer 5), pending the owner: the P0b scoring and P0a diagnostic run on EC2 if the
# projected KL-scoring hours are under about 10 h (roughly $75), otherwise on the 48 GB Mac overnight.
KL_EC2_MAX_H = 10.0
# Keys of the extras JSON, agreed with the harness in issue #1 / PR #7; rates_128k_x8 is proposed (audit gap 5).
EXTRAS = ["rates", "rates_128k_x8", "peak_gpu_gib", "energy_counter", "replay_match_rate", "sampler_echo_ok",
          "per_turn_energy_ok", "concurrency"]
RATE_KEYS = [key for key, _ in SCENARIOS.values()]


def attempt_stats(rows):
    by_runtime = defaultdict(list)
    for r in rows:
        if r["runtime"] not in RUNTIMES or r["profile"] not in PROFILES:
            raise ValueError(f"attempt {r.get('attempt_id')}: runtime {r['runtime']!r} / profile {r['profile']!r} "
                             f"not in {RUNTIMES} / {PROFILES}")
        by_runtime[r["runtime"]].append(r)
    return {rt: _stats(rs) for rt, rs in sorted(by_runtime.items())}


def _flagged(rec):
    """grading.test_files_touched, a required bool (as p0d.flagged reads it): missing raises, a null or 0/1 too."""
    v = rec["grading"]["test_files_touched"]
    if not isinstance(v, bool):
        raise TypeError(f"attempt {rec.get('attempt_id')!r}: grading.test_files_touched must be a bool, got {v!r}")
    return v


def _resolve(rows):
    """Resolve rate overall and excluding test-file-flagged attempts (04, tripwire: every resolve rate both ways)."""
    def rate(rs):
        return sum(bool(r["resolved"]) for r in rs) / len(rs) if rs else None
    return {"overall": rate(rows), "excluding_flagged": rate([r for r in rows if not _flagged(r)])}


def _p99(xs):
    """Linearly interpolated 99th percentile (numpy.percentile's default). With about 140 pilot attempts it
    sits close to the maximum, so one slow attempt moves it."""
    if len(xs) < 2:
        return xs[0] if xs else None
    return statistics.quantiles(xs, n=100, method="inclusive")[98]


def _wall(scored):
    """The runtime's infra cap: WALL_CAP_FACTOR x p99 over its CAP_PROFILE attempts only."""
    base = [r for r in scored if r["profile"] == CAP_PROFILE]
    p99 = _p99([r["wall_seconds"] for r in base])
    # An infra_timeout attempt stopped at the old cap, so its true time is longer and p99 is a lower bound.
    return {"profile": CAP_PROFILE, "p99": p99, "proposed_cap_s": None if p99 is None else WALL_CAP_FACTOR * p99,
            "censored": any(r["end_reason"] == "infra_timeout" for r in base)}


def _wall_cells(scored):
    cells = defaultdict(list)
    for r in scored:
        cells[(r["profile"], r["concurrency"])].append(r["end_reason"] == "infra_timeout")
    return [{"profile": p, "concurrency": c, "n": len(v), "share": sum(v) / len(v),
             "over_2pct": sum(v) / len(v) > WALL_HIT_FLAG} for (p, c), v in sorted(cells.items())]


def _union(spans):
    """Total length covered by a set of (start, end) intervals."""
    total, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b > end:
            total, end = total + b - max(a, end), b
    return total


def _pilot_rates(rows):
    """Aggregate tok/s each profile's server reached in the pilot, keyed by profile.

    Tokens over the time at least one request was in that phase: prefill [gen_t_start, gen_t_first_token),
    decode [gen_t_first_token, gen_t_end), the split energy_model (#13) uses; turns with no first token are
    left out. Prefill counts realized work when every record has the server's cache count: the per-turn ideal
    uncached tokens scaled by the attempts' realized / ideal ratio. With tool time fewer than all slots are
    busy, and the two phases overlap, so these sit below an all-slots-busy synthetic rate: with many slots the
    decode union covers nearly the whole span, so the rate is close to the pilot's wall-clock throughput, which
    already includes tool idleness. project_hours would then charge tool time and overlapped prefill twice
    (about 1.9x the pilot's own wall clock in a 14-slot simulation), so these are printed, never projected from.
    """
    by_profile = defaultdict(list)
    for r in rows:
        by_profile[r["profile"]].append(r)
    out = {}
    for profile, rs in sorted(by_profile.items()):
        turns = [t for r in rs for t in r["per_turn"] if t["gen_t_first_token"] is not None]
        pre = _union([(t["gen_t_start"], t["gen_t_first_token"]) for t in turns])
        dec = _union([(t["gen_t_first_token"], t["gen_t_end"]) for t in turns])
        tok = [r["tokens"] for r in rs]
        ideal = sum(t["input_uncached"] for t in tok)
        scale = 1.0
        if ideal and all(t.get("server_reported_cached") is not None for t in tok):
            scale = sum(t["input_cached"] + t["input_uncached"] - t["server_reported_cached"] for t in tok) / ideal
        if pre > 0 and dec > 0:
            out[profile] = {"decode_tok_s": sum(t["output_tokens"] for t in turns) / dec,
                            "prefill_tok_s": scale * sum(t["input_uncached"] for t in turns) / pre}
    return out


def _sampler_ok(sampler):
    def ok(v, want):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - want) <= 1e-9
    return isinstance(sampler, dict) and set(sampler) == set(SAMPLER) and all(ok(sampler[k], v)
                                                                              for k, v in SAMPLER.items())


def _sampler_check(rows):
    """Every record's sampler equals the pinned values and its seed is set, which catches a server default
    leaking in. Error rows count too: they were sampled with the same request."""
    bad = [r.get("attempt_id") for r in rows if not _sampler_ok(r.get("sampler"))]
    no_seed = [r.get("attempt_id") for r in rows
               if not isinstance(r.get("seed"), int) or isinstance(r.get("seed"), bool)]
    return {"ok": not bad and not no_seed, "n_mismatch": len(bad), "n_seed_missing": len(no_seed),
            "examples": sorted(set(bad + no_seed), key=(bad + no_seed).index)[:5]}


def _episode_tokens(rows):
    # Under full prefix reuse, the ideal uncached input plus the output is the whole token sequence.
    return sum(r["tokens"]["output"] + r["tokens"]["input_uncached"] for r in rows) / len(rows) if rows else None


def _stats(rows):
    n = len(rows)
    span = max(r["energy"]["t_end"] for r in rows) - min(r["energy"]["t_start"] for r in rows)
    ends = Counter(r["end_reason"] for r in rows)
    # end_reason error is an infrastructure fault, rerun as a fresh attempt (04), so those rows stay out of the
    # resolve, limit and flagged rates (as #6, #13 and #14 drop them); their count is reported.
    scored = [r for r in rows if r["end_reason"] != "error"]
    m = len(scored)
    by_profile, rows_by_profile = defaultdict(list), defaultdict(list)
    for r in scored:
        by_profile[r["profile"]].append(_flagged(r))
        rows_by_profile[r["profile"]].append(r)
    resolve = _resolve(scored)
    turns = sum(r["turns"] for r in rows)
    tok = {k: sum(r["tokens"][k] for r in rows) for k in ("output", "input_uncached", "input_cached")}
    # A missing or null count means the server did not report it (as in #13); the realized split then needs all.
    n_server_missing = sum(r["tokens"].get("server_reported_cached") is None for r in rows)
    has_server = n_server_missing == 0
    server_cached = sum(r["tokens"]["server_reported_cached"] for r in rows) if has_server else 0
    ctx_share = ends.get("context_limit", 0) / m if m else None
    successes = [r for r in scored if r["resolved"]]
    unflagged_successes = [r for r in successes if not _flagged(r)]
    wall_share = ends.get("infra_timeout", 0) / m if m else None
    hit = tok["input_cached"] / (tok["input_cached"] + tok["input_uncached"]) if tok["input_cached"] else 0.0
    return {
        "n_attempts": n,
        "span_s": span,
        "attempts_per_hour": n / span * 3600 if span else None,
        "n_error": n - m,
        "resolve_rate": resolve["overall"],
        "resolve_rate_excluding_flagged": resolve["excluding_flagged"],
        "resolve_by_profile": {p: _resolve(rs) for p, rs in sorted(rows_by_profile.items())},
        "flagged_share": sum(sum(v) for v in by_profile.values()) / m if m else None,
        "flagged_share_by_profile": {p: sum(v) / len(v) for p, v in sorted(by_profile.items())},
        "end_reason_share": {k: v / n for k, v in sorted(ends.items())},
        "context_limit_check": {"share": ctx_share, "over_10pct": ctx_share is not None and ctx_share > 0.10},
        "wall_clock_check": {"share": wall_share, "over_2pct": wall_share is not None and wall_share > WALL_HIT_FLAG},
        "wall_clock_by_cell": _wall_cells(scored),
        "wall_seconds": _wall(scored),
        # Per turn: a malformed turn executes no call, so a per-call rate can exceed 1.
        "malformed_per_turn": {k: sum(r["malformed"][k] for r in rows) / turns if turns else 0.0 for k in MALFORMED},
        "tokens_per_attempt": {k: v / n for k, v in tok.items()},
        # Prefill the server actually did: all input minus what it reported as cached.
        "realized_uncached_per_attempt": ((tok["input_cached"] + tok["input_uncached"] - server_cached) / n
                                          if has_server else None),
        "gen_seconds_per_attempt": sum(r.get("gen_seconds", 0.0) for r in rows) / n,
        "prefix_cache_hit_rate_ideal": hit,
        "prefix_cache_flag": hit < HERO_PREFIX_HIT - 0.12,
        "n_server_cached_missing": n_server_missing,
        "server_vs_ideal_cached": server_cached / tok["input_cached"] if tok["input_cached"] and has_server else None,
        "tool_seconds_per_attempt": sum(r["tool_seconds"] for r in rows) / n,
        "container_vcpus_busy": sum(r["container_cpu_seconds"] for r in rows) / span if span else None,
        "concurrency": Counter(r["concurrency"] for r in rows).most_common(1)[0][0],
        "pilot_rates": _pilot_rates(rows),
        "sampler_check": _sampler_check(rows),
        "episode_tokens_by_profile": {p: _episode_tokens(rs) for p, rs in sorted(rows_by_profile.items())},
        # KL scoring covers unflagged successes only (owner's agent on PR #7, answer 4); flagged ones are counted.
        "unflagged_success_share": len(unflagged_successes) / m if m else None,
        "flagged_success_share": (len(successes) - len(unflagged_successes)) / m if m else None,
        "unflagged_success_episode_tokens": _episode_tokens(unflagged_successes),
    }


def project_hours(stats, rates, n_attempts, concurrency, realized=True):
    """Instance-hours for n attempts from aggregate token rates (not from one wave's wall clock).

    realized=True charges prefill for what the server actually recomputed (when it reported its cache
    counts); realized=False uses the ideal split under full prefix reuse, a lower bound.

    rates are aggregate tokens per second with all slots busy, so server time is already shared
    across slots; tool time runs in parallel across `concurrency` containers. Following docs/04
    ("tokens per attempt / aggregate token rate plus tool time"), the two are added, which is
    conservative: in practice they partly overlap.
    """
    t = stats["tokens_per_attempt"]
    uncached = stats["realized_uncached_per_attempt"] if realized and stats["realized_uncached_per_attempt"] \
        is not None else t["input_uncached"]
    per_attempt = t["output"] / rates["decode_tok_s"] + uncached / rates["prefill_tok_s"]
    server_h = n_attempts * per_attempt / 3600
    tool_h = n_attempts * stats["tool_seconds_per_attempt"] / concurrency / 3600
    return {"server_hours": server_h, "tool_hours": tool_h, "hours": server_h + tool_h}


def _check_rates(extras):
    """Rates are extras[key][runtime][profile] = {decode_tok_s, prefill_tok_s}, as agreed in the #7 review.
    Any other shape raises: a missing profile level used to move every step under "not included"."""
    for key in RATE_KEYS:
        for runtime, by_profile in extras.get(key, {}).items():
            for profile, rate in (by_profile.items() if isinstance(by_profile, dict) else [(None, by_profile)]):
                if (runtime not in RUNTIMES or profile not in PROFILES or not isinstance(rate, dict)
                        or {"decode_tok_s", "prefill_tok_s"} - rate.keys()):
                    raise ValueError(f"extras {key}[{runtime!r}][{profile!r}] = {rate!r}: expected "
                                     f"{key}[runtime][profile] = {{decode_tok_s, prefill_tok_s}}, runtime in "
                                     f"{RUNTIMES}, profile in {PROFILES}")


def _rate(extras, key, runtime, profile, pilot):
    """(rate, profile it came from, source). The profile's own rate before its RATE_FALLBACK, and for each the
    synthetic-load rate (extras[key]) before the pilot's. The pilot ran at 64K, so it serves only "rates".
    Projections pass pilot={}: see _pilot_rates for why a pilot rate is not an all-slots-busy rate."""
    measured = extras.get(key, {}).get(runtime, {})
    for p in dict.fromkeys((profile, RATE_FALLBACK.get(profile, profile))):
        if measured.get(p):
            return measured[p], p, "synthetic"
        if key == "rates" and pilot.get(p):
            return pilot[p], p, "pilot"
    return None, None, None


def _tool_containers(extras):
    """(containers, where the number came from) for CONTAINERS_SCENARIO's tool time."""
    c = extras.get("concurrency")
    if c is None or (isinstance(c, dict) and "containers" not in c):
        return SCENARIOS[CONTAINERS_SCENARIO][1], "assumed until extras concurrency.containers is written"
    n = c.get("containers") if isinstance(c, dict) else None
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"extras concurrency.containers must be a positive int, got {n!r}")
    return n, "extras concurrency.containers"


def _scenario(stats, extras, key, slots, tool):
    """Instance-hours per P0 step at one context x slots shape. Tool time is shared over tool = (containers,
    source) containers."""
    containers, containers_from = tool
    steps, missing = {}, []
    for step, (runtime, profiles) in P0_STEPS.items():
        if runtime not in stats:
            missing.append(f"{step}: no {runtime} attempts in the pilot")
            continue
        s, rows = stats[runtime], {}
        for profile, (point, upper) in profiles.items():
            rate, rate_from, source = _rate(extras, key, runtime, profile, {})
            if rate is None:
                note = " (the pilot rate above is a cross-check only)" if profile in s["pilot_rates"] else ""
                missing.append(f"{step}: no synthetic rate (extras {key}) for {runtime} {profile}{note}")
                break
            rows[profile] = {"rate_from": rate_from, "rate_source": source,
                             "point_hours": project_hours(s, rate, point, containers)["hours"],
                             "upper_hours": project_hours(s, rate, upper, containers)["hours"],
                             "ideal_cache_hours": project_hours(s, rate, point, containers, realized=False)["hours"]}
        else:
            steps[step] = {"runtime": runtime, "profiles": rows,
                           **{k: sum(v[k] for v in rows.values())
                              for k in ("point_hours", "upper_hours", "ideal_cache_hours")}}
    if "llama.cpp" in stats:
        s = stats["llama.cpp"]
        h = SUB_RUN["attempts"] * (s["gen_seconds_per_attempt"] + s["tool_seconds_per_attempt"] + SUB_RUN["settle_s"])
        steps["energy_sub_run"] = {"runtime": "llama.cpp", "serial": True, "attempts": SUB_RUN["attempts"],
                                   "point_hours": h / 3600, "upper_hours": h / 3600, "ideal_cache_hours": h / 3600}
    total = {k: sum(v[k] for v in steps.values()) for k in ("point_hours", "upper_hours", "ideal_cache_hours")}
    return {"slots": slots, "rates_key": key, "tool_containers": containers, "tool_containers_from": containers_from,
            "steps": steps, "total": total if steps else None, "missing": missing}


def _bf16_ratio(stats, extras):
    """llama.cpp BF16 / vLLM BF16 aggregate rate at 64K x 16, for the optional P0b rule (the owner sets the
    margin). Sources are given, since a synthetic rate against a pilot one is not like for like."""
    got = {rt: _rate(extras, "rates", rt, "BF16", stats[rt]["pilot_rates"] if rt in stats else {}) for rt in RUNTIMES}
    (lc, _, lc_src), (vl, _, vl_src) = got["llama.cpp"], got["vllm"]
    if lc is None or vl is None:
        return None
    return {"decode": lc["decode_tok_s"] / vl["decode_tok_s"], "prefill": lc["prefill_tok_s"] / vl["prefill_tok_s"],
            "sources": {"llama.cpp": lc_src, "vllm": vl_src}}


def kl_hours(stats, extras):
    """Hours of teacher-forced KL scoring: scored runs x seconds per run on the stated machine (KL_SCORING, or
    extras["kl_scoring"]), scaled linearly by the runs' mean length over the measured run's 16.5K tokens.

    - P0b: every unflagged successful P0b run (4,800 x the vLLM pilot's unflagged-success share), BF16 vs
      Q3_K_M. Flagged successes are never selection candidates, so they are not scored (owner's agent on PR #7,
      answer 4); flagged_not_scored counts them.
    - P0b Q4_K_M probe: the same runs, BF16 vs Q4_K_M, for P0d's Q3_K_M vs Q4_K_M probe agreement (04, P0d
      score choice). 04's P0 step 3 names only BF16 and Q3_K_M, so this is listed as its own job; it errs high
      if only tasks with 3 or more successes are scored, or if the BF16 logits are reused.
    - P0a diagnostic: every P0a BF16 trajectory (600, 900 with the top-up) against each of the 4 quantized
      profiles, at the llama.cpp BF16 pilot length.
    The per-run figure includes both model loads, which a batched call amortizes, so this errs high too.
    """
    m = {**KL_SCORING, **extras.get("kl_scoring", {})}
    per_token_h = m["seconds_per_run"] / m["tokens_per_run"] / 3600
    jobs = {}
    vllm = stats.get("vllm")
    if vllm and vllm["unflagged_success_share"] and vllm["unflagged_success_episode_tokens"]:
        n = P0_STEPS["P0b"][1]["BF16"][0]
        runs, tokens = n * vllm["unflagged_success_share"], vllm["unflagged_success_episode_tokens"]
        h = runs * tokens * per_token_h
        for job, probe in (("P0b", "Q3_K_M"), ("P0b Q4_K_M probe", "Q4_K_M")):
            jobs[job] = {"what": f"unflagged successful runs, BF16 vs {probe}", "runs": runs,
                         "flagged_not_scored": n * vllm["flagged_success_share"], "pairs": 1,
                         "tokens_per_run": tokens, "point_hours": h, "upper_hours": h}
    bf16_tokens = stats.get("llama.cpp", {}).get("episode_tokens_by_profile", {}).get("BF16")
    if bf16_tokens:
        point, upper = P0_STEPS["P0a"][1]["BF16"]
        pairs = len(PROFILES) - 1
        jobs["P0a"] = {"what": "diagnostic, BF16 trajectories vs each quantized profile", "runs": point,
                       "pairs": pairs, "tokens_per_run": bf16_tokens,
                       "point_hours": point * pairs * bf16_tokens * per_token_h,
                       "upper_hours": upper * pairs * bf16_tokens * per_token_h}
    total = {k: sum(j[k] for j in jobs.values()) for k in ("point_hours", "upper_hours")}
    pick = None if not jobs else "EC2" if total["point_hours"] < KL_EC2_MAX_H else "the 48 GB Mac overnight"
    return {**m, "jobs": jobs, "total": total,
            "machine_recommendation": {"threshold_h": KL_EC2_MAX_H, "hours": total["point_hours"], "recommend": pick}}


def projections(stats, extras):
    """Projected hours: per P0 step at the point and upper attempt counts for each SCENARIOS shape, plus the
    energy sub-run; the llama.cpp / vLLM BF16 rate ratio; and KL scoring hours on its own machine.

    A step is projected only when every profile in it has a rate (its own or the fallback in RATE_FALLBACK);
    otherwise it is listed under the scenario's "missing".
    """
    _check_rates(extras)
    tools = {name: (slots, f"the scenario's {slots} slots") for name, (_, slots) in SCENARIOS.items()}
    tools[CONTAINERS_SCENARIO] = _tool_containers(extras)
    return {"scenarios": {name: _scenario(stats, extras, key, slots, tools[name])
                          for name, (key, slots) in SCENARIOS.items()},
            "bf16_throughput_ratio": _bf16_ratio(stats, extras),
            "kl_scoring": kl_hours(stats, extras),
            "not_included": NOT_INCLUDED}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _kernels(probe_row):
    """Whether the probe ran the fast DeltaNet kernels (kernels.in_use from #5); a --allow-slow-kernels run's
    s/step and memory come from the torch fallback. Probe files written before that field show unknown."""
    in_use = (probe_row.get("kernels") or {}).get("in_use")
    if not in_use:
        return "unknown"
    return "fast" if all(op["fast"] for op in in_use.values()) else "slow path"


def _md_sampler(stats):
    lines = ["", "## Sampler and seed", "", ("Records whose sampler differs from the pinned values in 04 "
             f"({', '.join(f'{k}={v}' for k, v in SAMPLER.items())}), and records without a seed."), "",
             "| runtime | sampler mismatches | seed missing | examples |", "|---|---|---|---|"]
    for rt, s in stats.items():
        c = s["sampler_check"]
        lines.append(f"| {rt} | {c['n_mismatch']}{' **FAIL**' if c['n_mismatch'] else ''} "
                     f"| {c['n_seed_missing']}{' **FAIL**' if c['n_seed_missing'] else ''} "
                     f"| {', '.join(map(str, c['examples'])) or '-'} |")
    return lines


def _md_wall(stats):
    lines = ["", f"Wall-clock hits by profile and concurrency (owner revisits the cap above {_pct(WALL_HIT_FLAG)}).",
             "", "| runtime | profile | concurrency | attempts | wall clock |", "|---|---|---|---|---|"]
    for rt, s in stats.items():
        lines += [f"| {rt} | {c['profile']} | {c['concurrency']} | {c['n']} | {_pct(c['share'])}"
                  f"{' **over**' if c['over_2pct'] else ''} |" for c in s["wall_clock_by_cell"]]
    lines += ["", (f"Proposed infra cap for P0: one cap per runtime, from its {CAP_PROFILE} pilot attempts, for "
              "every profile on that runtime."), "", f"| runtime | {CAP_PROFILE} p99 attempt time | 3 x p99 |",
              "|---|---|---|"]
    for rt, s in stats.items():
        w = s["wall_seconds"]
        if w["p99"] is None:
            lines.append(f"| {rt} | n/a (no {CAP_PROFILE} pilot attempts) | n/a |")
            continue
        note = " (lower bound: some attempts hit the old cap)" if w["censored"] else ""
        lines.append(f"| {rt} | {w['p99'] / 3600:.2f} h{note} | {w['proposed_cap_s'] / 3600:.2f} h |")
    return lines


def _md_rates(stats, extras, projections):
    lines = ["", "## Aggregate token rates", "", ("Synthetic load from the extras at each shape, and the rate each "
             "pilot server reached (tokens over the time a request was in that phase; below all-slots-busy)."), "",
             "| runtime | profile | source | decode tok/s | prefill tok/s |", "|---|---|---|---|---|"]
    sources = [(f"synthetic {name}", extras.get(key, {})) for name, (key, _) in SCENARIOS.items()]
    sources.append(("pilot", {rt: s["pilot_rates"] for rt, s in stats.items()}))
    for label, rates in sources:
        lines += [f"| {rt} | {p} | {label} | {r['decode_tok_s']:.1f} | {r['prefill_tok_s']:.1f} |"
                  for rt, by_profile in rates.items() for p, r in by_profile.items()]
    ratio = (projections or {}).get("bf16_throughput_ratio")
    if ratio:
        lines += ["", (f"llama.cpp BF16 / vLLM BF16: decode {ratio['decode']:.2f}, prefill {ratio['prefill']:.2f} "
                  f"(llama.cpp {ratio['sources']['llama.cpp']}, vLLM {ratio['sources']['vllm']}), for the optional "
                  "P0b rule; the owner sets the margin.")]
    return lines


def _md_projections(projections):
    lines = ["", "## Projected instance-hours", "",
             ("Point and upper attempt counts from 04, prefill at the realized cache split; the ideal-cache "
              "column is a lower bound. Tokens per attempt come from the 64K pilot, so 128K x 8 is a lower "
              "bound too (runs that hit 64K would have continued).")]
    for name, sc in projections["scenarios"].items():
        lines += ["", f"### {name}", "", (f"Rates from extras `{sc['rates_key']}`; tool time shared over "
                  f"{sc['tool_containers']} containers ({sc['tool_containers_from']})."), "",
                  "| step | runtime | rates | point h | upper h | ideal-cache h |",
                  "|---|---|---|---|---|---|"]
        for step, v in sc["steps"].items():
            src = ", ".join(f"{p}: {r['rate_from']} {r['rate_source']}" for p, r in v.get("profiles", {}).items())
            lines.append(f"| {step} | {v['runtime']} | {src or '-'} | {v['point_hours']:.1f} "
                         f"| {v['upper_hours']:.1f} | {v['ideal_cache_hours']:.1f} |")
        if sc["total"]:
            t = sc["total"]
            # A step without a rate is left out of the sum, so the row must not read as the whole bill.
            label = "partial total (see below)" if sc["missing"] else "total"
            lines.append(f"| {label} | | | {t['point_hours']:.1f} | {t['upper_hours']:.1f} "
                         f"| {t['ideal_cache_hours']:.1f} |")
        lines += ["", "Not included in the total:", ""]
        lines += [f"- {x}" for x in projections["not_included"] + sc["missing"]]
    return lines


def _md_kl(kl):
    lines = ["", "## KL scoring", "", (f"Machine: {kl['machine']}, {kl['seconds_per_run']:.0f} s per "
             f"{kl['tokens_per_run']}-token run, scaled linearly by run length. Not instance-hours unless the "
             "machine is the g6e."), "", ("04's P0 step 3 names KL scoring at BF16 and Q3_K_M only, while the P0d "
             "score choice needs the Q3_K_M vs Q4_K_M probe agreement on the P0b pool, so the Q4_K_M probe is its "
             "own row (owner question: the whole pool, or only tasks with 3 or more successes?)."), "",
             "| step | what | runs | pairs per run | tokens per run | point h | upper h |",
             "|---|---|---|---|---|---|---|"]
    for step, j in kl["jobs"].items():
        lines.append(f"| {step} | {j['what']} | {j['runs']:.0f} | {j['pairs']} | {j['tokens_per_run']:.0f} "
                     f"| {j['point_hours']:.1f} | {j['upper_hours']:.1f} |")
    t = kl["total"]
    lines.append(f"| total | | | | | {t['point_hours']:.1f} | {t['upper_hours']:.1f} |")
    if "P0b" in kl["jobs"]:
        lines += ["", (f"P0b: {kl['jobs']['P0b']['flagged_not_scored']:.0f} flagged successful runs are not scored "
                       "(never selection candidates).")]
    r = kl["machine_recommendation"]
    pick = (f"at {r['hours']:.1f} h (point total, on {kl['machine']}) that is {r['recommend']}, pending the owner"
            if r["recommend"] else "no KL job is projected yet")
    lines += ["", (f"Machine for the P0b scoring and the P0a diagnostic (owner decision): the owner's agent recommends "
                   f"EC2 if the projected KL-scoring hours are under about {r['threshold_h']:.0f} h, otherwise the "
                   f"48 GB Mac overnight; {pick}. The #11 acceptance criterion is judged on that backend.")]
    return lines


def render_markdown(stats, probe, extras, projections):
    lines = ["# Pilot exit report", ""]
    lines += ["## Attempts per hour", "", ("Resolve, limit and flagged shares leave out end_reason error rows "
              "(rerun as fresh attempts)."), "",
              ("| runtime | attempts | errors | per hour | resolve rate | resolve excl. flagged | flagged "
               "| concurrency |"), "|---|---|---|---|---|---|---|---|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | {s['n_attempts']} | {s['n_error']} | {s['attempts_per_hour']:.1f} "
                     f"| {_pct(s['resolve_rate'])} | {_pct(s['resolve_rate_excluding_flagged'])} "
                     f"| {_pct(s['flagged_share'])} | {s['concurrency']} |")
    lines += ["", "## Test-file tripwire", "", ("Share of attempts with grading.test_files_touched, by profile, "
              "with the resolve rate overall and excluding flagged attempts."), "",
              "| runtime | profile | flagged | resolve rate | resolve excl. flagged |", "|---|---|---|---|---|"]
    for rt, s in stats.items():
        lines += [f"| {rt} | {p} | {_pct(v)} | {_pct(s['resolve_by_profile'][p]['overall'])} "
                  f"| {_pct(s['resolve_by_profile'][p]['excluding_flagged'])} |"
                  for p, v in s["flagged_share_by_profile"].items()]
    lines += _md_sampler(stats)
    lines += ["", "## Limits hit", "", ("| runtime | end reasons | 64K context (owner check at 10%) "
              "| wall clock (cap check at 2%) |"), "|---|---|---|---|"]
    for rt, s in stats.items():
        ends = ", ".join(f"{k} {_pct(v)}" for k, v in s["end_reason_share"].items())
        ctx, wall = s["context_limit_check"], s["wall_clock_check"]
        lines.append(f"| {rt} | {ends} | {_pct(ctx['share'])}{' **over**' if ctx['over_10pct'] else ''} "
                     f"| {_pct(wall['share'])}{' **over**' if wall['over_2pct'] else ''} |")
    lines += _md_wall(stats)
    lines += ["", "## Malformed calls", "", "Per turn, by category (a malformed turn executes no call).", "",
              "| runtime | " + " | ".join(MALFORMED) + " |", "|---" * (len(MALFORMED) + 1) + "|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | " + " | ".join(_pct(s["malformed_per_turn"][k]) for k in MALFORMED) + " |")
    lines += ["", "## Tokens and prefix cache", "",
              ("| runtime | output | uncached input | cached input | ideal hit rate (flag below 80%) "
              "| server / ideal cached | server count missing "
              "| tool s | vCPUs busy |"), "|---|---|---|---|---|---|---|---|---|"]
    for rt, s in stats.items():
        t = s["tokens_per_attempt"]
        flag = " **below HERO's ~92%**" if s["prefix_cache_flag"] else ""
        ratio = "n/a" if s["server_vs_ideal_cached"] is None else f"{s['server_vs_ideal_cached']:.2f}"
        busy = "n/a" if s["container_vcpus_busy"] is None else f"{s['container_vcpus_busy']:.1f}"
        lines.append(f"| {rt} | {t['output']:.0f} | {t['input_uncached']:.0f} | {t['input_cached']:.0f} "
                     f"| {_pct(s['prefix_cache_hit_rate_ideal'])}{flag} | {ratio} | {s['n_server_cached_missing']} "
                     f"| {s['tool_seconds_per_attempt']:.0f} | {busy} |")
    lines += ["", "## Training probe", ""]
    if probe:
        lines += ["| loss | supervised | length | status | s/step | memory | kernels |",
                  "|---|---|---|---|---|---|---|"]
        for p in probe:
            sec = f"{p['sec_per_step']:.1f}" if p["status"] == "ok" else "-"
            mem = "n/a" if p.get("mem_gib") is None else f"{p['mem_gib']:.1f} GiB ({p['mem_kind']})"
            lines.append(f"| {p['loss_path']} | {p['supervised_frac']:.2f} | {p['length']} | {p['status']} "
                         f"| {sec} | {mem} | {_kernels(p)} |")
    else:
        lines.append("No probe results.")
    lines += _md_rates(stats, extras, projections)
    if projections:
        lines += _md_projections(projections) + _md_kl(projections["kl_scoring"])
    missing = [k for k in EXTRAS if k not in extras]
    lines += ["", "## Not yet provided", ""]
    lines += [f"- `{k}`" for k in missing] or ["Nothing: every section is filled."]
    for k in EXTRAS:
        if k in extras and k not in RATE_KEYS:  # rates have their own table
            lines += ["", f"`{k}`: `{extras[k]}`"]
    return "\n".join(lines) + "\n"
