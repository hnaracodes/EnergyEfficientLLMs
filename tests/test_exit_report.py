"""Tests for the pilot exit report. Run:  python -m pytest tests/test_exit_report.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from qtrace.exit_report import attempt_stats, project_hours, projections, render_markdown

SAMPLER = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
           "repetition_penalty": 1.0}


def rec(runtime, end_reason, t0, t1, out=1000, unc=3000, cached=20000, server_cached=20000, malformed=None,
        tool_s=60.0, cpu_s=30.0, gen_s=120.0, profile="BF16", touched=False, wall_s=None, concurrency=14):
    # One generation window per attempt: prefill over its first second, decode for the rest of gen_s.
    turn = {"turn": 0, "output_tokens": out, "input_uncached": unc, "input_cached": cached,
            "gen_t_start": t0, "gen_t_first_token": t0 + 1.0, "gen_t_end": t0 + gen_s}
    return {"attempt_id": f"{runtime}-{profile}-{t0}-{t1}", "seed": 7, "sampler": dict(SAMPLER),
            "runtime": runtime, "end_reason": end_reason,
            "resolved": end_reason == "submitted",
            "grading": {"test_files_touched": touched}, "per_turn": [turn],
            "wall_seconds": t1 - t0 if wall_s is None else wall_s,
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached,
                       "server_reported_cached": server_cached},
            "malformed": malformed or {"misnested_tag": 0, "wrong_name": 0, "text_after_call": 0,
                                       "truncated_call": 0, "no_call_no_answer": 0},
            "energy": {"t_start": t0, "t_end": t1}, "tool_seconds": tool_s, "container_cpu_seconds": cpu_s,
            "gen_seconds": gen_s,
            "turns": 10, "tool_calls": 10, "profile": profile, "concurrency": concurrency}


def test_attempt_stats_rates_and_shares():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "context_limit", 0, 900),
            rec("vllm", "turn_limit", 300, 1200), rec("vllm", "infra_timeout", 600, 3600,
                                                      malformed={"misnested_tag": 2, "wrong_name": 0,
                                                                 "text_after_call": 1, "truncated_call": 0,
                                                                 "no_call_no_answer": 0})]
    s = attempt_stats(rows)["vllm"]
    assert s["n_attempts"] == 4
    assert abs(s["attempts_per_hour"] - 4.0) < 1e-9  # 4 attempts in a 3,600 s span
    assert s["end_reason_share"]["context_limit"] == 0.25
    assert s["context_limit_check"] == {"share": 0.25, "over_10pct": True}
    assert s["malformed_per_turn"]["misnested_tag"] == 2 / 40
    assert s["prefix_cache_hit_rate_ideal"] == 20000 / 23000
    assert s["server_vs_ideal_cached"] == 1.0
    assert s["realized_uncached_per_attempt"] == 3000.0
    assert s["tokens_per_attempt"]["output"] == 1000


def test_project_hours_uses_token_rates_not_wave_time():
    stats = {"tokens_per_attempt": {"output": 1000, "input_uncached": 3000, "input_cached": 20000},
             "tool_seconds_per_attempt": 60.0, "realized_uncached_per_attempt": None}
    rates = {"decode_tok_s": 500.0, "prefill_tok_s": 5000.0}
    h = project_hours(stats, rates, n_attempts=1000, concurrency=14)
    # per attempt: 1000/500 + 3000/5000 = 2.6 s of server time; tool time overlaps across 14 slots
    assert abs(h["server_hours"] - 1000 * 2.6 / 3600) < 1e-9
    assert abs(h["tool_hours"] - 1000 * 60 / 14 / 3600) < 1e-9
    assert h["hours"] == h["server_hours"] + h["tool_hours"]  # 04: rate time plus tool time


def test_markdown_lists_every_required_section():
    rows = [rec("llama.cpp", "submitted", 0, 600)]
    md = render_markdown(attempt_stats(rows), probe=[{"length": 16384, "status": "ok", "sec_per_step": 9.0,
                                                      "mem_gib": 30.0, "mem_kind": "cuda_peak",
                                                      "supervised_frac": 0.15, "loss_path": "chunked"}],
                         extras={}, projections={})
    for heading in ("Attempts per hour", "Limits hit", "Malformed calls", "Tokens and prefix cache",
                    "Training probe", "Not yet provided"):
        assert heading in md
    for key in ("sampler_echo_ok", "per_turn_energy_ok", "concurrency"):
        assert f"`{key}`" in md  # listed as missing until the harness writes it


def test_realized_cache_misses_raise_the_projection():
    stats = {"tokens_per_attempt": {"output": 1000, "input_uncached": 3000, "input_cached": 20000},
             "tool_seconds_per_attempt": 0.0, "realized_uncached_per_attempt": 13000.0}
    rates = {"decode_tok_s": 500.0, "prefill_tok_s": 5000.0}
    real = project_hours(stats, rates, 100, 14)["hours"]
    ideal = project_hours(stats, rates, 100, 14, realized=False)["hours"]
    assert real > ideal and abs(real - 100 * (2 + 13000 / 5000) / 3600) < 1e-9


def test_projections_by_profile_with_fallback_and_sub_run():
    rows = [rec("llama.cpp", "submitted", 0, 600), rec("vllm", "submitted", 0, 600)]
    stats = attempt_stats(rows)
    extras = {"rates": {"llama.cpp": {"BF16": {"decode_tok_s": 400, "prefill_tok_s": 4000},
                                      "Q4_K_M": {"decode_tok_s": 600, "prefill_tok_s": 5000}}}}
    steps = projections(stats, extras)["scenarios"]["64K x 16"]["steps"]
    assert steps["P0a"]["profiles"]["Q3_K_M"]["rate_from"] == "Q4_K_M"
    assert steps["P0a"]["profiles"]["Q8_0"]["rate_from"] == "BF16"
    assert steps["P0a"]["profiles"]["BF16"]["rate_source"] == "synthetic"
    assert steps["P0a"]["upper_hours"] > steps["P0a"]["point_hours"]  # the 600-attempt top-up
    # No synthetic vLLM rate: P0b is not projected from the pilot rate (see the next test).
    assert "P0b" not in steps
    sub = steps["energy_sub_run"]
    assert sub["serial"] and abs(sub["point_hours"] - 150 * (120 + 60 + 45) / 3600) < 1e-9


def test_pilot_rates_are_not_projected_from():
    # Review of eadb97e: a 14-slot pilot, 30 turns of 1 s prefill + 19 s decode + 10 s tool per attempt. Its
    # decode union covers nearly the whole span, so the pilot rate already holds the tool idleness; projected
    # with tool time on top, P0b read about 1.9x the pilot's own wall clock as a plain "total" row.
    rows = []
    for slot in range(14):
        t = slot * 2.0
        for a in range(10):
            r = rec("vllm", "submitted", t, t + 900, out=12000, unc=45000, cached=300000, server_cached=300000,
                    tool_s=300.0, gen_s=600.0)
            r["attempt_id"] = f"{slot}-{a}"
            r["per_turn"] = [{"turn": k, "output_tokens": 400, "input_uncached": 1500, "input_cached": 0,
                              "gen_t_start": t + 30 * k, "gen_t_first_token": t + 30 * k + 1.0,
                              "gen_t_end": t + 30 * k + 20.0} for k in range(30)]
            rows.append(r)
            t += 900
    stats = attempt_stats(rows)
    assert stats["vllm"]["pilot_rates"]["BF16"]["decode_tok_s"] > 0  # still derived and printed
    sc = projections(stats, {})["scenarios"]["64K x 16"]
    assert "P0b" not in sc["steps"]
    assert any("no synthetic rate (extras rates) for vllm BF16 (the pilot rate above is a cross-check only)" in x
               for x in sc["missing"])
    md = render_markdown(stats, [], {}, projections(stats, {}))
    assert "| vllm | BF16 | pilot |" in md and "| P0b |" not in md.split("## KL scoring")[0]


def test_null_server_cached_counts_as_not_reported():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "submitted", 0, 600, server_cached=None)]
    s = attempt_stats(rows)["vllm"]
    assert s["n_server_cached_missing"] == 1
    assert s["realized_uncached_per_attempt"] is None and s["server_vs_ideal_cached"] is None
    assert "| 1 |" in render_markdown({"vllm": s}, probe=[], extras={}, projections={})


def test_error_rows_leave_the_rate_denominators():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "context_limit", 0, 600),
            rec("vllm", "infra_timeout", 0, 600), rec("vllm", "turn_limit", 0, 600),
            rec("vllm", "error", 0, 600), rec("vllm", "error", 0, 600)]
    for r in rows[4:]:
        r["grading"] = None  # an infrastructure fault may never reach the grader
    s = attempt_stats(rows)["vllm"]
    assert s["n_error"] == 2 and s["n_attempts"] == 6
    assert s["resolve_rate"] == 0.25
    assert s["context_limit_check"]["share"] == 0.25 and s["wall_clock_check"]["share"] == 0.25


def test_probe_table_marks_slow_kernel_rows():
    base = {"length": 16384, "status": "ok", "sec_per_step": 9.0, "mem_gib": 30.0, "mem_kind": "cuda_peak",
            "supervised_frac": 0.15, "loss_path": "chunked"}
    ops = lambda *fast: {"in_use": {f"op{i}": {"fast": f, "impl": "x"} for i, f in enumerate(fast)}}
    probe = [{**base, "kernels": ops(True, True)}, {**base, "length": 8192, "kernels": ops(True, False)},
             {**base, "length": 4096}]  # a probe file written before the kernels field
    md = render_markdown({}, probe=probe, extras={}, projections={})
    assert "| loss | supervised | length | status | s/step | memory | kernels |" in md
    assert "| chunked | 0.15 | 16384 | ok | 9.0 | 30.0 GiB (cuda_peak) | fast |" in md
    assert "| chunked | 0.15 | 8192 | ok | 9.0 | 30.0 GiB (cuda_peak) | slow path |" in md
    assert "| chunked | 0.15 | 4096 | ok | 9.0 | 30.0 GiB (cuda_peak) | unknown |" in md


def test_flagged_share_by_profile_and_strict_field():
    rows = [rec("llama.cpp", "submitted", 0, 600, profile="BF16", touched=True),
            rec("llama.cpp", "submitted", 0, 600, profile="BF16"),
            rec("llama.cpp", "turn_limit", 0, 600, profile="Q4_K_M")]
    s = attempt_stats(rows)["llama.cpp"]
    assert s["flagged_share_by_profile"] == {"BF16": 0.5, "Q4_K_M": 0.0}
    assert abs(s["flagged_share"] - 1 / 3) < 1e-12
    assert "| llama.cpp | BF16 | 50.0% |" in render_markdown({"llama.cpp": s}, probe=[], extras={}, projections={})
    del rows[2]["grading"]["test_files_touched"]
    with pytest.raises(KeyError):
        attempt_stats(rows)
    for bad in (None, 0, 1):  # a 0 would print a 0.0% flagged share instead of stopping
        rows[2]["grading"]["test_files_touched"] = bad
        with pytest.raises(TypeError, match="must be a bool"):
            attempt_stats(rows)


@pytest.mark.parametrize("field,value", [("runtime", "vLLM"), ("profile", "Q4_K_M_gates")])
def test_unknown_runtime_or_profile_raises(field, value):
    r = rec("vllm", "submitted", 0, 600)
    r[field] = value
    with pytest.raises(ValueError, match="not in"):
        attempt_stats([r])


RATE = {"decode_tok_s": 500.0, "prefill_tok_s": 5000.0}


def test_rates_are_printed_and_derived_from_the_pilot_records():
    # Audit gap 1. Two overlapping BF16 attempts: prefill windows [0, 1) and [0.5, 1.5) cover 1.5 s,
    # decode windows [1, 11) and [1.5, 11.5) cover 10.5 s.
    rows = [rec("llama.cpp", "submitted", 0, 600, out=500, unc=1000, gen_s=11.0),
            rec("llama.cpp", "submitted", 0.5, 600, out=500, unc=1000, gen_s=11.0)]
    pilot = attempt_stats(rows)["llama.cpp"]["pilot_rates"]["BF16"]
    assert abs(pilot["decode_tok_s"] - 1000 / 10.5) < 1e-9
    assert abs(pilot["prefill_tok_s"] - 2000 / 1.5) < 1e-9
    # The server reported 1,000 fewer cached tokens per attempt than the ideal split, so it prefilled twice the
    # ideal count in the same windows.
    for r in rows:
        r["tokens"]["server_reported_cached"] = 19000
    assert abs(attempt_stats(rows)["llama.cpp"]["pilot_rates"]["BF16"]["prefill_tok_s"] - 4000 / 1.5) < 1e-9
    stats = attempt_stats(rows)
    extras = {"rates": {"llama.cpp": {"Q4_K_M": {"decode_tok_s": 812.5, "prefill_tok_s": 9100.0}}}}
    md = render_markdown(stats, probe=[], extras=extras, projections=projections(stats, extras))
    assert "| llama.cpp | Q4_K_M | synthetic 64K x 16 | 812.5 | 9100.0 |" in md
    assert f"| llama.cpp | BF16 | pilot | {1000 / 10.5:.1f} | {4000 / 1.5:.1f} |" in md


def test_rates_in_the_per_runtime_shape_raise():
    # Audit gap 2: the old docstring shape {runtime: {decode_tok_s, prefill_tok_s}} has no profile level, so
    # every step used to drop silently to "not included".
    stats = attempt_stats([rec("llama.cpp", "submitted", 0, 600)])
    with pytest.raises(ValueError, match=r"rates\[runtime\]\[profile\]"):
        projections(stats, {"rates": {"llama.cpp": {"decode_tok_s": 400, "prefill_tok_s": 4000}}})
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "p0_exit_report.py")
    with open(script) as f:
        doc = f.read()
    assert '{"rates": {runtime: {profile: {"decode_tok_s", "prefill_tok_s"}}}' in doc


def test_wall_seconds_p99_and_proposed_cap():
    # Audit gap 3: 100 scored attempts of 1..100 minutes; an error row's wall time stays out.
    rows = [rec("vllm", "submitted", 0, 600, wall_s=60.0 * i) for i in range(1, 101)]
    rows.append(rec("vllm", "error", 0, 600, wall_s=1e6))
    rows[-1]["grading"] = None
    w = attempt_stats(rows)["vllm"]["wall_seconds"]
    assert abs(w["p99"] - 60 * 99.01) < 1e-9  # linear interpolation, as numpy.percentile
    assert abs(w["proposed_cap_s"] - 3 * 60 * 99.01) < 1e-9
    assert w["censored"] is False
    rows[0]["end_reason"] = "infra_timeout"
    assert attempt_stats(rows)["vllm"]["wall_seconds"]["censored"] is True  # hits stop at the old cap
    assert "Proposed infra cap" in render_markdown(attempt_stats(rows), probe=[], extras={}, projections={})


def test_one_infra_cap_per_runtime_from_its_bf16_attempts():
    # Owner's agent on PR #7 (answer 2): one cap per runtime, from that runtime's BF16 pilot attempts, applied
    # to every profile on it, so no within-runtime profile comparison runs under two caps.
    rows = [rec("llama.cpp", "submitted", 0, 600, wall_s=60.0 * i) for i in range(1, 101)]
    rows += [rec("llama.cpp", "submitted", 0, 600, wall_s=1e5, profile="Q4_K_M")] * 5  # not BF16: no say
    rows += [rec("llama.cpp", "infra_timeout", 0, 600, profile="Q3_K_M")]  # nor in the censoring flag
    rows += [rec("vllm", "submitted", 0, 600, wall_s=600.0 * i) for i in range(1, 101)]
    stats = attempt_stats(rows)
    lc, vl = stats["llama.cpp"]["wall_seconds"], stats["vllm"]["wall_seconds"]
    assert abs(lc["p99"] - 60 * 99.01) < 1e-9 and lc["censored"] is False and lc["profile"] == "BF16"
    assert abs(vl["proposed_cap_s"] - 3 * 600 * 99.01) < 1e-9
    md = render_markdown(stats, probe=[], extras={}, projections={})
    assert "the larger runtime" not in md
    assert f"| llama.cpp | {60 * 99.01 / 3600:.2f} h | {3 * 60 * 99.01 / 3600:.2f} h |" in md
    assert "one cap per runtime, from its BF16 pilot attempts, for every profile on that runtime" in md
    q4_only = attempt_stats([rec("llama.cpp", "submitted", 0, 600, profile="Q4_K_M")])
    assert q4_only["llama.cpp"]["wall_seconds"]["p99"] is None
    assert "| llama.cpp | n/a (no BF16 pilot attempts) | n/a |" in render_markdown(q4_only, [], {}, {})


def test_wall_clock_share_by_profile_and_concurrency():
    # Audit gap 4: one cell at 2.5% is over the 2% check while the runtime as a whole (3 of 200) is not.
    rows = ([rec("llama.cpp", "infra_timeout", 0, 600)] + [rec("llama.cpp", "submitted", 0, 600)] * 39
            + [rec("llama.cpp", "submitted", 0, 600, profile="Q4_K_M")] * 10
            + [rec("llama.cpp", "infra_timeout", 0, 600, concurrency=16)]
            + [rec("llama.cpp", "submitted", 0, 600, concurrency=16)] * 99
            + [rec("llama.cpp", "infra_timeout", 0, 600, profile="Q3_K_M")]
            + [rec("llama.cpp", "submitted", 0, 600, profile="Q3_K_M")] * 49)
    error = rec("llama.cpp", "error", 0, 600)
    error["grading"] = None
    rows.append(error)  # an infra fault stays out of its cell, or it would dilute the BF16/14 share
    s = attempt_stats(rows)["llama.cpp"]
    assert s["wall_clock_check"]["over_2pct"] is False
    cells = {(c["profile"], c["concurrency"]): c for c in s["wall_clock_by_cell"]}
    assert cells[("BF16", 14)]["n"] == 40 and cells[("BF16", 14)]["share"] == 1 / 40
    assert cells[("BF16", 14)]["over_2pct"] is True
    assert cells[("Q3_K_M", 14)]["share"] == 0.02 and cells[("Q3_K_M", 14)]["over_2pct"] is False  # strictly above
    assert cells[("BF16", 16)]["share"] == 1 / 100 and cells[("BF16", 16)]["over_2pct"] is False
    assert cells[("Q4_K_M", 14)]["n"] == 10 and cells[("Q4_K_M", 14)]["share"] == 0.0
    md = render_markdown({"llama.cpp": s}, probe=[], extras={}, projections={})
    assert "| llama.cpp | BF16 | 14 | 40 | 2.5% **over** |" in md


def test_projection_scenarios_64k_x16_and_128k_x8():
    # Audit gap 5: tool time is shared over the scenario's slots (16 or 8), not the pilot's 14, and 128K x 8
    # needs its own measured rates: nothing is scaled from 64K.
    stats = attempt_stats([rec("vllm", "submitted", 0, 600), rec("llama.cpp", "submitted", 0, 600),
                           rec("llama.cpp", "submitted", 0, 600, profile="Q4_K_M")])
    s = stats["vllm"]
    rates = {"vllm": {"BF16": RATE}, "llama.cpp": {"BF16": RATE, "Q4_K_M": RATE}}
    p = projections(stats, {"rates": rates})
    p64, p128 = p["scenarios"]["64K x 16"], p["scenarios"]["128K x 8"]
    assert p64["slots"] == 16 and p128["slots"] == 8
    assert p64["steps"]["P0b"]["point_hours"] == project_hours(s, RATE, 4800, 16)["hours"]
    assert "P0b" not in p128["steps"] and any("rates_128k_x8" in x and "vllm BF16" in x for x in p128["missing"])
    md = render_markdown(stats, probe=[], extras={}, projections=p)
    assert "| partial total (see below) |" in md.split("### 128K x 8")[1]
    assert "| total |" in md.split("### 128K x 8")[0].split("### 64K x 16")[1]
    half = {"decode_tok_s": 250.0, "prefill_tok_s": 2500.0}
    extras = {"rates": rates, "rates_128k_x8": {"vllm": {"BF16": half}}}
    p128 = projections(stats, extras)
    assert p128["scenarios"]["128K x 8"]["steps"]["P0b"]["point_hours"] == project_hours(s, half, 4800, 8)["hours"]
    md = render_markdown(stats, probe=[], extras=extras, projections=p128)
    assert "### 64K x 16" in md and "### 128K x 8" in md
    assert "| vllm | BF16 | synthetic 128K x 8 | 250.0 | 2500.0 |" in md


def test_64k_x16_tool_time_shares_over_the_containers_alive_at_once():
    # Owner's agent on PR #7 (answer 3): divide tool time by extras concurrency.containers, assuming 16 (and
    # saying so) until the harness writes it. 128K x 8 keeps its 8 slots.
    stats = attempt_stats([rec("vllm", "submitted", 0, 600)])
    s, rates = stats["vllm"], {"vllm": {"BF16": RATE}}
    extras = {"rates": rates, "rates_128k_x8": rates, "concurrency": {"containers": 12, "vcpus_per_container": 4}}
    p = projections(stats, extras)["scenarios"]
    assert p["64K x 16"]["steps"]["P0b"]["point_hours"] == project_hours(s, RATE, 4800, 12)["hours"]
    assert p["128K x 8"]["steps"]["P0b"]["point_hours"] == project_hours(s, RATE, 4800, 8)["hours"]
    md = render_markdown(stats, [], extras, projections(stats, extras))
    assert "tool time shared over 12 containers (extras concurrency.containers)" in md
    assumed = projections(stats, {"rates": rates})
    assert assumed["scenarios"]["64K x 16"]["tool_containers"] == 16
    assert assumed["scenarios"]["64K x 16"]["steps"]["P0b"]["point_hours"] == project_hours(s, RATE, 4800, 16)["hours"]
    assert ("tool time shared over 16 containers (assumed until extras concurrency.containers is written)"
            in render_markdown(stats, [], {"rates": rates}, assumed))
    for bad in (0, "16", True, None):
        with pytest.raises(ValueError, match="concurrency.containers"):
            projections(stats, {"rates": rates, "concurrency": {"containers": bad}})


def test_bf16_throughput_ratio_for_the_optional_p0b_rule():
    # Audit gap 6.
    stats = attempt_stats([rec("vllm", "submitted", 0, 600), rec("llama.cpp", "submitted", 0, 600)])
    extras = {"rates": {"vllm": {"BF16": {"decode_tok_s": 800.0, "prefill_tok_s": 10000.0}},
                        "llama.cpp": {"BF16": {"decode_tok_s": 400.0, "prefill_tok_s": 4000.0}}}}
    p = projections(stats, extras)
    r = p["bf16_throughput_ratio"]
    assert r["decode"] == 0.5 and r["prefill"] == 0.4
    assert r["sources"] == {"llama.cpp": "synthetic", "vllm": "synthetic"}
    assert "llama.cpp BF16 / vLLM BF16: decode 0.50, prefill 0.40" in render_markdown(stats, [], extras, p)


def test_kl_scoring_hours_on_the_stated_machine():
    # Audit gap 7: 2 of 4 vLLM attempts resolve, each 4,000 tokens long (output + ideal uncached input). The
    # failed ones are longer, as runs that hit a limit usually are, and must not enter the P0b length.
    rows = [rec("vllm", "submitted", 0, 600)] * 2 + [rec("vllm", "turn_limit", 0, 600, out=5000, unc=20000)] * 2
    rows += [rec("llama.cpp", "submitted", 0, 600)]
    k = projections(attempt_stats(rows), {})["kl_scoring"]
    assert k["seconds_per_run"] == 43.0 and k["tokens_per_run"] == 16500 and "M5 Pro" in k["machine"]
    p0b = k["jobs"]["P0b"]
    assert p0b["runs"] == 2400 and p0b["tokens_per_run"] == 4000
    assert abs(p0b["point_hours"] - 2400 * 43 * 4000 / 16500 / 3600) < 1e-9
    # P0d's probe agreement needs the pool scored with the Q4_K_M probe too (review of eadb97e).
    q4 = k["jobs"]["P0b Q4_K_M probe"]
    assert q4["what"] == "unflagged successful runs, BF16 vs Q4_K_M" and q4["point_hours"] == p0b["point_hours"]
    assert abs(k["total"]["point_hours"] - 2 * p0b["point_hours"] - k["jobs"]["P0a"]["point_hours"]) < 1e-9
    p0a = k["jobs"]["P0a"]  # BF16 against each of the 4 quantized profiles, on every P0a BF16 trajectory
    assert p0a["pairs"] == 4 and abs(p0a["upper_hours"] - 900 * 4 * 43 * 4000 / 16500 / 3600) < 1e-9
    ec2 = projections(attempt_stats(rows), {"kl_scoring": {"machine": "g6e.16xlarge", "seconds_per_run": 20.0,
                                                           "tokens_per_run": 16500}})
    assert ec2["kl_scoring"]["machine"] == "g6e.16xlarge"
    assert abs(ec2["kl_scoring"]["jobs"]["P0b"]["point_hours"] - 2400 * 20 * 4000 / 16500 / 3600) < 1e-9
    md = render_markdown(attempt_stats(rows), [], {}, ec2)
    assert "## KL scoring" in md and "g6e.16xlarge" in md and "| P0b Q4_K_M probe | unflagged successful runs" in md


def test_kl_scoring_counts_unflagged_successes_and_reports_the_flagged():
    # Owner's agent on PR #7 (answer 4): a flagged run is never a candidate, so P0b KL scoring covers unflagged
    # successes only and the report counts the flagged ones. 2 of 5 vLLM attempts are unflagged successes; the
    # flagged success is longer and must not enter the scored length either.
    rows = [rec("vllm", "submitted", 0, 600)] * 2 + [rec("vllm", "turn_limit", 0, 600)] * 2
    rows += [rec("vllm", "submitted", 0, 600, out=9000, unc=30000, touched=True)]
    stats = attempt_stats(rows)
    k = projections(stats, {})["kl_scoring"]
    for job in ("P0b", "P0b Q4_K_M probe"):
        j = k["jobs"][job]
        assert j["runs"] == 4800 * 2 / 5 and j["flagged_not_scored"] == 4800 / 5 and j["tokens_per_run"] == 4000
        assert abs(j["point_hours"] - 4800 * 2 / 5 * 43 * 4000 / 16500 / 3600) < 1e-9
    md = render_markdown(stats, [], {}, projections(stats, {}))
    assert "P0b: 960 flagged successful runs are not scored (never selection candidates)." in md


def test_kl_machine_line_names_the_recommendation_pending_the_owner():
    # Owner's agent on PR #7 (answer 5): EC2 if the projected KL-scoring hours are under about 10 h, otherwise
    # the 48 GB Mac overnight; the owner decides.
    rows = [rec("vllm", "submitted", 0, 600), rec("llama.cpp", "submitted", 0, 600)]
    stats = attempt_stats(rows)

    def machine(seconds_per_run):
        p = projections(stats, {"kl_scoring": {"machine": "m", "seconds_per_run": seconds_per_run,
                                               "tokens_per_run": 16500}})
        return p["kl_scoring"]["machine_recommendation"], render_markdown(stats, [], {}, p)

    low, md = machine(1.0)
    assert low["recommend"] == "EC2" and low["hours"] < 10 and low["threshold_h"] == 10
    assert "the owner's agent recommends EC2 if the projected KL-scoring hours are under about 10 h" in md
    assert f"at {low['hours']:.1f} h (point total, on m) that is EC2, pending the owner" in md
    high, md = machine(1000.0)
    assert high["hours"] >= 10 and high["recommend"] == "the 48 GB Mac overnight"
    assert "that is the 48 GB Mac overnight, pending the owner" in md
    none = projections(attempt_stats([rec("vllm", "turn_limit", 0, 600)]), {})["kl_scoring"]
    assert none["jobs"] == {} and none["machine_recommendation"]["recommend"] is None


def test_resolve_rates_overall_and_excluding_flagged():
    # Audit gap 8 (tripwire amendment): 2 of 4 resolve, one of them flagged; excluding it, 1 of 3.
    rows = [rec("llama.cpp", "submitted", 0, 600, touched=True), rec("llama.cpp", "submitted", 0, 600),
            rec("llama.cpp", "turn_limit", 0, 600), rec("llama.cpp", "turn_limit", 0, 600),
            rec("llama.cpp", "submitted", 0, 600, profile="Q4_K_M")]
    s = attempt_stats(rows)["llama.cpp"]
    assert s["resolve_rate"] == 0.6 and s["resolve_rate_excluding_flagged"] == 0.5
    assert s["resolve_by_profile"]["BF16"] == {"overall": 0.5, "excluding_flagged": 1 / 3}
    assert s["resolve_by_profile"]["Q4_K_M"] == {"overall": 1.0, "excluding_flagged": 1.0}
    md = render_markdown({"llama.cpp": s}, probe=[], extras={}, projections={})
    assert "| llama.cpp | BF16 | 25.0% | 50.0% | 33.3% |" in md


def test_sampler_and_seed_check():
    # Audit gap 9: a server default leaking in (temperature 0.7, top_k missing) or an unset seed is caught.
    rows = [rec("vllm", "submitted", i, 600) for i in range(5)]
    assert attempt_stats(rows)["vllm"]["sampler_check"] == {"ok": True, "n_mismatch": 0, "n_seed_missing": 0,
                                                            "examples": []}
    rows[1]["sampler"] = {**SAMPLER, "temperature": 0.7}
    del rows[2]["sampler"]["top_k"]
    rows[3]["seed"] = None
    rows[4]["end_reason"], rows[4]["grading"] = "error", None  # sampled with the same request, so it counts
    rows[4]["sampler"] = {**SAMPLER, "top_p": 1.0}
    c = attempt_stats(rows)["vllm"]["sampler_check"]
    assert c["ok"] is False and c["n_mismatch"] == 3 and c["n_seed_missing"] == 1
    assert c["examples"] == [rows[i]["attempt_id"] for i in (1, 2, 4, 3)]  # mismatches first, then seeds
    assert "| vllm | 3 **FAIL** | 1 **FAIL** |" in render_markdown(attempt_stats(rows), [], {}, {})


def test_sampler_keys_are_exactly_the_six_pinned_names():
    # Owner's agent on PR #7 (answer 6): record.sampler keys are exactly temperature, top_p, top_k, min_p,
    # presence_penalty, repetition_penalty. An extra key (a server-side knob, or another name for one) fails.
    rows = [rec("vllm", "submitted", i, 600) for i in range(3)]
    rows[1]["sampler"] = {**SAMPLER, "frequency_penalty": 0.0}
    rows[2]["sampler"] = {**{k: v for k, v in SAMPLER.items() if k != "repetition_penalty"}, "repeat_penalty": 1.0}
    c = attempt_stats(rows)["vllm"]["sampler_check"]
    assert c["n_mismatch"] == 2 and c["examples"] == [rows[1]["attempt_id"], rows[2]["attempt_id"]]
