# Related-work log

A dated record of papers checked against the Q-TRACE agentic plan (`docs/03`), newest first.
`docs/01` is the 13 Nov 2026 survey for the GSM8K plan; the neighbour table in `docs/03` ("What
others have done") is the summary for the paper. This file holds the checks in between, so the
related-work section and the preprint timing (`docs/03`, Phases) rest on a record, not on memory.

Each entry gives: the date checked, the paper, how much of it was read, what it does, where we
differ, and any change it asks of the plan. "Abstract only" or "grep of the full text" means just
that; check the paper before citing a number from such an entry.

The claim to protect (from `docs/03`): no published work changes only the fine-tuning data, keeps
stock quantization, and reports joules per resolved task on SWE-style agent tasks. Each sweep
states whether it still holds.

## 22 Nov 2026 sweep

Queries: arXiv API, submissions since 1 Nov 2026, sorted by date: quantization with agent and
energy; quantization with software-engineering agents; quantization with coding agents;
quantization with fine-tuning, robustness and data; quantization with trajectory selection;
SWE-bench with energy; SWE-bench with RL and efficiency; joules with LLM reward; tool calls with
quantization. Nine queries, 25 newest results each. Four papers are relevant.

The claim still holds. NariQ comes closest: it uses BF16 agent traces, but to calibrate a new
quantizer, not to choose fine-tuning data, and it reports no energy.

### NariQ: Sub-3-Bit Quantization for Long-Horizon Agentic Tasks (arXiv 2611.26157, 19 Nov 2026)

Read: abstract, plus a grep of the full HTML text for method, results and size.

What it does: a trellis-coded PTQ format for Qwen3.8-27B, calibrated on tokens the BF16 model wrote
in agent sessions. The loss is KL(BF16 || quantized) on model-written positions only, with tool
outputs kept as context. Bit widths are allocated by each module's share of that loss. At 3.0 and
2.5 bits per weight it matches BF16 on DeepSWE (113 tasks) and Terminal Bench 2.1. At 2.0 bits it
trails, with the largest drop, 11.7 points, on AIME. It ships as GGUF files that need the authors'
llama.cpp fork, tuned for one GPU generation (SM120). It reports tokens per second, not energy.

Where we differ: NariQ changes the quantizer and its calibration data; we change only the
fine-tuning data and keep stock quantizers unmodified. Its objective is the same quantity as our
stability score (per-position KL on model-written tokens, tool-output context included), which
supports the `model_mask` choice in `qtrace/kl.py`. A reviewer may ask why we do not simply use a
NariQ-style quantizer.

What it asks of the plan:

1. Effect size at 4 bits. NariQ states that "near 4 bits per weight, quantization already preserves
   such agentic performance" on the 27B model: NVIDIA's NVFP4 is within 1.6 points of BF16 on
   Terminal Bench, and Unsloth's UD-IQ4_XS solves 90% as many DeepSWE tasks as BF16. Damage starts
   below about 3.5 bits. This may not transfer to 4B (SteerDistill, 2611.13410, reports stock GPTQ
   losing 7.46 points at Qwen3.5-4B), but it is a direct risk that our primary `Q4_K_M` profile
   leaves too small a BF16 gap to move. The P0a pilot measures that gap at 4B. If it is small,
   `Q3_K_M` (already in the pinned table under `data/quant_tables/`) should become the main test.
   That is an owner decision; raised in issue #1.
2. A stronger quantizer baseline. If we report sub-4-bit results, NariQ is the PTQ-side method to
   compare with or combine with. Its fork targets SM120, so it is out of scope for P2 on our
   hardware. Cite it, and say so.
3. Their DeepSWE numbers use Claude Code with one attempt per task; Unsloth's use mini-swe-agent
   with at least three passes. Harness gaps of this size are why we compare arms only within our
   own harness.

### Beyond Accuracy: How Quantization Affects Coding Agents (arXiv 2611.02368, 2 Nov 2026)

Read: abstract, plus a grep of the full text.

What it does: BF16 vs 4-bit AWQ and GPTQ on Qwen2.5-Coder-Instruct 7B, 14B and 32B and
Devstral-Small-2507, on OJBench, three seeds, up to five executor calls per problem, served by vLLM.
No significant accuracy loss, but behaviour shifts: AWQ takes Qwen-7B one-shot solves from 17 to 0
and loop-based solves from 10 to 25. Under GPTQ, one model's tool-use rate fell from 8.7% to 1.2%.
Devstral's failures move from runtime errors to wrong answers and timeouts. It measures neither
energy nor tokens per solve.

Where we differ: it measures and we try a fix; competitive-programming problems with one tool, not
repository repair.

What it asks of the plan: nothing new. It is independent support for judging arms on trajectories
(turns, tool calls, failure modes) as well as resolve rate, which P0d and the attempt record
already do. Cite it with 2610.32417 for "accuracy hides behavioural change".

### Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks (arXiv 2611.06257, 4 Nov 2026)

Read: abstract, plus a grep of the full text.

What it does: 12 local configurations (3 generations, dense vs MoE, 3 precisions, down to 4B) on
10 CyberGym tasks, 3 runs each. "Quantization did not change what the models solved in any matched
pair, but it changed how fast they got there." In one case BF16 hit the 7200 s cap on two hard tasks
that FP8 solved because it ran faster. It reports wall clock, tokens and tool calls per solved
task, not energy.

Where we differ: it measures and we try a fix, on a different task family.

What it asks of the plan: nothing new. It shows how a wall-clock cap turns speed into resolve rate.
`docs/04` already treats wall clock as an infrastructure cap only, logs hits as `infra_timeout`,
reports the hit share, and runs a sensitivity analysis without them.

### Toward Sustainable Agentic AI: Carbon Tradeoffs Across Iterative Self-Refinement and Tool-Augmented Coding (arXiv 2611.12128, 9 Nov 2026)

Read: abstract only.

What it does: estimates carbon from output tokens for API models (GPT-4.1-mini, Claude Opus 4.6,
GPT-5.3-codex) on SWE-bench Lite, and for self-refinement on MATH. It notes that prompt tokens are
over 99% of tokens in the agent study and are not charged, so its figures are lower bounds.

Where we differ: we measure joules on our own hardware, and the P0a energy model has uncached and
cached prefill terms (`docs/04`).

What it asks of the plan: nothing new. A citation for "output-token accounting misses most agent
cost".

## 19 Nov 2026

Papers checked while writing `docs/03`, summarized in its neighbour table: SteerDistill
(2611.13410), HERO (2609.38885), the Texas A&M agent-energy pair (2610.32417, 2610.41672), and
Measured Joules, Learned Routes (2609.23085). The full read of 2610.32417, with its accounting
equations and the changes it suggested, is issue #1 comment 7001323852.

The RL-for-efficiency survey behind the Stage 2 design (length-penalty RL, adaptive compute,
routing, energy as a reward) was run the same week and not committed. Its conclusions are the
Stage 2 rows of the `docs/03` table: our reward shape reuses Arora and Zanette's (2502.04463) and
DLER's (2510.15110), and no length-RL paper reports joules or evaluates its model after
quantization.

## 13 Nov 2026

The GSM8K-era survey: `docs/01_literature_and_gaps.md`. Its gaps G1 (demonstration selection as a
lever for PTQ robustness) and G4 (energy unmeasured) carried over to the agentic plan.

## How to add a sweep

Add a dated section at the top, list the queries, and give one entry per relevant paper in the
format above. If a paper changes the plan, raise it in issue #1 as well; this file records the
check but does not replace that discussion. If a paper closes the claim at the top, say so in the
sweep's opening lines and in issue #1 the same day.
