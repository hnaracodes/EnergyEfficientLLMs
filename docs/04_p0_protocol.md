# P0 protocol: agentic scope (v3)

Status: v3, 19 Nov 2026. Nothing here has been run. Time figures are estimates to be replaced by
the pilot. v2 applied the owner's agent's review (issue #1). v3 records the owner's decisions:
thinking on at 64K with 16 concurrent runs, one `g6e.16xlarge`, and the 20-task pilot as the only
spend in scope, not yet approved to start (owner, 19 Nov 08:02Z). It also adds llama.cpp to the
pilot, a CPU headroom rule, and a pilot exit report from which the owner sets the ceiling for the
rest of P0.

## What P0 must answer before any training

| Gate | Question | If it fails |
|---|---|---|
| P0a | Does stock PTQ hurt a 4B coding agent on our tasks? | drop the robustness half (B); energy (A) continues |
| P0b | Can we build a pool of at least 4 successful trajectories per task, at a usable size? | fallbacks (below) |
| P0c | Does the pool contain the tasks quantization breaks? | fallbacks, or claim generalization only |
| P0d | Does a low stability score mean the quantized model can follow that trajectory? | drop KL from selection; beta = 0 in RL |

Order: the 20-task pilot first. **The pilot is the only spend in scope, and it is not yet approved
to start (owner, 19 Nov 08:02Z).** It starts when the entry criteria under 'Before the pilot' pass
and the owner gives the go. The owner sets the ceiling for the rest of P0 from its exit report. The
rest of P0 runs as serial blocks, because one 48 GB L40S cannot hold a vLLM BF16 server and a
llama.cpp server at 16 x 64K at once (35-41 GB each), and the NVML counter covers the whole device:

1. P0b on vLLM.
2. One llama.cpp block per profile, `BF16` first: P0a (`heldout100` and `verified100`, x3) and, for
   `BF16`, `Q4_K_M` and `Q3_K_M`, P0c (`pool200`, x4). Each block ends with that profile's energy
   sub-run while the GPU is otherwise idle. P0c uses P0b's counts only for binning, so it can start
   before P0b's analysis is done.
3. KL scoring of P0b's successful runs at `BF16` and `Q3_K_M`, on a stated machine (Mac or EC2),
   with stated hours.
4. P0d.

Optional pilot rule: if llama.cpp `BF16` aggregate throughput is within an owner-set margin of
vLLM's, P0b also runs on llama.cpp and vLLM leaves P0. P0c keeps its own BF16 attempts, because
binning must use draws that are independent of the pass rates it compares.

## Fixed setup for all of P0

### Model and decoding

- `Qwen/Qwen3.5-4B`, text-only (`Qwen3_5ForCausalLM`).
- Thinking mode and context (owner decision D-P0-1): **thinking on, 64K context, 16 concurrent
  runs.** The model card advises at least 128K in thinking mode; the owner chose 64K knowing that.
  Check: if more than 10% of pilot attempts end at the 64K limit, the question goes back to the
  owner before the main P0b run. No silent switch.
- Sampling, pinned per request in both runtimes and logged: `temperature=0.6, top_p=0.95,
  top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0` (card, thinking-mode coding).
  Server defaults are never relied on.
- No greedy decoding (it risks endless repetition in thinking mode). Every "first run" is a sampled
  run with a fixed seed. The pilot logs how often attempts hit the turn or context limit.

### Episodes are token sequences

- The harness stores each episode as a token sequence. Each turn appends the exact generated tokens,
  then the template's rendering of the tool response plus the next generation prompt. Prompts are
  sent as raw tokens or text. The conversation is never re-rendered from messages, because the
  template trims reasoning, re-wraps it, and rebuilds tool calls in canonical form. Re-rendering
  would break the prefix cache, inflate processed tokens and joules, and make the teacher-forced
  score condition on a context the model never saw.
- Consequence: the scored sequence is exactly the generated one, and one forward pass over the
  final sequence gives every position.
- Masking: sampled tokens, including the whole `<tool_call>` block, count as model-written.
  Template-inserted tokens don't. The logit at position i predicts token i+1.

### Action format, parsing, repair

- One bash tool, as in `mantra-energy` paper-2: 180 s tool timeout, output capped at 10,000
  characters, `--network none`, 4 CPUs and 16 GB per container.
- The model emits Qwen3.5's native XML-style call
  (`<tool_call><function=bash><parameter=command>...`). Tool definitions go into the system prompt
  via the official template.
- Parsing: raw completions from every server, server-side tool parsers off, one parser in our
  harness for every runtime. The raw text of every failed parse is logged.
- Malformed-call categories, fixed before P0a: missing or misnested closing tag; wrong function or
  parameter name; text after `</tool_call>`; call truncated at the token limit; no call and no
  final answer.
- Repair: "malformed call, try again" arrives **inside a `<tool_response>`**, never as a user
  message. A user message mid-episode moves the template's last-query index, after which earlier
  turns' reasoning is dropped from context. Each repair is a turn and is charged to the attempt
  (`llm-agent-recovery` accounting).

### Runtimes, budgets, profiles

- SWE-bench Verified runtime from paper-2 as is. A new R2E-Gym runtime with the same interface
  (`begin / context / tool / finalize / verify`), graded by R2E-Gym's own tests. Datacurve and
  Terminal-Bench are not in P0.
- Scored budgets per attempt, all independent of the runtime: 64K context, 100 turns, a per-turn
  `max_tokens` cap fixed before the pilot (the limit behind the 'call truncated' category), and the
  180 s tool timeout. Hitting one is a failure, logged with its reason. Wall clock is only an
  infrastructure cap, about 3x the pilot's 99th-percentile attempt time (2-3 h until the pilot
  measures it). A hit is logged as `infra_timeout` and charged as a failure in the primary analysis,
  with a sensitivity analysis that excludes hits. Only real infrastructure faults (container or
  server crash) are rerun, as fresh attempts.
- Serving: vLLM for the large BF16 sampling in P0b; llama.cpp at one pinned commit for every
  comparison between precisions, BF16 included. llama.cpp splits `-c` across `-np` slots, so
  N slots at context C need `-c N*C` (16 x 64K = 1,048,576).
- Profiles: `BF16` GGUF, `Q8_0`, `Q4_K_M`, `Q3_K_M`, and `Q4_K_M` with `ssm_alpha`/`ssm_beta`
  overridden to Q8_0 (the gate-tensor ablation). Default `llama-quantize` flags, no imatrix in P0.
  The per-tensor type table of every file is saved with its results.

### Task sets (seeded ID lists saved before anything runs)

- Pool repos (6): pandas, numpy, pillow, orange3, aiohttp, tornado (3,887 tasks).
- Held-out repos (4, never in any pool): scrapy, pyramid, datalad, coveragepy (691 tasks).
- `pool200`: 200 tasks stratified across the pool repos. `heldout100`: 100 tasks stratified
  across the held-out repos. `verified100`: 100 random SWE-bench Verified tasks.
- `inrepo_test`: a seeded within-repo test list (for example 300 tasks; the owner sets the size with
  P1's cost in mind), stratified over the 6 pool repos and split by source commit where tasks share
  one. It is drawn first and excluded from `pool200`, from every pool, and from the pool projection.

### Logged per attempt

Resolved, failure reason, turns, tool calls, malformed calls by category, repairs, output tokens,
input tokens split into cached and uncached (computed from the stored token sequence under full
prefix reuse, with the server's own per-request cache counts logged next to it), generation seconds
and tool seconds, wall time, container CPU-seconds, the repo-state fingerprint after every tool call
(hash of `git status --porcelain` plus `git diff`, run outside the model's context), sampler
parameters, seed, and the token sequence with model-written spans marked. GPU joules are logged per
attempt only at concurrency 1. In concurrent runs, log raw counter readings with timestamps at
attempt start and end, plus run totals, marked `not_attributable`.

## P0b: pool feasibility (first)

- Pilot (the only spend in scope; not yet approved to start):
  - vLLM `BF16`: 20 `pool200` tasks x 4 attempts.
  - llama.cpp `BF16` and `Q4_K_M`: the same 20 tasks x 1 attempt each. About 6,350-7,750 of the P0
    attempts run on llama.cpp (P0a 3,000-3,600, P0c 2,400, P0d 800-1,600, energy sub-run about
    150), and its throughput with 16 slots of 64K on a hybrid model is the least certain number in
    this plan.
  - Proposed, pending owner approval: 3 `verified100` tasks x 1 attempt on llama.cpp `BF16`, because
    `verified100` is half of P0a and uses a different runtime path.
  - Concurrency: 14 containers at 4 vCPUs, leaving 8 vCPUs for the inference server, the harness
    and Docker. The measured container CPU load decides between 14-15 containers at 4 vCPUs or
    16 at 3 vCPUs for the rest of P0.
- **Pilot exit report** (what the owner reads to set the ceiling):
  - attempts per hour, separately for vLLM and llama.cpp
  - peak GPU memory per runtime
  - share of attempts hitting each limit: turns, 64K context (the 10% check), wall clock
  - malformed-call rate by category
  - container CPU load, and the concurrency chosen
  - aggregate decode and prefill tokens per second per runtime at 14-16 busy slots, from about 15
    minutes of synthetic load (llama-batched-bench, or vLLM's serving benchmark at 16 x 64K) plus
    the pilot runs
  - output, uncached and cached prefill tokens per attempt, and the prefix-cache hit rate per
    runtime, flagged if far below HERO's about 92%
  - the energy counter: readable and monotonic on the g6e L40S, its update period from a tight read
    loop, and idle watts with no process and with each server loaded
  - the replay match rate (P0d rule) on the pilot's successful runs, first halves re-run in fresh
    containers on CPU
  - the wall-clock hit share by runtime, profile and concurrency; the owner revisits the cap if any
    cell exceeds about 2%
  - projected instance-hours for each P0 step and the total, computed as attempts x tokens per
    attempt / aggregate token rate plus tool time (not wall-clock attempts per hour from one wave),
    for 64K x 16 and for 128K x 8 (a lower bound, since runs that hit 64K would have continued)
- Proposed, pending owner approval: a **training probe** in the first hour, before the pilot
  attempts, on synthetic sequences of 16K, 32K and 64K tokens (memory and step time depend on
  length, not content). It uses the chosen LoRA targets, gradient checkpointing, and the fla and
  causal-conv1d kernels confirmed in use rather than the torch fallback, and it reports peak GPU
  memory and seconds per step. The method depends on LoRA SFT over 64K-token thinking-mode runs,
  which nothing else in P0 tests. If it doesn't fit on 48 GB, P1 changes shape (truncated runs, a
  bigger GPU, or a per-turn loss). A few minutes on the box.
- Main: BF16 on vLLM, 16 attempts on each `pool200` task and each `heldout100` task (held-out
  trajectories are measurement only and never enter a pool).
- Report: successes out of 16 per task and per repo, tasks with at least 4 successes, and the
  projected pool size = (share of `pool200` with at least 4 successes) x (3,887 minus the size of
  `inrepo_test`).
- For RL: the share of tasks with 1 to 15 successes out of 16. GRPO learns only from tasks whose
  rollouts disagree, so this, not the 4-success pool, sizes the RL task set.
- Also report, once the energy model exists, an arm-overlap table among successful runs. For each
  pair of rankings (turns, output tokens, estimated `Q4_K_M` joules, HERO's reward cost of total
  tokens, and HERO's API-price cost of uncached + 0.1 x cached + 4 x output tokens), report the
  share of tasks where both pick the same run, with the chance rate (1 / number of candidates) next
  to it, and the median within-task joule regret of each pick (its estimated `Q4_K_M` joules over
  the task minimum). `fewest_joules` and `fewest_turns` stay separate arms unless they pick the same
  run in at least 85% of pool tasks or the median regret of the `fewest_turns` pick is below 5%. If
  the energy ranking matches the API-price ranking in at least 85% of tasks, the selection cost is
  described as a calibrated token-mix cost, not 'joules vs tokens'. Once scores exist (P0d), the
  same table covers every arm pair. Pairs above 85% are flagged as not separable, and their
  hypothesis is reported as untestable instead of training both.
- Pass, all three:
  1. Among `pool200` tasks with at least 1 success, more than half reach 4.
  2. The projected pool is at least 500 tasks (one selected trajectory per task per arm), which
     means at least 13% of `pool200` reach 4 successes.
  3. `heldout100` supports P2: from P0b's 16 attempts per task, the expected BF16 resolves per
     arm-seed run in the P2 confirmatory contrast are at least about 100 (mean pass@1 of about 15%
     or more on 691 tasks), with a simulated minimum detectable effect for joules per resolved task
     reported next to it. If it fails, the options (more attempts per task, 9B, a different held-out
     set) go back to the owner; no automatic switch.
- Fallbacks, in order: `min_candidates` 2; a teacher pool from DeepSWE-Preview (Qwen3-32B) on the
  same tasks, scored with the 4B BF16/quantized pair; the 9B model. Before using the DeepSWE-Preview
  pool, check DeepSWE's training task list against the 691 held-out IDs. If they overlap, `03`
  notes teacher-level exposure of the held-out repos, and SWE-bench Verified is reported next to any
  base-model comparison.

## P0a: headroom

- Run the 5 profiles on llama.cpp on `heldout100` and `verified100`, 3 seeded sampled attempts
  each (3,000 attempts).
- **Pre-registered primary test:** `Q4_K_M` vs `BF16` on `heldout100`. R = (processed tokens summed
  over all attempts / resolved attempts) at `Q4_K_M`, divided by the same at `BF16`. Processed
  tokens = output tokens plus all input tokens per turn (cache hits and misses), from the stored
  token sequence. This pooled form is 2610.32417 Eq. 4 (failed attempts charged), and it stays
  defined for tasks with 0 of 3 resolves, which a per-task ratio would drop. 5,000-draw bootstrap
  resampling tasks, keeping all attempts and both profiles of a task together.
  - **Pass:** R >= 1.15 and the 95% interval excludes 1.
  - **Fail (drop B):** the upper 95% bound is below 1.15.
  - **Inconclusive:** otherwise. Add 3 attempts per task at `BF16` and `Q4_K_M` on `heldout100` (600
    attempts) and re-test once. If still inconclusive, B continues, `Q3_K_M` is reported as the
    stress profile, and P1 measures B directly, since a false drop of B costs more than a false
    keep.
- Sensitivity: the geometric mean of per-task ratios over tasks resolved at least once under both
  profiles (the length factor), and the pooled resolve-rate ratio (the completion factor).
- Secondary (reported, not used for the gate): every other profile, turns per resolved task,
  `verified100`, and resolve rate per profile with a paired permutation test on per-task pass counts
  (0-3). At 100 tasks the resolve-rate interval is about plus or minus 8 points.
- Diagnostic: teacher-forced KL between BF16 and each quantized profile on BF16 trajectories, with
  the chunked wrapper at model-written positions only. Position effect, measured within each
  trajectory: KL in the late half vs the early half inside the same BF16-entropy bins, with the
  median paired difference across trajectories up to 64K. Task phase (exploration early, edits and
  tests late) is a known confound. Thinking tokens and `<tool_call>` tokens are reported separately,
  also within entropy bins.
- Energy sub-run, per profile at concurrency 1 with 45 s settling: 15 tasks x 2 attempts (the same
  count as 30 x 1, and it gives measured within-task pairs), with the same seeded task IDs for every
  profile. The counter is read when each request is sent and when its last token arrives, so
  generation windows and tool windows are separate. Tool-window energy stays in per-attempt totals
  but not in the token regression. Idle power is measured per profile with no process and with
  llama-server loaded using that profile's flags, before and after the sub-run, and every table
  gives raw and idle-subtracted joules.
- Per-turn energy model (for `fewest_joules` and the RL reward), fitted on generation windows only,
  separately per profile, with these terms: output tokens, uncached prefill tokens, context position
  at turn start (the cached prefill under full prefix reuse, replacing a separate cached-token
  term), uncached prefill x context position, output x context position, and an intercept for fixed
  per-turn overhead. Coefficients are non-negative. Terms are chosen by leave-one-attempt-out CV
  error, and CIs come from a bootstrap over attempts. Cached and uncached counts come from the
  stored token sequence under full prefix reuse, not from server counts. The sub-run reports how
  often llama.cpp's per-request cached-prompt counts differ from that ideal split; if misses occur,
  each attempt is pinned to one slot (`id_slot`) and re-checked. **Runs are ranked with the `Q4_K_M`
  coefficients**, the primary deployment profile. Report per profile: CV error, residuals, whether
  the model orders each measured within-task pair correctly, and the within-task agreement between
  the `BF16` and `Q4_K_M` coefficient sets. **Pre-registered fallback:** `fewest_joules` falls back
  to `fewest_turns` if the held-out median absolute error on attempt joules exceeds 10%, and the
  report says so. A forced-decode replay check is optional, only if the CV error is borderline.
- Batched coefficients: each profile's main llama.cpp block runs llama-server with `--metrics`,
  logging the energy counter and the server's cumulative prompt and predicted token counters
  every 1 s. Regress each window's joules on its prefill and decode tokens, and report how often pool
  rankings under the batched and concurrency-1 `Q4_K_M` coefficients pick the same cheapest run. If
  they agree on fewer than 85% of tasks, `fewest_joules`, the `qtrace` cost rank and `E_Q4` name
  their regime, and P2 reports energy in both.
- Concurrent runs report measured joules only as run-level totals, since energy can't be split
  between concurrent attempts.
- The SteerDistill GPTQ gap (-7.46 points at 4B) is a prior, not a substitute for this test.

## P0c: does the pool reach the tasks quantization breaks?

- On `pool200`, on llama.cpp: `BF16`, `Q4_K_M` and `Q3_K_M`, 4 seeded sampled attempts each.
- Breakage per bin, separately for `Q4_K_M` and `Q3_K_M`: L_b = sum over tasks in bin b of
  (llama.cpp `BF16` pass rate minus quantized pass rate), in the same runtime. The vLLM 16-attempt
  counts are used only to bin tasks by BF16 successes (0, 1-3, 4-8, 9-16). They are independent
  draws, so E[L_b] = 0 under no effect.
- Pass: L_total has a 95% task-bootstrap lower bound above 0, and L_(4+) / L_total >= 0.4, reported
  with its bootstrap CI. If L_total is not above 0, record 'no detectable breakage on pool200' for
  that profile and read it together with P0a. If L_total is above 0 but the share is below 0.4, use
  the P0b fallbacks or state that any effect works through generalization.

## P0d: does the score mean what we claim?

- Tasks: every `pool200` task with at least 2 successful BF16 trajectories (the test uses only the
  lowest- and highest-scoring run). Report the count and the score spread between the two. If fewer
  than about 50 qualify, run on what qualifies and report the minimum detectable effect; extra
  pool-repo tasks need owner approval and a stated attempt count.
- **Primary score (pre-registered):** decision-point KL with the `Q3_K_M` probe, the mean KL over
  decision positions: the `<tool_call>` token, the first 8 tokens of the `command` parameter value
  (after `<parameter=command>\n`), and the top 10% BF16-entropy positions among model-written
  tokens. Tokens fixed by the template are never decision points. `kl_mean` over all model-written
  tokens is secondary, reported on the same replays. `stable`, `qtrace` and the 2x2 arms use the
  score chosen by the rule below, which is the primary score unless it fails.
- **Candidate scores, frozen at three (2026-11-20, before any P0 data):** (1) `kl_decision`, the
  primary score above; (2) `kl_command`, the mean KL over the model-written tokens of each
  `command` value (`command_spans`; whole tool-call blocks are not used, because their
  near-deterministic wrapper tokens would make many short commands score low); (3) `kl_mean`.
  All use the `Q3_K_M` probe and come from one scoring pass (`qtrace/kl.py`). No other variant is added after seeing data. The reason
  for the ordering: on T1 proxy trajectories (issue #1, 4B and 9B, MLX), within-task rank agreement
  across probes rises from about 0.35-0.43 for `kl_decision` to 0.62-0.67 for `kl_command` and
  0.68-0.76 for `kl_mean`, while the within-task Spearman against BF16 entropy rises from about
  0.0-0.2 to 0.4-0.5 and 0.6.
- Before any continuation runs, report for each candidate on the P0b pool: the within-task Spearman
  against turns, written tokens, estimated `Q4_K_M` joules and BF16 entropy; the `Q3_K_M` vs
  `Q4_K_M` within-task rank agreement; and the turn-parity split-half reliability (within-task
  Spearman between the even-turn and odd-turn scores, Spearman-Brown corrected). Within-task
  values are means over tasks with at least 3 successful runs, with the median alongside. A
  candidate passes when |rho| <= 0.8 against all four, split-half reliability >= 0.5 and probe
  agreement >= 0.5 (a ranking that flips with the probe is not a property of the trajectory).
- The score is the first passing candidate in the order above. If none passes, H3 is answered
  negatively and the stability arms are dropped before P1 (the rule from
  `docs/02_research_plan.md`). The chosen score and every candidate's numbers are written here
  before continuations start.
- For each task, take the lowest- and highest-scoring trajectory and replay the first half of its
  turns in a fresh container. A replayed prefix is valid when every replayed command returns the
  same exit code and the final repo-state fingerprint matches the recording. Output text is compared
  after a fixed normalization (durations, timestamps, hex addresses, tmp paths, PIDs) only as a
  logged secondary flag. The continuation keeps the recorded context tokens. Report the pair drop
  rate.
- Continue each prefix with **both `BF16` and `Q3_K_M`, 4 seeded attempts each**. Before running,
  compute power from pilot and P0c pass rates against a minimum effect of interest of 10 points of
  DiD. Extra budget goes to 8 seeds per prefix or more tasks, never to middle-ranked trajectories.
  `Q4_K_M` continuations are a secondary result if budget allows.
- Test: DiD = (Q3 success minus BF16 success) on low-score prefixes minus the same on high-score
  prefixes, paired over tasks, expected positive; one-sided, task bootstrap. This controls for
  low-score prefixes simply being further along. Controls without extra replays: regress the
  per-task DiD on the chosen pair's differences in BF16 entropy at decision positions, turns and
  malformed calls, and report the adjusted estimate.
- Decision:
  - **Pass:** one-sided p < 0.05, and the adjusted estimate keeps its sign and at least half its
    size.
  - **Fail:** the upper 95% bound is below the 10-point minimum effect. The kill rule applies.
  - **Inconclusive:** anything else. The stability arms stay in P1 as exploratory, out of the
    headline and the RL default; beta stays as an ablation.
- Auto-bench found model-level KL did not predict coding-agent performance, so this is the gate that
  matters most.

## Kill rules (same as the gates table)

- P0a fails: drop the robustness half (stability arms, KL term). The energy half continues.
- P0d fails: drop KL from selection, and set beta = 0 in the RL reward.
- P0b or P0c fails: apply the fallbacks in order. If all fail, stop.
- Inconclusive outcomes follow the P0a and P0d decision rules; after a P0d fail, the owner's Stage 2
  choice and the A-only claim are in `03`, Kill rules.
- In every case, the constrained-decoding and calibrated-quantizer baselines still run.

## EC2 (decided: D-P0-2)

Facts checked by the owner's agent against the AWS g6e page and `config.json`:

- **Decided (D-P0-2): one `g6e.16xlarge`**: 1x L40S 48 GB, 64 vCPUs, 512 GiB RAM, about 1,900 GB of
  local NVMe (ephemeral).
- KV cache for Qwen3.5-4B (8 of 32 layers full attention, 4 KV heads, head_dim 256): 32 KiB per token,
  2.0 GiB per 64K sequence, 4.0 GiB per 128K sequence. DeltaNet state is 24-48 MiB per sequence.
  Weights about 9 GB. So 16 rollouts at 64K, or 8 at 128K, fit in 48 GB with headroom; 16 at 128K
  does not. Hence D-P0-1.
- Sandboxes: 16 containers at 4 vCPUs would use all 64 vCPUs and leave nothing for the server,
  the harness or Docker, so the pilot runs 14 and decides from measured load (see P0b).
- Disk: about 400 images (`pool200`, `heldout100` and `verified100` each need one per task) at
  0.4-1.3 GB compressed each, roughly 160-520 GB. The local NVMe covers P0. Pull with an
  authenticated Docker Hub account, in the background ahead of each block (Docker Hub rate-limits
  pulls), or mirror to same-region ECR. Keep results on EBS or S3, since NVMe is wiped on stop.
- If CPU becomes the bottleneck: keep the GPU box and add a separate CPU instance for containers.
  There is no 2-GPU g6e size, and `g6e.12xlarge` (4 GPUs, 48 vCPUs) has fewer vCPUs than the 16xlarge.
- 9B and the main experiment: revisit after P0.
- Prices: check the EC2 console before approving. Not checked here.

### Energy boundary on EC2

GPU via NVML (Zeus), read every 0.1 s. Virtualized instances generally don't expose CPU energy
counters, so every energy table states "GPU only" and reports container CPU-seconds next to it.
2610.32417 cites a 41-45% system-energy gap for GPU-only telemetry, and the paper should say so.

## Compute estimate (replace after the pilot)

About 11,000-12,500 attempts: P0b 4,800; P0a 3,000, plus 600 if the top-up runs; P0c 2,400; P0d
800-1,600, set by its power calculation; and the energy sub-run, about 150 attempts at concurrency 1
(roughly 1 day at the 10-minute guess). KL scoring of the pool at `BF16` and `Q3_K_M` comes on top,
with its machine and hours stated. At 14-16 concurrent attempts and a guessed 10 minutes each, that
is roughly 6-7 days on one machine; treat it as a floor. The pilot exit report replaces this with
projections from token rates.

## Before the pilot (ordered, free work first)

1. Owner: assign the unowned components (`03`, Who does what), and decide on the training probe and
   the 3 extra Verified attempts.
2. Free, on the Macs: the episode store, XML parser and repair path, unit-tested on real Qwen3.5-4B
   completions; the R2E-Gym runtime and the Verified adaptation; per-tool-call repo-state
   fingerprints; a Rosetta smoke test on 3-5 R2E-Gym tasks and 1 Verified task with the real parser,
   repair path and grader; the chunked KL wrapper (PyTorch `logits_to_keep`, then the llama.cpp
   low-level API), checked equal on one 16K trajectory; MLX cross-check scoring; a GGUF round trip
   on a LoRA-merged Qwen3.5-4B (`convert_hf_to_gguf`, then `llama-quantize` to `Q8_0`, `Q4_K_M`,
   `Q3_K_M` and the gate override, per-tensor tables, a Metal smoke test), after the T1 queue
   finishes and with LordKarV's go-ahead; the probe script, tested on a tiny model; the analysis
   scripts, on synthetic data; the frozen seeded lists, including `inrepo_test`.
3. Entry criteria, all checkable off the box: the smoke test passes; pilot images are reachable
   (authenticated Docker Hub pulls or a mirror); S3 sync is tested; the GGUFs build at the pinned
   commit. Then the owner gives the go.
4. First hour on the box, stop if any check fails: server logs echo the pinned sampler parameters; a
   1-attempt run shows per-turn energy reads; peak memory per runtime is recorded; the synthetic
   load per runtime and, if approved, the training probe run before the pilot attempts.
5. The pilot attempts, the replay check and the exit report. Then the owner sets the P0 ceiling,
   settles the 64K check, chooses the P0b runtime and plans the Stage 2 (S2, `03`) instance.

## Selection and training rules for P1

Moved here from `03` (Method, Baselines, Who does what) so the plan stays short. Fixed before P1
starts.

### Selection (`03`, Method step 3)

- Every arm picks exactly one run per task from the same candidates.
- `fewest_turns` ties: fewest output tokens. `fewest_joules` ties: fewest turns.
- `qtrace`: ties in the rank mix take average ranks, then a seeded tie-break on the combined score.
- 2x2 length controls: split on the cost measure `qtrace` uses as its cost rank (turns, or
  estimated `Q4_K_M` joules if P0b switches it). The lower floor(n/2) runs by that cost are short,
  with ties broken by output tokens, then by a shuffle seeded with the arm seed. Each arm's mean
  turns, written tokens and estimated joules are reported, and the per-task length difference is a
  covariate in the H2/H3 analysis.
- Remaining ties are broken at random with the arm's seed. Seed s sets both the selection RNG
  (including the `random` draw) and the training RNG.
- The current `qtrace/select.py` breaks ties by input order and splits on `n_tokens`; the
  migration below fixes both.

### Training (`03`, Method step 4)

- The loss is the sum over model-written tokens divided by a fixed constant set in the config.
- Logged per arm and seed: supervised tokens seen, and drift from the base model (BF16 resolve-rate
  change, KL to base on held-out text).
- If the headline arm saw more supervised tokens than `random`, one token-matched rerun of that
  contrast is added; the H3 comparisons get an optional token-matched run in P1.

### `random_fq` baseline (`03`, Baselines)

Fake-quant 4-bit group-wise weights in the forward pass, applied to the merged weight (W + BA) with
a straight-through estimator to the LoRA parameters and a skip list from the saved `Q4_K_M`
per-tensor table, then stock `llama-quantize` like every arm.

### Migrating `scripts/04_train_lora.py` (`03`, Who does what)

It was built for single-turn GSM8K: it drops examples over `max_seq_len` (768), masks only the
prompt, never enables gradient checkpointing, and the LoRA targets in `configs/base.yaml` skip the
DeltaNet mixers in 24 of 32 layers. The migration needs:

- the scorer's loss mask over every span the model didn't write;
- no length drop, or an explicit truncation policy that reports the truncated token share per arm
  and prefers per-turn loss chunks to cutting the tail;
- gradient checkpointing;
- explicit LoRA targets (add `in_proj_qkv`, `in_proj_z` and `out_proj`, and decide on
  `in_proj_a`/`in_proj_b`);
- steps set by epochs over the pool, not GSM8K's 1,500 x 16;
- the fla and causal-conv1d kernels installed and confirmed in use;
- the tie and split rules for `qtrace/select.py` from the Selection rules above.

## Analysis plan for P1 and P2

Moved here from `03` (Metrics and Stage 2) so the plan stays short. Everything in this section is
frozen before P1 starts, except the Stage 2 subsection.

### Primary test (goal A)

- Endpoint: GPU joules per resolved task at `Q4_K_M` on the 4 held-out repos, failed attempts charged
  to resolved ones (2610.32417 Eq. 4).
- Unit: the attempt. The `Q4_K_M` per-turn energy model, frozen at the end of P0a, is applied to each
  attempt's token mix.
- Test: a paired bootstrap over tasks, stratified by repo, of the P1-chosen headline arm vs `random`.
- Guard: the arm's `Q4_K_M` resolve rate is not worse than `random`'s by more than a margin set from
  P0b's measured standard error. The guard applies to every hypothesis: an arm that wins on joules
  but fails it does not count.
- Frozen here before P1 starts: the test statistic, attempts per task per seed (1) and the margin.

### Estimator check

`fewest_joules`, the joules-ranked `qtrace` and the RL reward optimize the same energy model that
scores them, so the estimate is checked against measurement.

- 30 held-out tasks x {headline arm, `random`, base} x 1 seed also run at `Q4_K_M`, concurrency 1,
  with the counter read per turn as in P0a.
- The estimate stands if the 95% CI of the between-arm difference in mean log(measured / estimated)
  per attempt lies within +/-5%. Otherwise only the measured subset is claimed.
- Processed tokens per resolved task are co-reported as an estimator-free check.

### Seeds, repos and multiplicity

- The primary analysis averages each task over the 3 seed models. Per-seed estimates, a two-stage
  (seed, then task) bootstrap and per-repo effects are reported.
- Inference is conditional on the 4 held-out repos; the sign should hold in at least 3 of 4.
- Secondary analyses (other profiles, H3/H4 controls, the within-repo split, the external sets) use
  Holm within each hypothesis family, or are labelled exploratory.

### Energy upper bound

The GPU-only number is reported in every table. An energy win (H1b, H5, Stage 2) is claimed only if
it also holds under the upper bound: GPU joules + container CPU-seconds x the per-vCPU share of the
host CPU's published TDP, pooled per resolved task. Both are reported.

### Stage 2 reward and compute (`03`, Stage 2)

A draft, like `03` Stage 2, and fixed before S2 starts rather than before P1.

- `KL(y)` decision positions, by a fixed rule: the `<tool_call>` token, the first 8 tokens of each
  `command` value, and positions above a fixed BF16-entropy threshold set from the P0b pool.
  If P0d chose `kl_command`, `KL(y)` sums over the command-value tokens instead; if it chose `kl_mean`,
  over all model-written tokens, which overlaps the energy term most.
- `KL(y)` is a sum over those positions, never a per-token or per-decision mean, which can be
  lowered by adding predictable tokens or easy actions. The P0d selection score can be a mean
  because it ranks fixed runs, which cannot pad themselves. A sum grows with turns, so it overlaps
  the energy term; the within-group Spearman of `KL(y)` with turns and with `E_Q4` is reported.
- Each step, the rollouts' token mix (output share, context positions) is checked against the energy
  model's calibration range. Outside it, re-measure on llama.cpp `Q4_K_M` and refit.
- Compute: one L40S at 16 tasks x 8 rollouts per step is at least 80 minutes of rollouts per step.
  The training pass over 128 sequences of up to 64K tokens adds about 25 minutes by FLOP count
  (128 x 32K tokens x 6 x 4e9, at 40% of the L40S's dense BF16 peak), so a step takes about 105
  minutes or more and 100 steps take about 7-8 days per run, before model swaps and
  re-quantization.

## Owner decisions

- D-P0-1: thinking on, 64K, 16 concurrent runs (with the 10% context-limit check).
- D-P0-2: one `g6e.16xlarge`.
- Spending: the 20-task pilot only, not yet approved to start (owner, 19 Nov 08:02Z). The ceiling
  for the rest of P0 is set from the pilot exit report.
- Open:
  - who builds the pilot harness and the other unowned components (`03`, Who does what);
  - the claim regime for energy (`03`, Metrics);
  - the P2 secondary tier, options a-c (`03`, Cost);
  - the order of S2 relative to P2 (`03`, Phases);
  - the P1 selection set (`03`, Phases);
  - the size of `inrepo_test`, the within-repo list P1 runs on (Task sets);
  - the training probe and the 3 extra Verified attempts in the pilot (P0b).
