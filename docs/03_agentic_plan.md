# Q-TRACE, agentic version: the plan

Status: 19 Nov 2026. Agreed in issue #1 between the owner's agent and LordKarV's agent, with the
owner's decisions. Open decisions are listed in `04`, "Owner decisions". This plan replaces the
GSM8K plan in `docs/02_research_plan.md` for the main study. The detailed protocol for the first
experiments is `docs/04_p0_protocol.md`.

## The idea in one paragraph

Small coding agents are usually run quantized, to save memory and energy. Quantized agents make more
malformed tool calls and take more turns, so they can end up using more energy per solved task, not
less (arXiv 2610.32417, 2610.41672). We test a cheap training-side fix: from many successful runs of
the full-precision model, pick the runs that a quantized copy of the model can also follow, fine-tune
on those, then quantize with ordinary stock tools. If it works, a deployer gets a model that keeps its
accuracy and its energy savings after quantization with unmodified stock quantizers (llama.cpp GGUF
and AWQ tested). Stage 1 uses a quantized copy only to choose training data, with no
quantization-aware training or teacher loop; Stage 2 adds a quantized probe to the reward.

## The goal (owner decision, 19 Nov 2026): A + B

- **A:** train the agent to use less energy per fixed bug, measured in joules, not just tokens.
  Selection and the RL reward use estimated joules; evaluation checks them against measured joules
  (Metrics).
- **B:** keep it efficient after stock quantization.

Energy is the headline. Robustness to stock PTQ is half of the goal, not the whole story. Two stages:
selection plus SFT (cheap, and the baseline RL must beat), then RL. RL is in scope.

## The question

Can a small coding agent be trained to use fewer joules per resolved task, and keep that advantage
under stock post-training quantization? Stage 1 asks whether choosing **which** successful runs to
fine-tune on gets there; Stage 2 asks whether RL with an energy and stability reward gets further.

Hypotheses, each able to fail:

- H1a: fine-tuning on the fewest-turn successful runs lowers joules per resolved task at `Q4_K_M`
  compared with random successful runs. Secondary: a smaller `Q4_K_M`/`BF16` ratio of tokens per
  resolved task.
- H1b: fine-tuning on the lowest-energy successful runs (`fewest_joules`) lowers joules per resolved
  task at 4-bit compared with random successful runs.
- H2: picking runs by stability under quantization (`qtrace`) lowers joules per resolved task at
  `Q4_K_M` more than fewest-turns alone. Secondary: a smaller `Q4_K_M`/`BF16` ratio.
- H3: the stability score is more than length, entropy or clean formatting: `stable` beats
  `low_entropy` and `clean_format`, and `short_stable` beats `short_unstable` (likewise for long).
- H4: the differences between arms survive grammar-constrained tool calls, so the gain is not just
  formatting.
- H5: `qtrace`, with the cost rank chosen at P0b, uses fewer joules per resolved task at `Q4_K_M`
  than `fewest_joules`.

Every hypothesis is subject to the resolve-rate guard in Metrics: an arm that wins on joules but loses
resolve rate beyond the margin does not count.

## What others have done, and where we differ

| Work | What it does | Our difference |
|---|---|---|
| SteerDistill (2611.13410) | trains the quantized agent with a teacher; at Qwen3.5-4B, stock GPTQ loses 7.46 points and SteerDistill recovers 6.73 of them | we train in full precision and keep unmodified stock quantizers (llama.cpp GGUF and AWQ tested); no teacher loop; coding tasks |
| 2610.32417, 2610.41672 (Texas A&M) | measure that quantized agents cost more energy per completed task, including SWE-bench Verified | they measure, we try a fix; we reuse their harness and energy accounting |
| HERO (2609.38885) | RL that keeps tokens near the base model's level (25-48% fewer than plain GRPO) while raising resolve rate; its reward uses total tokens | full precision only; no quantization; run 1 of our Stage 2 ladder |
| Munkhbat et al. 2025 | fine-tune on the shortest correct solution | math, never quantized; our `fewest_turns` arm |
| OnQ, on-policy distillation | train the quantized model with a teacher | same distinction as SteerDistill |
| Arora and Zanette (2502.04463); DLER (2510.15110) | length-penalty RL | our reward shape reuses theirs |
| QeRL | RL with a quantized model (-2.44 at Qwen3.5-4B in SteerDistill's Table 1) | our "roll out the quantized copy" ablation is the QeRL-like design |
| auto-bench | model-level KL did not predict coding-agent performance | why P0d tests the score on behavior |
| Catalan-Tatjer et al. (2510.06213) | training dynamics change PTQ robustness | we change only the fine-tuning data |
| Measured Joules, Learned Routes (2609.23085) | measured GPU joules in an RL reward, for a router | nearest energy-reward precedent; not an agent policy |

What is still open: no published work changes only the fine-tuning data, keeps stock quantization,
and reports joules per resolved task on SWE-style agent tasks. For Stage 2: no RL work on SWE agents
uses an energy-calibrated cost in the reward or evaluates the trained agent after stock PTQ.

## Setup

- **Model:** Qwen3.5-4B, with 9B for the scale check. Thinking on, 64K context, 16 concurrent
  runs (owner decision; the card advises 128K, so the pilot checks how often runs hit 64K).
- **Training tasks:** R2E-Gym (the environment set of Agentica's DeepSWE recipe), 6 repos: pandas,
  numpy, pillow, orange3, aiohttp, tornado.
- **Primary test:** 4 R2E-Gym repos never used in training (scrapy, pyramid, datalad, coveragepy;
  691 tasks). A within-repo split (`inrepo_test`, a seeded list carved out of the 6 pool repos; see
  `04`, Task sets) is reported next to it, to show how much is repo memorization.
- **External checks:** SWE-bench Verified (500), Datacurve DeepSWE (113, reported as Python vs
  non-Python), Terminal-Bench 2.1 (89, on its own harness).
- **Quantization profiles:** llama.cpp GGUF at one pinned version: `BF16`, `Q8_0`, `Q4_K_M`
  (primary), `Q3_K_M`, plus `Q4_K_M` with the two DeltaNet gate tensors kept at 8-bit. MLX and AWQ
  are secondary. Our uniform round-to-nearest quantizer is a harsh ablation only.
- **Harness:** one bash tool in Qwen3.5's native XML call format; episodes stored as raw tokens;
  one parser for every runtime; "try again" sent as a tool response. Built on the MIT-licensed
  `mantra-energy` paper-2 harness, with new plugins for R2E-Gym and Datacurve.
- **Hardware:** one EC2 `g6e.16xlarge` (1x L40S 48 GB, 64 vCPUs, 512 GiB) for sandboxes, runs and
  energy; all task images are amd64-only. The Macs do score development, MLX cross-checks and
  harness smoke tests.

## Method

1. **Pool:** run the full-precision model 16 times on each training task. Keep tasks with at least
   4 successful runs.
2. **Score:** for each successful run, measure how much a quantized copy (`Q3_K_M`) disagrees with
   the full-precision model on the tokens the model wrote (teacher-forced KL). The primary score is
   decision-point KL, defined and pre-registered in `04` P0d; the whole-run average (`kl_mean`) is
   reported as a secondary variant.
3. **Select:** build one training set per arm by picking one run per task:
   - `random`: any successful run
   - `fewest_turns`: the successful run with the fewest turns
   - `fewest_joules`: the successful run with the lowest estimated energy. Energy is estimated from
     each run's token mix with a per-turn model calibrated on measured joules (see `04`, P0a energy
     sub-run). If the energy model fails its pre-registered accuracy check (`04`, P0a), this arm
     falls back to `fewest_turns`.
   - `stable`: the run with the lowest primary score
   - `qtrace`: a mix of fewest turns and most stable (rank mix weighted by `select.alpha`, fixed at
     0.5 before P1). If `fewest_joules` and `fewest_turns` pick different runs under the `04` P0b
     overlap rule, `qtrace` uses estimated joules as its cost rank instead of turns (decided from the
     P0b data).
   - `low_entropy`: the run the full-precision model was most certain about (control)
   - `clean_format`: the run with the fewest malformed tool calls (control)
   - 2x2 length controls: `short_stable`, `short_unstable`, `long_stable`, `long_unstable`, split
     into short and long halves on the cost `qtrace` ranks by. They answer "did `stable` just pick
     shorter runs?"

   Every arm picks exactly one run per task from the same candidates, so arms differ only in which
   run. Tie-breaks, the 2x2 split, seeding and the fixes to `qtrace/select.py` are in `04`,
   "Selection and training rules for P1".
4. **Train:** LoRA fine-tune on each set, same number of steps for every arm, 3 seeds. Supervised
   tokens and drift from the base model are logged per arm and seed, with a token-matched rerun if
   the headline arm (the arm P1 picks, see Phases) saw more tokens than `random`; the loss and
   logging rules are in `04`, "Selection and training rules for P1".
5. **Quantize and evaluate:** merge, quantize with each stock profile, evaluate everything.

## Stage 2: RL (draft, for discussion)

- **Start from** one fixed checkpoint of the arm P1 chose on the within-repo split by the primary
  endpoint. Every RL variant starts from it with the same task set, rollouts per step and step budget.
- **Reward per rollout:** `r = 1[resolved, clean end] x (1 - lambda x clip(E_Q4(y) / E_ref - 1, -1, 1) - beta x clip(KL(y) / KL_ref - 1, -1, 1))`,
  with `lambda + beta < 1`. Every failure gets 0.
  - `E_Q4(y)`: the run's estimated joules from the per-turn energy model with `Q4_K_M` coefficients
    (non-negative, see `04`). `E_ref`: the mean over the resolved runs in the same rollout group.
  - `KL(y)`: the sum of teacher-forced KL between the policy and its quantized copy over decision
    positions (tool calls, the start of each command, high-entropy positions) chosen by a fixed rule
    (unlike the P0d selection score, which ranks fixed runs, a reward can be gamed by padding, so it
    uses a sum). `KL_ref`: the group mean over resolved runs. The probe is a merged copy,
    re-quantized to `Q3_K_M` every k steps.
  - Every resolved run then scores above every failure, and all-fail groups give no gradient, so the
    share of mixed groups sizes the RL task set (`04`, P0b). With one resolved run in a group, both
    terms are 0.
  - The exact decision-position rule, the overlap report between `KL(y)` and the energy term, and
    the per-step check of the energy model's calibration range are in `04`, "Stage 2 reward and
    compute".
- **Robustness alternative:** roll out the quantized copy as well and reward its episodes. That
  matches the goal directly, but it doubles sandbox cost, and updating full-precision weights from
  quantized rollouts is off-policy. HiFloat4 (2607.26515) reports that rollout-training precision
  mismatch is the main failure mode in low-precision GRPO. Default: the KL term; the alternative is
  an ablation if budget allows.
- **RL ladder: three nested runs from the same init, under our budget.** (1) The reward above with
  HERO's cost (total tokens) in place of `E_Q4` and beta = 0, labelled "HERO-style token cost under
  our budget". (2) The same with `E_Q4` (joules vs tokens). (3) Run 2 plus the KL term (plus
  stability). "RL beats Stage 1" is run 3 vs its own init on the primary endpoint, paired over
  held-out tasks; "joules vs tokens" is run 2 vs run 1; "plus stability" is run 3 vs run 2. If P0d
  fails, run 3 is dropped. List the setting differences from HERO's paper (group 8 vs 16, 100 vs 200
  turns, 64K vs 131K context), and cite its published 4B result (32.8% to 40.0% on SWE-bench
  Verified at 4.5M tokens) as an outside anchor. If HERO's checkpoints are public, evaluate them at
  every profile.
- **Compute is the open risk.** On one L40S at 16 tasks x 8 rollouts per step, a step takes about
  105 minutes or more (rollouts plus the training pass; arithmetic in `04`, "Stage 2 reward and
  compute"), so 100 steps take about 7-8 days per run, before model swaps and re-quantization.
  DeepSWE used 512 rollouts per step for 200 steps on 64 H100s. Stage 2 is sized after P0 and may
  need a bigger instance.

## Baselines

- The untouched base model, quantized the same way.
- The base model with thinking off, at `BF16` and `Q4_K_M`, on `heldout100` (100 held-out-repo
  tasks, `04`; about 600 attempts at 3 per task).
- A post hoc turn-cap curve for the base model and the headline arm, from logged per-turn tokens
  (resolved after the cap counts as a failure; no new runs). A is claimed only if the headline arm
  lies below the base model's whole joules-vs-resolve-rate frontier, not just below its 100-turn
  point.
- `random_fq`: the `random` set trained with fake-quant 4-bit weights in the forward pass (mechanics
  in `04`, "Selection and training rules for P1"), then stock `llama-quantize` like every arm.
  Required in P2 at `BF16` and `Q4_K_M`. It answers "why select data instead of training with
  noise?"
- A cross-family 4-bit profile, required for the headline arm, `random` and the base model: AWQ int4
  on vLLM, keeping `in_proj_a`/`in_proj_b` at bf16. GPTQ W4A16 g128 is added only if a P0 smoke test
  shows vLLM runs Qwen3.5-4B's DeltaNet layers under it.
- Grammar-constrained tool calls at decode time (tests H4), crossed with the arms in the P2 tier that
  tests H4.
- A calibrated quantizer: llama.cpp importance matrix computed on the pool's trajectories.
- RL: HERO-style token cost, as run 1 of the Stage 2 ladder.
- Optional if budget allows: a direct SteerDistill run on our tasks.

## Metrics

The exact test, the estimator check, the bootstrap and multiplicity rules and the energy upper bound
are in `04`, "Analysis plan for P1 and P2", frozen before P1 starts.

- **Primary endpoint (A, pre-registered):** GPU joules per resolved task at `Q4_K_M` on the 4
  held-out repos, failed attempts charged (2610.32417 Eq. 4), for the P1-chosen headline arm vs
  `random`, estimated per attempt with the `Q4_K_M` energy model frozen at the end of P0a. Guard: the
  arm's `Q4_K_M` resolve rate is not worse than `random`'s by more than a margin set from P0b's
  measured standard error.
- **B, always next to it:** the `Q4_K_M`/`BF16` ratio of processed tokens and of joules per resolved
  task (the P0a quantity), and the quantization gap in resolve rate, each printed with the absolute
  `BF16` and `Q4_K_M` values and the base model. A method can shrink the gap by lowering the BF16
  score, so a smaller gap is reported only with the arm's BF16 resolve-rate difference from `random`
  and its CI.
- **Claim regime (proposed, owner to confirm):** a single-user local agent, llama.cpp `Q4_K_M` at
  concurrency 1, the regime the energy model is fitted in. Concurrency-16 run-level totals are a
  secondary regime.
- **Estimator check:** `fewest_joules`, the joules-ranked `qtrace` and the RL reward optimize the
  same energy model that scores them, so a held-out subset is also measured per turn at
  concurrency 1. If the estimate fails that check, only the measured subset is claimed. Processed
  tokens per resolved task are co-reported as an estimator-free check.
- **Everything else is secondary:** other profiles, H3/H4 controls, the within-repo split and the
  external sets, corrected for multiplicity within each hypothesis family, or labelled exploratory.
- **Seeds and repos:** the primary analysis averages each task over the 3 seed models; per-seed and
  per-repo effects are reported too. Inference is conditional on the 4 held-out repos.
- Also reported: resolve rate, turns, processed tokens (output plus all input tokens, cache hits and
  misses, from the stored token sequence), malformed calls by type, and the Eq. 2-3 breakdown
  (per-token cost x extra tokens x extra failures).
- **Energy boundary:** GPU only on EC2, stated in every table. An energy win (H1b, H5, Stage 2) is
  claimed only if it also holds under an upper bound that adds the container's share of host CPU
  energy. Both are reported.
- **Training cost:** GPU-hours and joules for pool generation, scoring, SFT and RL, and the
  break-even count of resolved tasks (training joules / joules saved per resolved task at `Q4_K_M`
  vs the base model).

## Phases

| Phase | What | Gate |
|---|---|---|
| P0 | pilot, pool size (P0b), headroom (P0a), pool reach (P0c), score check (P0d); see `docs/04_p0_protocol.md` | four gates (P0a and P0d with pass/fail/inconclusive rules) |
| P1 | one-seed pilot of all arms, including the 2x2 length controls, on `inrepo_test` | picks the headline arm by a fixed rule: among `fewest_turns`, `fewest_joules`, `stable` and `qtrace`, the lowest estimated `Q4_K_M` joules per resolved task, with a BF16 resolve rate not clearly below `random`; controls are not eligible. The headline is then tested on the held-out repos, whose tasks are never used for any arm decision |
| P2 | confirmatory tier: the headline arm, `random`, the strongest cost-only arm (`fewest_joules`, or `fewest_turns` if the headline is `fewest_joules`) and the base model, 3 seeds, `BF16` and `Q4_K_M`, all 691 held-out tasks. Secondary tier for the other arms, profiles and H4 on a frozen held-out subset, from the options under Cost | the primary test in Metrics, frozen before P1 starts |
| P3 | external checks: SWE-bench Verified, Datacurve, Terminal-Bench | none, reported as is |
| S2 | RL ladder (Stage 2) from the P1 headline arm's checkpoint, one seed; order relative to P2 is an owner decision | the training probe's seconds per step at S2 lengths; the mixed-group share at group size 8, from subsampling P0b's 16 per task; a verified LoRA-merge, `convert_hf_to_gguf`, `llama-quantize` round trip (also needed before P1) |
| P4 | 9B replication and the remaining optional baselines | budget |

Post a preprint as soon as P2 has a clean answer. Related papers are appearing monthly.

## Kill rules

- Quantization doesn't hurt the base model (P0a fails under its three-way rule): drop the robustness
  half (B, the KL term and the stability arms). The energy half (A) continues, and the paper then
  claims measured joules per resolved task for `fewest_joules` vs `fewest_turns` vs `random`, and for
  RL ladder runs 1 and 2, on the held-out repos, with a null result reported as such.
- The stability score doesn't predict whether the quantized model can follow a run (P0d fails under
  its three-way rule): drop the score from selection and set beta = 0 in the RL reward. If P0a
  passed, B then has no training signal; before Stage 2 the owner chooses between the
  quantized-rollout reward and narrowing the claim to A with B measured only. If P0d is
  inconclusive, the stability arms stay exploratory, out of the headline and the RL default.
- The pool is too small or misses the tasks quantization breaks: use the fallbacks (fewer required
  successes, a DeepSWE-Preview teacher pool, the 9B model), then stop if all fail. DeepSWE-Preview
  trained on R2E-Gym-Subset's train split, which holds all 691 held-out tasks
  (`data/task_lists/deepswe_heldout_overlap.json`; the tasks its run actually drew are not
  published). A pool from it therefore carries teacher-level exposure of the held-out repos, and
  SWE-bench Verified is reported next to any base-model comparison.

## Cost

The 20-task pilot is the only spend in scope, and it is not yet approved to start (owner, 19 Nov
08:02Z). It starts when the entry criteria in `04` pass and the owner gives the go. Its exit report
gives measured token rates, and the owner sets the ceiling for the rest of P0 from it. Current
estimate: P0 is about 11,000-12,500 agent runs plus KL scoring, roughly 6-7 days on the one machine.
That is a floor until the pilot measures throughput.

Sizing for P1, P2 and S2, with minutes per attempt, concurrency and training seconds per step filled
in from the pilot and the probe: attempts = models x profiles x tasks x (2 if grammar-crossed). P1 is
11 arms plus base at 1 seed on `inrepo_test`, plus 11 LoRA runs. P2 as first written (11 arms x 3
seeds + base, 5 profiles, 691 tasks, grammar crossed with every arm) is about 117,000-235,000
attempts, or 53-107 days at the P0 rate, before 33 LoRA runs and RL. The owner picks the P2
secondary tier after the pilot and before any P1 result:

- (a) as written;
- (b) the headline arm, `random`, `fewest_turns`, `fewest_joules` and the base model at `BF16`,
  `Q4_K_M` and `Q3_K_M`; `Q8_0` and the gate-tensor `Q4_K_M` profile (Setup) on the base model
  only; grammar for headline vs `random` at `Q4_K_M`. About 31,000 attempts before the added
  baselines, with H3 resting on P1 evidence only;
- (c) option b plus `stable`, `low_entropy` and the 2x2 arms at `Q4_K_M` only, so H3 keeps 3 seeds.

## Who does what (proposed)

- Owner: the spending ceiling after the pilot; who builds the pilot harness; merges this plan.
- Unowned, for the owner to assign: GGUF profile builds including the `ssm_alpha`/`ssm_beta`
  override; pinned vLLM and llama.cpp builds; EC2 operations and image pulls; per-turn energy hooks;
  the P0d replay path and repo-state fingerprints; the frozen seeded lists (including
  `inrepo_test`); the training-probe script; the SFT migration; the GGUF round trip on a LoRA-merged
  model. Proposed split: the owner's agent takes the harness side (episode store, parser, runtimes,
  replay, energy hooks, GGUF builds, EC2); LordKarV's agent takes the KL wrapper, the SFT migration
  and probe, the seeded lists, the analysis scripts and the exit-report generator.
- Owner's agent: the R2E-Gym runtime plugin, the token-level episode store, the parser and repair
  path, the per-tensor quantization tables.
- Before P1, whoever owns training: migrate `scripts/04_train_lora.py`, built for single-turn
  GSM8K, to multi-turn runs (checklist in `04`, "Selection and training rules for P1"). Before the
  pilot, only the probe script is needed (`04`).
- LordKarV's agent: the chunked KL scoring wrapper (PyTorch first, then llama.cpp), the P0 analysis
  scripts (bootstrap, P0c breakage tables, the P0d difference-in-differences), and the related-work
  tracking.
- Both: review each other's code in PRs, discuss in issue #1.
