"""LoRA SFT on stored multi-turn agent runs (docs/04, "Selection and training rules for P1"; docs/03 Method step 4).

The training set is one arm file, outputs/arms/<arm>_seed<s>.jsonl, with one selected run per task. Each
run is the episode the harness stored: input_ids, model_mask (true exactly on sampled tokens) and the
turn spans. The loss covers only model-written tokens and is the sum of their cross-entropies over a
step's episodes divided by a fixed constant, never a per-batch mean, so a long run is not down-weighted
per token and the losses of two arms are on one scale. Steps come from epochs over the arm's runs, which
gives every arm the same count because every arm has one run per task.
"""

import collections
import glob
import itertools
import os
import random

import numpy as np
import torch

from qtrace.common import read_jsonl
from qtrace.probe import GATE_TARGETS, attach_lora, chunked_ce_loss

ARM_FIELDS = ("arm", "seed", "task_id", "attempt_id", "episode_path")
EPISODE_ARRAYS = ("input_ids", "model_mask", "turn_spans")
GRAD_CLIP = 1.0
# The Qwen3.5 checkpoint's only non-bf16 text tensors (F32, 24 of each in the 4B). A bf16 load rounds them; the
# saved merge takes them back from the checkpoint so it matches a straight GGUF conversion (issue #1).
F32_RESTORED = ("linear_attn.norm.weight", "linear_attn.A_log")
TEXT_PREFIX = "model.language_model."  # the multimodal checkpoint's text tower; Qwen3_5ForCausalLM names it model.
NON_TEXT_PREFIXES = ("model.visual.", "mtp.")


def load_arm(path, arm, seed):
    """Rows of one arm file, checked: the fields the trainer reads, this arm and seed, one run per task."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"arm file {path} not found")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    for i, row in enumerate(rows):
        missing = [k for k in ARM_FIELDS if row.get(k) is None]
        if missing:
            raise ValueError(f"{path} row {i}: missing {missing}")
        if row["arm"] != arm or row["seed"] != seed:
            raise ValueError(f"{path} row {i}: arm {row['arm']!r} seed {row['seed']!r}, expected {arm!r} seed {seed}")
        if not os.path.isabs(row["episode_path"]):
            raise ValueError(f"{path} row {i}: episode_path must be absolute, got {row['episode_path']!r}")
    for key in ("task_id", "attempt_id"):
        dup = sorted(k for k, n in collections.Counter(r[key] for r in rows).items() if n > 1)
        if dup:
            raise ValueError(f"{path}: duplicate {key} {dup[:5]}; an arm has exactly one run per task")
    return rows


def load_episode(path):
    """input_ids (int64 [n]), model_mask (bool [n]) and turn_spans (int [m, 2], end exclusive) of one episode.

    Raises ValueError on missing arrays, wrong dtypes or shapes, or turn spans that are unordered,
    overlapping or out of range.
    """
    with np.load(path) as z:
        missing = [k for k in EPISODE_ARRAYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        ids, mask, turns = z["input_ids"], z["model_mask"], z["turn_spans"]
    if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{path}: input_ids must be a non-empty 1-D integer array, got {ids.dtype} {ids.shape}")
    if mask.dtype != np.bool_ or mask.shape != ids.shape:
        raise ValueError(f"{path}: model_mask must be bool of shape {ids.shape}, got {mask.dtype} {mask.shape}")
    if turns.ndim != 2 or turns.shape[1] != 2 or not np.issubdtype(turns.dtype, np.integer):
        raise ValueError(f"{path}: turn_spans must be an integer [m, 2] array, got {turns.dtype} {turns.shape}")
    if len(turns) and ((turns[:, 0] >= turns[:, 1]).any() or turns.min() < 0 or turns.max() > len(ids)
                       or (turns[1:, 0] < turns[:-1, 1]).any()):
        raise ValueError(f"{path}: turn_spans must be ordered, non-overlapping (start, end) within [0, {len(ids)}]")
    return {"input_ids": ids.astype(np.int64), "model_mask": mask, "turn_spans": turns}


def truncation_cut(turn_spans, n, max_tokens):
    """Tokens to keep: all n if they fit, else up to the end of the last turn ending at or before max_tokens.

    Returns 0 when no turn ends in range. A boundary cut only drops the tail; docs/04 prefers per-turn
    loss chunks, left until the training probe says whether 64K fits (max_tokens null = no cut).
    """
    if max_tokens is None or n <= max_tokens:
        return n
    return max((int(end) for _, end in turn_spans if end <= max_tokens), default=0)


def supervised_tokens(model_mask, cut):
    """Model-written tokens among the first `cut` that are loss targets. Position 0 never is: nothing predicts it."""
    return int(model_mask[1:cut].sum())


def build_examples(rows, max_tokens=None):
    """Load every row's episode, apply the truncation policy, and report what it cut.

    Returns (examples, report). truncated_supervised_share is the share of the arm's supervised tokens
    cut off, counted once per run (not per epoch). Raises ValueError if a run keeps no model-written token.
    """
    examples, full, kept, n_cut = [], 0, 0, 0
    for row in rows:
        ep = load_episode(row["episode_path"])
        n = len(ep["input_ids"])
        cut = truncation_cut(ep["turn_spans"], n, max_tokens)
        n_sup = supervised_tokens(ep["model_mask"], cut)
        if n_sup == 0:
            where = f" within max_tokens={max_tokens}" if cut < n else ""
            raise ValueError(f"{row['episode_path']}: no model-written tokens to train on{where}")
        examples.append({"attempt_id": row["attempt_id"], "n_sup": n_sup,
                         "input_ids": torch.from_numpy(ep["input_ids"][:cut]).unsqueeze(0),
                         "model_mask": torch.from_numpy(ep["model_mask"][:cut])})
        full += supervised_tokens(ep["model_mask"], n)
        kept += n_sup
        n_cut += cut < n
    report = {"truncated_supervised_share": (full - kept) / full, "truncated_episodes": n_cut,
              "supervised_tokens_per_epoch": kept}
    return examples, report


def _shuffled_epochs(n, seed):
    rng = random.Random(seed)
    while True:
        order = list(range(n))
        rng.shuffle(order)
        yield from order


def plan_steps(n_sup, epochs, per_step, seed, match_tokens=None):
    """Example indices for each optimizer step, from per-run supervised token counts n_sup.

    The order is a fresh shuffle of all runs per epoch from one random.Random(seed), consumed per_step at
    a time across epoch boundaries. Normally the plan covers exactly `epochs` passes, so steps =
    ceil(epochs * n / per_step) and only the last step can be short; it depends on n alone, not on
    lengths. With match_tokens the plan stops at the first full step whose cumulative supervised tokens
    reach match_tokens (docs/04 token-matched rerun): the same order as the original run, cut where the
    budget is met, which may lie past `epochs`.
    """
    n = len(n_sup)
    if n == 0 or not isinstance(epochs, int) or epochs < 1 or not isinstance(per_step, int) or per_step < 1:
        raise ValueError(f"need runs, integer epochs >= 1 and per_step >= 1; got n={n} {epochs=} {per_step=}")
    stream = _shuffled_epochs(n, seed)
    if match_tokens is None:
        flat = list(itertools.islice(stream, epochs * n))
        return [flat[i:i + per_step] for i in range(0, len(flat), per_step)]
    if match_tokens <= 0:
        raise ValueError(f"match_tokens must be positive, got {match_tokens}")
    plan, seen = [], 0
    while seen < match_tokens:  # every run has n_sup >= 1 (build_examples), so this ends
        plan.append(list(itertools.islice(stream, per_step)))
        seen += sum(n_sup[i] for i in plan[-1])
    return plan


def lr_factor(step, warmup_steps, total_steps):
    """Linear warmup to 1 over warmup_steps, then linear decay that reaches 1 / (total - warmup) at the last step."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return (total_steps - step) / max(1, total_steps - warmup_steps)


def attach(model, train_cfg, seed):
    """LoRA with gradient checkpointing on the configured targets; the seed sets the LoRA init.

    in_proj_a / in_proj_b go only through include_gates, so the open decision stays one visible switch.
    Raises ValueError for a target that matches no module, which peft skips silently when others match.
    """
    targets = list(train_cfg["targets"])
    if not targets or set(targets) & set(GATE_TARGETS):
        raise ValueError(f"targets must be non-empty and leave {GATE_TARGETS} to include_gates, got {targets}")
    torch.manual_seed(seed)
    model = attach_lora(model, include_gates=train_cfg["include_gates"], r=train_cfg["lora_r"],
                        alpha=train_cfg["lora_alpha"], targets=targets)
    have = {n.split(".lora_A.")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters() if ".lora_A." in n}
    missing = [t for t in targets + (GATE_TARGETS if train_cfg["include_gates"] else []) if t not in have]
    if missing:
        raise ValueError(f"LoRA targets not found in the model: {missing}")
    return model


def save_merged(model, out_dir, base_path):
    """Fold W <- W + scaling * B @ A in float32 on the CPU and save the plain model in F32 to out_dir.

    A bf16 merge rounds away the part of the update below half a bf16 step, about half of it at
    ||dW||/||W|| = 1e-3 (issue #1, the owner's round trip), and every GGUF profile built from that file
    inherits the loss. In f32 each saved weight is exactly W.float() + scaling * B.float() @ A.float().
    The LoRA model is moved to the CPU in f32 in place (the GPU never holds an f32 copy); config.json
    records dtype float32. About 16 GB on disk for the 4B. The checkpoint's F32 tensors (base_f32_tensors of
    base_path, the folder or HF id the model was loaded from) replace their bf16-rounded copies; training still
    saw the rounded ones. Returns the merged model.
    """
    merged = model.to(device="cpu", dtype=torch.float32).merge_and_unload()
    with torch.no_grad():
        for name, tensor in base_f32_tensors(base_path).items():
            merged.get_parameter(name).copy_(tensor)
    merged.save_pretrained(out_dir)
    return merged


def base_f32_tensors(base_path):
    """The checkpoint's F32_RESTORED tensors as float32, keyed as in the text-only Qwen3_5ForCausalLM state dict.

    Text tensors are all but the vision tower and the MTP head. Raises ValueError unless the F32_RESTORED tensors
    are exactly the checkpoint's non-bf16 text tensors and all F32: a checkpoint with other wide tensors must
    fail here rather than be saved half-restored.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    folder = base_path if os.path.isdir(base_path) else snapshot_download(base_path, allow_patterns=["*.safetensors"])
    restored, wide = {}, {}
    for file in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        with safe_open(file, "pt") as f:
            for key in f.keys():  # noqa: SIM118, a safe_open handle is not iterable
                if key.startswith(NON_TEXT_PREFIXES):
                    continue
                name = "model." + key.removeprefix(TEXT_PREFIX) if key.startswith(TEXT_PREFIX) else key
                dtype = f.get_slice(key).get_dtype()
                if dtype != "BF16":
                    wide[name] = dtype
                if name.endswith(F32_RESTORED):
                    restored[name] = f.get_tensor(key)
    if set(wide) != set(restored) or set(wide.values()) != {"F32"}:
        raise ValueError(f"{folder}: non-bf16 text tensors {sorted(wide.items())[:5]} ({len(wide)}) must be exactly "
                         f"the F32 {F32_RESTORED} tensors ({len(restored)})")
    return restored


def episode_loss(model, example, loss_norm, chunk=2048):
    """Summed cross-entropy over the episode's model-written tokens divided by loss_norm (chunked logits)."""
    device = next(model.parameters()).device
    total = chunked_ce_loss(model, example["input_ids"].to(device), example["model_mask"], chunk=chunk,
                            reduction="sum")
    return total / loss_norm


def train(model, examples, plan, lr, warmup_steps, loss_norm, chunk=2048, on_step=None):
    """AdamW on the LoRA parameters over the planned steps; one record per step, also passed to on_step.

    One episode per micro-batch (no padding), gradients accumulated over the step's episodes and clipped
    to GRAD_CLIP. A record has the step loss (summed CE / loss_norm), the mean CE per supervised token,
    the supervised tokens, the learning rate used and the gradient norm before clipping.
    """
    if loss_norm <= 0:
        raise ValueError(f"loss_norm must be positive, got {loss_norm}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    records = []
    for step, batch in enumerate(plan):
        step_lr = lr * lr_factor(step, warmup_steps, len(plan))
        for group in opt.param_groups:
            group["lr"] = step_lr
        loss, n_sup = 0.0, 0
        for i in batch:
            part = episode_loss(model, examples[i], loss_norm, chunk)
            part.backward()
            loss += float(part.detach())
            n_sup += examples[i]["n_sup"]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP))
        opt.step()
        opt.zero_grad(set_to_none=True)
        record = {"step": step, "loss": loss, "loss_per_token": loss * loss_norm / n_sup, "supervised_tokens": n_sup,
                  "lr": step_lr, "grad_norm": grad_norm}
        records.append(record)
        if on_step is not None:
            on_step(record)
    return records
