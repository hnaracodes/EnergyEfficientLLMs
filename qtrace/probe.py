"""Training probe: can one LoRA step on a long agent run fit and how long does it take? (docs/04, P0b)

Memory and step time depend on sequence length, not content, so the probe trains on random tokens.
The LoRA targets here are the ones the SFT migration will use (docs/04, "Migrating
scripts/04_train_lora.py"): attention, MLP and the DeltaNet mixers. The old GSM8K config targeted
only attention and MLP names, which left the mixers in 24 of 32 layers untouched.
"""

import importlib.util
import inspect
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
DELTANET_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj"]
GATE_TARGETS = ["in_proj_a", "in_proj_b"]  # DeltaNet write-strength and decay gates; an open decision

# The two DeltaNet ops on the training path. Without a cache, Qwen3_5GatedDeltaNet.forward always calls
# causal_conv1d_fn and torch_chunk_gated_delta_rule (the *_update / recurrent ops serve one-token decode).
TRAINING_OPS = {"chunk_gated_delta_rule": "torch_chunk_gated_delta_rule", "causal_conv1d_fn": "causal_conv1d_fn"}


def kernel_status():
    """Whether the fast DeltaNet kernels are importable. Without them transformers falls back to slow torch code."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}


def kernel_paths():
    """Which implementation each training-path DeltaNet op calls: the fast package or the torch fallback.

    transformers 5.23 decides once, when modeling_qwen3_5 is imported: use_kernel_func_from_hub_with_fallback
    (integrations/hub_kernels.py) tries to import fla / causal_conv1d and closes over the result, falling
    back to the torch function on any import error, with only a warning at the first call. So importable
    is not enough: a broken install (say fla without a working triton) trains on the slow path. This reads
    the closure the model will call. An unreadable closure (another transformers version) reports not fast.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5

    out = {}
    for op, attr in TRAINING_OPS.items():
        found = inspect.getclosurevars(getattr(modeling_qwen3_5, attr)).nonlocals
        impl = found.get("implementation")
        out[op] = {"fast": bool(found.get("is_new_implementation")),
                   "impl": f"{impl.__module__}.{impl.__qualname__}" if callable(impl) else "unknown"}
    return out


def attach_lora(model, include_gates=False, r=16, alpha=32, grad_ckpt=True):
    from peft import LoraConfig, get_peft_model

    targets = ATTENTION_TARGETS + MLP_TARGETS + DELTANET_TARGETS + (GATE_TARGETS if include_gates else [])
    if grad_ckpt:
        # transformers 5.x renamed the method and its kwargs; keep older installs working.
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def _share(num, den):
    return num / den if den > 0 else None


def lora_update_size(model, adapter="default"):
    """Size of the trained LoRA update per adapted module, and how much of it a bf16 file would keep.

    For each module, dW = scaling * B @ A in float32 and rel = ||dW||_F / ||W||_F, summarized by module
    type (the last name part: q_proj, in_proj_qkv, out_proj, ...) as median, p10 and p90 (numpy's linear
    percentiles). carried_bf16 = <bf16(W + dW) - bf16(W), dW> / ||dW||^2: 1 when a bf16 merge keeps the
    whole update, 0 when it rounds back to W. With a bf16 base bf16(W) is W itself, so this is the share
    the owner's round trip measured (issue #1); aggregates are the ratio of sums over modules, so a
    module weighs by ||dW||^2. Works one module at a time in f32 on the module's device, with no full
    model copy. carried_bf16 is None when the update is exactly zero (an untrained adapter).
    """
    by_type = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if adapter not in getattr(mod, "lora_A", {}):
                continue
            w = mod.base_layer.weight.float()
            dw = mod.scaling[adapter] * (mod.lora_B[adapter].weight.float() @ mod.lora_A[adapter].weight.float())
            kept = (w + dw).bfloat16().float() - w.bfloat16().float()
            t = by_type.setdefault(name.rsplit(".", 1)[-1], {"rel": [], "num": 0.0, "den": 0.0})
            t["rel"].append(float(dw.norm() / w.norm()))
            t["num"] += float((kept * dw).sum())
            t["den"] += float((dw * dw).sum())
    out = {}
    for kind, t in sorted(by_type.items()):
        p10, p50, p90 = np.percentile(t["rel"], [10, 50, 90])
        out[kind] = {"modules": len(t["rel"]), "rel_median": float(p50), "rel_p10": float(p10),
                     "rel_p90": float(p90), "carried_bf16": _share(t["num"], t["den"])}
    return {"modules": sum(t["modules"] for t in out.values()),
            "carried_bf16": _share(sum(t["num"] for t in by_type.values()), sum(t["den"] for t in by_type.values())),
            "by_type": out}


def supervised_mask(length, frac, period=2048):
    """Synthetic assistant turns: one contiguous span of round(frac * period) tokens per period.

    Real episodes supervise only model-written tokens (docs/04), so the loss never needs logits
    elsewhere. Position 0 is never supervised (nothing predicts it).
    """
    mask = torch.zeros(length, dtype=torch.bool)
    for start in range(0, length, period):
        span = min(period, length - start)
        n = round(frac * span)
        if n:
            mask[start + span - n:start + span] = True
    mask[0] = False
    return mask


def _ce_sum(hidden, targets, head):
    return F.cross_entropy(head(hidden).float(), targets, reduction="sum")


def chunked_ce_loss(model, input_ids, sup_mask, chunk=2048):
    """Mean cross-entropy over supervised tokens, with logits built per chunk and recomputed in backward.

    Logit memory is O(chunk x vocab) instead of O(length x vocab): at 64K x 248,320 the full tensor is
    about 30 GiB in bf16 and twice that once the loss upcasts it. Matches the transformers loss with
    labels = -100 outside the mask (the logit at i predicts token i + 1).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids, use_cache=False).last_hidden_state[0]
    t = torch.nonzero(sup_mask.to(hidden.device)).squeeze(1)
    t = t[t >= 1]
    idx, targets = t - 1, input_ids[0, t].to(hidden.device)
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, len(idx), chunk):
        total = total + checkpoint(_ce_sum, hidden[idx[s:s + chunk]], targets[s:s + chunk], base.lm_head,
                                   use_reentrant=False)
    return total / max(len(idx), 1)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _memory(device):
    """(GiB, kind). CUDA gives a true peak; MPS only the current allocation (smoke tests only)."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30, "cuda_peak"
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30, "mps_current"
    return None, None


def probe_length(model, length, steps, device, vocab_size, seed=0, lr=1e-4, supervised_frac=0.15,
                 loss_path="chunked", chunk=2048):
    """Run 1 warm-up step plus `steps` timed LoRA steps on one random sequence of `length` tokens.

    loss_path "chunked" is the loss SFT will use. "full_logits" is the transformers loss over the full
    vocabulary at every position, kept only as a comparison point.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (1, length), generator=g).to(device)
    sup = supervised_mask(length, supervised_frac)
    labels = torch.where(sup.to(device), ids[0], torch.full_like(ids[0], -100)).unsqueeze(0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    record = {"length": length, "steps": steps, "device": device, "loss_path": loss_path,
              "supervised_frac": supervised_frac, "n_supervised": int(sup.sum()), "chunk": chunk}
    done = 0
    try:
        times, loss, grad_norm = [], float("nan"), 0.0
        for step in range(steps + 1):
            _sync(device)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if loss_path == "chunked":
                loss = chunked_ce_loss(model, ids, sup, chunk=chunk)
            else:
                loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            grad_norm = float(torch.norm(torch.stack([p.grad.float().norm() for p in params if p.grad is not None])))
            opt.step()
            done += 1
            _sync(device)
            if step > 0:  # step 0 is warm-up (kernel compilation, allocator growth)
                times.append(time.perf_counter() - start)
        mem, kind = _memory(device)
        record.update(status="ok", sec_per_step=sorted(times)[len(times) // 2],
                      loss_finite=math.isfinite(float(loss.detach())), lora_grad_norm=grad_norm,
                      mem_gib=mem, mem_kind=kind)
    except torch.OutOfMemoryError:
        mem, kind = _memory(device)
        record.update(status="oom", mem_gib=mem, mem_kind=kind)
    finally:
        opt.zero_grad(set_to_none=True)
    record["optimizer_steps"] = done  # warm-up included: every step moves the adapter
    return record
