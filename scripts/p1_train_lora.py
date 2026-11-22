"""P1 LoRA fine-tune on one arm's selected agent runs (docs/04, "Selection and training rules for P1").

The multi-turn replacement for scripts/04_train_lora.py, which stays as the single-turn GSM8K archive.
Loss mask, sum-over-constant loss, truncation policy and step plan are in qtrace/sft.py.

Run on the box:
    python scripts/p1_train_lora.py --config configs/p1_train.yaml --arm qtrace --seed 0
Token-matched rerun (docs/04, Training), stopping once the supervised tokens of another run are reached:
    python scripts/p1_train_lora.py --arm qtrace --seed 0 --match-supervised-tokens <its supervised_tokens_seen>
Writes outputs/runs/<arm>_seed<s>[_tokmatch<N>]/: loss.jsonl, train_meta.json and merged/ (model and
tokenizer, the input of the GGUF round trip). With fake_quant.enabled in the config (the random_fq baseline,
qtrace/fake_quant.py) the run is named <arm>_fq, so --arm random writes outputs/runs/random_fq_seed<s>/.
Training runs in bf16 on a GPU; the merge and the saved model are F32 (merged_dtype), so no part of the update is
rounded away before the GGUF conversion. train_meta.json also has update_size (qtrace.probe.lora_update_size),
measured on the trained adapter before the merge.
Training cost (docs/03, Metrics; owner answer on #15): train_meta.json's training_cost has two metered windows,
whole_run (model load + train + merge + save, the primary, since break-even counts everything needed to produce
the deployed model) and train_loop (sft.train alone, secondary). Each has wall_seconds, gpu_hours (null on CPU),
gpu_joules (the NVML energy counter of GPU gpu_index, qtrace.energy.EnergyMeter, raw, idle draw included, GPU only)
and gpu_joules_idle_subtracted (gpu_joules minus the idle draw times wall_seconds). training_cost.idle has two
idle readings (qtrace.energy.measure_idle_watts over window_s, each after settle_s of settling, as on the inference
side): none_w before the model is loaded, subtracted from whole_run, and loaded_w with the model and adapter on the
GPU before step 1, subtracted from train_loop (owner review on #15). Without a CUDA device NVML is not read: the
joules and both readings are null and null_reason says why.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace import fake_quant, sft
from qtrace.common import load_config
from qtrace.energy import EnergyMeter, measure_idle_watts
from qtrace.models import pick_device
from qtrace.probe import kernel_paths, kernel_status, lora_update_size

GPU_INDEX = 0  # the g6e.16xlarge has one GPU (owner answer on #15); recorded so the record says which was metered
IDLE_WINDOW_S = 10.0
IDLE_SETTLE_S = 45.0  # GPU power states lag process start and model load (docs/04's energy sub-run settle)
NO_NVML = "torch sees no CUDA device, so the NVML energy counter was not read"


def idle_reading():
    """Average idle GPU watts after settling; None without CUDA (no wait then, nothing to read)."""
    if not torch.cuda.is_available():
        return None
    time.sleep(IDLE_SETTLE_S)
    return measure_idle_watts(IDLE_WINDOW_S, GPU_INDEX)


def window_cost(meter, idle_watts, on_gpu):
    """One metered window's wall time, GPU-hours and GPU joules, raw and idle-subtracted."""
    raw = meter.joules
    return {"wall_seconds": meter.seconds, "gpu_hours": meter.seconds / 3600 if on_gpu else None, "gpu_joules": raw,
            "gpu_joules_idle_subtracted": None if raw is None or idle_watts is None
            else raw - idle_watts * meter.seconds}


def check_kernels(allow_slow):
    """Kernel record for the metadata; exits unless both training-path ops resolved to the fast packages."""
    import transformers

    kernels = {"in_use": kernel_paths(), "importable": kernel_status(), "transformers": transformers.__version__}
    if not all(op["fast"] for op in kernels["in_use"].values()) and not allow_slow:
        raise SystemExit(f"fla and causal_conv1d must be in use, transformers resolved {kernels['in_use']} "
                         "(docs/04 requires the fast kernels); pass --allow-slow-kernels only for a smoke test")
    return kernels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_train.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the arm file's selection seed; also seeds training")
    ap.add_argument("--match-supervised-tokens", type=int, default=None,
                    help="train until this many supervised tokens are seen; sets the step count up front")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="run on the torch DeltaNet fallback (tests and smoke runs only)")
    ap.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tr, model_path = cfg["train"], cfg["model"]["path"]
    fq_cfg = cfg.get("fake_quant") or {}
    table = shapes = None
    if fq_cfg.get("enabled"):  # random_fq baseline (docs/04): read the table before the model, to fail fast
        if not fq_cfg.get("table"):
            raise ValueError("fake_quant.enabled needs fake_quant.table (the Q4_K_M GGUF or the pinned JSON table)")
        table, shapes = fake_quant.load_table(fq_cfg["table"])
    kernels = check_kernels(args.allow_slow_kernels)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    arm_file = os.path.join(cfg["paths"]["arms_dir"], f"{args.arm}_seed{args.seed}.jsonl")
    rows = sft.load_arm(arm_file, args.arm, args.seed)
    examples, report = sft.build_examples(rows, tr["max_tokens"])
    plan = sft.plan_steps([e["n_sup"] for e in examples], tr["epochs"], tr["episodes_per_step"], args.seed,
                          match_tokens=args.match_supervised_tokens)
    warmup = round(tr["warmup_ratio"] * len(plan))
    suffix = "" if args.match_supervised_tokens is None else f"_tokmatch{args.match_supervised_tokens}"
    name = args.arm if table is None else f"{args.arm}_fq"
    run_dir = os.path.join(cfg["paths"]["runs_dir"], f"{name}_seed{args.seed}{suffix}")
    print(f"{len(examples)} runs, {len(plan)} steps, {report}")

    device = args.device or pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    none_w = idle_reading()  # before the model is loaded: nothing resident, the whole_run baseline
    # Whole-run window, closed after the save. Not a with-block: it would re-indent the whole run; on an exception
    # no train_meta.json is written, so an unclosed meter loses nothing.
    run_meter = EnergyMeter(GPU_INDEX).__enter__()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    kernels["attn_implementation"] = model.config._attn_implementation
    top = max(int(e["input_ids"].max()) for e in examples)
    if top >= model.config.vocab_size:
        raise ValueError(f"token id {top} is outside the model's vocabulary ({model.config.vocab_size})")
    model = sft.attach(model, tr, args.seed)
    model.print_trainable_parameters()
    fq_summary = None if table is None else {"table": fq_cfg["table"], **fake_quant.apply(model, table, shapes)}

    loaded_w = idle_reading()  # weights, adapter and CUDA context resident, before step 1: the train_loop baseline
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "loss.jsonl"), "w") as f:
        def log(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"step {rec['step']:5d}  loss {rec['loss']:.4f}  per token {rec['loss_per_token']:.4f}  "
                  f"tokens {rec['supervised_tokens']}  grad {rec['grad_norm']:.3f}")

        with EnergyMeter(GPU_INDEX) as meter:
            records = sft.train(model, examples, plan, tr["learning_rate"], warmup, tr["loss_norm"], tr["chunk"],
                                log)

    # Measured on the trained adapter before the merge drops it (issue #1): how big the update is and how
    # much of it a bf16 file would keep.
    update_size = lora_update_size(model)
    # Fold W <- W + BA in f32 and save the plain model: quantization (GGUF) must see the deployed weights.
    merged_dir = os.path.join(run_dir, "merged")
    sft.save_merged(model, merged_dir)
    tokenizer.save_pretrained(merged_dir)
    run_meter.__exit__(None, None, None)
    on_gpu = device != "cpu"
    meta = {
        "arm": args.arm, "seed": args.seed, "arm_file": arm_file, "examples": len(examples), "steps": len(plan),
        "epochs": tr["epochs"], "episodes_per_step": tr["episodes_per_step"],
        "match_supervised_tokens": args.match_supervised_tokens,
        "supervised_tokens_seen": sum(r["supervised_tokens"] for r in records),
        "supervised_tokens_per_epoch": report["supervised_tokens_per_epoch"], "max_tokens": tr["max_tokens"],
        "truncated_supervised_share": report["truncated_supervised_share"],
        "truncated_episodes": report["truncated_episodes"], "loss_norm": tr["loss_norm"],
        "learning_rate": tr["learning_rate"], "warmup_steps": warmup, "grad_clip": sft.GRAD_CLIP,
        "targets": list(tr["targets"]), "include_gates": tr["include_gates"], "lora_r": tr["lora_r"],
        "lora_alpha": tr["lora_alpha"], "kernels": kernels, "model": model_path, "device": device,
        "dtype": str(dtype), "merged": merged_dir, "merged_dtype": "float32", "update_size": update_size,
        "fake_quant": fq_summary,
        "training_cost": {
            "primary": "whole_run", "gpu_index": GPU_INDEX,
            "idle": {"none_w": none_w, "loaded_w": loaded_w, "settle_s": IDLE_SETTLE_S, "window_s": IDLE_WINDOW_S},
            "null_reason": None if run_meter.joules is not None else NO_NVML,
            "whole_run": window_cost(run_meter, none_w, on_gpu),
            "train_loop": window_cost(meter, loaded_w, on_gpu)},
    }
    with open(os.path.join(run_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
