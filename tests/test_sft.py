"""Tests for the multi-turn SFT trainer on a tiny Qwen3.5 and synthetic episodes. Run:  python -m pytest tests -q"""

import importlib.util
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from test_probe import plant_update, tiny_qwen35

from qtrace import sft

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
           "out_proj"]
TRAIN = {"targets": TARGETS, "include_gates": False, "lora_r": 4, "lora_alpha": 8}


def make_episode(seed, turn_lens, prompt=12, tool=6, vocab=300):
    """A stored episode in the agreed format: prompt, then per turn a model-written span and a tool response."""
    rng = np.random.default_rng(seed)
    ids, mask, turns = list(rng.integers(0, vocab, prompt)), [False] * prompt, []
    for n in turn_lens:
        turns.append((len(ids), len(ids) + n))
        ids += list(rng.integers(0, vocab, n))
        mask += [True] * n
        ids += list(rng.integers(0, vocab, tool))
        mask += [False] * tool
    turns = np.array(turns, dtype=np.int32).reshape(-1, 2)
    return {"input_ids": np.array(ids, dtype=np.int32), "model_mask": np.array(mask), "turn_spans": turns,
            "tool_call_positions": turns[:, 1] - 2, "command_spans": np.stack([turns[:, 0], turns[:, 0] + 2], 1),
            "tool_response_spans": np.stack([turns[:, 1], turns[:, 1] + tool], 1)}


def as_example(ep):
    return {"attempt_id": "a", "input_ids": torch.from_numpy(ep["input_ids"].astype(np.int64)).unsqueeze(0),
            "model_mask": torch.from_numpy(ep["model_mask"]), "n_sup": int(ep["model_mask"][1:].sum())}


def write_arm(tmp_path, arm, seed, turn_lens_per_task):
    """Episode npz files plus outputs/arms/<arm>_seed<s>.jsonl, rows in the agreed P1 arm format."""
    ep_dir, arms_dir = tmp_path / "attempts", tmp_path / "arms"
    ep_dir.mkdir(exist_ok=True)
    arms_dir.mkdir(exist_ok=True)
    rows = []
    for i, lens in enumerate(turn_lens_per_task):
        path = ep_dir / f"att{i}.npz"
        np.savez(path, **make_episode(i, lens))
        rows.append({"arm": arm, "seed": seed, "task_id": f"task{i}", "attempt_id": f"att{i}",
                     "episode_path": str(path), "turns": len(lens), "output_tokens": sum(lens),
                     "est_joules_q4": None, "score_name": None, "score": None, "entropy_mean": None,
                     "malformed_total": 0})
    path = arms_dir / f"{arm}_seed{seed}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def reference_loss(model, ids, mask, loss_norm):
    """Full-vocab logits, labels = input ids on model-written tokens and -100 elsewhere, summed, over the constant."""
    logits = model(input_ids=ids).logits[0].float()
    labels = torch.where(mask, ids[0], torch.full_like(ids[0], -100))
    return F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100, reduction="sum") / loss_norm


def perturbed_lora_model(seed=0):
    model = sft.attach(tiny_qwen35().float(), TRAIN, seed)
    for name, p in model.named_parameters():
        if "lora_B" in name:  # B starts at zero; make the LoRA path matter
            p.data.normal_(0, 0.05)
    return model


def lora_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def test_episode_loss_is_the_masked_sum_over_the_constant_with_matching_gradients():
    model = perturbed_lora_model()
    ep = make_episode(0, [9, 14, 5])
    ep["model_mask"][0] = True  # position 0 is never a target, whatever the mask says
    ex = as_example(ep)
    ours = sft.episode_loss(model, ex, loss_norm=50.0, chunk=7)
    g_ours = torch.autograd.grad(ours, lora_params(model))
    ref = reference_loss(model, ex["input_ids"], ex["model_mask"], 50.0)
    g_ref = torch.autograd.grad(ref, lora_params(model))
    assert torch.allclose(ours, ref, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_ours, g_ref))


def test_step_loss_sums_episodes_without_a_per_batch_mean():
    model = perturbed_lora_model()
    exs = [as_example(make_episode(1, [5])), as_example(make_episode(2, [20, 30]))]
    with torch.no_grad():
        refs = [float(reference_loss(model, e["input_ids"], e["model_mask"], 64.0)) for e in exs]
    rec = sft.train(model, exs, [[0, 1]], lr=0.0, warmup_steps=0, loss_norm=64.0, chunk=16)[0]
    assert rec["supervised_tokens"] == 55
    assert rec["loss"] == pytest.approx(sum(refs), rel=1e-5)
    assert rec["loss_per_token"] == pytest.approx(sum(refs) * 64.0 / 55, rel=1e-5)
    assert rec["grad_norm"] > 0 and rec["lr"] == 0.0


def test_tokens_outside_model_mask_are_context_not_targets():
    model = perturbed_lora_model()
    ep = make_episode(3, [8, 8])
    base = as_example(ep)
    loss = sft.episode_loss(model, base, 10.0, chunk=5)
    grads = torch.autograd.grad(loss, lora_params(model))

    tail = {**ep, "input_ids": ep["input_ids"].copy()}
    tail["input_ids"][-6:] = (tail["input_ids"][-6:] + 1) % 300  # the trailing tool response: no target follows it
    tail_ex = as_example(tail)
    tail_loss = sft.episode_loss(model, tail_ex, 10.0, chunk=5)
    assert torch.equal(loss, tail_loss)
    assert all(torch.equal(a, b) for a, b in zip(grads, torch.autograd.grad(tail_loss, lora_params(model))))

    mid = {**ep, "input_ids": ep["input_ids"].copy()}
    end_first_turn = int(ep["turn_spans"][0][1])
    mid["input_ids"][end_first_turn] = (mid["input_ids"][end_first_turn] + 1) % 300  # tool token before turn 2
    assert not torch.equal(loss, sft.episode_loss(model, as_example(mid), 10.0, chunk=5))


def test_steps_come_from_epochs_and_each_epoch_is_a_fresh_shuffle():
    plan = sft.plan_steps([5] * 10, epochs=3, per_step=4, seed=1)
    assert len(plan) == math.ceil(3 * 10 / 4)
    assert all(len(s) == 4 for s in plan[:-1]) and len(plan[-1]) == 2
    flat = [i for s in plan for i in s]
    passes = [flat[k * 10:(k + 1) * 10] for k in range(3)]
    assert all(sorted(p) == list(range(10)) for p in passes)
    assert passes[0] != passes[1]
    assert plan == sft.plan_steps([900] * 10, epochs=3, per_step=4, seed=1)  # every arm with 10 tasks: same steps
    assert plan != sft.plan_steps([5] * 10, epochs=3, per_step=4, seed=2)
    with pytest.raises(ValueError):
        sft.plan_steps([5] * 10, epochs=0, per_step=4, seed=1)


def test_match_supervised_tokens_stops_at_the_first_step_reaching_the_budget():
    n_sup = [3, 5, 7, 11]
    long_plan = sft.plan_steps(n_sup, epochs=3, per_step=2, seed=0)
    cum = np.cumsum([sum(n_sup[i] for i in s) for s in long_plan])
    assert sft.plan_steps(n_sup, 1, 2, 0, match_tokens=int(cum[1])) == long_plan[:2]
    assert sft.plan_steps(n_sup, 1, 2, 0, match_tokens=int(cum[1]) + 1) == long_plan[:3]
    past_one_epoch = sft.plan_steps(n_sup, 1, 2, 0, match_tokens=int(cum[3]) + 1)  # one epoch has 26 tokens
    assert past_one_epoch == long_plan[:5]
    with pytest.raises(ValueError):
        sft.plan_steps(n_sup, 1, 2, 0, match_tokens=0)


def test_truncation_cuts_at_the_last_turn_boundary_and_reports_the_share(tmp_path):
    turns = np.array([[10, 30], [40, 70], [80, 100]], dtype=np.int32)
    assert sft.truncation_cut(turns, 100, None) == 100
    assert sft.truncation_cut(turns, 100, 100) == 100
    assert sft.truncation_cut(turns, 100, 75) == 70
    assert sft.truncation_cut(turns, 100, 69) == 30
    assert sft.truncation_cut(turns, 100, 5) == 0
    assert sft.supervised_tokens(np.array([True, True, False, True]), 3) == 1  # position 0 is never a target

    # 12-token prompt, 6-token tool responses: turns end at 32, 68 and 94 of 100 tokens; the second run has 28
    _, rows = write_arm(tmp_path, "random", 0, [[20, 30, 20], [10]])
    examples, report = sft.build_examples(rows, max_tokens=70)
    assert [e["input_ids"].shape[1] for e in examples] == [12 + 20 + 6 + 30, 28]
    assert [e["n_sup"] for e in examples] == [50, 10]
    assert report["truncated_episodes"] == 1
    assert report["truncated_supervised_share"] == pytest.approx(20 / 80)
    assert sft.build_examples(rows, None)[1]["truncated_supervised_share"] == 0.0
    with pytest.raises(ValueError, match="max_tokens"):
        sft.build_examples(rows, max_tokens=20)


def test_same_seed_and_data_give_the_same_loss_sequence():
    exs = [as_example(make_episode(i, [6 + i, 9])) for i in range(4)]

    def losses(seed):
        model = sft.attach(tiny_qwen35().float(), TRAIN, seed)
        plan = sft.plan_steps([e["n_sup"] for e in exs], epochs=2, per_step=2, seed=seed)
        return [r["loss"] for r in sft.train(model, exs, plan, lr=5e-3, warmup_steps=1, loss_norm=32.0, chunk=8)]

    first = losses(0)
    assert first == losses(0)
    assert first != losses(1)


def test_seed_sets_the_lora_init_whatever_the_global_rng():
    # tiny_qwen35() reseeds torch, but nothing does after from_pretrained in the script: scramble it here
    def lora_a(seed, ambient):
        model = tiny_qwen35().float()
        torch.manual_seed(ambient)
        return [p for n, p in sft.attach(model, TRAIN, seed).named_parameters() if "lora_A" in n]

    assert all(torch.equal(a, b) for a, b in zip(lora_a(0, 1), lora_a(0, 2)))
    assert not all(torch.equal(a, b) for a, b in zip(lora_a(0, 1), lora_a(1, 1)))


def test_gradients_are_clipped_to_grad_clip_before_the_step():
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    model = perturbed_lora_model()
    exs = [as_example(make_episode(4, [10, 10]))]
    seen = []

    def hook(opt, args, kwargs):
        grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
        seen.append(float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))))

    handle = register_optimizer_step_pre_hook(hook)
    try:
        rec = sft.train(model, exs, [[0]], lr=1e-3, warmup_steps=0, loss_norm=0.01, chunk=8)[0]
    finally:
        handle.remove()
    assert rec["grad_norm"] > 10 * sft.GRAD_CLIP  # a tiny loss_norm makes the raw norm large
    assert len(seen) == 1 and seen[0] <= sft.GRAD_CLIP + 1e-6


def test_inputs_are_validated_at_the_boundary(tmp_path):
    path, rows = write_arm(tmp_path, "stable", 2, [[5], [6]])
    assert len(sft.load_arm(str(path), "stable", 2)) == 2
    with pytest.raises(ValueError, match="seed"):
        sft.load_arm(str(path), "stable", 3)
    for bad, match in (({**rows[1], "task_id": "task0"}, "duplicate task_id"),
                       ({k: v for k, v in rows[1].items() if k != "episode_path"}, "missing"),
                       ({**rows[1], "episode_path": "attempts/att1.npz"}, "absolute")):
        path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match=match):
            sft.load_arm(str(path), "stable", 2)

    ep = make_episode(0, [5])
    np.savez(tmp_path / "short_mask.npz", **{**ep, "model_mask": ep["model_mask"][:-1]})
    np.savez(tmp_path / "no_turns.npz", **{k: v for k, v in ep.items() if k != "turn_spans"})
    np.savez(tmp_path / "no_model_tokens.npz", **{**ep, "model_mask": np.zeros_like(ep["model_mask"])})
    with pytest.raises(ValueError, match="model_mask"):
        sft.load_episode(str(tmp_path / "short_mask.npz"))
    with pytest.raises(ValueError, match="turn_spans"):
        sft.load_episode(str(tmp_path / "no_turns.npz"))
    with pytest.raises(ValueError, match="no model-written tokens"):
        sft.build_examples([{"attempt_id": "x", "episode_path": str(tmp_path / "no_model_tokens.npz")}])

    with pytest.raises(ValueError, match="in_proj_zz"):
        sft.attach(tiny_qwen35(), {**TRAIN, "targets": TARGETS + ["in_proj_zz"]}, 0)
    with pytest.raises(ValueError, match="include_gates"):
        sft.attach(tiny_qwen35(), {**TRAIN, "targets": TARGETS + ["in_proj_a"]}, 0)


def test_save_merged_writes_the_exact_f32_merge_not_the_bf16_one(tmp_path):
    import copy

    from safetensors.torch import load_file

    model = sft.attach(tiny_qwen35().to(torch.bfloat16), TRAIN, 0)  # bf16 base, as trained on the GPU
    plant_update(model, 1e-3)  # a small update: about half of it is below half a bf16 step
    expected, lora = {}, set()
    for name, mod in model.named_modules():
        if "default" in getattr(mod, "lora_A", {}):
            key = name.removeprefix("base_model.model.") + ".weight"
            a, b = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
            expected[key] = mod.base_layer.weight.float() + mod.scaling["default"] * (b @ a)
            lora.add(key)
    expected.update({k: v.float() for k, v in model.get_base_model().state_dict().items() if k not in expected})
    bf16_merge = copy.deepcopy(model).merge_and_unload().state_dict()

    sft.save_merged(model, tmp_path / "merged")
    saved = load_file(tmp_path / "merged" / "model.safetensors")
    assert {v.dtype for v in saved.values()} == {torch.float32}
    assert json.loads((tmp_path / "merged" / "config.json").read_text())["dtype"] == "float32"
    assert lora <= set(saved) and set(saved) <= set(expected)
    assert all(torch.equal(saved[k], expected[k]) for k in saved)  # bit-exact in f32
    differ = [k for k in lora if not torch.equal(saved[k], bf16_merge[k].float())]
    assert sorted(differ) == sorted(lora)


def load_script():
    spec = importlib.util.spec_from_file_location("p1_train_lora", os.path.join(ROOT, "scripts", "p1_train_lora.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_setup(tmp_path):
    """A tiny Qwen3.5 checkpoint with a tokenizer, one arm file and a config pointing at them."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    ckpt = tmp_path / "ckpt"
    tiny_qwen35().save_pretrained(ckpt)
    tok = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>").save_pretrained(ckpt)
    write_arm(tmp_path, "fewest_turns", 0, [[6, 9], [12], [5, 5, 5], [20]])
    cfg = {"model": {"path": str(ckpt)},
           "train": {**TRAIN, "learning_rate": 1e-2, "warmup_ratio": 0.25, "epochs": 2, "episodes_per_step": 2,
                     "loss_norm": 32, "max_tokens": None, "chunk": 16},
           "paths": {"arms_dir": str(tmp_path / "arms"), "runs_dir": str(tmp_path / "runs")}}
    path = tmp_path / "p1_train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, ckpt


def test_script_end_to_end_writes_logs_meta_and_a_merged_model(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_path, ckpt = write_setup(tmp_path)
    args = ["--config", str(cfg_path), "--arm", "fewest_turns", "--seed", "0", "--allow-slow-kernels",
            "--device", "cpu"]
    load_script().main(args)
    run = tmp_path / "runs" / "fewest_turns_seed0"
    log = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
    meta = json.loads((run / "train_meta.json").read_text())
    assert len(log) == meta["steps"] == math.ceil(2 * 4 / 2)
    assert set(log[0]) == {"step", "loss", "loss_per_token", "supervised_tokens", "lr", "grad_norm"}
    # 4 steps, warmup round(0.25 * 4) = 1, lr 1e-2: one warmup step, then linear decay
    assert [r["lr"] for r in log] == pytest.approx([1e-2, 1e-2, 2e-2 / 3, 1e-2 / 3])
    assert meta["arm"] == "fewest_turns" and meta["seed"] == 0 and meta["examples"] == 4 and meta["epochs"] == 2
    assert meta["supervised_tokens_seen"] == sum(r["supervised_tokens"] for r in log) == 2 * (15 + 12 + 15 + 20)
    assert meta["truncated_supervised_share"] == 0.0 and meta["loss_norm"] == 32
    assert meta["targets"] == TARGETS and meta["include_gates"] is False and meta["model"] == str(ckpt)
    assert set(meta["kernels"]["in_use"]) == {"chunk_gated_delta_rule", "causal_conv1d_fn"}
    assert meta["merged_dtype"] == "float32" and meta["update_size"]["modules"] == 4 * 3 + 3 * 3 + 1 * 4
    assert set(meta["update_size"]["by_type"]) == set(TARGETS)  # measured on the adapter, before the merge
    assert 0 < meta["update_size"]["carried_bf16"] <= 1.05

    base = AutoModelForCausalLM.from_pretrained(ckpt).state_dict()
    merged_model = AutoModelForCausalLM.from_pretrained(run / "merged")
    merged = merged_model.state_dict()
    assert merged_model.config.architectures == ["Qwen3_5ForCausalLM"]
    assert merged_model.config.dtype == torch.float32
    assert not any("lora" in k for k in merged)
    assert not torch.equal(base["model.layers.0.linear_attn.in_proj_qkv.weight"],
                           merged["model.layers.0.linear_attn.in_proj_qkv.weight"])
    conv = "model.layers.0.linear_attn.conv1d.weight"  # not a LoRA target
    assert torch.equal(base[conv], merged[conv])
    AutoTokenizer.from_pretrained(run / "merged")

    budget = meta["supervised_tokens_seen"] // 2
    load_script().main(args + ["--match-supervised-tokens", str(budget)])
    matched = json.loads((tmp_path / "runs" / f"fewest_turns_seed0_tokmatch{budget}" / "train_meta.json").read_text())
    assert matched["match_supervised_tokens"] == budget and matched["steps"] < meta["steps"]
    assert matched["supervised_tokens_seen"] >= budget


def test_kernel_guard_refuses_slow_kernels_without_the_flag(tmp_path, monkeypatch):
    cfg_path, _ = write_setup(tmp_path)
    script = load_script()
    # the trainer guards on the probe's check (qtrace.probe.kernel_paths), not a copy of it
    monkeypatch.setattr(script, "kernel_paths", lambda: {"chunk_gated_delta_rule": {"fast": False, "impl": "torch"},
                                                         "causal_conv1d_fn": {"fast": True, "impl": "causal_conv1d"}})
    with pytest.raises(SystemExit, match="allow-slow-kernels"):
        script.main(["--config", str(cfg_path), "--arm", "fewest_turns", "--seed", "0", "--device", "cpu"])
    assert not (tmp_path / "runs").exists()


def test_train_meta_records_whole_run_and_training_loop_cost(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    cfg_path, _ = write_setup(tmp_path)
    script = load_script()
    spans, meters = [], []
    readings = [(9000.0, 20.0), (5400.0, 7.2)]  # whole run, then training loop, in the order they are opened

    class FakeMeter:
        """Stands in for the NVML meter and notes which steps ran inside its block."""

        def __init__(self, gpu_index=0):
            self.gpu_index, self.n = gpu_index, len(meters)
            meters.append(self)

        def __enter__(self):
            spans.append(f"enter{self.n}")
            self.joules, self.seconds = None, None
            return self

        def __exit__(self, *exc):
            spans.append(f"exit{self.n}")
            self.joules, self.seconds = readings[self.n]
            return False

    idle = iter([50.0, 80.0])  # empty GPU, then loaded

    def fake_idle():
        spans.append("idle")
        return next(idle)

    def note(label, fn):
        return lambda *a, **k: spans.append(label) or fn(*a, **k)

    monkeypatch.setattr(script, "EnergyMeter", FakeMeter)
    monkeypatch.setattr(script, "idle_reading", fake_idle)
    monkeypatch.setattr(script.sft, "train", note("train", script.sft.train))
    monkeypatch.setattr(script.sft, "attach", note("attach", script.sft.attach))
    monkeypatch.setattr(script.sft, "save_merged", note("save", script.sft.save_merged))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", note("load", AutoModelForCausalLM.from_pretrained))
    script.main(["--config", str(cfg_path), "--arm", "fewest_turns", "--seed", "0", "--allow-slow-kernels",
                 "--device", "cpu"])
    meta = json.loads((tmp_path / "runs" / "fewest_turns_seed0" / "train_meta.json").read_text())
    # empty-GPU idle first, then the whole run spans load, train, merge and save; the loaded idle reading comes
    # after the adapter is attached and before the loop window, which spans train only
    assert spans == ["idle", "enter0", "load", "attach", "idle", "enter1", "train", "exit1", "save", "exit0"]
    assert [m.gpu_index for m in meters] == [0, 0]
    cost = meta["training_cost"]
    assert cost["primary"] == "whole_run" and cost["gpu_index"] == 0 and cost["null_reason"] is None
    assert cost["idle"] == {"none_w": 50.0, "loaded_w": 80.0, "settle_s": 45.0, "window_s": 10.0}
    assert cost["whole_run"] == {"wall_seconds": 20.0, "gpu_hours": None, "gpu_joules": 9000.0,
                                 "gpu_joules_idle_subtracted": 9000.0 - 50.0 * 20.0}
    assert cost["train_loop"] == {"wall_seconds": 7.2, "gpu_hours": None, "gpu_joules": 5400.0,
                                  "gpu_joules_idle_subtracted": pytest.approx(5400.0 - 80.0 * 7.2)}


def test_window_cost_gpu_hours_on_gpu_and_null_subtraction_without_idle():
    script = load_script()

    class Meter:
        joules, seconds = 7200.0, 1800.0

    assert script.window_cost(Meter, 2.0, True) == {"wall_seconds": 1800.0, "gpu_hours": 0.5, "gpu_joules": 7200.0,
                                                    "gpu_joules_idle_subtracted": 3600.0}
    assert script.window_cost(Meter, None, True)["gpu_joules_idle_subtracted"] is None


def test_idle_reading_settles_then_reads_and_is_null_without_cuda(monkeypatch):
    script = load_script()
    calls = []
    monkeypatch.setattr(script.time, "sleep", lambda s: calls.append(("sleep", s)))
    monkeypatch.setattr(script, "measure_idle_watts", lambda s, i: calls.append(("read", s, i)) or 61.5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert script.idle_reading() == 61.5
    assert calls == [("sleep", script.IDLE_SETTLE_S), ("read", script.IDLE_WINDOW_S, 0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls.clear()
    assert script.idle_reading() is None and calls == []


def test_train_meta_cost_off_cuda_is_wall_time_and_null_joules_with_a_reason(tmp_path, monkeypatch):
    cfg_path, _ = write_setup(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    script = load_script()
    script.main(["--config", str(cfg_path), "--arm", "fewest_turns", "--seed", "0", "--allow-slow-kernels",
                 "--device", "cpu"])
    cost = json.loads((tmp_path / "runs" / "fewest_turns_seed0" / "train_meta.json").read_text())["training_cost"]
    assert cost["idle"]["none_w"] is None and cost["idle"]["loaded_w"] is None and cost["null_reason"] == script.NO_NVML and cost["gpu_index"] == 0
    for window in ("whole_run", "train_loop"):
        assert cost[window]["wall_seconds"] > 0 and cost[window]["gpu_hours"] is None
        assert cost[window]["gpu_joules"] is None and cost[window]["gpu_joules_idle_subtracted"] is None
    assert cost["whole_run"]["wall_seconds"] > cost["train_loop"]["wall_seconds"]
