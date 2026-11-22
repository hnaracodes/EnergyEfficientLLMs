"""Tests for the training probe on a tiny Qwen3.5. Run:  python -m pytest tests/test_probe.py -q"""

import importlib.util
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import torch
import yaml

from qtrace.common import load_config
from qtrace.probe import (
    ATTENTION_TARGETS,
    DELTANET_TARGETS,
    GATE_TARGETS,
    MLP_TARGETS,
    attach_lora,
    chunked_ce_loss,
    kernel_paths,
    kernel_status,
    lora_update_size,
    probe_length,
    supervised_mask,
)


def tiny_qwen35(seed=0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(cfg)


def test_lora_covers_deltanet_mixers_and_attention():
    model = attach_lora(tiny_qwen35(), include_gates=False)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for target in DELTANET_TARGETS + ["q_proj", "o_proj", "down_proj"]:
        assert any(f"{target}.lora_A" in n for n in names), target
    assert not any(f"{g}.lora_A" in n for g in GATE_TARGETS for n in names)


def test_lora_gates_are_optional():
    model = attach_lora(tiny_qwen35(), include_gates=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(any(f"{g}.lora_A" in n for n in names) for g in GATE_TARGETS)


def test_probe_length_reports_time_memory_and_trains():
    model = attach_lora(tiny_qwen35(), include_gates=False)
    for path in ("chunked", "full_logits"):
        rec = probe_length(model, length=96, steps=2, device="cpu", vocab_size=300, seed=0,
                           supervised_frac=0.5, loss_path=path, chunk=16)
        assert rec["status"] == "ok" and rec["length"] == 96 and rec["loss_path"] == path
        assert rec["sec_per_step"] > 0 and rec["loss_finite"] and rec["lora_grad_norm"] > 0
        assert rec["n_supervised"] > 0


def test_supervised_mask_hits_the_fraction_in_contiguous_spans():
    m = supervised_mask(8192, 0.15, period=2048)
    assert not m[0]
    assert abs(m.float().mean().item() - 0.15) < 0.01
    assert (m[1:] != m[:-1]).sum().item() <= 2 * 4  # at most one span per period


def test_chunked_loss_matches_full_logits_loss_and_gradients():
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35().float(), include_gates=False, grad_ckpt=False)
    ids = torch.randint(0, 300, (1, 64))
    mask = supervised_mask(64, 0.5, period=16)
    labels = torch.where(mask, ids[0], torch.full_like(ids[0], -100)).unsqueeze(0)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # LoRA B starts at zero; make the comparison non-trivial
        p.data.normal_(0, 0.02)

    full = model(input_ids=ids, labels=labels).loss
    g_full = torch.autograd.grad(full, params)
    chunked = chunked_ce_loss(model, ids, mask, chunk=7)
    g_chunk = torch.autograd.grad(chunked, params)
    assert torch.allclose(full, chunked, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_full, g_chunk))


def test_kernel_status_names_both_kernels():
    status = kernel_status()
    assert set(status) == {"fla", "causal_conv1d"}


def test_kernel_paths_report_what_transformers_resolved():
    paths = kernel_paths()
    assert set(paths) == {"chunk_gated_delta_rule", "causal_conv1d_fn"}
    status = kernel_status()
    if not status["fla"]:
        assert not paths["chunk_gated_delta_rule"]["fast"]
        assert paths["chunk_gated_delta_rule"]["impl"].endswith("modeling_qwen3_5.torch_chunk_gated_delta_rule")
    if not status["causal_conv1d"]:
        assert not paths["causal_conv1d_fn"]["fast"]


P1_CONFIG = os.path.join(ROOT, "configs", "p1_train.yaml")


def test_p1_config_and_probe_defaults_agree():
    # attach_lora's defaults are the lists the probe module documents; P1 must train the same adapter.
    tr = load_config(P1_CONFIG)["train"]
    defaults = inspect.signature(attach_lora).parameters
    assert tr["targets"] == ATTENTION_TARGETS + MLP_TARGETS + DELTANET_TARGETS
    assert tr["lora_r"] == defaults["r"].default and tr["lora_alpha"] == defaults["alpha"].default
    assert tr["include_gates"] is defaults["include_gates"].default


def load_script():
    path = os.path.join(ROOT, "scripts", "p0_training_probe.py")
    spec = importlib.util.spec_from_file_location("p0_training_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SLOW_IN_USE = {"chunk_gated_delta_rule": {"fast": False, "impl": "torch"},
               "causal_conv1d_fn": {"fast": True, "impl": "causal_conv1d"}}


def test_script_refuses_importable_kernels_that_are_not_in_use(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "kernel_status", lambda: {"fla": True, "causal_conv1d": True})
    monkeypatch.setattr(script, "kernel_paths", lambda: SLOW_IN_USE)
    out = tmp_path / "probe.jsonl"
    with pytest.raises(SystemExit, match="allow-slow-kernels"):
        script.main(["--model", str(tmp_path / "no_model"), "--config", P1_CONFIG, "--out", str(out)])
    assert not out.exists()


def test_script_records_kernels_in_use_in_every_row(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    tiny_qwen35().save_pretrained(ckpt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    out = tmp_path / "probe.jsonl"
    load_script().main(["--model", str(ckpt), "--lengths", "48", "64", "--steps", "1", "--chunk", "16",
                        "--supervised-frac", "0.5", "--allow-slow-kernels", "--config", P1_CONFIG, "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
    assert all(r["kernels"] == {"in_use": kernel_paths(), "importable": kernel_status()} for r in rows)


def test_script_trains_and_records_the_config_lora(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    tiny_qwen35().save_pretrained(ckpt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cfg = tmp_path / "p1_train.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"targets": ["q_proj", "in_proj_qkv"], "include_gates": False,
                                             "lora_r": 4, "lora_alpha": 8}}))
    script = load_script()
    seen, real = [], script.probe_length

    def spy(model, *args, **kwargs):
        names = {n.split(".lora_A.")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters() if ".lora_A." in n}
        seen.append((names, model.peft_config["default"].r, model.peft_config["default"].lora_alpha))
        return real(model, *args, **kwargs)

    monkeypatch.setattr(script, "probe_length", spy)
    out = tmp_path / "probe.jsonl"
    base = ["--model", str(ckpt), "--lengths", "48", "--steps", "1", "--chunk", "16", "--supervised-frac", "0.5",
            "--allow-slow-kernels", "--config", str(cfg), "--out", str(out)]
    script.main(base)
    script.main(base + ["--include-gates"])
    assert seen == [({"q_proj", "in_proj_qkv"}, 4, 8), ({"q_proj", "in_proj_qkv", *GATE_TARGETS}, 4, 8)]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["include_gates"] for r in rows] == [False, True]
    assert all(r["targets"] == ["q_proj", "in_proj_qkv"] and r["lora_r"] == 4 and r["lora_alpha"] == 8
               and r["config"] == str(cfg) for r in rows)


def plant_update(model, rel, seed=0):
    """Set every LoRA B so each module's scaling * B @ A has ||.||_F / ||W||_F = rel (the owner's plant, issue #1).

    rel is one size for every module, or a function of the module name.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if "default" not in getattr(mod, "lora_A", {}):
                continue
            a, b, s = mod.lora_A["default"].weight, mod.lora_B["default"].weight, mod.scaling["default"]
            bn = torch.randn(b.shape, generator=g)
            size = rel(name) if callable(rel) else rel
            c = size * mod.base_layer.weight.float().norm() / (s * (bn @ a.float())).norm()
            b.copy_((bn * c).to(b.dtype))


def test_lora_update_size_gives_back_a_planted_size_by_module_type():
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35().to(torch.bfloat16), include_gates=True, grad_ckpt=False)
    assert lora_update_size(model)["carried_bf16"] is None  # B = 0 at init: no update yet
    plant_update(model, 3e-3)
    size = lora_update_size(model)
    kinds = set(ATTENTION_TARGETS + MLP_TARGETS + DELTANET_TARGETS + GATE_TARGETS)
    assert set(size["by_type"]) == kinds
    assert size["modules"] == sum(t["modules"] for t in size["by_type"].values()) == 4 * 3 + 3 * 5 + 1 * 4
    for t in size["by_type"].values():  # B is stored in bf16, so the planted size holds to its rounding
        assert t["rel_p10"] == pytest.approx(3e-3, rel=1e-2) and t["rel_p90"] == pytest.approx(3e-3, rel=1e-2)
        assert t["rel_p10"] <= t["rel_median"] <= t["rel_p90"]


def test_lora_update_size_carried_share_is_near_0_below_half_a_bf16_step_and_near_1_above():
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35().to(torch.bfloat16), include_gates=False, grad_ckpt=False)
    plant_update(model, 1e-5)  # far below half a bf16 step (2**-9 of each weight): rounds back to W
    small = lora_update_size(model)
    plant_update(model, 1e-3)  # the owner's round trip kept 0.52 of a 1e-3 update in the BF16 file
    mid = lora_update_size(model)
    plant_update(model, 1e-1)
    large = lora_update_size(model)
    assert small["carried_bf16"] < 0.05 and 0.4 < mid["carried_bf16"] < 0.65 and large["carried_bf16"] > 0.99
    assert all(t["carried_bf16"] < 0.1 for t in small["by_type"].values())
    assert all(t["carried_bf16"] > 0.98 for t in large["by_type"].values())


def carried_terms(model, suffix="", reference="bf16"):
    """(<kept, dW>, ||dW||^2) per adapted module, kept = bf16(W + dW) - ref with ref = bf16(W) or W itself."""
    terms = []
    with torch.no_grad():
        for name, mod in model.named_modules():
            if "default" not in getattr(mod, "lora_A", {}) or not name.endswith(suffix):
                continue
            w = mod.base_layer.weight.float()
            dw = mod.scaling["default"] * (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float())
            ref = w.bfloat16().float() if reference == "bf16" else w
            kept = (w + dw).bfloat16().float() - ref
            terms.append((float((kept * dw).sum()), float((dw * dw).sum())))
    return terms


def test_lora_update_size_percentiles_and_carried_share_weigh_modules_of_different_size():
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35().to(torch.bfloat16), include_gates=False, grad_ckpt=False)
    sizes = [1e-5, 1e-3, 3e-3, 1e-1]  # one per layer's down_proj, across the bf16 rounding edge

    def size_of(name):
        return sizes[int(name.split("layers.")[1].split(".")[0])] if name.endswith("down_proj") else 1e-3

    plant_update(model, size_of)
    down = lora_update_size(model)["by_type"]["down_proj"]
    p10, p50, p90 = np.percentile(sizes, [10, 50, 90])
    assert down["modules"] == 4
    assert down["rel_p10"] == pytest.approx(p10, rel=1e-2) and down["rel_median"] == pytest.approx(p50, rel=1e-2)
    assert down["rel_p90"] == pytest.approx(p90, rel=1e-2)
    terms = carried_terms(model, "down_proj")
    ratio_of_sums = sum(n for n, _ in terms) / sum(d for _, d in terms)
    mean_of_ratios = float(np.mean([n / d for n, d in terms]))
    assert down["carried_bf16"] == pytest.approx(ratio_of_sums, rel=1e-6)
    assert abs(ratio_of_sums - mean_of_ratios) > 0.2  # the 1e-1 module dominates the sum


def test_lora_update_size_carried_share_uses_bf16_w_as_reference_on_an_f32_base():
    # An f32 W lies off the bf16 grid, so a small dW still moves some elements across a rounding edge and
    # the share comes out near 1 on average. Against W itself the rounding error of W swamps a 1e-6 update.
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35(), include_gates=False, grad_ckpt=False)
    plant_update(model, 1e-6)
    carried = lora_update_size(model)["carried_bf16"]
    bf16_ref, w_ref = (carried_terms(model, reference=r) for r in ("bf16", "w"))
    assert carried == pytest.approx(sum(n for n, _ in bf16_ref) / sum(d for _, d in bf16_ref), rel=1e-6)
    assert abs(carried - sum(n for n, _ in w_ref) / sum(d for _, d in w_ref)) > 1


def test_script_reports_the_update_size_after_each_row(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    tiny_qwen35().save_pretrained(ckpt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    out = tmp_path / "probe.jsonl"
    load_script().main(["--model", str(ckpt), "--lengths", "48", "64", "--steps", "1", "--chunk", "16",
                        "--supervised-frac", "0.5", "--allow-slow-kernels", "--config", P1_CONFIG, "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["optimizer_steps"] for r in rows] == [2, 2] and [r["adapter_steps"] for r in rows] == [2, 4]
    for r in rows:
        size = r["update_size"]
        assert set(size["by_type"]) == set(ATTENTION_TARGETS + MLP_TARGETS + DELTANET_TARGETS)
        assert 0 <= size["carried_bf16"] <= 1.05
        assert all(t["rel_median"] > 0 for t in size["by_type"].values())
    # the adapter keeps training across rows, so the update grows
    q = [r["update_size"]["by_type"]["q_proj"]["rel_median"] for r in rows]
    assert q[1] > q[0]
