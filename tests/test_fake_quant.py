"""Tests for the random_fq fake quant on a tiny Qwen3.5. Run:  python -m pytest tests/test_fake_quant.py -q"""

import copy
import json
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
from test_sft import TRAIN, as_example, load_script, make_episode, perturbed_lora_model, write_arm, write_setup

from qtrace import fake_quant as fq
from qtrace import sft
from qtrace.probe import attach_lora

gguf = pytest.importorskip("gguf")


def tiny_table(quant="Q4_K", keep="F32", overrides=None):
    """The Q4_K_M GGUF layout of the tiny model, written out by hand from the real file's tensor list."""
    t = {"token_embd.weight": "Q6_K", "output_norm.weight": keep}
    for i in range(3):  # linear-attention layers
        for n in ("attn_qkv_gate", "ssm_beta_alpha", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"):
            t[f"blk.{i}.{n}.weight"] = quant
        for n in ("ssm_conv1d.weight", "ssm_norm.weight", "ssm_a", "ssm_dt.bias", "attn_norm.weight",
                  "post_attention_norm.weight"):
            t[f"blk.{i}.{n}"] = keep
    for n in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        t[f"blk.3.{n}.weight"] = quant
    for n in ("attn_q_norm", "attn_k_norm", "attn_norm", "post_attention_norm"):
        t[f"blk.3.{n}.weight"] = keep
    return {**t, **(overrides or {})}


def pinned_table(model, types):
    """types in the pinned data/quant_tables format, as its Q4_K_M profile, with the shapes model implies.

    Only Linear/Embedding shapes are checked, so the other entries carry a placeholder.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    found, _ = fq.tensor_types(base, types)
    shapes = fq.ggml_shapes(found)
    profile = {k: {"type": v, "shape": shapes.get(k, [1]), "bytes": 0} for k, v in types.items()}
    return {"llama_cpp_commit": "0" * 40, "model": "tiny", "note": "", "argv": {}, "profiles": {"Q4_K_M": profile}}


def layer(model, path):
    return model.get_base_model().get_submodule(path)


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_quantizer_levels_error_idempotence_and_constant_groups():
    torch.manual_seed(0)
    w = torch.randn(16, 256) * torch.logspace(-3, 1, 256)  # groups at very different scales
    w[3, 32:64] = 0.37
    w[5, :32] = 0.0
    q = fq.fake_quant(w)
    g, gq = w.reshape(16, 8, 32), q.reshape(16, 8, 32)
    step = (g.amax(-1) - g.amin(-1)) / 15
    assert all(len(torch.unique(row)) <= 16 for row in gq.reshape(-1, 32))
    assert ((g - gq).abs().amax(-1) <= step / 2 * (1 + 1e-5) + 1e-12).all()
    assert torch.equal(q[3, 32:64], w[3, 32:64]) and torch.equal(q[5, :32], w[5, :32])
    # A second pass recomputes the step from the rounded max, so fp32 agrees to rounding; bf16 exactly.
    assert torch.allclose(fq.fake_quant(q), q, rtol=0, atol=1e-6 * float(step.max()))
    qb = fq.fake_quant(w.bfloat16())
    assert qb.dtype == torch.bfloat16 and torch.equal(fq.fake_quant(qb), qb)
    with pytest.raises(ValueError, match="multiple"):
        fq.fake_quant(torch.randn(4, 48))


def test_ste_gradient_to_lora_params_equals_the_manual_computation():
    model = perturbed_lora_model()
    fq.apply(model, tiny_table())
    mod = layer(model, "model.layers.0.mlp.down_proj")
    a, b, s = mod.lora_A["default"].weight, mod.lora_B["default"].weight, mod.scaling["default"]
    torch.manual_seed(1)
    x = torch.randn(2, 5, 128, requires_grad=True)
    g = torch.randn(2, 5, 64)
    out = mod(x)
    ga, gb, gx = torch.autograd.grad((out * g).sum(), [a, b, x])

    w_eff = mod.base_layer.weight + s * b @ a
    wq = fq.fake_quant(w_eff)
    gw = g.reshape(-1, 64).T @ x.reshape(-1, 128)  # STE: dL/dW_eff = dL/dW_q
    assert torch.allclose(out, x @ wq.T, atol=1e-6)
    assert torch.allclose(gb, s * gw @ a.T, atol=1e-5) and torch.allclose(ga, s * b.T @ gw, atol=1e-5)
    assert torch.allclose(gx, g @ wq, atol=1e-6)

    literal = x @ (w_eff + (wq - w_eff).detach()).T  # the textbook STE form gives the same gradients
    la, lb = torch.autograd.grad((literal * g).sum(), [a, b])
    assert torch.allclose(ga, la, atol=1e-5) and torch.allclose(gb, lb, atol=1e-5)


def test_a_table_with_no_4_bit_module_fails():
    # e.g. the BF16 GGUF given as the table: the run would be a plain random run saved as random_fq
    table = {k: "BF16" if v in ("Q4_K", "Q6_K") else v for k, v in tiny_table().items()}
    with pytest.raises(ValueError, match="no Linear/Embedding weight at a 4-bit type"):
        fq.apply(perturbed_lora_model(), table)


def test_forward_uses_fq_of_the_merged_weight_and_skipped_layers_stay_unquantized():
    plain = perturbed_lora_model()
    model = copy.deepcopy(plain)
    summary = fq.apply(model, tiny_table(overrides={"blk.1.ffn_down.weight": "Q6_K"}))
    assert "model.layers.1.mlp.down_proj" in summary["skipped"] and "lm_head" in summary["skipped"]
    assert {"model.layers.0.linear_attn.in_proj_a", "model.layers.0.linear_attn.in_proj_b",
            "model.layers.3.self_attn.q_proj"} <= set(summary["quantized"])
    assert "model.embed_tokens" in summary["skipped"]  # Q6_K, like the stock Q4_K_M file

    skipped = layer(model, "model.layers.1.mlp.down_proj")
    x = torch.randn(3, 128)
    assert "forward" not in vars(skipped)
    assert torch.equal(skipped(x), layer(plain, "model.layers.1.mlp.down_proj")(x))

    # Reference: merge, then overwrite each quantized weight with its fake quant; plain forward after that.
    ref = copy.deepcopy(plain).merge_and_unload()
    for name in summary["quantized"]:
        w = ref.get_submodule(name).weight
        w.data = fq.fake_quant(w.data)
    ids = torch.randint(0, 300, (1, 40))
    ours = logits(model, ids)
    assert torch.allclose(ours, logits(ref, ids), atol=1e-5)
    assert not torch.allclose(ours, logits(plain, ids), atol=1e-3)


def test_tied_head_and_embedding_follow_the_token_embd_type():
    model = perturbed_lora_model()
    summary = fq.apply(model, tiny_table(overrides={"token_embd.weight": "Q4_K"}))
    assert {"model.embed_tokens", "lm_head"} <= set(summary["quantized"])
    base = model.get_base_model()
    ids = torch.arange(300).unsqueeze(0)
    assert torch.equal(base.model.embed_tokens(ids)[0], fq.fake_quant(base.model.embed_tokens.weight))
    h = torch.randn(4, 64)
    assert torch.allclose(base.lm_head(h), h @ fq.fake_quant(base.lm_head.weight).T, atol=1e-6)


def test_saved_merged_weights_are_w_plus_scaled_ba_exactly_not_fake_quantized(tmp_path):
    from transformers import AutoModelForCausalLM

    model = sft.attach(tiny_qwen35().float(), TRAIN, 0)
    original = {k: v.clone() for k, v in model.get_base_model().state_dict().items() if "lora_" not in k}
    summary = fq.apply(model, tiny_table())
    exs = [as_example(make_episode(i, [7, 9])) for i in range(2)]
    sft.train(model, exs, [[0, 1], [1, 0]], lr=5e-2, warmup_steps=0, loss_norm=16.0, chunk=8)
    ids = torch.randint(0, 300, (1, 40))
    trained = logits(model, ids)
    base = model.get_base_model()
    expected = {}
    for name, mod in base.named_modules():
        if hasattr(mod, "lora_A"):
            a, b = mod.lora_A["default"].weight, mod.lora_B["default"].weight
            assert torch.equal(mod.base_layer.weight, original[f"{name}.base_layer.weight"])  # W never changed
            expected[f"{name}.weight"] = (mod.base_layer.weight + (b @ a) * mod.scaling["default"]).detach()
    sft.save_merged(model, tmp_path / "merged")  # the trainer's save path, with fake quant still applied
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
    saved = {k: v.clone() for k, v in reloaded.state_dict().items()}
    assert len(expected) == 3 * 6 + 7  # every LoRA target in 3 DeltaNet layers and 1 attention layer
    assert all(torch.equal(saved[k], v) for k, v in expected.items())
    for k in ("model.layers.0.linear_attn.in_proj_a.weight", "model.layers.3.self_attn.q_norm.weight"):
        assert torch.equal(saved[k], original[k])  # frozen, and in_proj_a was fake-quantized in the forward
    w = saved["model.layers.0.mlp.up_proj.weight"].reshape(-1, 32)
    assert max(len(torch.unique(row)) for row in w) > 16

    # After the optimizer steps the forward is fq of the current W + BA, not of the weights at apply time.
    for name in summary["quantized"]:
        q = reloaded.get_submodule(name).weight
        q.data = fq.fake_quant(q.data)
    assert torch.allclose(trained, logits(reloaded, ids), atol=1e-5)


def test_bf16_forward_quantizes_the_f32_merge_that_save_merged_writes(tmp_path):
    from safetensors.torch import load_file

    model = sft.attach(tiny_qwen35().to(torch.bfloat16), TRAIN, 0)  # bf16 base, as trained on the GPU
    plant_update(model, 1e-3)  # small: about half of it is lost if W + BA is rounded to bf16 first
    fq.apply(model, tiny_table())
    base = model.get_base_model()
    ours = {n: fq.merged_weight(m, "default") for n, m in base.named_modules() if hasattr(m, "lora_A")}
    assert {w.dtype for w in ours.values()} == {torch.float32}
    up = base.get_submodule("model.layers.0.mlp.up_proj")
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(up(x), F.linear(x, fq.fake_quant(ours["model.layers.0.mlp.up_proj"]).bfloat16()))
        bf16_eff = (up.base_layer.weight + up.get_delta_weight("default")).bfloat16()
        assert not torch.equal(up(x), F.linear(x, fq.fake_quant(bf16_eff)))
    sft.save_merged(model, tmp_path / "merged")  # the deployed weights llama-quantize will see
    saved = load_file(tmp_path / "merged" / "model.safetensors")
    assert all(torch.equal(saved[f"{n}.weight"], w) for n, w in ours.items())


def test_gradient_checkpointing_gives_the_same_gradients():
    def grads(ckpt):
        model = attach_lora(tiny_qwen35().float(), r=4, alpha=8, grad_ckpt=ckpt)
        torch.manual_seed(3)
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.data.normal_(0, 0.05)
        fq.apply(model, tiny_table())
        model.train()
        assert model.get_base_model().model.gradient_checkpointing is ckpt
        loss = sft.episode_loss(model, as_example(make_episode(5, [9, 12])), 10.0, chunk=8)
        return torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])

    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(grads(True), grads(False)))


def write_gguf(path, table, file_type=gguf.LlamaFileType.MOSTLY_Q4_K_M):
    """A GGUF with the table's names and types; 4-bit as Q4_0 and 8-bit as Q8_0 (the tiny dims fit 32-blocks)."""
    w = gguf.GGUFWriter(str(path), "qwen35")
    if file_type is not None:
        w.add_file_type(file_type)
    for name, kind in table.items():
        qtype = gguf.GGMLQuantizationType[kind]
        if qtype == gguf.GGMLQuantizationType.F32:
            w.add_tensor(name, np.zeros(4, dtype=np.float32))
        else:
            w.add_tensor(name, gguf.quants.quantize(np.zeros((2, 64), dtype=np.float32), qtype), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_table_parses_from_a_gguf_file_and_from_the_pinned_json(tmp_path):
    table = {k: {"Q4_K": "Q4_0", "Q6_K": "Q8_0"}.get(v, v) for k, v in tiny_table().items()}
    write_gguf(tmp_path / "t.gguf", table)
    types, shapes = fq.load_table(str(tmp_path / "t.gguf"))
    assert types == table
    assert shapes["blk.0.ffn_up.weight"] == [64, 2] and shapes["output_norm.weight"] == [4]  # ggml order
    pinned = pinned_table(perturbed_lora_model(), tiny_table())
    (tmp_path / "t.json").write_text(json.dumps(pinned))
    types, shapes = fq.load_table(str(tmp_path / "t.json"))
    assert types == tiny_table()
    assert shapes == {k: v["shape"] for k, v in pinned["profiles"]["Q4_K_M"].items()}
    a, b = perturbed_lora_model(), perturbed_lora_model()
    assert fq.apply(a, fq.load_table(str(tmp_path / "t.gguf"))[0]) == fq.apply(b, tiny_table())  # Q4_0 is 4-bit
    bad = copy.deepcopy(pinned)
    bad["profiles"]["Q4_K_M"]["blk.0.ffn_up.weight"]["type"] = "Q4_KM"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown ggml types"):
        fq.load_table(str(tmp_path / "bad.json"))
    bad["profiles"]["Q4_K_M"]["blk.0.ffn_up.weight"]["type"] = [1]  # unhashable: no raw TypeError
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown ggml types"):
        fq.load_table(str(tmp_path / "bad.json"))
    bad["profiles"]["Q4_K_M"]["blk.0.ffn_up.weight"] = {"type": "Q4_K"}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="without a ggml shape.*blk.0.ffn_up.weight"):
        fq.load_table(str(tmp_path / "bad.json"))
    for name, doc in (("flat", tiny_table()), ("q3", {**pinned, "profiles": {"Q3_K_M": {}}}),
                      ("list", {**pinned, "profiles": []})):
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))  # a flat table has no shapes to check
        with pytest.raises(ValueError, match="profiles.Q4_K_M"):
            fq.load_table(str(tmp_path / f"{name}.json"))
    for name, ftype in (("q3_k_m", gguf.LlamaFileType.MOSTLY_Q3_K_M), ("none", None)):
        write_gguf(tmp_path / f"{name}.gguf", table, ftype)
        with pytest.raises(ValueError, match="file_type is .*MOSTLY_Q4_K_M"):
            fq.load_table(str(tmp_path / f"{name}.gguf"))


def test_table_shapes_must_match_the_model():
    model = perturbed_lora_model()
    pinned = pinned_table(model, tiny_table())["profiles"]["Q4_K_M"]
    shapes = {k: v["shape"] for k, v in pinned.items()}
    # torch dims reversed; fused rows summed (qkv 2*2*16 + 2*16 = 96 plus z 32; b 2 plus a 2); tied head counted once
    assert shapes["blk.0.attn_qkv_gate.weight"] == [64, 128] and shapes["blk.0.ssm_beta_alpha.weight"] == [64, 4]
    assert shapes["token_embd.weight"] == [64, 300] and shapes["blk.3.ffn_down.weight"] == [128, 64]
    assert fq.apply(model, tiny_table(), shapes)["quantized"]
    for name, wrong in (("blk.0.attn_qkv_gate.weight", [64, 96]),  # the table of a converter that kept parts apart
                        ("blk.3.ffn_down.weight", [64, 128]),  # torch order instead of ggml order
                        ("token_embd.weight", [64, 301])):  # another vocabulary
        with pytest.raises(ValueError, match=f"shapes .*another model.*{name}"):
            fq.apply(perturbed_lora_model(), tiny_table(), {**shapes, name: wrong})


def test_unfused_names_and_the_mtp_block_are_handled():
    unfused = {k: v for k, v in tiny_table().items() if "qkv_gate" not in k and "beta_alpha" not in k}
    for i in range(3):
        unfused.update({f"blk.{i}.attn_qkv.weight": "Q4_K", f"blk.{i}.attn_gate.weight": "Q6_K",
                        f"blk.{i}.ssm_beta.weight": "Q4_K", f"blk.{i}.ssm_alpha.weight": "Q4_K"})
    unfused["blk.4.nextn.eh_proj.weight"] = "Q4_K"  # MTP block: past num_hidden_layers, no module
    summary = fq.apply(perturbed_lora_model(), unfused)
    assert "model.layers.0.linear_attn.in_proj_z" in summary["skipped"]
    assert "model.layers.0.linear_attn.in_proj_qkv" in summary["quantized"]
    assert summary["ignored_table_tensors"] == ["blk.4.nextn.eh_proj.weight"]


def test_coverage_errors_fail_loudly():
    no_entry = {k: v for k, v in tiny_table().items() if k != "blk.2.ffn_up.weight"}
    with pytest.raises(ValueError, match=r"no table entry.*layers\.2\.mlp\.up_proj"):
        fq.apply(perturbed_lora_model(), no_entry)
    with pytest.raises(ValueError, match="map to no module.*blk.1.ffn_extra.weight"):
        fq.apply(perturbed_lora_model(), tiny_table(overrides={"blk.1.ffn_extra.weight": "Q4_K"}))
    with pytest.raises(ValueError, match="not Linear/Embedding.*ssm_conv1d"):
        fq.apply(perturbed_lora_model(), tiny_table(overrides={"blk.0.ssm_conv1d.weight": "Q4_0"}))
    untied = tiny_table(overrides={"output.weight": "Q6_K", "token_embd.weight": "Q4_K"})  # an untied model's table
    with pytest.raises(ValueError, match=r"map to no module.*'output\.weight'"):
        fq.apply(perturbed_lora_model(), untied)


def test_lora_dropout_above_zero_fails():
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1, target_modules=["up_proj"], task_type="CAUSAL_LM")
    with pytest.raises(ValueError, match="lora_dropout"):
        fq.apply(get_peft_model(tiny_qwen35(), config), tiny_table())


def test_script_trains_random_fq_and_saves_plain_merged_weights(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    cfg_path, ckpt = write_setup(tmp_path)
    write_arm(tmp_path, "random", 0, [[6, 9], [12], [5, 5, 5], [20]])
    (tmp_path / "table.json").write_text(json.dumps(pinned_table(tiny_qwen35(), tiny_table())))
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["fake_quant"] = {"enabled": True, "table": str(tmp_path / "table.json")}
    cfg_path.write_text(yaml.safe_dump(cfg))
    script, order = load_script(), []
    real_idle, real_apply = script.idle_reading, script.fake_quant.apply
    monkeypatch.setattr(script, "idle_reading", lambda: order.append("idle") or real_idle())
    monkeypatch.setattr(script.fake_quant, "apply", lambda *a: order.append("fq") or real_apply(*a))
    script.main(["--config", str(cfg_path), "--arm", "random", "--seed", "0", "--allow-slow-kernels",
                 "--device", "cpu"])
    assert order == ["idle", "fq", "idle"]  # the loaded-idle reading sees the fake-quantized model, before step 1
    run = tmp_path / "runs" / "random_fq_seed0"
    meta = json.loads((run / "train_meta.json").read_text())
    assert meta["arm"] == "random" and meta["fake_quant"]["table"] == str(tmp_path / "table.json")
    assert "model.layers.0.linear_attn.in_proj_a" in meta["fake_quant"]["quantized"]
    # the same F32 save and update-size record as every other arm (issue #1)
    assert meta["merged_dtype"] == "float32" and set(meta["update_size"]["by_type"]) == set(TRAIN["targets"])
    assert 0 < meta["update_size"]["carried_bf16"] <= 1.05
    # the same training-cost record (owner answer on #15): whole run primary, training loop secondary
    cost = meta["training_cost"]
    assert cost["primary"] == "whole_run" and cost["gpu_index"] == 0 and "null_reason" in cost
    assert set(cost["idle"]) == {"none_w", "loaded_w", "settle_s", "window_s"}
    for window in ("whole_run", "train_loop"):
        assert set(cost[window]) == {"wall_seconds", "gpu_hours", "gpu_joules", "gpu_joules_idle_subtracted"}
    assert cost["whole_run"]["wall_seconds"] > cost["train_loop"]["wall_seconds"] > 0
    base = AutoModelForCausalLM.from_pretrained(ckpt).state_dict()
    merged = AutoModelForCausalLM.from_pretrained(run / "merged").state_dict()
    assert json.loads((run / "merged" / "config.json").read_text())["dtype"] == "float32"
    gate = "model.layers.0.linear_attn.in_proj_a.weight"  # frozen, 4-bit in the table, saved unquantized
    assert torch.equal(base[gate], merged[gate])
    up = "model.layers.0.mlp.up_proj.weight"
    assert not torch.equal(base[up], merged[up]) and not torch.equal(merged[up], fq.fake_quant(merged[up]))

    # Fake quant was on while sft.train ran: LoRA B starts at 0, so step 0 differs from the plain run only by fq.
    cfg["fake_quant"] = {"enabled": False, "table": None}
    cfg_path.write_text(yaml.safe_dump(cfg))
    load_script().main(["--config", str(cfg_path), "--arm", "random", "--seed", "0", "--allow-slow-kernels",
                        "--device", "cpu"])
    fq_loss, plain_loss = (json.loads((tmp_path / "runs" / r / "loss.jsonl").read_text().splitlines()[0])["loss"]
                           for r in ("random_fq_seed0", "random_seed0"))
    assert abs(fq_loss - plain_loss) > 1e-5  # about 5e-4 here; equal seeds make the plain rerun bit-identical

    cfg["fake_quant"] = {"enabled": True, "table": None}
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fake_quant.table"):
        load_script().main(["--config", str(cfg_path), "--arm", "random", "--seed", "0", "--allow-slow-kernels",
                            "--device", "cpu"])


@pytest.mark.skipif(not (os.environ.get("QTRACE_Q4KM_GGUF") and os.environ.get("QTRACE_MODEL_DIR")),
                    reason="set QTRACE_Q4KM_GGUF and QTRACE_MODEL_DIR to check the real Qwen3.5-4B table")
def test_real_q4_k_m_table_covers_qwen35_4b():
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(os.environ["QTRACE_MODEL_DIR"]).text_config
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    found, ignored = fq.tensor_types(model, *fq.load_table(os.environ["QTRACE_Q4KM_GGUF"]))  # GGUF or pinned JSON
    n = config.num_hidden_layers
    assert all(t.startswith(f"blk.{n}.") for t in ignored)  # empty for the pinned table (--no-mtp)
    assert found["lm_head"][1] == "token_embd.weight"
    assert found["model.layers.0.linear_attn.in_proj_z"][1] == "blk.0.attn_qkv_gate.weight"
