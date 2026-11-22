"""Fake-quant 4-bit weights in the LoRA forward pass: the random_fq baseline (docs/04, "random_fq baseline").

Every Linear and Embedding whose GGUF tensor the saved Q4_K_M table stores at a 4-bit type runs on
fake-quantized weights; every other tensor (Q5_K, Q6_K, Q8_0, F16, BF16, F32) is the skip list and runs
unquantized. A LoRA layer quantizes its merged weight W + scaling * B @ A, the weight llama-quantize will
see, with a straight-through estimator to the LoRA parameters. Only the forward changes: parameters,
state_dict and merge_and_unload are untouched, so the saved merged model holds the plain W + BA like
every other arm, and the stock llama-quantize run on it is the same as for every arm.

Table names follow convert_hf_to_gguf.py for Qwen3.5 (llama.cpp conversion/qwen.py, Qwen3_5TextModel).
What it does to the tensors that matter here:
- in_proj_qkv + in_proj_z are concatenated into attn_qkv_gate, in_proj_b + in_proj_a into ssm_beta_alpha
  (row concatenation, older converters wrote the parts): both source modules take the fused type.
- V-head reorder (16 key heads, 32 value heads): rows of in_proj_qkv (V part), in_proj_z, in_proj_a/b
  are permuted, which leaves every row's input-dim groups intact. out_proj's columns, its input dim,
  are permuted in whole 128-wide heads, so each 32-wide group still holds the same weights; only the
  pairing of heads inside a 256-wide Q4_K super-block changes, and this quantizer has no super-blocks.
- Value changes (A_log -> -exp(A_log), norm weights + 1, conv1d squeeze) and the dt_bias rename touch
  only F32 tensors that are not Linear weights, so nothing quantized here.
- The MTP block (blk.<num_hidden_layers>, nextn.*) has no module: Qwen3_5ForCausalLM drops it, so table
  tensors at that block index are ignored and listed in the summary.
"""

import json
import re

import torch
import torch.nn.functional as F

GROUP = 32  # llama.cpp Q4_K: 32-wide sub-blocks inside 256-wide super-blocks, along the input dim
LEVELS = 16
FOUR_BIT_TYPES = frozenset({"Q4_0", "Q4_1", "Q4_K", "IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4"})
FUSED = {"attn_qkv": "attn_qkv_gate", "attn_gate": "attn_qkv_gate",
         "ssm_beta": "ssm_beta_alpha", "ssm_alpha": "ssm_beta_alpha"}
BLOCK = re.compile(r"^blk\.(\d+)\.")
PROFILE = "Q4_K_M"  # the stock profile whose skip list random_fq uses (docs/04), not Q4_K_M_gate8


def fake_quant(w):
    """w rounded per row in groups of GROUP input columns to LEVELS evenly spaced values from the group min to max.

    Asymmetric min/max, round to nearest, computed in fp32 and returned in w's dtype; a constant group
    is returned exactly. Q4_K also quantizes each sub-block's scale and min to 6 bits against fp16
    super-block scales, clamps the min to <= 0 and searches the scale iteratively (make_qkx2_quants);
    none of that is reproduced, so this is close to, not equal to, what llama-quantize produces.
    """
    rows, cols = w.shape
    if cols % GROUP:
        raise ValueError(f"input dim {cols} is not a multiple of the group size {GROUP}")
    g = w.float().reshape(rows, cols // GROUP, GROUP)
    lo = g.amin(-1, keepdim=True)
    step = (g.amax(-1, keepdim=True) - lo) / (LEVELS - 1)
    q = torch.round((g - lo) / torch.where(step > 0, step, 1.0)).clamp(0, LEVELS - 1)
    return (lo + q * step).reshape(rows, cols).to(w.dtype)


def load_table(path):
    """({gguf_tensor_name: ggml_type_name}, {gguf_tensor_name: ggml shape}) from the stock Q4_K_M GGUF or the pinned JSON.

    The JSON is the data/quant_tables format (top-level llama_cpp_commit, model, note, argv, profiles), read
    at profiles["Q4_K_M"], each entry {type, shape, bytes}. Shapes are in ggml order, torch dims reversed.
    A flat {name: type} JSON is refused: without shapes, a table from another model could not be caught.
    """
    import gguf

    if path.endswith(".gguf"):
        reader = gguf.GGUFReader(path)
        field = reader.fields.get(gguf.Keys.General.FILE_TYPE)
        ftype = field.contents() if field else None
        # Any GGUF of this model covers the same names, so only the file type tells Q4_K_M from Q3_K_M or BF16.
        if ftype != gguf.LlamaFileType.MOSTLY_Q4_K_M:
            raise ValueError(f"{path}: general.file_type is {ftype}, not {int(gguf.LlamaFileType.MOSTLY_Q4_K_M)} "
                             "(MOSTLY_Q4_K_M); the skip list must come from the stock Q4_K_M file")
        return ({t.name: t.tensor_type.name for t in reader.tensors},
                {t.name: [int(d) for d in t.shape] for t in reader.tensors})
    with open(path) as f:
        doc = json.load(f)
    profiles = doc.get("profiles") if isinstance(doc, dict) else None
    profile = profiles.get(PROFILE) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict) or not profile:
        raise ValueError(f"{path}: expected the pinned table format with a non-empty profiles.{PROFILE} "
                         "{gguf_tensor_name: {type, shape, bytes}} (data/quant_tables)")

    def shaped(v):
        dims = v.get("shape") if isinstance(v, dict) else None
        return isinstance(dims, list) and dims and all(type(d) is int and d > 0 for d in dims)

    unshaped = sorted(k for k, v in profile.items() if not shaped(v))
    if unshaped:
        raise ValueError(f"{path}: entries without a ggml shape (positive ints): {unshaped[:5]}")
    bad = {k: v.get("type") for k, v in profile.items()
           if not isinstance(v.get("type"), str) or v["type"] not in gguf.GGMLQuantizationType.__members__}
    if bad:
        raise ValueError(f"{path}: unknown ggml types {dict(list(bad.items())[:5])}")
    return {k: v["type"] for k, v in profile.items()}, {k: v["shape"] for k, v in profile.items()}


def ggml_shapes(found):
    """{table name: ggml shape} the modules in `found` (tensor_types) imply: torch dims reversed, and a fused
    tensor's rows the sum of its parts' rows (the converter concatenates rows). A tied head counts once."""
    weights = {}
    for mod, hit in found.values():
        weights.setdefault(hit, {})[id(mod.weight)] = tuple(mod.weight.shape)
    return {hit: [*reversed(next(iter(ws.values()))[1:]), sum(s[0] for s in ws.values())]
            for hit, ws in weights.items()}


def _resolve(table, gguf_name):
    """The table entry holding gguf_name: itself, or the fused tensor the converter concatenates it into."""
    if gguf_name in table:
        return gguf_name
    stem, _, suffix = gguf_name.rpartition(".")
    blk, _, kind = stem.rpartition(".")
    fused = f"{blk}.{FUSED[kind]}.{suffix}" if kind in FUSED else None
    return fused if fused in table else None


def _hf_name(name):
    # peft keeps the original weight under <module>.base_layer; the converter renames dt_bias before mapping
    return name.replace(".base_layer", "").replace(".dt_bias", ".dt_proj.bias")


def tensor_types(base, table, shapes=None):
    """Table entry for each Linear / Embedding of the HF model `base` (a LoRA layer stands for its base layer).

    Returns {module_name: (module, gguf_name)} and the ignored MTP table names. Raises ValueError when a
    Linear or Embedding has no table entry, when its table shape (`shapes`, ggml order, as load_table
    returns it; None skips the check) differs from its weight's, when a table tensor (outside the MTP
    block) maps to no parameter (output.weight for a tied head among them), or when a 4-bit table tensor
    is not a Linear / Embedding weight.
    """
    import gguf
    from peft.tuners.lora import LoraLayer

    n_layers = base.config.num_hidden_layers
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, n_layers)
    tied = base.get_input_embeddings().weight
    found, missing = {}, []
    for name, mod in base.named_modules():
        if ".lora_" in name or name.endswith(".base_layer"):
            continue
        if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding, LoraLayer)):
            continue
        if name == "lm_head" and mod.weight is tied:
            # llama.cpp reads token_embd for a tied head; an output.weight entry is left unclaimed and raises
            # below, since it means the table is from an untied model.
            hit = _resolve(table, "token_embd.weight")
        else:
            hit = _resolve(table, tmap.get_name(f"{name}.weight", try_suffixes=(".weight",)) or "")
        if hit is None:
            missing.append(name)
        else:
            found[name] = (mod, hit)
    if missing:
        raise ValueError(f"no table entry for Linear/Embedding modules {missing[:8]} ({len(missing)} in all)")
    if shapes is not None:
        wrong = {h: (list(shapes[h]), s) for h, s in ggml_shapes(found).items() if list(shapes[h]) != s}
        if wrong:
            raise ValueError(f"table shapes (ggml order) differ from the model's weights, so the table is for "
                             f"another model: {dict(list(wrong.items())[:5])} as (table, model), {len(wrong)} in all")

    claimed = {hit for _, hit in found.values()}
    for name, _ in base.named_parameters():
        if ".lora_" not in name:
            claimed.add(_resolve(table, tmap.get_name(_hf_name(name), try_suffixes=(".weight", ".bias")) or ""))
    ignored = sorted(t for t in table if (m := BLOCK.match(t)) and int(m.group(1)) >= n_layers)
    orphan = sorted(set(table) - claimed - set(ignored))
    if orphan:
        raise ValueError(f"table tensors map to no module of the model: {orphan[:8]} ({len(orphan)} in all)")
    stray = sorted(t for t in claimed - {hit for _, hit in found.values()} if table.get(t) in FOUR_BIT_TYPES)
    if stray:
        raise ValueError(f"4-bit table tensors that are not Linear/Embedding weights: {stray[:8]}")
    return found, ignored


def merged_weight(mod, adapter):
    """W + scaling * B @ A in float32, bit for bit the weight sft.save_merged writes and llama-quantize reads.

    Not in W's dtype: with a bf16 base a bf16 W + BA drops the part of the update below half a bf16 step
    (issue #1), which the F32 merge keeps, so fake quant would round a different weight than the deployed one.
    """
    a, b = mod.lora_A[adapter].weight.float(), mod.lora_B[adapter].weight.float()
    return mod.base_layer.weight.float() + (b @ a) * mod.scaling[adapter]


def _lora_forward(mod):
    from peft.tuners.lora import Linear

    if not isinstance(mod, Linear) or len(mod.active_adapters) != 1:
        raise ValueError(f"fake quant supports one active adapter on lora.Linear, got {type(mod).__name__} "
                         f"with {mod.active_adapters}")
    adapter = mod.active_adapters[0]
    if getattr(mod.lora_dropout[adapter], "p", 0.0) > 0:  # peft uses nn.Identity for p = 0
        raise ValueError("fake quant needs lora_dropout = 0: the merged weight W + BA cannot express dropout on x")
    lora_a, lora_b, scaling, bias = mod.lora_A[adapter], mod.lora_B[adapter], mod.scaling[adapter], mod.base_layer.bias

    def forward(x, *args, **kwargs):
        # W changes every optimizer step, so quantize on the fly; under gradient checkpointing this reruns
        # in the recompute with the same W, A and B, so the saved tensors match.
        with torch.no_grad():
            wq = fake_quant(merged_weight(mod, adapter)).to(mod.base_layer.weight.dtype)
        y = F.linear(x, wq, bias)
        # Straight-through estimator: d out / d W_eff is taken as d out / d wq. z - z.detach() is exactly 0,
        # so the value is x @ wq^T, while A and B get the gradient of x @ (scaling * B @ A)^T, which is
        # the STE gradient of W_q = W_eff + (fq(W_eff) - W_eff).detach() without forming a full-size
        # weight gradient per layer. x is detached in z so that x's gradient comes only from wq.
        z = lora_b(lora_a(x.detach().to(lora_a.weight.dtype))) * scaling
        return y + (z - z.detach()).to(y.dtype)

    return forward


def _frozen_forward(mod, wq):
    if isinstance(mod, torch.nn.Embedding):
        return lambda ids: F.embedding(ids, wq, mod.padding_idx, mod.max_norm, mod.norm_type,
                                       mod.scale_grad_by_freq, mod.sparse)
    return lambda x: F.linear(x, wq, mod.bias)


def apply(model, table, shapes=None):
    """Make the forward of `model` (peft LoRA model or plain HF model) fake-quantize the table's 4-bit tensors.

    `table` and `shapes` are what load_table returns; shapes are checked against the weights (tensor_types).

    Frozen 4-bit layers use fq(W) computed once (W never changes; with the default targets that is
    only in_proj_a/in_proj_b, a few MB, and a tied 4-bit head shares the embedding's copy). Returns a
    summary for train_meta.json: the quantized and skipped module names and the ignored MTP tensors.
    merge_and_unload drops the LoRA wrappers and their override, but frozen modules keep theirs, so the
    merged model in memory still fake-quantizes them; its saved weights are plain.
    """
    from peft.tuners.lora import LoraLayer

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    found, ignored = tensor_types(base, table, shapes)
    cache, quantized, skipped = {}, [], []
    for name, (mod, hit) in found.items():
        if table[hit] not in FOUR_BIT_TYPES:
            skipped.append(name)
            continue
        if isinstance(mod, LoraLayer):
            mod.forward = _lora_forward(mod)
        else:
            if id(mod.weight) not in cache:
                cache[id(mod.weight)] = fake_quant(mod.weight.detach())
            mod.forward = _frozen_forward(mod, cache[id(mod.weight)])
        quantized.append(name)
    if not quantized:
        raise ValueError("the table stores no Linear/Embedding weight at a 4-bit type, so fake quant would change "
                         "nothing; it is not a Q4_K_M table")
    return {"group_size": GROUP, "levels": LEVELS, "quantized": quantized, "skipped": skipped,
            "ignored_table_tensors": ignored}
