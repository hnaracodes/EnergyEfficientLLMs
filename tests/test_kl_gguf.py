"""Tests for the llama.cpp KL path.

The tiny-model tests run by default. They build qtrace/kl_score.cpp against a llama.cpp tree, convert
two tiny random Qwen3.5 hybrids (P and a perturbed Q) with that tree's convert_hf_to_gguf.py --no-mtp,
and check kl_gguf against kl.compare on the same HF models, per position, on the CPU backend. The tree
is QTRACE_LLAMA_CPP, else the one the installed gguf-py package comes from (the converter needs the
gguf-py of its own commit); QTRACE_KL_SCORE replaces the build with an existing helper. They skip if
no tree is found.

The real-model tests also need two GGUF files of one model:

    QTRACE_KL_SCORE=path/to/kl_score QTRACE_GGUF_P=bf16.gguf QTRACE_GGUF_Q=q3_k_m.gguf \
        python -m pytest tests/test_kl_gguf.py -q
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qtrace import kl_gguf
from qtrace.kl import compare, hidden_at, logit_index, score_stats
from qtrace.kl_gguf import compare_gguf, score_trajectories_gguf

BIN, P, Q = (os.environ.get(k) for k in ("QTRACE_KL_SCORE", "QTRACE_GGUF_P", "QTRACE_GGUF_Q"))
real = pytest.mark.skipif(not (BIN and P and Q), reason="QTRACE_KL_SCORE, QTRACE_GGUF_P and QTRACE_GGUF_Q not set")

# Per-position absolute tolerances for GGUF (f32 weights, CPU, F16 KV cache) against PyTorch (f32).
# Observed maxima on an M5 Pro: kl 8e-5, entropy 4e-5, nll 6e-4. An off-by-one target moves nll by
# about 0.9 on average and a swapped P and Q moves kl by 0.03 and entropy by 0.05.
ATOL = {"kl": 1e-3, "entropy_p": 1e-3, "nll_p": 5e-3, "nll_q": 5e-3}
VOCAB = 320

# convert_hf_to_gguf names the BPE pre-tokenizer by hashing the encoding of a fixed text, and refuses
# a tokenizer it does not know. The tiny byte-level tokenizer stands in for Qwen3.5's, so it is given
# Qwen3.5's pre-tokenizer name; the pre-tokenizer only matters for text, and these tests pass ids.
CONVERT = """
import runpy, sys
tree, model, out = sys.argv[1:4]
sys.path.insert(0, tree)
from conversion import base
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
sys.argv = ["convert_hf_to_gguf.py", model, "--no-mtp", "--outtype", "f32", "--outfile", out]
runpy.run_path(tree + "/convert_hf_to_gguf.py", run_name="__main__")
"""


def episode(n=700, seed=0, vocab=150_000):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, n), generator=g)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[50:300] = True
    mask[420:690] = True
    return {"input_ids": ids, "model_mask": mask, "command_spans": [(100, 140), (500, 520)],
            "tool_call_positions": [99, 499], "turn_spans": [(50, 300), (420, 690)]}


def llama_tree():
    tree = os.environ.get("QTRACE_LLAMA_CPP")
    if not tree:
        try:
            import gguf
        except ImportError:
            return None
        tree = Path(gguf.__file__).resolve().parents[2]
    tree = Path(tree)
    return tree if (tree / "convert_hf_to_gguf.py").exists() and (tree / "build" / "bin").is_dir() else None


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Paths of the helper and of the tiny GGUFs, and the HF models they were converted from."""
    tree = llama_tree()
    if tree is None:
        pytest.skip("no llama.cpp tree: set QTRACE_LLAMA_CPP or install its gguf-py")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError:
        pytest.skip("transformers without Qwen3.5")
    d = tmp_path_factory.mktemp("tiny_gguf")
    binary = BIN or build_helper(tree, d / "kl_score")

    def tokenizer(rename=None):
        merges = [("\u0120", "t"), ("h", "e"), ("\u0120t", "he")]  # \u0120 is byte-level space; llama.cpp needs merges
        tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["".join(m) for m in merges]
        tokens += ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        if rename:
            tokens[tokens.index(rename[0])] = rename[1]
        tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")

    # The 4B's layout at toy size: 3 DeltaNet layers per attention layer, the attention output gate
    # and interleaved M-RoPE. initializer_range 0.1 gives peaked logits, so a wrong target shows.
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=4096, initializer_range=0.1,
        tie_word_embeddings=True, attn_output_gate=True, bos_token_id=None, pad_token_id=None, eos_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [2, 1, 1]})
    torch.manual_seed(0)
    model_p = Qwen3_5ForCausalLM(cfg).eval()
    model_q = Qwen3_5ForCausalLM(cfg).eval()
    model_q.load_state_dict(model_p.state_dict())
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for w in model_q.parameters():
            if w.numel() > 1:
                w.add_(torch.randn(w.shape, generator=g) * 0.3 * w.std())

    def convert(name, model, tok):
        model.save_pretrained(d / name)
        tok.save_pretrained(d / name)
        out = d / f"{name}.gguf"
        r = subprocess.run([sys.executable, "-c", CONVERT, str(tree), str(d / name), str(out)], capture_output=True,
                           text=True, check=False)
        assert r.returncode == 0, r.stderr[-3000:]
        return str(out)

    # other_vocab is P with one token's text changed: same size and type, different vocabulary.
    return {"bin": binary, "p": convert("p", model_p, tokenizer()), "q": convert("q", model_q, tokenizer()),
            "other_vocab": convert("v", model_p, tokenizer(rename=("<|im_start|>", "<|im_begin|>"))), "model_p": model_p,
            "model_q": model_q}


def build_helper(tree, out):
    """The build command of qtrace/kl_gguf.py's docstring."""
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    if not cxx:
        pytest.skip("no C++ compiler to build kl_score")
    lib = tree / "build" / "bin"
    r = subprocess.run([cxx, "-std=c++17", "-O3", str(REPO / "qtrace" / "kl_score.cpp"), f"-I{tree}/include",
                        f"-I{tree}/ggml/include", f"-L{lib}", "-lllama", "-lggml", "-lggml-base",
                        f"-Wl,-rpath,{lib}", "-o", str(out)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    return str(out)


def reference(t, e):
    """kl.compare() on the HF models, the PyTorch path P0d's GGUF scores must equal."""
    idx = logit_index(e["model_mask"])
    p, q = t["model_p"], t["model_q"]
    return idx, compare(p.lm_head, hidden_at(p, e["input_ids"], idx), q.lm_head, hidden_at(q, e["input_ids"], idx),
                        e["input_ids"][0, idx + 1])


@pytest.mark.parametrize("n_batch", [512, 7])
def test_tiny_hybrid_matches_pytorch(tiny, n_batch):
    # n_batch 7 splits the 700 tokens into 100 chunks, so the DeltaNet state is carried 99 times.
    e = episode(vocab=VOCAB)
    idx, ref = reference(tiny, e)
    s, = compare_gguf(tiny["bin"], tiny["p"], tiny["q"], [(e["input_ids"], idx)], n_batch=n_batch, cpu=True)
    # The test can only catch a wrong target or a swapped P and Q if those change the numbers.
    assert (ref["nll_p"][1:] - ref["nll_p"][:-1]).abs().mean() > 100 * ATOL["nll_p"]
    assert (ref["nll_p"] - ref["nll_q"]).abs().mean() > 100 * ATOL["nll_p"]
    for k, atol in ATOL.items():
        assert (s[k] - ref[k]).abs().max() <= atol, (k, float((s[k] - ref[k]).abs().max()))
    assert (s["flip"] != ref["flip"]).float().mean() <= 0.01
    assert s["runtime"]["devices"] == ["CPU"] and s["runtime"]["n_batch"] == n_batch
    assert s["runtime"]["type_k"] == "f16" and s["runtime"]["flash_attn"] == "enabled"


def test_tiny_top_k_matches_pytorch(tiny):
    e = episode(vocab=VOCAB)
    idx = logit_index(e["model_mask"])
    s, = compare_gguf(tiny["bin"], tiny["p"], tiny["q"], [(e["input_ids"], idx)], cpu=True, top_k=8)
    lp = torch.log_softmax(tiny["model_p"].lm_head(hidden_at(tiny["model_p"], e["input_ids"], idx)), -1)
    top = lp.topk(8, dim=-1)
    assert torch.equal(s["topk_ids"][:, 0], top.indices[:, 0])
    assert (s["topk_logprob"] - top.values).abs().max() <= ATOL["entropy_p"]
    assert (s["topk_logprob"][:, 1:] <= s["topk_logprob"][:, :-1]).all()


def test_tiny_numpy_episode_scores_like_torch(tiny):
    e = episode(vocab=VOCAB)
    as_numpy = {**e, "input_ids": e["input_ids"][0].numpy(), "model_mask": e["model_mask"].numpy()}
    a, b = score_trajectories_gguf(tiny["bin"], tiny["p"], tiny["q"], [e, as_numpy], cpu=True)
    assert a == b


def test_tiny_bad_job_exits_1_and_names_it(tiny):
    # Not --cpu: on Metal, exit() used to abort in the backend teardown, so every error was exit 134.
    good, bad = episode(vocab=VOCAB), episode(vocab=VOCAB)
    bad["input_ids"][0, 5] = VOCAB
    with pytest.raises(RuntimeError, match=r"exited with 1:(.|\n)*job 1: token id 320 is outside the vocabulary"):
        compare_gguf(tiny["bin"], tiny["p"], tiny["q"], [(e["input_ids"], logit_index(e["model_mask"]))
                                                          for e in (good, bad)])


def test_tiny_bad_token_fails_before_any_job_is_scored(tiny, tmp_path):
    # The helper writes each job's output as it finishes, so an output for job 0 means it was scored
    # before the bad id in job 1 was seen (PR #11 review minor 4: fail in seconds, not after the batch).
    good, bad = episode(vocab=VOCAB), episode(vocab=VOCAB)
    bad["input_ids"][0, 5] = VOCAB
    lines = []
    for i, e in enumerate((good, bad)):
        ids, idx = kl_gguf.check_run(i, e["input_ids"], logit_index(e["model_mask"]))
        ids.tofile(tmp_path / f"{i}.tok")
        idx.tofile(tmp_path / f"{i}.idx")
        lines.append(f"{tmp_path / f'{i}.tok'} {tmp_path / f'{i}.idx'} {tmp_path / f'{i}.out'}\n")
    (tmp_path / "jobs.txt").write_text("".join(lines))
    r = subprocess.run([tiny["bin"], "--cpu", tiny["p"], tiny["q"], str(tmp_path / "jobs.txt")], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 1 and "job 1: token id 320 is outside the vocabulary" in r.stderr
    assert not (tmp_path / "0.out").exists()


def test_tiny_different_vocabularies_fail(tiny):
    e = episode(vocab=VOCAB)
    with pytest.raises(RuntimeError, match="different vocabularies: token"):
        compare_gguf(tiny["bin"], tiny["p"], tiny["other_vocab"], [(e["input_ids"], logit_index(e["model_mask"]))],
                     cpu=True)


@pytest.mark.parametrize("ids, idx, match", [
    (np.zeros((2, 10), dtype=np.int64), [1, 2], r"run 0: input_ids must be .*\(2, 10\)"),
    (np.zeros(10, dtype=np.float32), [1, 2], "run 0: input_ids must be integer"),
    (np.array([1, -1, 2]), [0], r"run 0: token ids must be in \[0, 2\*\*31\)"),
    (np.arange(10), [5, 3], "run 0: idx must be sorted"),
    (np.arange(10), [9], "run 0: idx must be sorted"),
    (np.arange(10), [[1, 2]], "run 0: idx must be a 1-D integer array"),
])
def test_bad_runs_fail_before_the_helper_runs(ids, idx, match):
    # The binary does not exist, so the error must come from the checks before it is called.
    with pytest.raises(ValueError, match=match):
        compare_gguf("/nonexistent/kl_score", "p.gguf", "q.gguf", [(ids, np.array(idx))])


def test_non_finite_rows_raise(tmp_path):
    fake = tmp_path / "kl_score"
    fake.write_text(f"#!{sys.executable}\n"
                    "import sys, numpy as np\n"
                    "for line in open(sys.argv[-2]):\n"
                    "    tok, idx, out = line.split()\n"
                    "    rows = np.zeros((len(np.fromfile(idx, dtype=np.int32)), 5), dtype=np.float32)\n"
                    "    rows[1, 0] = np.nan\n"
                    "    rows.tofile(out)\n"
                    "print('{}')\n")
    fake.chmod(0o755)
    with pytest.raises(ValueError, match=r"run 0: non-finite stats at logit positions \[3\]"):
        compare_gguf(str(fake), "p.gguf", "q.gguf", [(np.arange(10), np.array([2, 3, 4]))])


@real
def test_same_model_gives_zero_kl():
    e = episode()
    s, = compare_gguf(BIN, P, P, [(e["input_ids"], logit_index(e["model_mask"]))])
    assert s["kl"].abs().max() < 1e-6
    assert s["flip"].sum() == 0
    assert torch.equal(s["nll_p"], s["nll_q"])


@real
def test_batch_size_and_subsets_do_not_change_stats():
    e = episode()
    idx = logit_index(e["model_mask"])
    small, sub = compare_gguf(BIN, P, Q, [(e["input_ids"], idx), (e["input_ids"], idx[::5])], n_batch=64)
    big, = compare_gguf(BIN, P, Q, [(e["input_ids"], idx)], n_batch=512)
    assert small["kl"].mean() > 0
    # Equal to float tolerance, not bitwise: llama.cpp picks matmul kernels by batch size and by the
    # number of output rows.
    for k in kl_gguf.FIELDS:
        assert torch.allclose(small[k], big[k], rtol=2e-3, atol=2e-3), k
        assert torch.allclose(small[k][::5], sub[k], rtol=2e-3, atol=2e-3), k


@real
def test_episode_scores_match_score_stats(monkeypatch):
    # The stats are captured from the same helper call, not recomputed by a second one, which is only
    # bitwise repeatable where the backend is deterministic.
    seen = []

    def spy(*args, **kwargs):
        seen.extend(helper(*args, **kwargs))
        return seen

    helper = kl_gguf.compare_gguf
    monkeypatch.setattr(kl_gguf, "compare_gguf", spy)
    e = episode()
    idx = logit_index(e["model_mask"])
    out, = score_trajectories_gguf(BIN, P, Q, [e])
    want = score_stats(seen[0], idx, e["tool_call_positions"], e["command_spans"], None, e["turn_spans"])
    assert out == {**want, "runtime": seen[0]["runtime"]}
    assert out["n_scored"] == len(idx) and out["n_command"] == 60 and set(out["halves"]) == {"even", "odd"}


@real
def test_bad_positions_fail_loudly():
    e = episode()
    with pytest.raises(ValueError, match="idx must be sorted"):
        compare_gguf(BIN, P, Q, [(e["input_ids"], torch.tensor([5, 3]))])


def load_check_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_kl_runtime", REPO / "scripts" / "check_kl_runtime.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_noise_floor_is_exact_with_the_whole_vocabulary_and_a_lower_bound_with_top_k():
    check = load_check_script()
    g = torch.Generator().manual_seed(0)
    head = torch.nn.Linear(8, 50, bias=False)
    hidden, targets = torch.randn(20, 8, generator=g), torch.randint(0, 50, (20,), generator=g)
    with torch.no_grad():
        lp = torch.log_softmax(head(hidden).double(), -1)
    lq = torch.log_softmax(lp + 0.5 * torch.randn(lp.shape, generator=g, dtype=torch.float64), -1)
    exact = (lp.exp() * (lp - lq)).sum(-1).numpy()
    top = lq.topk(50, dim=-1)
    full = check.hf_stats(head, hidden, targets, top.indices, top.values)
    assert np.allclose(full["kl_floor"], exact, atol=1e-6) and np.allclose(full["tail_mass_hf"], 0, atol=1e-6)
    assert np.allclose(full["nll"], -lp.gather(1, targets[:, None]).squeeze(1).numpy())
    part = check.hf_stats(head, hidden, targets, top.indices[:, :5], top.values[:, :5])
    assert (part["kl_floor"] <= exact + 1e-9).all() and (part["kl_floor"] > 0).all()


def test_runtime_criterion_flags_each_failure():
    # Each case breaks exactly one criterion, so a comparison hard-wired to True fails its case.
    check = load_check_script()
    n = 1000
    pos = np.arange(n)
    alt = np.where(pos % 2, 1.0, -1.0)  # alternating signs keep the run mean
    few = np.where(pos < 20, alt, 0.0)  # 2% of positions, above p99's 1%
    one = (pos == 0).astype(float)
    hf = {"nll": np.full(n, 0.5), "entropy": np.full(n, 0.5), "argmax": pos, "kl_floor": np.full(n, 1e-4),
          "tail_mass_hf": np.zeros(n)}
    gguf = {"nll_p": hf["nll"] + 1e-3, "entropy_p": hf["entropy"].copy(), "topk_ids": pos[:, None]}
    exact = np.full(128, 2e-4)
    ok = check.judge(gguf, hf, 0.019, exact)
    assert ok["all_pass"] and ok["noise_floor_over_q4_kl_mean"] == pytest.approx(2e-4 / 0.019)
    assert ok["noise_floor_kl"]["n_positions"] == 128 and ok["noise_floor_kl_topk"]["mean"] == pytest.approx(1e-4)
    cases = {
        "abs_dnll_median": {"nll_p": hf["nll"] + 0.02 * alt},
        "abs_dnll_p99": {"nll_p": hf["nll"] + 0.5 * few},
        "abs_dnll_max": {"nll_p": hf["nll"] + 2.0 * one},  # moves the mean by 0.4%
        "abs_dentropy_median": {"entropy_p": hf["entropy"] + 0.02 * alt},
        "abs_dentropy_p99": {"entropy_p": hf["entropy"] + 0.5 * few},
        "abs_dentropy_max": {"entropy_p": hf["entropy"] + 2.0 * one},
        "rel_mean_change_nll": {"nll_p": hf["nll"] + 0.008},  # 1.6% of the mean, 0.008 per position
        "rel_mean_change_entropy": {"entropy_p": hf["entropy"] + 0.008},
        "top1_agreement": {"topk_ids": (pos + (pos < 30))[:, None]},  # 3% disagree
    }
    for key, change in cases.items():
        assert {k for k, v in check.judge({**gguf, **change}, hf, 0.019, exact)["pass"].items() if not v} == {key}, key
    # The ceiling is judged on the exact KL: a top-k lower bound far under 10% must not pass it.
    for q4, kl_exact in ((1e-4, exact), (0.019, np.full(128, 0.01))):
        failed = {k for k, v in check.judge(gguf, hf, q4, kl_exact)["pass"].items() if not v}
        assert failed == {"noise_floor_over_q4_kl_mean"}
    assert set(cases) | {"noise_floor_over_q4_kl_mean"} == set(ok["pass"])


def test_check_script_runs_on_the_tiny_models(tiny, tmp_path):
    check = load_check_script()
    e = episode(vocab=VOCAB)
    ep = tmp_path / "episode.npz"
    np.savez(ep, input_ids=e["input_ids"][0].numpy(), model_mask=e["model_mask"].numpy())
    row = check.main([str(ep), "--hf", tiny["p"].removesuffix(".gguf"), "--kl-score", tiny["bin"], "--gguf-bf16",
                      tiny["p"], "--gguf-q4", tiny["q"], "--device", "cpu", "--top-k", "16",
                      "--machine", "test box", "--out", str(tmp_path / "row.json")])
    assert row["n_scored"] == len(logit_index(e["model_mask"])) and len(row["pass"]) == 10
    assert row["machine_label"] == "test box"
    assert row["noise_floor_kl"]["n_positions"] == len(row["noise_floor_positions"]) == 128
    # The script loads HF in bf16 against f32 GGUFs, so only the noise floor is checked: top-1 agreement
    # is low on a tiny model's near-tied logits.
    assert row["noise_floor_over_q4_kl_mean"] < 0.1
