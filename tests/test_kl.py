"""Tests for the chunked teacher-forced KL scorer. Run:  python -m pytest tests/test_kl.py -q"""

import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.kl import (
    compare,
    decision_mask,
    hidden_at,
    logit_index,
    pool_entropy_threshold,
    score_trajectory,
    span_mask,
    summarize,
)
from qtrace.quantize import quantize_model_rtn


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
    return Qwen3_5ForCausalLM(cfg).eval()


def episode(n=80, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 300, (1, n), generator=g)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[10:30] = True  # first assistant turn
    mask[45:70] = True  # second assistant turn; 30-44 is a tool response
    return ids, mask


def reference(model_p, model_q, ids, mask):
    with torch.no_grad():
        lp = torch.log_softmax(model_p(ids).logits[0].float(), -1)
        lq = torch.log_softmax(model_q(ids).logits[0].float(), -1)
    t = torch.nonzero(mask).squeeze(1)
    t = t[t >= 1]
    kl = (lp[t - 1].exp() * (lp[t - 1] - lq[t - 1])).sum(-1)
    nll = -lp[t - 1].gather(1, ids[0, t].unsqueeze(1)).squeeze(1)
    return kl, nll


def test_logit_index_shifts_by_one_and_skips_position_zero():
    mask = torch.tensor([True, False, True, True])
    assert logit_index(mask).tolist() == [1, 2]


def test_self_comparison_is_zero():
    m = tiny_qwen35()
    ids, mask = episode()
    s = score_trajectory(m, m, ids, mask, chunk=16)
    assert s["kl_mean"] == 0.0 and s["flip_rate"] == 0.0


def test_chunked_matches_full_logits_for_any_chunk_size():
    p = tiny_qwen35(0)
    q = copy.deepcopy(p)
    quantize_model_rtn(q, bits=3, group_size=32)
    ids, mask = episode()
    kl_ref, nll_ref = reference(p, q, ids, mask)
    idx = logit_index(mask)
    targets = ids[0, idx + 1]
    hp, hq = hidden_at(p, ids, idx), hidden_at(q, ids, idx)
    for chunk in (1, 7, 1000):
        out = compare(p.lm_head, hp, q.lm_head, hq, targets, chunk=chunk)
        assert torch.allclose(out["kl"], kl_ref, atol=1e-5)
        assert torch.allclose(out["nll_p"], nll_ref, atol=1e-5)


def test_numpy_episode_scores_like_torch():
    # The episode store saves .npz arrays: input_ids [n] and a bool model_mask [n].
    model_p = tiny_qwen35()
    model_q = copy.deepcopy(model_p)
    quantize_model_rtn(model_q, bits=3, group_size=32)
    ids, mask = episode()
    want = score_trajectory(model_p, model_q, ids, mask, command_spans=[(20, 30)])
    assert score_trajectory(model_p, model_q, ids[0].numpy(), mask.numpy(), command_spans=[(20, 30)]) == want
    with pytest.raises(ValueError, match=r"input_ids must be \[n\] or \[1, n\]"):
        score_trajectory(model_p, model_q, ids.repeat(2, 1), mask)


def test_quantized_copy_disagrees():
    p = tiny_qwen35(0)
    q = copy.deepcopy(p)
    quantize_model_rtn(q, bits=3, group_size=32)
    ids, mask = episode()
    s = score_trajectory(p, q, ids, mask, chunk=16)
    assert s["kl_mean"] > 0
    assert abs(s["kl_sum_all"] - s["kl_mean"] * s["n_scored"]) < 1e-4


def test_decision_mask_rule():
    mask = torch.zeros(40, dtype=torch.bool)
    mask[5:35] = True
    idx = logit_index(mask)  # scored token positions are idx + 1, here 5..34 (30 positions)
    entropy = torch.arange(len(idx), dtype=torch.float32) * 0.01  # rising, so the top 3 are 32-34
    dm = decision_mask(idx, entropy, tool_call_positions=[6], command_spans=[(10, 25)], command_tokens=8)
    picked = set((idx[dm] + 1).tolist())
    assert 6 in picked  # the <tool_call> token
    assert set(range(10, 18)) <= picked and 18 not in picked  # first 8 command tokens only
    assert {32, 33, 34} <= picked and 31 not in picked  # top ceil(10% of 30) = 3 by entropy
    assert all(mask[p] for p in picked)  # never a template position
    dm_none = decision_mask(idx, entropy, tool_call_positions=[2], command_spans=[(38, 40)], command_tokens=8)
    assert not {2, 38, 39} & set((idx[dm_none] + 1).tolist())  # outside model-written spans


def test_short_command_window_stops_at_command_end():
    mask = torch.zeros(40, dtype=torch.bool)
    mask[5:35] = True
    idx = logit_index(mask)
    entropy = torch.zeros(len(idx))
    entropy[0] = 1.0  # keep the top-k pick away from the command region
    dm = decision_mask(idx, entropy, tool_call_positions=[], command_spans=[(12, 14)], command_tokens=8,
                       top_frac=0.0)
    assert set((idx[dm] + 1).tolist()) == {12, 13}  # a 2-token command; closing tags at 14+ are excluded


def test_fixed_entropy_threshold_replaces_top_k():
    mask = torch.ones(21, dtype=torch.bool)
    idx = logit_index(mask)  # 20 scored tokens, positions 1..20
    entropy = torch.tensor([0.1] * 15 + [0.9] * 5)
    dm = decision_mask(idx, entropy, [], [], entropy_threshold=0.5)
    assert set((idx[dm] + 1).tolist()) == {16, 17, 18, 19, 20}


def test_pool_entropy_threshold_is_the_pooled_token_quantile():
    runs = [np.arange(10) * 0.1, torch.arange(10, 20) * 0.1]  # npz arrays or tensors, pooled 0.0 .. 1.9
    t = pool_entropy_threshold(runs)
    assert abs(t - 1.71) < 1e-6  # default 0.90, linear interpolation at 0.9 * 19 = 17.1
    assert abs(pool_entropy_threshold(runs, quantile=0.5) - 0.95) < 1e-6
    idx = torch.arange(20)
    dm = decision_mask(idx, torch.arange(20) * 0.1, [], [], entropy_threshold=t)
    assert dm.nonzero().squeeze(1).tolist() == [18, 19]  # the pool's top 10%


def test_pool_entropy_threshold_rejects_bad_input():
    with pytest.raises(ValueError, match="quantile"):
        pool_entropy_threshold([np.ones(3)], quantile=1.0)
    with pytest.raises(ValueError, match="no entropies"):
        pool_entropy_threshold([np.zeros(0)])
    with pytest.raises(ValueError, match="finite"):
        pool_entropy_threshold([np.array([0.1, np.nan])])


def test_empty_run_is_not_scored_as_stable():
    empty = {k: torch.zeros(0) for k in ("kl", "flip", "entropy_p", "nll_p", "nll_q")}
    s = summarize(empty, torch.zeros(0, dtype=torch.bool))
    assert s["kl_mean"] is None and s["kl_decision"] is None and s["n_scored"] == 0
    assert s["entropy_mean"] is None and s["entropy_decision"] is None


def test_summarize_reports_decision_and_whole_run():
    stats = {"kl": torch.tensor([1.0, 2.0, 3.0, 4.0]), "flip": torch.tensor([0.0, 1.0, 0.0, 0.0]),
             "entropy_p": torch.tensor([0.1, 0.2, 0.3, 0.5]), "nll_p": torch.zeros(4), "nll_q": torch.zeros(4)}
    s = summarize(stats, torch.tensor([False, True, False, True]))
    assert s["kl_mean"] == 2.5 and s["kl_decision"] == 3.0
    assert abs(s["entropy_mean"] - 0.275) < 1e-6 and abs(s["entropy_decision"] - 0.35) < 1e-6  # positions 1 and 3
    assert s["kl_sum_all"] == 10.0 and s["kl_sum_decision"] == 6.0
    assert s["n_scored"] == 4 and s["n_decision"] == 2 and s["flip_rate"] == 0.25


def test_span_mask_selects_scored_tokens_inside_spans():
    mask = torch.zeros(40, dtype=torch.bool)
    mask[5:35] = True
    idx = logit_index(mask)
    m = span_mask(idx, [(8, 11), (30, 50)])
    assert set((idx[m] + 1).tolist()) == {8, 9, 10, 30, 31, 32, 33, 34}  # end exclusive, unscored 35+ skipped


def test_summarize_reports_command_kl_and_turn_halves():
    stats = {"kl": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "flip": torch.zeros(6),
             "entropy_p": torch.zeros(6), "nll_p": torch.zeros(6), "nll_q": torch.zeros(6)}
    decision = torch.tensor([True, False, False, True, False, True])
    command = torch.tensor([False, True, True, False, False, True])
    even = torch.tensor([True, True, True, False, False, False])
    s = summarize(stats, decision, command=command, halves={"even": even, "odd": ~even})
    assert abs(s["kl_command"] - 11.0 / 3) < 1e-6 and s["n_command"] == 3  # float32 mean
    assert s["kl_sum_command"] == 11.0  # positions 1, 2 and 5
    assert s["halves"]["even"] == {"n_scored": 3, "kl_mean": 2.0, "kl_decision": 1.0, "kl_command": 2.5}
    assert s["halves"]["odd"] == {"n_scored": 3, "kl_mean": 5.0, "kl_decision": 5.0, "kl_command": 6.0}


def test_no_command_spans_means_no_command_score():
    stats = {k: torch.ones(3) for k in ("kl", "flip", "entropy_p", "nll_p", "nll_q")}
    s = summarize(stats, torch.zeros(3, dtype=torch.bool))
    assert s["kl_command"] is None and s["n_command"] == 0 and "halves" not in s
    assert s["kl_sum_command"] == 0.0


def test_score_trajectory_command_and_halves_match_reference():
    model_p = tiny_qwen35()
    model_q = copy.deepcopy(model_p)
    quantize_model_rtn(model_q, bits=3, group_size=32)
    ids, mask = episode()
    kl_ref, _ = reference(model_p, model_q, ids, mask)
    pos = logit_index(mask) + 1
    s = score_trajectory(model_p, model_q, ids, mask, command_spans=[(20, 30), (60, 70)],
                         turn_spans=[(10, 30), (45, 70)])
    in_command = ((pos >= 20) & (pos < 30)) | ((pos >= 60) & (pos < 70))
    assert abs(s["kl_command"] - float(kl_ref[in_command].mean())) < 1e-4
    assert abs(s["halves"]["even"]["kl_mean"] - float(kl_ref[pos < 30].mean())) < 1e-4
    assert abs(s["halves"]["odd"]["kl_mean"] - float(kl_ref[pos >= 45].mean())) < 1e-4
    assert s["halves"]["even"]["n_scored"] + s["halves"]["odd"]["n_scored"] == s["n_scored"]
