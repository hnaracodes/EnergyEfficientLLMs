"""Tests for the per-arm drift from base (scripts/p1_drift.py) on a tiny Qwen3.5. Run:  python -m pytest tests -q"""

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np
import pytest
import torch
from test_probe import tiny_qwen35
from test_sft import make_episode

from qtrace.kl import score_trajectory


def load_script():
    spec = importlib.util.spec_from_file_location(
        "p1_drift", os.path.join(ROOT, "scripts", "p1_drift.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_attempts(tmp_path):
    """A P0b run file: heldout100 BF16 attempts (2 tasks x 2), plus rows the drift set must skip."""
    att_dir = tmp_path / "attempts"
    (att_dir / "ep").mkdir(parents=True)
    rows = []
    for i, (task, attempt, lst, profile) in enumerate(
        [
            ("h1", "p0b-h1-b", "heldout100", "BF16"),
            ("h1", "p0b-h1-a", "heldout100", "BF16"),
            ("h2", "p0b-h2-a", "heldout100", "BF16"),
            ("h2", "p0b-h2-b", "heldout100", "BF16"),
            ("p1", "p0b-p1-a", "pool200", "BF16"),
            ("h1", "p0b-h1-q", "heldout100", "Q4_K_M"),
        ]
    ):
        np.savez(att_dir / "ep" / f"{attempt}.npz", **make_episode(i, [5 + i, 7]))
        rows.append(
            {
                "run_id": "p0b",
                "attempt_id": attempt,
                "task_id": task,
                "list": lst,
                "profile": profile,
                "episode_path": f"ep/{attempt}.npz",
                "resolved": i % 2 == 0,
            }
        )
    path = att_dir / "p0b.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def write_run(tmp_path, arm_model):
    """A P1 run folder with train_meta.json naming the base checkpoint and the arm's merged model."""
    base, run = tmp_path / "base", tmp_path / "runs" / "fewest_turns_seed0"
    tiny_qwen35(0).save_pretrained(base)
    arm_model.save_pretrained(run / "merged")
    meta = {
        "arm": "fewest_turns",
        "seed": 0,
        "model": str(base),
        "merged": str(run / "merged"),
    }
    (run / "train_meta.json").write_text(json.dumps(meta))
    return run


def perturbed(seed=0, scale=0.05):
    model = tiny_qwen35(seed)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * p.std() * torch.randn(p.shape, generator=g))
    return model


def test_heldout_set_is_the_first_bf16_heldout100_attempt_per_task(tmp_path):
    eps = load_script().heldout_episodes(write_attempts(tmp_path))
    assert [(e["task_id"], e["attempt_id"]) for e in eps] == [
        ("h1", "p0b-h1-a"),
        ("h2", "p0b-h2-a"),
    ]
    assert (
        eps[0]["input_ids"].dtype == np.int64 and eps[0]["model_mask"].dtype == np.bool_
    )


def test_drift_json_is_token_weighted_hf_f32_kl_of_arm_against_base(tmp_path):
    # scale 3: the tiny model's top-1 survives smaller noise, and a zero flip rate pins nothing.
    attempts, arm = write_attempts(tmp_path), perturbed(scale=3.0)
    run = write_run(tmp_path, arm)
    script = load_script()
    script.main(["--run-dir", str(run), "--attempts", str(attempts), "--device", "cpu"])
    drift = json.loads((run / "drift.json").read_text())
    assert drift["arm"] == "fewest_turns" and drift["seed"] == 0
    assert (
        drift["heldout"]["list"] == "heldout100"
        and drift["heldout"]["profile"] == "BF16"
    )
    assert drift["heldout"]["n_episodes"] == 2 and "owner" in drift["heldout"]["note"]
    hf = drift["primary"]
    assert hf["method"] == "hf_f32" and hf["dtype"] == "float32"
    assert [e["attempt_id"] for e in hf["per_episode"]] == ["p0b-h1-a", "p0b-h2-a"]

    base = tiny_qwen35(0).eval()
    ref = [
        score_trajectory(
            base,
            arm.eval(),
            torch.from_numpy(e["input_ids"]).unsqueeze(0),
            torch.from_numpy(e["model_mask"]),
        )
        for e in script.heldout_episodes(attempts)
    ]
    n = sum(r["n_scored"] for r in ref)
    assert hf["n_scored"] == n
    assert hf["kl_mean"] == pytest.approx(
        sum(r["kl_mean"] * r["n_scored"] for r in ref) / n, rel=1e-5
    )
    for key, ref_key in [
        ("flip_rate", "flip_rate"),
        ("nll_base_mean", "nll_p_mean"),
        ("nll_arm_mean", "nll_q_mean"),
    ]:
        assert hf[key] == pytest.approx(
            sum(r[ref_key] * r["n_scored"] for r in ref) / n, rel=1e-5
        )
    assert hf["kl_mean"] > 0 and 0 < hf["flip_rate"] <= 1
    assert hf["nll_base_mean"] != pytest.approx(hf["nll_arm_mean"])
    assert drift["secondary_gguf"] is None


def test_arm_equal_to_base_has_zero_drift(tmp_path):
    attempts = write_attempts(tmp_path)
    run = write_run(tmp_path, tiny_qwen35(0))
    load_script().main(
        ["--run-dir", str(run), "--attempts", str(attempts), "--device", "cpu"]
    )
    hf = json.loads((run / "drift.json").read_text())["primary"]
    assert hf["kl_mean"] == 0.0 and hf["flip_rate"] == 0.0


def gguf_file(tmp_path, kls, ids=("p0b-h1-a", "p0b-h2-a")):
    rows = [
        {"attempt_id": a, "n_scored": 10, "kl_mean": k, "runtime": {"devices": ["CPU"]}}
        for a, k in zip(ids, kls)
    ]
    path = tmp_path / "kl_gguf.json"
    path.write_text(json.dumps(rows))
    return path


def test_gguf_drift_is_reported_only_above_1e_3_mean_kl(tmp_path):
    script = load_script()
    ids = ["p0b-h1-a", "p0b-h2-a"]
    low = script.gguf_drift(gguf_file(tmp_path, [4e-4, 6e-4]), ids)
    assert (
        low["kl_mean"] == pytest.approx(5e-4)
        and low["reported"] is False
        and low["threshold"] == 1e-3
    )
    high = script.gguf_drift(gguf_file(tmp_path, [1e-3, 3e-3]), ids)
    assert high["kl_mean"] == pytest.approx(2e-3) and high["reported"] is True
    assert high["runtime"] == {"devices": ["CPU"]} and high["n_scored"] == 20
    at = script.gguf_drift(gguf_file(tmp_path, [1e-3, 1e-3]), ids)
    assert at["kl_mean"] == 1e-3 and at["reported"] is False


def test_gguf_drift_must_cover_the_same_episodes(tmp_path):
    with pytest.raises(ValueError, match="same held-out episodes"):
        load_script().gguf_drift(
            gguf_file(tmp_path, [1e-2], ids=("p0b-h1-a",)), ["p0b-h1-a", "p0b-h2-a"]
        )


@pytest.mark.parametrize(
    "ids", [("p0b-h1-a", "p0b-h1-b"), ("p0b-h1-a", "p0b-h1-a")], ids=["wrong", "dup"]
)
def test_gguf_drift_rejects_same_count_with_wrong_or_duplicate_ids(tmp_path, ids):
    with pytest.raises(ValueError, match="same held-out episodes"):
        load_script().gguf_drift(
            gguf_file(tmp_path, [1e-2, 1e-2], ids=ids), ["p0b-h1-a", "p0b-h2-a"]
        )


def test_script_writes_the_gguf_block_when_given_a_result_file(tmp_path):
    attempts = write_attempts(tmp_path)
    run = write_run(tmp_path, tiny_qwen35(0))
    load_script().main(
        [
            "--run-dir",
            str(run),
            "--attempts",
            str(attempts),
            "--device",
            "cpu",
            "--gguf-kl",
            str(gguf_file(tmp_path, [2e-3, 2e-3])),
        ]
    )
    gguf = json.loads((run / "drift.json").read_text())["secondary_gguf"]
    assert gguf["reported"] is True and gguf["kl_mean"] == pytest.approx(2e-3)
