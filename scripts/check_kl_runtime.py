"""Cross-runtime check of the KL reference: HF BF16 (PyTorch) against GGUF BF16 (llama.cpp) on one episode.

docs/04 asks for the llama.cpp KL path to be "checked equal" to PyTorch on a long trajectory before
the pilot. P0d's scores use the GGUF BF16 as P, so the check is whether llama.cpp's BF16 is the same
model as PyTorch's to well below the quantization signal P0d measures. The criterion, declared before
the CUDA run that decides it (PR #11 review, should-fix 2), is CRITERION below, on the scored
(model-written) positions of the episode:

- per-position |dNLL| and |dEntropy| (GGUF minus HF, nats): median <= 0.01, p99 <= 0.1, max <= 1.0
- run means of NLL and entropy: relative change <= 1%
- top-1 agreement between runtimes >= 98%
- noise floor: mean KL(HF BF16 || GGUF BF16) <= 10% of kl_mean(Q4_K_M), the smaller of the two P0d
  probes' signals, on the same episode; the mean is of the exact KL over a seeded sample of
  NOISE_FLOOR_POSITIONS scored positions (below)

The noise floor is a ceiling, so it is judged on the exact KL, not on a lower bound. The exact KL
needs the GGUF's whole distribution (248,320 log-probabilities per position), so a second helper call
dumps it (kl_score --top-k n_vocab) at NOISE_FLOOR_POSITIONS scored positions drawn with
NOISE_FLOOR_SEED, and the criterion's mean is the mean over that sample. That call asks for logits at
fewer positions per batch than the scoring run, which can move llama.cpp's kernels (qtrace/kl_gguf.py);
the result is still GGUF BF16 against HF BF16. Over every scored position the row also reports
noise_floor_kl_topk: the KL over the GGUF's top-k tokens plus one cell for the rest, a lower bound on
the full KL by the data-processing inequality, with tail_mass_hf, the HF probability in that cell.
These two are informative and are not judged.

    python scripts/check_kl_runtime.py EPISODE.npz --hf models/Qwen3.5-4B --kl-score bin/kl_score \
        --gguf-bf16 bf16.gguf --gguf-q4 q4_k_m.gguf [--device cuda] [--machine ec2-g6e.16xlarge] [--out row.json]

EPISODE.npz is the episode store's format (input_ids [n], bool model_mask [n]). The row printed and
written to --out holds every number, the thresholds, a pass flag per criterion, the helper's runtime
settings, the platform string ("machine") and the optional --machine label as given ("machine_label").
No hostname is recorded: committed rows should not carry personal machine names (PR #11 review).
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.kl import hidden_at, logit_index
from qtrace.kl_gguf import compare_gguf

CRITERION = {
    "abs_dnll": {"median": 0.01, "p99": 0.1, "max": 1.0},
    "abs_dentropy": {"median": 0.01, "p99": 0.1, "max": 1.0},
    "rel_mean_change": 0.01,
    "top1_agreement": 0.98,
    "noise_floor_over_q4_kl_mean": 0.10,
}
NOISE_FLOOR_POSITIONS = 128  # positions with the exact KL; declared with CRITERION
NOISE_FLOOR_SEED = 0


@torch.no_grad()
def hf_stats(lm_head, hidden, targets, topk_ids, topk_logprob, chunk=256):
    """HF entropy, NLL and argmax per position, and the top-k-coarsened KL(HF || GGUF)."""
    out = {k: [] for k in ("entropy", "nll", "argmax", "kl_floor", "tail_mass_hf")}
    for s in range(0, len(targets), chunk):
        lp = torch.log_softmax(lm_head(hidden[s:s + chunk]).float().cpu().double(), -1)  # MPS has no float64
        p = lp.exp()
        out["entropy"].append(-(p * lp).sum(-1))
        out["nll"].append(-lp.gather(1, targets[s:s + chunk, None]).squeeze(1))
        out["argmax"].append(lp.argmax(-1))
        ids, lq = topk_ids[s:s + chunk].long(), topk_logprob[s:s + chunk].double()
        lp_s = lp.gather(1, ids)
        p_tail = (1 - lp_s.exp().sum(-1)).clamp(min=0)
        q_tail = (1 - lq.exp().sum(-1)).clamp(min=1e-300)
        tail = torch.where(p_tail > 0, p_tail * (p_tail.clamp(min=1e-300).log() - q_tail.log()), 0)
        out["kl_floor"].append((lp_s.exp() * (lp_s - lq)).sum(-1) + tail)
        out["tail_mass_hf"].append(p_tail)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summary(x):
    return {"median": float(np.median(x)), "p99": float(np.quantile(x, 0.99)), "max": float(np.max(x)),
            "mean": float(np.mean(x))}


def judge(gguf, hf, q4_kl_mean, kl_exact):
    """Every declared number and whether it meets CRITERION.

    kl_exact: the exact KL(HF || GGUF) at the noise-floor sample; its mean is the judged noise floor.
    hf["kl_floor"] is the top-k lower bound at every position, reported only.
    """
    d_nll, d_ent = summary(np.abs(gguf["nll_p"] - hf["nll"])), summary(np.abs(gguf["entropy_p"] - hf["entropy"]))
    rel = {k: float(abs(gguf[g].mean() - hf[h].mean()) / abs(hf[h].mean()))
           for k, g, h in (("nll", "nll_p", "nll"), ("entropy", "entropy_p", "entropy"))}
    top1 = float((gguf["topk_ids"][:, 0] == hf["argmax"]).mean())
    floor = summary(kl_exact)
    ratio = floor["mean"] / q4_kl_mean
    c = CRITERION
    passes = {f"abs_dnll_{q}": d_nll[q] <= c["abs_dnll"][q] for q in ("median", "p99", "max")}
    passes |= {f"abs_dentropy_{q}": d_ent[q] <= c["abs_dentropy"][q] for q in ("median", "p99", "max")}
    passes |= {f"rel_mean_change_{k}": v <= c["rel_mean_change"] for k, v in rel.items()}
    passes |= {"top1_agreement": top1 >= c["top1_agreement"],
               "noise_floor_over_q4_kl_mean": ratio <= c["noise_floor_over_q4_kl_mean"]}
    return {"abs_dnll": d_nll, "abs_dentropy": d_ent, "rel_mean_change": rel, "top1_agreement": top1,
            "noise_floor_kl": {**floor, "n_positions": len(kl_exact)},
            "noise_floor_kl_topk": summary(hf["kl_floor"]), "tail_mass_hf": summary(hf["tail_mass_hf"]),
            "q4_kl_mean": q4_kl_mean,
            "noise_floor_over_q4_kl_mean": ratio, "criterion": CRITERION, "pass": passes,
            "all_pass": all(passes.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--hf", required=True, help="the HF BF16 checkpoint the GGUFs were converted from")
    ap.add_argument("--kl-score", required=True)
    ap.add_argument("--gguf-bf16", required=True)
    ap.add_argument("--gguf-q4", required=True)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    ap.add_argument("--machine", help="free-text label for the machine, e.g. mac-48gb; recorded as given")
    args = ap.parse_args(argv)
    with np.load(args.episode) as z:
        ids, mask = torch.from_numpy(z["input_ids"].astype(np.int64)), torch.from_numpy(z["model_mask"])
    idx = logit_index(mask)

    t = time.time()
    g, = compare_gguf(args.kl_score, args.gguf_bf16, args.gguf_q4, [(ids, idx)], top_k=args.top_k)
    sub = torch.from_numpy(np.sort(np.random.default_rng(NOISE_FLOOR_SEED).choice(
        len(idx), min(NOISE_FLOOR_POSITIONS, len(idx)), replace=False)))
    full, = compare_gguf(args.kl_score, args.gguf_bf16, args.gguf_q4, [(ids, idx[sub])],
                         top_k=g["runtime"]["n_vocab"])
    gguf_s = time.time() - t
    gguf = {k: v.numpy() for k, v in g.items() if k != "runtime"}

    from transformers import AutoModelForCausalLM
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(args.device).eval()
    with torch.no_grad():
        hidden, targets = hidden_at(model, ids[None], idx), ids[idx + 1]
        hf = hf_stats(model.lm_head, hidden, targets, g["topk_ids"], g["topk_logprob"])
        # chunk 16: each chunk holds [chunk, n_vocab] float64 log-probabilities and their gather
        kl_exact = hf_stats(model.lm_head, hidden[sub.to(hidden.device)], targets[sub], full["topk_ids"], full["topk_logprob"],
                            chunk=16)["kl_floor"]
    hf_s = time.time() - t

    row = {"episode": os.path.basename(args.episode), "n_tokens": len(ids), "n_scored": len(idx),
           "gguf_bf16": os.path.basename(args.gguf_bf16), "gguf_q4": os.path.basename(args.gguf_q4),
           "top_k": args.top_k, "noise_floor_positions": idx[sub].tolist(), "noise_floor_seed": NOISE_FLOOR_SEED,
           **judge(gguf, hf, float(gguf["kl"].mean()), kl_exact), "runtime": g["runtime"],
           "hf_device": args.device, "torch": torch.__version__, "machine": platform.platform(),
           "machine_label": args.machine,
           "seconds": {"gguf": round(gguf_s, 1), "hf": round(hf_s, 1)}}
    print(json.dumps(row, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=1)
    return row


if __name__ == "__main__":
    main()
