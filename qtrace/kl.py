"""Chunked teacher-forced disagreement between a full-precision model and a quantized copy.

Used by P0d, the `stable`/`qtrace` arms and the Stage 2 KL term (docs/04, P0d and "Selection and
training rules for P1"). Only model-written tokens are scored: the logit at position i predicts
token i+1, so token t is scored from logits at t-1.

Memory: a full logits tensor is n_tokens x 248,320 floats (about 60 GiB at 64K), so each model runs
one forward pass, keeps its final hidden states at the scored positions only, and the output heads
are applied in chunks. Each model uses its own head, because a quantizer may change it.
"""

import math

import torch


def logit_index(model_mask):
    """Positions of the logits that predict the model-written tokens (t - 1 for each t >= 1)."""
    t = torch.nonzero(model_mask).squeeze(1)
    return t[t >= 1] - 1


@torch.no_grad()
def hidden_at(model, input_ids, idx):
    """Final (normed) hidden states at positions idx, from one forward pass.

    One forward over a 64K-token episode assumes the fla and causal-conv1d kernels and SDPA on CUDA;
    the torch fallback is only fast enough for the CPU tests.
    """
    hidden = model.model(input_ids.to(model.device), use_cache=False).last_hidden_state[0]
    return hidden[idx.to(hidden.device)]


@torch.no_grad()
def compare(head_p, h_p, head_q, h_q, targets, chunk=384):
    """Per-position KL(P||Q), top-1 flip, entropy of P and NLL of the actual token under P and Q."""
    out = {k: [] for k in ("kl", "flip", "entropy_p", "nll_p", "nll_q")}
    for s in range(0, len(targets), chunk):
        lp = torch.log_softmax(head_p(h_p[s:s + chunk]).float(), dim=-1)
        lq = torch.log_softmax(head_q(h_q[s:s + chunk].to(h_p.device)).float().to(lp.device), dim=-1)
        y = targets[s:s + chunk].to(lp.device).unsqueeze(1)
        p = lp.exp()
        out["kl"].append((p * (lp - lq)).sum(-1))
        out["flip"].append((lp.argmax(-1) != lq.argmax(-1)).float())
        out["entropy_p"].append(-(p * lp).sum(-1))
        out["nll_p"].append(-lp.gather(1, y).squeeze(1))
        out["nll_q"].append(-lq.gather(1, y).squeeze(1))
    return {k: torch.cat(v).cpu() if v else torch.zeros(0) for k, v in out.items()}


def decision_mask(idx, entropy, tool_call_positions, command_spans, command_tokens=8, top_frac=0.10,
                  entropy_threshold=None):
    """Decision positions among the scored tokens (docs/04, P0d primary score and Stage 2 KL term).

    A scored token (position idx + 1) is a decision point if it is a `<tool_call>` token, one of the
    first `command_tokens` tokens of a command value (never past the command's end, so short
    commands don't spill onto the near-deterministic closing tags), or a high-entropy token.

    High entropy means the ceil(top_frac * n) highest-entropy scored tokens, ties broken by position
    (P0d), unless `entropy_threshold` is given: then every token at or above that fixed threshold
    (Stage 2, where a per-run top-k would move with run length and with the policy).

    Spans are (start, end) with end exclusive, from the episode store. Tokens outside model-written
    spans are never scored, so they can never be decision points.
    """
    pos = (idx + 1).tolist()
    where = {p: i for i, p in enumerate(pos)}
    mask = torch.zeros(len(pos), dtype=torch.bool)
    for p in tool_call_positions:
        if p in where:
            mask[where[p]] = True
    for start, end in command_spans:
        for p in range(start, min(start + command_tokens, end)):
            if p in where:
                mask[where[p]] = True
    if entropy_threshold is not None:
        mask |= entropy >= entropy_threshold
        return mask
    k = math.ceil(top_frac * len(pos))
    if k:
        order = sorted(range(len(pos)), key=lambda i: (-float(entropy[i]), pos[i]))
        mask[order[:k]] = True
    return mask


def span_mask(idx, spans):
    """Scored tokens (positions idx + 1) inside any (start, end) span, end exclusive."""
    pos = idx + 1
    mask = torch.zeros(len(pos), dtype=torch.bool)
    for start, end in spans:
        mask |= (pos >= start) & (pos < end)
    return mask


def summarize(stats, decision, command=None, halves=None):
    """Whole-run and decision-point scores.

    kl_sum_decision is the Stage 2 RL term when the mask uses a fixed entropy threshold. kl_sum_all is
    a diagnostic only: it grows with every written token and would duplicate the energy term.
    Means are None when nothing is scored, so an empty run can never rank as most stable.

    kl_command is the mean over the command-value tokens, the second candidate in docs/04's ordered
    score rule; it is None without command spans. Whole tool-call blocks would be diluted by about a
    dozen near-deterministic wrapper tokens per call, so many short commands would score low. halves
    maps a name to a mask of scored tokens (turn parity in P0b) and gets the three candidate scores
    per half for split-half reliability. The decision mask is computed over the whole run first, so
    each half scores the same positions the full-run score uses.

    entropy_decision is the mean BF16 entropy over the decision positions, the P0d DiD covariate
    (docs/04: "differences in BF16 entropy at decision positions"); entropy_mean is the whole-run mean.
    """
    kl = stats["kl"]
    n = len(kl)

    def mean(x):
        return float(x.mean()) if len(x) else None

    command = torch.zeros(n, dtype=torch.bool) if command is None else command
    out = {
        "n_scored": n,
        "n_decision": int(decision.sum()),
        "n_command": int(command.sum()),
        "kl_mean": mean(kl),
        "kl_decision": mean(kl[decision]),
        "kl_command": mean(kl[command]),
        "kl_sum_decision": float(kl[decision].sum()),
        "kl_sum_all": float(kl.sum()),
        "flip_rate": mean(stats["flip"]),
        "entropy_mean": mean(stats["entropy_p"]),
        "entropy_decision": mean(stats["entropy_p"][decision]),
        "nll_p_mean": mean(stats["nll_p"]),
        "nll_q_mean": mean(stats["nll_q"]),
    }
    if halves:
        out["halves"] = {name: {"n_scored": int(m.sum()), "kl_mean": mean(kl[m]),
                                "kl_decision": mean(kl[decision & m]), "kl_command": mean(kl[command & m])}
                         for name, m in halves.items()}
    return out


def score_trajectory(model_p, model_q, input_ids, model_mask, tool_call_positions=(), command_spans=(),
                     chunk=384, entropy_threshold=None, turn_spans=()):
    """Score one stored episode. model_p is full precision, model_q the quantized copy.

    command_spans are the command values (also used for kl_command); turn_spans are the assistant
    turns in order, split into even and odd halves for split-half reliability. Spans are
    (start, end), end exclusive. input_ids ([n] or [1, n]) and model_mask may be numpy arrays, as in
    the episode store's .npz files.
    """
    input_ids, model_mask = torch.as_tensor(input_ids), torch.as_tensor(model_mask)
    input_ids = input_ids[None] if input_ids.ndim == 1 else input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [n] or [1, n], not {tuple(input_ids.shape)}")
    idx = logit_index(model_mask)
    targets = input_ids[0, (idx + 1).to(input_ids.device)]
    stats = compare(model_p.lm_head, hidden_at(model_p, input_ids, idx),
                    model_q.lm_head, hidden_at(model_q, input_ids, idx), targets, chunk=chunk)
    return score_stats(stats, idx, tool_call_positions, command_spans, entropy_threshold, turn_spans)


def score_stats(stats, idx, tool_call_positions=(), command_spans=(), entropy_threshold=None, turn_spans=()):
    """Scores for one episode from its per-position stats, from compare() or qtrace/kl_gguf.py."""
    dm = decision_mask(idx, stats["entropy_p"], tool_call_positions, command_spans,
                       entropy_threshold=entropy_threshold)
    halves = None
    if turn_spans:
        halves = {"even": span_mask(idx, turn_spans[0::2]), "odd": span_mask(idx, turn_spans[1::2])}
    command = span_mask(idx, command_spans) if command_spans else None
    return summarize(stats, dm, command=command, halves=halves)
