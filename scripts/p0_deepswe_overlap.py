"""Check DeepSWE-Preview's training task list against heldout691 (docs/04, P0b fallbacks).

DeepSWE-Preview publishes no per-task list of what its RL run sampled. What it does publish is the
training source: the model card names R2E-Gym/R2E-Gym-Subset, and rLLM's DeepSWE recipe trains on
that dataset's whole train split, loaded without a revision. So the list joined here is that train
split at the revision live during the July 2025 run, keyed by docker_image like heldout691. It is
the candidate pool, an upper bound on what the run saw (see EVIDENCE).

    python scripts/p0_deepswe_overlap.py
"""

import json
import os
import sys

from huggingface_hub import HfFileSystem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0_make_task_lists import OUT, read_columns

from qtrace.tasklists import overlap_check

# The only data upload; the later 2e8108ff commit (which heldout691 reads) changed the card only,
# and all 8 parquet files have the same LFS sha256 at both revisions.
DEEPSWE_TRAIN = ("R2E-Gym/R2E-Gym-Subset", "e8b9fcbce43eaca0dc2c0d4798ee6f3e965f590a")
MODEL_CARD = "huggingface.co/agentica-org/DeepSWE-Preview@4887205c533cd162baac7ba758159cfc3304cf94"
RLLM = "github.com/agentica-project/rllm@0ffd38407f30d492563bc950df0658386372c415"
EVIDENCE = [
    (
        f"{MODEL_CARD} README.md: metadata 'datasets: R2E-Gym/R2E-Gym-Subset'; Data section: '4.5K problems "
        "from a subset of R2E-Gym', with repos shared with SWE-Bench-Verified filtered out."
    ),
    (
        f"{RLLM} examples/swe/train_deepswe_32b.sh: data.train_files=R2E_Gym_Subset.parquet, written by "
        "scripts/data/swe_dataset.py from load_dataset('R2E-Gym/R2E-Gym-Subset') train split, no "
        "revision and no id filter."
    ),
    (
        "Not published: the tasks the run actually drew. The same script sets data.filter_overlong_prompts=True "
        "(max_prompt_length 4096) and train_batch_size 8, and the card reports 200 RL steps, so the visited "
        "set can be smaller than the train split. The blog post (Notion) renders only with JavaScript and "
        "was not read; the card mirrors its Data section."
    ),
]


def main():
    with open(os.path.join(OUT, "heldout691.json")) as f:
        heldout = json.load(f)
    train = [
        r["docker_image"]
        for r in read_columns(HfFileSystem(), *DEEPSWE_TRAIN, ["docker_image"])
    ]
    out = overlap_check(train, heldout["ids"])
    out["sources"] = {
        "deepswe_train": f"{DEEPSWE_TRAIN[0]}@{DEEPSWE_TRAIN[1]}",
        "heldout": heldout["sources"]["r2e_gym"],
    }
    out["evidence"] = EVIDENCE
    with open(os.path.join(OUT, "deepswe_heldout_overlap.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"DeepSWE train {out['train_size']}, held-out {out['heldout_size']}, overlap {out['overlap_size']}"
    )


if __name__ == "__main__":
    main()
