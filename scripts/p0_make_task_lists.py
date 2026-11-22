"""Build and freeze the P0/P1 task lists (docs/04_p0_protocol.md, "Task sets").

Reads only the needed columns of two pinned dataset revisions and writes one JSON file per list to
data/task_lists/. Run once; the files are committed and never regenerated with a different seed.

    python scripts/p0_make_task_lists.py [--inrepo-size 300]
"""

import argparse
import json
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.tasklists import HELDOUT_REPOS, POOL_REPOS, build_lists

SEED = 20261119
HELDOUT_ALL_SIZE = 691  # docs/04 "Task sets"
R2E = ("R2E-Gym/R2E-Gym-Subset", "2e8108ff942f24fcb5686badfaf7f9a8808566d5")
VERIFIED = ("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "task_lists")


def read_columns(fs, repo, revision, columns):
    files = sorted(fs.glob(f"datasets/{repo}@{revision}/**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found for {repo}@{revision}")
    rows = []
    for path in files:
        with fs.open(path) as f:
            rows.extend(pq.read_table(f, columns=columns).to_pylist())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inrepo-size", type=int, default=300)
    args = ap.parse_args()

    fs = HfFileSystem()
    r2e = read_columns(fs, *R2E, ["docker_image", "repo_name"])
    rows = [(r["docker_image"], r["repo_name"]) for r in r2e]
    repos = {repo for _, repo in rows}
    expected = set(POOL_REPOS + HELDOUT_REPOS)
    if repos != expected:
        raise SystemExit(f"R2E-Gym repos changed: got {sorted(repos)}, expected {sorted(expected)}")
    verified = [r["instance_id"] for r in read_columns(fs, *VERIFIED, ["instance_id"])]

    lists = build_lists(rows, verified, SEED, inrepo_size=args.inrepo_size)
    if lists["heldout691"]["size"] != HELDOUT_ALL_SIZE:
        raise SystemExit(f"held-out repos have {lists['heldout691']['size']} tasks, 04 says {HELDOUT_ALL_SIZE}")
    os.makedirs(OUT, exist_ok=True)
    for name, payload in lists.items():
        payload["sources"] = {"r2e_gym": f"{R2E[0]}@{R2E[1]}", "swe_bench_verified": f"{VERIFIED[0]}@{VERIFIED[1]}"}
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=1)
        print(f"{name}: {payload['size']} tasks {payload['by_repo']}")
    print(f"R2E-Gym tasks: {len(rows)}, Verified tasks: {len(verified)}, seed {SEED}")


if __name__ == "__main__":
    main()
