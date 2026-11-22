"""Pin the P3 external-check task lists (docs/03 Setup, External checks).

Writes verified500, datacurve113 and terminal_bench89 to data/task_lists/, each from a pinned
revision. Datacurve's Hugging Face repo is gated: its file listing is public and gives the task ids,
but task.toml (which holds the language) is not, so the language comes from the same tasks in
Datacurve's public GitHub repo at a pinned commit, and each Hugging Face task id must resolve to a
GitHub task.toml that names the same id.

    python scripts/p3_make_external_lists.py
"""

import json
import os
import sys
import tomllib
import urllib.request

from huggingface_hub import HfFileSystem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0_make_task_lists import OUT, VERIFIED, read_columns

from qtrace.tasklists import external_lists

DATACURVE = ("datacurve/deep-swe", "6d6f134460c137e24c6bb7e1e69954116ea9dbb3")
DATACURVE_GITHUB = ("datacurve-ai/deep-swe", "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea")
TERMINAL_BENCH = (
    "harborframework/terminal-bench-2.1",
    "3e235dff6880252a587fa479c09bcb1e16edf2eb",
)
SIZES = {
    "verified500": 500,
    "datacurve113": 113,
    "terminal_bench89": 89,
}  # docs/03 External checks


def task_ids(fs, repo, revision):
    """Harbor-format datasets keep one tasks/<id>/task.toml per task."""
    return sorted(
        path.split("/")[-2]
        for path in fs.glob(f"datasets/{repo}@{revision}/tasks/*/task.toml")
    )


def datacurve_language(task_id):
    repo, commit = DATACURVE_GITHUB
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/tasks/{task_id}/task.toml"
    with urllib.request.urlopen(url, timeout=60) as resp:
        meta = tomllib.loads(resp.read().decode())["metadata"]
    if meta["task_id"] != task_id:
        raise SystemExit(f"{url} names task {meta['task_id']}, expected {task_id}")
    return meta["language"]


def main():
    fs = HfFileSystem()
    verified = [r["instance_id"] for r in read_columns(fs, *VERIFIED, ["instance_id"])]
    langs = {
        task_id: datacurve_language(task_id) for task_id in task_ids(fs, *DATACURVE)
    }
    terminal = task_ids(fs, *TERMINAL_BENCH)

    lists = external_lists(verified, langs, terminal)
    sources = {
        "verified500": {"swe_bench_verified": f"{VERIFIED[0]}@{VERIFIED[1]}"},
        "datacurve113": {
            "datacurve_hf": f"{DATACURVE[0]}@{DATACURVE[1]}",
            "datacurve_github_language": f"github.com/{DATACURVE_GITHUB[0]}@{DATACURVE_GITHUB[1]}",
        },
        "terminal_bench89": {
            "terminal_bench": f"{TERMINAL_BENCH[0]}@{TERMINAL_BENCH[1]}"
        },
    }
    for name, payload in lists.items():
        if payload["size"] != SIZES[name]:
            raise SystemExit(
                f"{name} has {payload['size']} tasks, 03 says {SIZES[name]}"
            )
        payload["sources"] = sources[name]
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=1)
        print(f"{name}: {payload['size']} tasks")
    print(
        f"datacurve113: {lists['datacurve113']['python']} Python, {lists['datacurve113']['non_python']} non-Python"
    )


if __name__ == "__main__":
    main()
