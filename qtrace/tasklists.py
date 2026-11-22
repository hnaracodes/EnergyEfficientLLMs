"""Frozen, seeded task lists for P0 and P1 (docs/04_p0_protocol.md, "Task sets").

Pure functions only, so the lists can be rebuilt and checked without network access.
A task is identified by its R2E-Gym docker image, which is unique per task.
"""

import random

POOL_REPOS = ["pandas", "numpy", "pillow", "orange3", "aiohttp", "tornado"]
HELDOUT_REPOS = ["scrapy", "pyramid", "datalad", "coveragepy"]


def allocate(counts, n):
    """Split n draws across groups in proportion to their sizes (largest remainder).

    Never gives a group more than it has. Raises ValueError if n exceeds the total.
    """
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"asked for {n} tasks but only {total} exist")
    exact = {g: n * c / total for g, c in counts.items()}
    alloc = {g: min(int(x), counts[g]) for g, x in exact.items()}
    # Hand out the remaining draws by largest fractional part, skipping full groups.
    order = sorted(counts, key=lambda g: (-(exact[g] - int(exact[g])), g))
    while sum(alloc.values()) < n:
        for g in order:
            if sum(alloc.values()) == n:
                break
            if alloc[g] < counts[g]:
                alloc[g] += 1
    return alloc


def stratified_sample(ids_by_group, n, seed):
    """Draw n ids, stratified by group, independent of input order."""
    alloc = allocate({g: len(ids) for g, ids in ids_by_group.items()}, n)
    picked = []
    for group in sorted(ids_by_group):
        rng = random.Random(f"{seed}:{group}")
        picked.extend(sorted(rng.sample(sorted(ids_by_group[group]), alloc[group])))
    return picked


def _by_repo(rows, repos, exclude=()):
    out = {repo: [] for repo in repos}
    for task_id, repo in rows:
        if repo in out and task_id not in exclude:
            out[repo].append(task_id)
    return out


def _summary(name, ids, rows, seed, size_note):
    repo_of = dict(rows)
    by_repo = {}
    for task_id in ids:
        repo = repo_of.get(task_id, task_id.split("__")[0])
        by_repo[repo] = by_repo.get(repo, 0) + 1
    return {"name": name, "seed": seed, "size": len(ids), "size_note": size_note,
            "by_repo": dict(sorted(by_repo.items())), "ids": ids}


def build_lists(rows, verified_ids, seed, inrepo_size=300, pool_size=200, heldout_size=100, verified_size=100,
                pilot_size=20, pilot_verified_size=3, subrun_size=15):
    """Build inrepo_test, pool200, heldout100, heldout691, verified100 and the three subsets below.

    rows: (task_id, repo_name) pairs for R2E-Gym. inrepo_test is drawn first from the pool repos
    and excluded from pool200, as 04 requires. heldout100 comes only from the held-out repos.
    heldout691 is every held-out-repo task (P2 runs on all of them), so it is not sampled.
    pilot20 is stratified by repo from pool200 and energy_subrun15 from pool200 minus pilot20;
    pilot_verified3 is a plain draw from verified100. Each has its own seed string, so adding them
    leaves the other lists unchanged.
    """
    ids = [task_id for task_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids in R2E-Gym rows")
    inrepo = stratified_sample(_by_repo(rows, POOL_REPOS), inrepo_size, f"{seed}:inrepo_test")
    pool = stratified_sample(_by_repo(rows, POOL_REPOS, exclude=set(inrepo)), pool_size, f"{seed}:pool200")
    held_by_repo = _by_repo(rows, HELDOUT_REPOS)
    held = stratified_sample(held_by_repo, heldout_size, f"{seed}:heldout100")
    every_held = [task_id for repo in sorted(held_by_repo) for task_id in sorted(held_by_repo[repo])]
    rng = random.Random(f"{seed}:verified100")
    verified = sorted(rng.sample(sorted(verified_ids), verified_size))
    pool_rows = [row for row in rows if row[0] in set(pool)]
    pilot = stratified_sample(_by_repo(pool_rows, POOL_REPOS), pilot_size, f"{seed}:pilot20")
    pilot_verified = sorted(random.Random(f"{seed}:pilot_verified3").sample(verified, pilot_verified_size))
    subrun = stratified_sample(_by_repo(pool_rows, POOL_REPOS, exclude=set(pilot)), subrun_size,
                               f"{seed}:energy_subrun15")
    return {
        "inrepo_test": _summary("inrepo_test", inrepo, rows, seed,
                                "working default 300; the owner sets the final size (04, Owner decisions)"),
        "pool200": _summary("pool200", pool, rows, seed, "fixed by 04"),
        "heldout100": _summary("heldout100", held, rows, seed, "fixed by 04"),
        "heldout691": _summary("heldout691", every_held, rows, seed,
                               "every held-out-repo task, a superset of heldout100; 04 gives 691"),
        "verified100": _summary("verified100", verified, rows, seed, "fixed by 04"),
        "pilot20": _summary("pilot20", pilot, rows, seed, "fixed by 04 (P0b pilot), drawn from pool200"),
        "pilot_verified3": _summary("pilot_verified3", pilot_verified, rows, seed,
                                    "fixed by 04 (P0b pilot extra, owner-approved), drawn from verified100"),
        "energy_subrun15": _summary("energy_subrun15", subrun, rows, seed,
                                    "fixed by 04 (P0a energy sub-run), drawn from pool200 minus pilot20: the "
                                    "Q4_K_M coefficients first rank pool200 runs for fewest_joules, and their "
                                    "held-out use gets its own estimator check (heldout691 minus heldout100)"),
    }


def _whole_set(name, ids, size_note):
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids in {name}")
    return {"name": name, "size": len(ids), "size_note": size_note, "ids": sorted(ids)}


def external_lists(verified_ids, datacurve_langs, terminal_bench_ids):
    """Build verified500, datacurve113 and terminal_bench89 (docs/03 Setup, External checks).

    Each is a whole benchmark, so nothing is sampled and there is no seed. datacurve_langs maps a
    Datacurve task id to the language in its task.toml; 03 reports Datacurve as Python vs
    non-Python, so every task also carries that tag. Raises ValueError on duplicate ids.
    """
    verified = _whole_set("verified500", list(verified_ids), "every SWE-bench Verified task; 03 gives 500")
    by_repo = {}
    for task_id in verified["ids"]:
        repo = task_id.split("__")[0]
        by_repo[repo] = by_repo.get(repo, 0) + 1
    verified["by_repo"] = dict(sorted(by_repo.items()))
    datacurve = _whole_set("datacurve113", list(datacurve_langs),
                           "every Datacurve DeepSWE task; 03 gives 113, reported as Python vs non-Python")
    tasks = {t: {"language": datacurve_langs[t], "python": datacurve_langs[t].lower() == "python"}
             for t in datacurve["ids"]}
    by_language = {}
    for task in tasks.values():
        by_language[task["language"]] = by_language.get(task["language"], 0) + 1
    n_python = sum(task["python"] for task in tasks.values())
    datacurve.update(by_language=dict(sorted(by_language.items())), python=n_python,
                     non_python=len(tasks) - n_python, tasks=tasks)
    terminal = _whole_set("terminal_bench89", list(terminal_bench_ids),
                          "every Terminal-Bench 2.1 task; 03 gives 89, run on its own harness")
    return {"verified500": verified, "datacurve113": datacurve, "terminal_bench89": terminal}


def overlap_check(train_ids, heldout_ids):
    """Join a teacher's training task ids with held-out ids (docs/04, Fallbacks: DeepSWE-Preview).

    Returns both set sizes, the overlap size and the held-out ids the teacher never trained on.
    """
    train, heldout = set(train_ids), set(heldout_ids)
    return {"train_size": len(train), "heldout_size": len(heldout), "overlap_size": len(train & heldout),
            "heldout_not_in_train": sorted(heldout - train)}
