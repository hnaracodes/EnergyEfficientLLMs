"""R2E-Gym grading for the harness's verify() (docs/04: R2E-Gym runtime, graded by its own tests).

Mirrors R2E-Gym's reward (r2egym/agenthub/runtime/docker.py, `_calculate_reward_r2e`, commit 0d94c4e):
run the task's `run_tests.sh` (pytest -rA r2e_tests from /testbed), parse pytest's short summary, and
count the task resolved only if the parsed test -> status map matches the row's `expected_output_json`
exactly (same keys, same statuses, keys compared up to the first " - ").

Grading never happens in the agent's container. The agent's shell runs as root, so anything left there
could be read or edited, which would contaminate P0 and give Stage 2 RL a reward hack. Instead:
  1. `prepare_episode` deletes the tests and run_tests.sh from the episode container and snapshots the
     repo as a git tree whose hash stays on the host.
  2. `final_diff` diffs the end state against that snapshot (new files included, test infrastructure
     excluded) without touching the agent's HEAD or index.
  3. `grade_fresh` applies the diff in a fresh container from the same image, where the original tests
     are intact, and runs them there. This matches the SWE-bench harness, which also grades a patch in
     a fresh container.
R2E-Gym's reward is all-or-nothing and the dataset's old-commit results carry no per-test names, so the
fail-to-pass / pass-to-pass split comes from a baseline: `grade_fresh` with an empty diff, cached per task.
"""

import json
import re
import uuid
from dataclasses import dataclass

from qtrace.agent.docker_exec import DockerContainer

TEST_TIMEOUT = 300
EXCLUDE = "-- . ':(exclude).venv' ':(exclude)r2e_tests' ':(exclude)run_tests.sh'"
TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$|pytest\.ini$|tox\.ini$|setup\.cfg$)")


@dataclass(frozen=True)
class Grading:
    resolved: bool
    fail_to_pass_ok: int
    fail_to_pass_total: int
    pass_to_pass_ok: int
    pass_to_pass_total: int
    tests_timed_out: bool = False
    test_files_touched: bool = False  # tripwire: the diff touched test files or pytest configuration
    grading_error: str | None = None  # e.g. the diff did not apply; such an attempt is unresolved


def parse_pytest_log(log):
    """R2E-Gym's parse_log_pytest, same behavior: statuses from the short test summary."""
    status = {}
    if not log or "short test summary info" not in log:
        return status
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            status[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            status[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            status[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return status


def _normalize(status):
    decolored = {re.sub(r"\u001b\[\d+m", "", k): v for k, v in status.items()}
    return {k.split(" - ")[0]: decolored[k] for k in sorted(decolored)}


def r2e_resolved(parsed, expected):
    """R2E-Gym's rule: same number of tests and every parsed status equals the expected one."""
    parsed, expected = _normalize(parsed), _normalize(expected)
    if len(parsed) != len(expected):
        return False
    return all(not k or (k in expected and parsed[k] == expected[k]) for k in parsed)


def grading(parsed, expected, baseline, **diagnostics):
    """Grading for the harness. F2P: expected to pass but not passing at baseline; P2P: passing at both."""
    parsed, expected, baseline = _normalize(parsed), _normalize(expected), _normalize(baseline)
    should_pass = [k for k, v in expected.items() if v == "PASSED"]
    f2p = [k for k in should_pass if baseline.get(k) != "PASSED"]
    p2p = [k for k in should_pass if baseline.get(k) == "PASSED"]
    return Grading(
        resolved=r2e_resolved(parsed, expected) and not diagnostics.get("grading_error"),
        fail_to_pass_ok=sum(parsed.get(k) == "PASSED" for k in f2p),
        fail_to_pass_total=len(f2p),
        pass_to_pass_ok=sum(parsed.get(k) == "PASSED" for k in p2p),
        pass_to_pass_total=len(p2p),
        **diagnostics,
    )


def touched_test_files(diff):
    """Paths in a git diff that look like tests or pytest configuration."""
    paths = re.findall(r"^diff --git a/(\S+) b/", diff.decode("utf-8", errors="replace"), flags=re.M)
    return sorted(p for p in paths if TEST_PATH.search(p))


def _snapshot_tree(container, index):
    script = (f"export GIT_INDEX_FILE={index}; rm -f $GIT_INDEX_FILE; git add -A {EXCLUDE} >/dev/null "
              f"&& git write-tree; rc=$?; rm -f $GIT_INDEX_FILE; exit $rc")
    out, code = container.exec(script, workdir="/testbed", timeout=120)
    if code != 0:
        raise RuntimeError(f"snapshot failed (exit {code}): {out[-500:]!r}")
    return out.decode().strip().splitlines()[-1]


def prepare_episode(container):
    """Remove the tests from the episode container and return the baseline tree hash (keep it on the host)."""
    out, code = container.exec("rm -rf /r2e_tests /testbed/run_tests.sh /testbed/r2e_tests", workdir="/")
    if code != 0:
        raise RuntimeError(f"removing tests failed (exit {code}): {out[-500:]!r}")
    return _snapshot_tree(container, "/tmp/.qtrace_base_index")


def final_diff(container, base_tree):
    """Binary diff from the baseline snapshot to the current tree, untracked files included."""
    final = _snapshot_tree(container, "/tmp/.qtrace_final_index")
    out, code = container.exec(f"git diff --binary {base_tree} {final}", workdir="/testbed", timeout=120)
    if code != 0:
        raise RuntimeError(f"diff failed (exit {code}): {out[-500:]!r}")
    return out


def _run_tests(container, timeout):
    script = "ln -sfn /r2e_tests /testbed/r2e_tests && bash /testbed/run_tests.sh"
    out, code = container.exec(script, workdir="/testbed", timeout=timeout)
    log = re.sub(r"\x1b\[[0-9;]*m|\r", "", out.decode("utf-8", errors="replace"))
    return parse_pytest_log(log), code == 124


def grade_fresh(image, diff, expected_output_json, baseline, timeout=TEST_TIMEOUT, docker="docker"):
    """Apply `diff` in a fresh container from `image`, run the original tests there, and grade.

    With an empty diff and baseline=None this returns the baseline statuses instead (cache per task).
    """
    c = DockerContainer.start(image, f"qtrace-grade-{uuid.uuid4().hex[:10]}", docker=docker)
    try:
        error = None
        if diff:
            out, code = c.exec("git apply --binary --whitespace=nowarn -", workdir="/testbed", stdin=diff)
            if code != 0:
                error = f"git apply failed: {out[-300:].decode('utf-8', errors='replace')}"
        parsed, timed_out = ({}, False) if error else _run_tests(c, timeout)
    finally:
        c.stop()
    if baseline is None:
        return parsed
    expected = json.loads(expected_output_json) if isinstance(expected_output_json, str) else expected_output_json
    return grading(parsed, expected, baseline, tests_timed_out=timed_out,
                   test_files_touched=bool(touched_test_files(diff)), grading_error=error)


def baseline_statuses(image, timeout=TEST_TIMEOUT, docker="docker"):
    """Statuses of the untouched repo, in a fresh container. Deterministic per image: cache per task."""
    return grade_fresh(image, b"", None, None, timeout=timeout, docker=docker)


def verify(container, image, base_tree, expected_output_json, baseline, timeout=TEST_TIMEOUT):
    """Grade the episode container's current repo state, without running anything the agent could touch."""
    return grade_fresh(image, final_diff(container, base_tree), expected_output_json, baseline, timeout=timeout)
