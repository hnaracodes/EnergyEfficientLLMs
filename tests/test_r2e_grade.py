"""Tests for the R2E-Gym grader and the repo-state fingerprint.

Unit tests always run. Docker tests need amd64 R2E-Gym images and run only with R2E_DOCKER_TESTS=1:
    R2E_DOCKER_TESTS=1 python -m pytest tests/test_r2e_grade.py -q
"""

import json
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.agent.r2e_grade import TEST_PATH, grading, parse_pytest_log, r2e_resolved, touched_test_files

LOG = """============================= test session starts ==============================
r2e_tests/test_1.py ..F.
=========================== short test summary info ============================
PASSED r2e_tests/test_1.py::TestA::test_one
PASSED r2e_tests/test_1.py::TestA::test_two
FAILED r2e_tests/test_1.py::TestA::test_three - AssertionError: boom
ERROR r2e_tests/test_1.py::test_four - ImportError
"""


def test_parse_matches_r2e_names_and_statuses():
    assert parse_pytest_log(LOG) == {"TestA.test_one": "PASSED", "TestA.test_two": "PASSED",
                                     "TestA.test_three": "FAILED", "test_four": "ERROR"}
    assert parse_pytest_log("no summary here") == {} and parse_pytest_log(None) == {}


def test_resolved_needs_an_exact_match():
    expected = {"TestA.test_one": "PASSED", "TestA.test_two": "PASSED"}
    assert r2e_resolved({"TestA.test_one": "PASSED", "TestA.test_two": "PASSED"}, expected)
    assert not r2e_resolved({"TestA.test_one": "PASSED", "TestA.test_two": "FAILED"}, expected)
    assert not r2e_resolved({"TestA.test_one": "PASSED"}, expected)  # a missing test fails the task
    assert not r2e_resolved({**expected, "extra": "PASSED"}, expected)
    assert r2e_resolved({"\u001b[31mTestA.test_one\u001b[0m": "PASSED", "TestA.test_two - x": "PASSED"}, expected)


def test_grading_splits_fail_to_pass_and_pass_to_pass():
    expected = {"a": "PASSED", "b": "PASSED", "c": "PASSED"}
    baseline = {"a": "FAILED", "b": "PASSED", "c": "PASSED"}
    g = grading({"a": "PASSED", "b": "PASSED", "c": "FAILED"}, expected, baseline)
    assert (g.fail_to_pass_ok, g.fail_to_pass_total, g.pass_to_pass_ok, g.pass_to_pass_total) == (1, 1, 1, 2)
    assert not g.resolved
    assert grading(expected, expected, baseline).resolved
    assert not grading(expected, expected, baseline, grading_error="git apply failed").resolved


def test_touched_test_files_tripwire():
    diff = (b"diff --git a/pkg/core.py b/pkg/core.py\n+x\n"
            b"diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n"
            b"diff --git a/pkg/tests/test_core.py b/pkg/tests/test_core.py\n")
    assert touched_test_files(diff) == ["conftest.py", "pkg/tests/test_core.py"]
    assert touched_test_files(b"diff --git a/pkg/core.py b/pkg/core.py\n") == []


docker = pytest.mark.skipif(os.environ.get("R2E_DOCKER_TESTS") != "1", reason="set R2E_DOCKER_TESTS=1")

# Two small tasks from R2E-Gym-Subset@2e8108f, from different repos (first orange3 and tornado rows).
IMAGES = ["namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d",
          "namanjain12/tornado_final:b5ec807edc83c8e7d1d12553d635ebe765e5c614"]


def _row(image):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    base = "datasets/R2E-Gym/R2E-Gym-Subset@2e8108ff942f24fcb5686badfaf7f9a8808566d5/data"
    for path in sorted(fs.glob(f"{base}/*.parquet")):
        t = pq.read_table(fs.open(path), columns=["docker_image", "expected_output_json", "parsed_commit_content"])
        for r in t.to_pylist():
            if r["docker_image"] == image:
                return r
    raise KeyError(image)


def _apply_gold(container, row):
    """Write the commit's new file contents (non-binary, non-test files) into /testbed."""
    import base64

    for d in json.loads(row["parsed_commit_content"])["file_diffs"]:
        path = d["header"]["file"]["path"]
        if d["is_binary_file"] or TEST_PATH.search(path) or d.get("new_file_content") is None:
            continue
        data = base64.b64encode(d["new_file_content"].encode()).decode()
        _, code = container.exec(f"echo {data} | base64 -d > '{path}'")
        assert code == 0, path


@pytest.fixture(params=IMAGES, ids=lambda img: img.split("/")[1].split("_")[0])
def task(request):
    from qtrace.agent.docker_exec import DockerContainer
    from qtrace.agent.r2e_grade import prepare_episode

    c = DockerContainer.start(request.param, f"qtrace-test-{uuid.uuid4().hex[:8]}")
    c.image = request.param
    try:
        c.base_tree = prepare_episode(c)
        yield c
    finally:
        c.stop()


@docker
def test_tests_are_gone_from_the_episode_container(task):
    out, _ = task.exec("ls /r2e_tests /testbed/run_tests.sh /testbed/r2e_tests 2>&1", workdir="/")
    assert out.count(b"No such file") == 3
    found, _ = task.exec("find / \\( -name run_tests.sh -o -name r2e_tests \\) -not -path '/proc/*' 2>/dev/null",
                         workdir="/")
    assert found.strip() == b""


@docker
def test_fresh_container_grading(task):
    from qtrace.agent.r2e_grade import baseline_statuses, final_diff, verify

    row = _row(task.image)
    base = baseline_statuses(task.image)
    assert base, "baseline produced no parsed tests"
    g0 = verify(task, task.image, task.base_tree, row["expected_output_json"], base)
    assert not g0.resolved and g0.grading_error is None
    _apply_gold(task, row)
    task.exec("echo 'new helper' > NEW_FILE_FROM_AGENT.txt")
    assert b"NEW_FILE_FROM_AGENT.txt" in final_diff(task, task.base_tree)  # untracked files are in the diff
    g = verify(task, task.image, task.base_tree, row["expected_output_json"], base)
    assert g.resolved and g.fail_to_pass_ok == g.fail_to_pass_total > 0
    assert not g.test_files_touched and not g.tests_timed_out
    task.exec("echo 'import pytest' > conftest.py")
    assert verify(task, task.image, task.base_tree, row["expected_output_json"], base).test_files_touched
    status, _ = task.exec("git status --porcelain --untracked-files=no")
    assert b"conftest.py" not in status  # grading never staged anything in the agent's repo


@docker
def test_fingerprint_tracks_repo_state(task):
    import time

    from qtrace.agent.fingerprint import fingerprint

    start = time.perf_counter()
    a = fingerprint(task)
    print(f"fingerprint took {time.perf_counter() - start:.2f} s")
    assert fingerprint(task) == a  # same tree, same hash
    task.exec("echo 'x' > untracked_note.txt")
    b = fingerprint(task)
    assert b != a  # untracked files count
    task.exec("echo 'y' > untracked_note.txt")
    assert fingerprint(task) != b  # and so does their content
    task.exec("rm untracked_note.txt")
    assert fingerprint(task) == a
    tracked = task.exec("git ls-files '*.py' | head -1")[0].decode().strip()
    task.exec(f"printf ' ' >> {tracked}")
    assert fingerprint(task) != a  # a whitespace-only change differs
