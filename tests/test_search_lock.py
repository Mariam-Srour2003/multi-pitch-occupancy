"""The preprocessing search lock.

Two searches writing one state file silently mix their results - that happened, and the
survivor held evaluations scored on different frame counts with no way to tell them apart.
The lock prevents it. These tests cover the two ways a lock itself goes wrong: refusing to
release after a crash, and being committed to the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from experiments.preprocess_search import SearchLock, process_alive


def test_the_lock_blocks_a_second_search(tmp_path) -> None:
    path = tmp_path / ".search.lock"
    with SearchLock(path):
        with pytest.raises(SystemExit, match="already holds"):
            with SearchLock(path):
                pass


def test_the_lock_is_released_on_the_way_out(tmp_path) -> None:
    path = tmp_path / ".search.lock"
    with SearchLock(path):
        assert path.exists()
    assert not path.exists()


def test_the_lock_is_released_even_when_the_search_raises(tmp_path) -> None:
    path = tmp_path / ".search.lock"
    with pytest.raises(ValueError):
        with SearchLock(path):
            raise ValueError("an embedding pass failed")
    assert not path.exists()


def test_a_lock_held_by_a_dead_process_is_cleared(tmp_path) -> None:
    """A killed search never reaches __exit__, so its lock outlives it. Before this, that
    blocked every future run until someone deleted a hidden file by hand."""
    path = tmp_path / ".search.lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])  # exits immediately
    dead.wait()
    path.write_text(str(dead.pid), encoding="utf-8")  # a real pid that is really gone
    with SearchLock(path):
        assert path.read_text(encoding="utf-8") == str(os.getpid())


def test_a_lock_holding_junk_is_treated_as_held(tmp_path) -> None:
    """A truncated or corrupt lock is not evidence that nothing is running, so it refuses
    rather than assuming it is safe to proceed."""
    path = tmp_path / ".search.lock"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        with SearchLock(path):
            pass


def test_liveness_check_does_not_kill_the_process_it_asks_about() -> None:
    """os.kill(pid, 0) routes through TerminateProcess on Windows, so the obvious
    implementation of this check would kill the process it was probing - including, when
    asked about itself, the test runner."""
    assert process_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert process_alive(proc.pid)
        assert process_alive(proc.pid)  # asked twice; still alive
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()
    assert not process_alive(proc.pid)


def test_no_lock_file_is_tracked_in_git() -> None:
    """One was, naming a pid from the machine that committed it."""
    out = subprocess.run(
        ["git", "ls-files", "results/"], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ).stdout
    assert not [ln for ln in out.splitlines() if ln.endswith(".lock")]
