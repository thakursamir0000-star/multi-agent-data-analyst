"""
Tests for the sandbox's process isolation (tools/sandbox.py + tools/sandbox_worker.py).
"""

import time

import pandas as pd
import pytest

from tools import sandbox
from tools.sandbox import RESULT_MARKER, run_code


@pytest.fixture(params=["process", "thread"])
def isolation(request, monkeypatch):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, request.param)
    return request.param


def test_snippets_cannot_change_the_callers_data(isolation):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = run_code("df['a'] = 0\ndel df['b']\nprint(df.to_dict('list'))", {"df": df})
    assert result["error"] is None and "{'a': [0, 0]}" in result["stdout"]
    assert df.to_dict("list") == {"a": [1, 2], "b": [3, 4]}


def test_runaway_code_is_stopped_on_time(monkeypatch):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    start = time.perf_counter()
    result = run_code("while True: pass", timeout=2)
    assert "timed out after 2 seconds" in result["error"]
    assert time.perf_counter() - start < 2 + sandbox.STARTUP_GRACE_SECONDS


def test_child_environment_has_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_should_not_leak")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_should_not_leak")
    env = sandbox._child_env("/tmp/work")
    assert "GROQ_API_KEY" not in env and "LANGSMITH_API_KEY" not in env
    assert set(env) <= set(sandbox._ENV_ALLOWLIST) | {
        "MPLBACKEND", "MPLCONFIGDIR", "PYTHONIOENCODING",
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    }


def test_printed_fake_result_cannot_spoof_the_parent(monkeypatch):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    fake = RESULT_MARKER + '{"stdout": "spoofed", "error": null}'
    result = run_code(f"print({fake!r})")
    assert result["stdout"] != "spoofed"
    assert "spoofed" in result["stdout"]  # it is just captured output


def test_errors_and_plots_survive_the_process_boundary(monkeypatch):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    assert "ZeroDivisionError" in run_code("x = 1 / 0")["error"]
    plot = run_code("plt.plot([1, 2, 3]); plt.title('t')")
    assert plot["error"] is None and plot["plot_base64"]
    assert run_code("print('₹ 中文')")["stdout"].strip() == "₹ 中文"


def test_a_crashing_child_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    broken = tmp_path / "worker.py"
    broken.write_text("import sys\nsys.stderr.write('boom')\nsys.exit(3)\n", encoding="utf-8")
    monkeypatch.setattr(sandbox, "_WORKER", broken)
    result = run_code("print(1)")
    assert "exited with code 3" in result["error"] and "boom" in result["error"]


def test_static_checks_still_run_before_any_process_starts(monkeypatch):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    monkeypatch.setattr(sandbox, "_WORKER", "/nonexistent/worker.py")  # would fail if reached
    with pytest.raises(sandbox.SecurityError):
        run_code("import os")


# ── run-time guard (tools/sandbox_guard.py) ─────────────────────────
# A review found that df.agg(<name built at run time>, 0, path) reached
# DataFrame.to_csv and wrote a file outside the sandbox: the static scan only
# saw string literals. These tests keep both layers honest.

@pytest.mark.parametrize("built", [
    '"to_" + "csv"',
    '"".join(["to_", "csv"])',
    'f"to_{\'csv\'}"',
])
def test_writer_names_built_from_literals_are_rejected_by_the_scan(built):
    with pytest.raises(sandbox.SecurityError, match="to_csv"):
        sandbox._scan_ast(f"df.agg({built}, 0, 'out.csv')")


@pytest.mark.parametrize("build", [
    'name = "".join(reversed(["csv", "to_"]))',
    'name = "to-csv".replace("-", "_")',
])
def test_a_name_built_at_run_time_cannot_write_outside_the_sandbox(monkeypatch, tmp_path, build):
    monkeypatch.setenv(sandbox.ISOLATION_ENV, "process")
    target = tmp_path / "escaped.csv"
    result = run_code(f"{build}\ndf.agg(name, 0, r'{target}')", {"df": pd.DataFrame({"a": [1]})})
    assert not target.exists()
    assert "outside the sandbox" in result["error"]


GUARD_PROBE = r"""
import os, socket, subprocess, sys, tempfile
sys.path.insert(0, sys.argv[1])
from tools.sandbox_guard import install_guard

work, outside = tempfile.mkdtemp(), sys.argv[2]
install_guard(work)

def attempt(label, action):
    try:
        action()
        print(label, "allowed")
    except PermissionError:
        print(label, "blocked")

attempt("write-inside", lambda: open(os.path.join(work, "ok.txt"), "w").write("x"))
attempt("read-inside", lambda: open(os.path.join(work, "ok.txt")).read())
attempt("write-outside", lambda: open(os.path.join(outside, "new.txt"), "w").write("x"))
attempt("read-outside", lambda: open(os.path.join(outside, "secret.txt")).read())
attempt("delete-outside", lambda: os.remove(os.path.join(outside, "secret.txt")))
attempt("network", lambda: socket.create_connection(("127.0.0.1", 9), timeout=1))
attempt("start-program", lambda: subprocess.run([sys.executable, "-c", "pass"]))
attempt("stdlib-import", lambda: __import__("fractions"))
"""


def test_guard_blocks_outside_files_network_and_programs(tmp_path):
    # In its own process: an audit hook can't be removed and would stay in pytest's.
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "secret.txt").write_text("GROQ_API_KEY=not-a-real-key", encoding="utf-8")
    project = str(Path(sandbox.__file__).resolve().parents[1])
    probe = subprocess.run([sys.executable, "-I", "-c", GUARD_PROBE, project, str(tmp_path)],
                           capture_output=True, text=True, timeout=60)
    outcome = dict(line.split() for line in probe.stdout.splitlines() if line.strip())
    assert outcome == {
        "write-inside": "allowed", "read-inside": "allowed", "stdlib-import": "allowed",
        "write-outside": "blocked", "read-outside": "blocked", "delete-outside": "blocked",
        "network": "blocked", "start-program": "blocked",
    }, probe.stderr
    assert (tmp_path / "secret.txt").exists() and not (tmp_path / "new.txt").exists()
