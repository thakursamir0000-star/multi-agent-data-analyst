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
