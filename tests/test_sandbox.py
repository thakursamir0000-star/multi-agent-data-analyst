"""
Tests for tools/sandbox.py — sandboxed code executor.

Validates both allowed operations and security restrictions.
"""

import pytest
import pandas as pd

from tools.sandbox import SecurityError, run_code


# ── Successful Execution Tests ──────────────────────────────────────

class TestAllowedExecution:
    def test_simple_print(self):
        result = run_code('print("hello world")')
        assert result["error"] is None
        assert "hello world" in result["stdout"]

    def test_pandas_operations(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = run_code(
            'print(df["a"].sum())',
            local_vars={"df": df},
        )
        assert result["error"] is None
        assert "6" in result["stdout"]

    def test_numpy_operations(self):
        result = run_code(
            "import numpy as np\nprint(np.mean([1, 2, 3, 4, 5]))"
        )
        assert result["error"] is None
        assert "3.0" in result["stdout"]

    def test_matplotlib_plot_generation(self):
        result = run_code(
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "plt.plot([1, 2, 3], [1, 4, 9])\n"
            "plt.title('Test')\n"
        )
        assert result["error"] is None
        assert result["plot_base64"] is not None
        assert len(result["plot_base64"]) > 100  # Non-trivial base64 string

    def test_math_operations(self):
        result = run_code(
            "import math\nprint(math.sqrt(144))"
        )
        assert result["error"] is None
        assert "12.0" in result["stdout"]

    def test_json_operations(self):
        result = run_code(
            'import json\nprint(json.dumps({"key": "value"}))'
        )
        assert result["error"] is None
        assert '"key"' in result["stdout"]

    def test_datetime_operations(self):
        result = run_code(
            "import datetime\nprint(datetime.datetime(2024, 1, 15).strftime('%Y-%m-%d'))"
        )
        assert result["error"] is None
        assert "2024-01-15" in result["stdout"]

    def test_list_comprehension(self):
        result = run_code(
            "result = [x**2 for x in range(5)]\nprint(result)"
        )
        assert result["error"] is None
        assert "[0, 1, 4, 9, 16]" in result["stdout"]

    def test_statistics_module(self):
        result = run_code(
            "import statistics\nprint(statistics.mean([1, 2, 3, 4, 5]))"
        )
        assert result["error"] is None
        assert "3" in result["stdout"]


# ── Security: Blocked Imports ───────────────────────────────────────

class TestBlockedImports:
    def test_import_os_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*os"):
            run_code("import os")

    def test_import_sys_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*sys"):
            run_code("import sys")

    def test_import_subprocess_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*subprocess"):
            run_code("import subprocess")

    def test_import_shutil_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*shutil"):
            run_code("import shutil")

    def test_import_socket_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*socket"):
            run_code("import socket")

    def test_import_requests_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*requests"):
            run_code("import requests")

    def test_import_urllib_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*urllib"):
            run_code("import urllib")

    def test_from_os_import_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*os"):
            run_code("from os import path")

    def test_import_pathlib_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*pathlib"):
            run_code("import pathlib")

    def test_import_pickle_blocked(self):
        with pytest.raises(SecurityError, match="Blocked import.*pickle"):
            run_code("import pickle")

    def test_unknown_module_blocked(self):
        with pytest.raises(SecurityError, match="not in whitelist"):
            run_code("import some_unknown_module")


# ── Security: Blocked Function Calls ────────────────────────────────

class TestBlockedFunctions:
    def test_open_blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*open"):
            run_code('open("/etc/passwd", "r")')

    def test_exec_blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*exec"):
            run_code('exec("import os")')

    def test_eval_blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*eval"):
            run_code('eval("1+1")')

    def test_compile_blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*compile"):
            run_code('compile("import os", "<str>", "exec")')

    def test___import___blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*__import__"):
            run_code('__import__("os")')


# ── Security: Blocked Attribute Access ──────────────────────────────

class TestBlockedAttributes:
    def test_builtins_access_blocked(self):
        with pytest.raises(SecurityError, match="Blocked attribute.*__builtins__"):
            run_code('x = "".__class__.__builtins__')

    def test_subclasses_blocked(self):
        with pytest.raises(SecurityError, match="Blocked attribute.*__subclasses__"):
            run_code("object.__subclasses__()")

    def test_import_via_attribute_blocked(self):
        with pytest.raises(SecurityError, match="Blocked function call.*getattr"):
            run_code('getattr(None, "__import__")')


# ── Security: Timeout ──────────────────────────────────────────────

class TestTimeout:
    def test_infinite_loop_times_out(self):
        result = run_code("while True: pass", timeout=2)
        assert result["error"] is not None
        assert "timed out" in result["error"].lower()


# ── Error Handling ──────────────────────────────────────────────────

class TestErrorHandling:
    def test_syntax_error(self):
        with pytest.raises(SecurityError, match="Syntax error"):
            run_code("def f(\n")

    def test_runtime_error_captured(self):
        result = run_code("x = 1 / 0")
        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]

    def test_name_error_captured(self):
        result = run_code("print(undefined_variable)")
        assert result["error"] is not None
        assert "NameError" in result["error"]

    def test_empty_code(self):
        result = run_code("")
        assert result["error"] is None
        assert result["stdout"] == ""
