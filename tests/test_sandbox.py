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


# ── Security: File-System Escapes via Allowed Libraries ────────────

_DF_VARS = {"df": pd.DataFrame({"a": [1, 2], "b": [3, 4]})}

BLOCKED_ESCAPES = {
    # pandas readers (direct, imported, aliased, pickle)
    "pd.read_csv": "pd.read_csv('data.csv')",
    "pd.read_pickle": "pd.read_pickle('data.pkl')",
    "pd.read_excel": "pd.read_excel('data.xlsx')",
    "from pandas import read_csv": "from pandas import read_csv",
    "from pandas import *": "from pandas import *",
    "alias pd.read_csv": "r = pd.read_csv",
    # pandas writers that always target a file
    "df.to_pickle": "df.to_pickle('out.pkl')",
    "df.to_parquet": "df.to_parquet('out.parquet')",
    "df.to_excel": "df.to_excel('out.xlsx')",
    "pd.to_pickle": "pd.to_pickle(df, 'out.pkl')",
    "pd.ExcelWriter": "w = pd.ExcelWriter('out.xlsx')",
    "pd.HDFStore": "s = pd.HDFStore('out.h5')",
    # path-optional serializers given a path / buffer / unknown args
    "df.to_csv(path)": "df.to_csv('out.csv')",
    "df.to_json(path)": "df.to_json('out.json')",
    "df.to_string(path)": "df.to_string('out.txt')",
    "df.to_csv(path_or_buf=)": "df.to_csv(path_or_buf='out.csv')",
    "df.to_html(buf=)": "df.to_html(buf='out.html')",
    "df.to_csv(**kwargs)": "kw = {}\ndf.to_csv(**kw)",
    "df.to_csv(*args)": "args = ['out.csv']\ndf.to_csv(*args)",
    "alias df.to_csv": "g = df.to_csv",
    "unbound DataFrame.to_csv": "df.pipe(pd.DataFrame.to_csv, 'out.csv')",
    # pandas string dispatch
    "df.agg('to_csv')": "df.agg('to_csv', 0, 'out.csv')",
    "df.apply('to_pickle')": "df.apply('to_pickle')",
    # pandas.io
    "import pandas.io.pickle": "import pandas.io.pickle",
    "pd.io.common": "pd.io.common",
    # NumPy
    "np.load": "np.load('arr.npy', allow_pickle=True)",
    "np.save": "np.save('arr.npy', [1, 2])",
    "np.loadtxt": "np.loadtxt('data.txt')",
    "np.genfromtxt": "np.genfromtxt('data.txt')",
    "np.fromfile": "np.fromfile('data.bin')",
    "np.memmap": "np.memmap('data.bin')",
    "ndarray.tofile": "np.array([1]).tofile('out.bin')",
    "ndarray.dump": "np.array([1]).dump('out.pkl')",
    "from numpy import load": "from numpy import load",
    "np.ctypeslib": "np.ctypeslib",
    "import numpy.ctypeslib": "import numpy.ctypeslib",
    "np.lib.npyio": "np.lib.npyio",
    # io
    "io.open": "import io\nio.open('data.txt')",
    "alias io.open": "import io\nf = io.open",
    "io.FileIO": "import io\nio.FileIO('data.txt')",
    "from io import open": "from io import open",
    # matplotlib
    "plt.savefig": "plt.plot([1, 2])\nplt.savefig('out.png')",
    "figure.savefig": "plt.gcf().savefig('out.png')",
    "plt.imread": "plt.imread('img.png')",
    "canvas.print_png": "plt.gcf().canvas.print_png('out.png')",
    "import matplotlib.animation": "import matplotlib.animation",
    "import matplotlib.cbook": "import matplotlib.cbook",
    "from matplotlib import cbook": "from matplotlib import cbook",
    # system modules re-exported as attributes
    "matplotlib.os": "import matplotlib\nmatplotlib.os.getcwd()",
    "plt.matplotlib.os": "plt.matplotlib.os",
    "from matplotlib import os": "from matplotlib import os",
    "statistics.random._os": "import statistics\nstatistics.random._os",
    "json.codecs": "import json\njson.codecs",
    "pd.core.generic.gc": "pd.core.generic.gc",
    # dunder / frame introspection
    "__globals__": "pd.DataFrame.head.__globals__",
    "__traceback__": "try:\n    1 / 0\nexcept Exception as e:\n    t = e.__traceback__",
    "gi_frame": "g = (x for x in [1])\nfr = g.gi_frame",
    "f_back": "def f(fr):\n    return fr.f_back",
    "tb_frame": "def f(tb):\n    return tb.tb_frame",
    # dynamic execution via alias
    "alias pd.eval": "e = pd.eval",
}


class TestBlockedFileSystemEscapes:
    @pytest.mark.parametrize(
        "code", BLOCKED_ESCAPES.values(), ids=BLOCKED_ESCAPES.keys()
    )
    def test_escape_blocked(self, code):
        with pytest.raises(SecurityError, match="Blocked"):
            run_code(code, local_vars=dict(_DF_VARS))

    def test_blocked_write_creates_no_file(self, tmp_path):
        target = tmp_path / "out.json"
        with pytest.raises(SecurityError):
            run_code(f"df.to_json({str(target)!r})", local_vars=dict(_DF_VARS))
        assert not target.exists()

    def test_error_message_suggests_column_access(self):
        with pytest.raises(SecurityError, match=r"df\['<name>'\]"):
            run_code("df.load.sum()", local_vars=dict(_DF_VARS))


# ── Legitimate In-Memory Operations Still Work ─────────────────────

ALLOWED_OPERATIONS = {
    "df.to_string()": "print(df.to_string(index=False))",
    "df.to_csv()": "print(df.to_csv(index=False))",
    "df.to_csv(None)": "print(df.to_csv(None, index=False))",
    "df.to_csv(path_or_buf=None)": "print(df.to_csv(path_or_buf=None))",
    "df.to_json(orient=)": "print(df.to_json(orient='records'))",
    "df.to_html()": "print(df.to_html())",
    "df.to_dict()": "print(df.to_dict())",
    "series.to_list()": "print(df['a'].to_list())",
    "df.to_numpy()": "print(df.to_numpy())",
    "pd.to_datetime": "print(pd.to_datetime('2024-01-15').year)",
    "pd.to_numeric": "print(pd.to_numeric(pd.Series(['1', '2'])).sum())",
    "io.StringIO": "import io\nbuf = io.StringIO('x,y\\n1,2')\nprint(buf.read())",
    "func.__name__": "print(np.mean.__name__)",
    "json.dumps/loads": "import json\nprint(json.loads(json.dumps({'k': 1})))",
    "groupby.agg('sum')": "print(df.groupby('a')['b'].agg('sum'))",
}


class TestAllowedInMemoryOperations:
    @pytest.mark.parametrize(
        "code", ALLOWED_OPERATIONS.values(), ids=ALLOWED_OPERATIONS.keys()
    )
    def test_operation_allowed(self, code):
        result = run_code(code, local_vars=dict(_DF_VARS))
        assert result["error"] is None
        assert result["stdout"]

    def test_dataframe_plot_still_captured(self):
        result = run_code(
            "df.plot(kind='bar', x='a', y='b')\nplt.title('t')",
            local_vars=dict(_DF_VARS),
        )
        assert result["error"] is None
        assert result["plot_base64"] is not None


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
