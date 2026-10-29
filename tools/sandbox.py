"""
Sandboxed Code Executor — runs pandas/matplotlib code in a restricted namespace.

Security model:
1. AST pre-scan blocks dangerous function calls (open, exec, eval, compile, __import__).
2. AST pre-scan blocks file-system access through allowed libraries
   (pd.read_*, df.to_* with a path, np.load/save, io.open, plt.savefig, ...),
   dunder / frame introspection, and system modules re-exported as
   attributes (e.g. matplotlib.os) — including aliased references.
3. Custom import function whitelist blocks os, sys, subprocess, etc.
4. No access to real builtins beyond a curated safe set.
5. Process isolation (default): each snippet runs in a fresh child Python
   process (tools/sandbox_worker.py) that is killed on timeout, gets memory
   and CPU limits on Linux/macOS, starts in an empty temp directory, and
   inherits no secrets (API keys are not in its environment). The child
   works on its own copy of the data and reports back as JSON only, so a
   compromised snippet can't touch the app's state or run code in the app.
   SANDBOX_ISOLATION=thread runs in-process instead (faster, weaker; for
   local development).
"""

from __future__ import annotations

import ast
import base64
import copy
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
import threading
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any


class SecurityError(Exception):
    """Raised when sandboxed code attempts a forbidden operation."""


# ── Whitelist / Blocklist Configuration ──────────────────────────────

ALLOWED_MODULES = frozenset({
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "math",
    "statistics",
    "datetime",
    "time",
    "json",
    "re",
    "io",
    "base64",
    "collections",
    "itertools",
    "functools",
})

BLOCKED_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "http.client",
    "http.server",
    "urllib",
    "urllib.request",
    "requests",
    "importlib",
    "ctypes",
    "signal",
    "pathlib",
    "builtins",
    "code",
    "codeop",
    "compileall",
    "pickle",
    "shelve",
    "multiprocessing",
    "threading",
    "webbrowser",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
})

BLOCKED_FUNCTION_NAMES = frozenset({
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "execfile",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "type",
    "super",
    "classmethod",
    "staticmethod",
    "property",
})

BLOCKED_ATTRIBUTE_NAMES = frozenset({
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
})

# Every other dunder attribute is blocked too (e.g. __globals__, which
# exposes the real builtins); only these harmless ones are allowed.
ALLOWED_DUNDER_NAMES = frozenset({"__name__", "__doc__"})

# Frame / generator / traceback introspection — walking frames reaches
# the sandbox's own module globals.
BLOCKED_INTROSPECTION_NAMES = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom",
    "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code",
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace",
    "tb_frame", "tb_next",
})

# System modules that allowed libraries re-export as attributes, e.g.
# matplotlib.os, json.codecs, statistics.random._os, pd.core.generic.gc.
# Leading underscores are stripped before matching.
BLOCKED_MODULE_ATTRIBUTES = frozenset({
    "os", "nt", "posix", "ntpath", "posixpath", "genericpath",
    "sys", "subprocess", "shutil", "pathlib", "glob", "tempfile",
    "importlib", "pkgutil", "zipimport", "runpy", "builtins", "ctypes",
    "pickle", "shelve", "marshal", "codecs", "fileinput", "mmap",
    "zipfile", "tarfile", "gzip", "bz2", "lzma",
    "socket", "urllib", "webbrowser", "multiprocessing", "threading",
    "signal", "gc", "inspect", "traceback", "linecache", "tokenize",
    "winapi", "winreg", "msvcrt",
})

# pandas writers that always target a file (readers: see BLOCKED_IO_PREFIXES).
BLOCKED_PANDAS_WRITERS = frozenset({
    "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql",
    "to_stata", "to_excel", "to_orc", "to_clipboard", "to_gbq",
})

# Serializers that return a string when called without a path. Allowed only
# as a direct, pathless call: ``print(df.to_csv(index=False))``.
PATH_OPTIONAL_WRITERS = frozenset({
    "to_csv", "to_json", "to_string", "to_html", "to_markdown",
    "to_latex", "to_xml",
})
_PATH_KEYWORDS = frozenset({"path_or_buf", "path_or_buffer", "buf", "path"})

# File-system access reachable through pandas, NumPy, io and matplotlib.
# Blocked as attribute references (calls *and* aliases such as
# ``f = io.open``), as imported names, and as parts of import paths.
BLOCKED_IO_NAMES = BLOCKED_PANDAS_WRITERS | frozenset({
    # pandas
    "ExcelWriter", "ExcelFile", "HDFStore", "show_versions",
    "io",  # pandas.io (file handles, pickle, parsers)
    # NumPy
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "fromregex", "tofile", "dump", "memmap",
    "open_memmap", "write_array", "DataSource", "npyio", "ctypeslib",
    "f2py",
    # io
    "open", "FileIO", "open_code",
    # matplotlib
    "savefig", "imread", "imsave", "cbook", "animation", "backends",
    "PdfPages", "rc_file",
})
BLOCKED_IO_PREFIXES = ("read_", "print_")  # pd.read_*, canvas.print_png()

# Dynamic-execution builtins that must not be referenced as attributes
# either (e.g. ``e = pd.eval``); calls are caught by BLOCKED_FUNCTION_NAMES.
BLOCKED_CALLABLE_ATTRIBUTES = frozenset({
    "exec", "eval", "compile", "execfile", "breakpoint", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
})

_INTERNALS_HINT = (
    "interpreter internals and system modules are not available in the sandbox"
)
_FILE_IO_HINT = (
    "reading or writing files is not allowed; work with the provided `df` "
    "(for a DataFrame column, use df['<name>'] instead of attribute access)"
)

DEFAULT_TIMEOUT_SECONDS = 30


# ── AST Security Scanner ────────────────────────────────────────────

def _blocked_name_reason(name: str) -> str | None:
    """Return why an attribute or imported name is blocked, or None if allowed."""
    if name in BLOCKED_ATTRIBUTE_NAMES or name in BLOCKED_INTROSPECTION_NAMES:
        return _INTERNALS_HINT
    if (
        name.startswith("__")
        and name.endswith("__")
        and name not in ALLOWED_DUNDER_NAMES
    ):
        return _INTERNALS_HINT
    bare = name.lstrip("_")
    if bare in BLOCKED_MODULE_ATTRIBUTES:
        return _INTERNALS_HINT
    if bare in BLOCKED_IO_NAMES or bare.startswith(BLOCKED_IO_PREFIXES):
        return _FILE_IO_HINT
    if name in PATH_OPTIONAL_WRITERS:
        return (
            f"call it directly without a path to get a string, "
            f"e.g. print(df.{name}())"
        )
    if name in BLOCKED_CALLABLE_ATTRIBUTES:
        return "dynamic code execution is not allowed"
    return None


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_pathless_call(node: ast.Call) -> bool:
    """True if a serializer call has no path: no positional args, no
    ``**kwargs`` unpacking, and no path keyword (other than ``None``)."""
    if any(not _is_none(arg) for arg in node.args):
        return False
    for kw in node.keywords:
        if kw.arg is None:  # **kwargs unpacking
            return False
        if kw.arg in _PATH_KEYWORDS and not _is_none(kw.value):
            return False
    return True


class _ASTSecurityVisitor(ast.NodeVisitor):
    """Walks the AST and raises SecurityError for forbidden patterns."""

    def __init__(self) -> None:
        # ids of Attribute nodes vetted as pathless serializer calls
        self._allowed_attributes: set[int] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_module(node.module, node)
        for alias in node.names:
            if alias.name == "*":
                raise SecurityError(
                    f"Blocked import: 'from {node.module} import *' "
                    f"at line {node.lineno}"
                )
            reason = _blocked_name_reason(alias.name)
            if reason:
                raise SecurityError(
                    f"Blocked import: '{alias.name}' from '{node.module}' "
                    f"at line {node.lineno}. {reason}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Direct function calls: open(...), exec(...), etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_FUNCTION_NAMES:
                raise SecurityError(
                    f"Blocked function call: '{node.func.id}()' "
                    f"at line {node.lineno}"
                )
        # Attribute calls: __builtins__.__import__(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_FUNCTION_NAMES:
                raise SecurityError(
                    f"Blocked attribute call: '.{node.func.attr}()' "
                    f"at line {node.lineno}"
                )
            # df.to_csv() is fine (returns a string); df.to_csv("x.csv") is not
            if node.func.attr in PATH_OPTIONAL_WRITERS:
                if not _is_pathless_call(node):
                    raise SecurityError(
                        f"Blocked file write: '.{node.func.attr}()' with a "
                        f"path or buffer at line {node.lineno}. Call it "
                        f"without a path to get a string, e.g. "
                        f"print(df.{node.func.attr}())"
                    )
                self._allowed_attributes.add(id(node.func))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Checked on every reference, not just calls, so aliases such as
        # `f = io.open` or `g = df.to_csv` are caught too.
        if id(node) not in self._allowed_attributes:
            reason = _blocked_name_reason(node.attr)
            if reason:
                raise SecurityError(
                    f"Blocked attribute access: '.{node.attr}' "
                    f"at line {node.lineno}. {reason}"
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # pandas can invoke methods by name, e.g. df.agg("to_csv", 0, path)
        if isinstance(node.value, str) and (
            node.value in BLOCKED_PANDAS_WRITERS
            or node.value in PATH_OPTIONAL_WRITERS
        ):
            raise SecurityError(
                f"Blocked method name string: '{node.value}' "
                f"at line {node.lineno}. Calling file I/O methods by name "
                f"is not allowed."
            )

    def _check_module(self, module_name: str, node: ast.stmt) -> None:
        # Check the module and all parent packages
        parts = module_name.split(".")
        for i in range(len(parts)):
            prefix = ".".join(parts[: i + 1])
            if prefix in BLOCKED_MODULES:
                raise SecurityError(
                    f"Blocked import: '{module_name}' "
                    f"at line {node.lineno}"
                )
        # Submodules are checked like attributes, e.g. pandas.io, numpy.ctypeslib
        for part in parts[1:]:
            reason = _blocked_name_reason(part)
            if reason:
                raise SecurityError(
                    f"Blocked import: '{module_name}' "
                    f"at line {node.lineno}. {reason}"
                )
        # Also check against the whitelist
        root_module = parts[0]
        if root_module not in {m.split(".")[0] for m in ALLOWED_MODULES}:
            raise SecurityError(
                f"Import not in whitelist: '{module_name}' "
                f"at line {node.lineno}. "
                f"Allowed top-level modules: "
                f"{sorted({m.split('.')[0] for m in ALLOWED_MODULES})}"
            )


def _scan_ast(code_str: str) -> ast.Module:
    """Parse and security-scan the code. Returns the AST if safe."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise SecurityError(f"Syntax error in code: {e}") from e
    _ASTSecurityVisitor().visit(tree)
    return tree


# ── Sandboxed Import ────────────────────────────────────────────────

def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Replacement for __import__ that enforces the module whitelist."""
    root = name.split(".")[0]
    allowed_roots = {m.split(".")[0] for m in ALLOWED_MODULES}

    if name in BLOCKED_MODULES or root not in allowed_roots:
        raise SecurityError(
            f"Import blocked at runtime: '{name}'. "
            f"Allowed: {sorted(allowed_roots)}"
        )
    return __builtins__["__import__"](name, *args, **kwargs) if isinstance(
        __builtins__, dict
    ) else getattr(__builtins__, "__import__")(name, *args, **kwargs)


# ── Safe Builtins ───────────────────────────────────────────────────

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "oct", "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "True", "False", "None",
]


def _build_safe_builtins() -> dict[str, Any]:
    """Build a dict of safe builtins for the exec namespace."""
    import builtins as _b

    safe = {}
    for name in _SAFE_BUILTIN_NAMES:
        val = getattr(_b, name, None)
        if val is not None:
            safe[name] = val
    safe["__import__"] = _safe_import
    safe["__build_class__"] = _b.__build_class__
    return safe


# ── Matplotlib Capture ──────────────────────────────────────────────

def _capture_matplotlib_plot() -> str | None:
    """If matplotlib has any open figures, render to base64 PNG and close."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    figs = [plt.figure(n) for n in plt.get_fignums()]
    if not figs:
        return None

    buf = io.BytesIO()
    # Save the last figure
    figs[-1].savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close("all")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Main Executor ───────────────────────────────────────────────────

def run_code(
    code_str: str,
    local_vars: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute pandas/matplotlib code in a sandboxed namespace.

    Parameters
    ----------
    code_str : str
        The Python code to execute.
    local_vars : dict, optional
        Variables to inject into the namespace (e.g. ``{"df": my_dataframe}``).
    timeout : int
        Maximum execution time in seconds.

    Returns
    -------
    dict with keys:
        - stdout: str — captured print output
        - plot_base64: str | None — base64-encoded PNG if a plot was created
        - error: str | None — traceback string if execution failed
        - result: Any | None — the value of the last expression (if simple)
    """
    # 1. AST security scan (in the parent, before anything runs)
    _scan_ast(code_str)

    if os.environ.get(ISOLATION_ENV, "process").strip().lower() == "thread":
        return _run_in_thread(code_str, local_vars, timeout)
    return _run_in_subprocess(code_str, local_vars, timeout)


ISOLATION_ENV = "SANDBOX_ISOLATION"
MAX_MEMORY_MB = int(os.environ.get("SANDBOX_MAX_MEMORY_MB", "2048"))
STARTUP_GRACE_SECONDS = 15  # interpreter + pandas/matplotlib import in the child
RESULT_MARKER = "\n<<<SANDBOX_RESULT>>>\n"
_WORKER = Path(__file__).with_name("sandbox_worker.py")
# Only what a Python process needs to start; API keys and other secrets stay behind.
_ENV_ALLOWLIST = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")
_RESULT_KEYS = ("stdout", "plot_base64", "error", "result")


def _empty_result() -> dict[str, Any]:
    return {"stdout": "", "plot_base64": None, "error": None, "result": None}


def _timeout_result(timeout: int) -> dict[str, Any]:
    return {**_empty_result(), "error": (
        f"Execution timed out after {timeout} seconds. "
        "Consider optimizing your code or reducing data size."
    )}


def execute_snippet(code_str: str, local_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run already-scanned code in a restricted namespace in *this* process.

    Used by the child worker (process mode) and by thread mode.
    """
    namespace: dict[str, Any] = {"__builtins__": _build_safe_builtins()}
    if local_vars:
        namespace.update(local_vars)

    # Pre-import common modules into namespace for convenience
    try:
        import pandas as pd
        import numpy as np

        namespace["pd"] = pd
        namespace["np"] = np
    except ImportError:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        namespace["plt"] = plt
    except ImportError:
        pass

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = _empty_result()
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code_str, "<sandbox>", "exec"), namespace)
    except SecurityError as e:  # e.g. an import blocked at runtime
        result["error"] = f"SecurityError: {e}"
    except Exception:
        result["error"] = traceback.format_exc()

    result["stdout"] = stdout_capture.getvalue()
    plot_b64 = _capture_matplotlib_plot()
    if plot_b64:
        result["plot_base64"] = plot_b64

    # Try to capture 'result' variable if set by user code
    if "result" in namespace and namespace["result"] is not result:
        try:
            result["result"] = str(namespace["result"])
        except Exception:
            pass
    return result


def _run_in_thread(code_str: str, local_vars: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    """In-process mode: a daemon thread. A timed-out snippet keeps running in the
    background, and stdout capture is process-wide, so this is for development."""
    # Copies, so a snippet that edits df can't change the caller's data.
    local_vars = {k: copy.deepcopy(v) for k, v in (local_vars or {}).items()}
    box: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: box.append(execute_snippet(code_str, local_vars)), daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive() or not box:
        return _timeout_result(timeout)
    return box[0]


def _child_env(workdir: str) -> dict[str, str]:
    """The child's whole environment: allowlisted basics only, so no API keys."""
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.update({
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": workdir, "PYTHONIOENCODING": "utf-8",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    return env


def _run_in_subprocess(code_str: str, local_vars: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    """Process mode: one fresh, killable child per snippet."""
    payload = pickle.dumps({
        "code": code_str,
        "local_vars": local_vars or {},
        "timeout": timeout,
        "max_memory_mb": MAX_MEMORY_MB,
    })
    with tempfile.TemporaryDirectory(prefix="sandbox_") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(_WORKER)],
                input=payload, capture_output=True, cwd=workdir, env=_child_env(workdir),
                timeout=timeout + STARTUP_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired:  # subprocess.run has already killed the child
            return _timeout_result(timeout)

    stdout = proc.stdout.decode("utf-8", errors="replace")
    if RESULT_MARKER in stdout:
        try:
            reported = json.loads(stdout.rsplit(RESULT_MARKER, 1)[1])
        except json.JSONDecodeError:
            reported = None
        if isinstance(reported, dict):
            result = _empty_result()
            for key in _RESULT_KEYS:
                value = reported.get(key)
                if value is None or isinstance(value, str):
                    result[key] = value if key != "stdout" else (value or "")
            return result

    detail = proc.stderr.decode("utf-8", errors="replace").strip()[-1500:]
    if "MemoryError" in detail:
        reason = f"ran out of memory (limit {MAX_MEMORY_MB} MB)"
    else:
        reason = f"exited with code {proc.returncode}"
    return {**_empty_result(), "error": f"Sandbox process {reason}.\n{detail}".strip()}
