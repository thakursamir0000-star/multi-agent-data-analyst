"""
Sandboxed Code Executor — runs pandas/matplotlib code in a restricted namespace.

Security model:
1. AST pre-scan blocks dangerous function calls (open, exec, eval, compile, __import__).
2. Custom import function whitelist blocks os, sys, subprocess, etc.
3. Execution timeout prevents infinite loops.
4. No access to real builtins beyond a curated safe set.
"""

from __future__ import annotations

import ast
import base64
import io
import sys
import traceback
import threading
from contextlib import redirect_stdout, redirect_stderr
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

DEFAULT_TIMEOUT_SECONDS = 30


# ── AST Security Scanner ────────────────────────────────────────────

class _ASTSecurityVisitor(ast.NodeVisitor):
    """Walks the AST and raises SecurityError for forbidden patterns."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_module(node.module, node)
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
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_ATTRIBUTE_NAMES:
            raise SecurityError(
                f"Blocked attribute access: '.{node.attr}' "
                f"at line {node.lineno}"
            )
        self.generic_visit(node)

    def _check_module(self, module_name: str, node: ast.AST) -> None:
        # Check the module and all parent packages
        parts = module_name.split(".")
        for i in range(len(parts)):
            prefix = ".".join(parts[: i + 1])
            if prefix in BLOCKED_MODULES:
                raise SecurityError(
                    f"Blocked import: '{module_name}' "
                    f"at line {node.lineno}"
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
    # 1. AST security scan
    _scan_ast(code_str)

    # 2. Build restricted namespace
    namespace: dict[str, Any] = {
        "__builtins__": _build_safe_builtins(),
    }
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

    # 3. Execute with timeout and output capture
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result: dict[str, Any] = {
        "stdout": "",
        "plot_base64": None,
        "error": None,
        "result": None,
    }
    exec_error: list[str | None] = [None]

    def _execute():
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(code_str, "<sandbox>", "exec"), namespace)
        except SecurityError:
            raise
        except Exception:
            exec_error[0] = traceback.format_exc()

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result["error"] = (
            f"Execution timed out after {timeout} seconds. "
            "Consider optimizing your code or reducing data size."
        )
        return result

    # 4. Collect results
    result["stdout"] = stdout_capture.getvalue()

    if exec_error[0]:
        result["error"] = exec_error[0]

    # Capture any matplotlib figures
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
