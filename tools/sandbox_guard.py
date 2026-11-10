"""
Run-time guard for the sandbox child process (tools/sandbox_worker.py).

The AST scan in tools/sandbox.py rejects the known ways to reach the file
system, but Python is dynamic: pandas, for one, calls methods by a name that
can be built while the code runs (``df.agg(name, 0, path)``). This guard
enforces the rules where the access happens, with a PEP 578 audit hook that
the worker installs just before the snippet runs. Audit hooks can't be
removed, so the snippet can't turn it off. Once installed, the process may

- write, create, rename and delete files only inside its own temporary
  directory;
- read files only there, in the Python installation and in font / time-zone
  directories, so it can't read the project's .env or other user files;
- not open network connections or start other programs.

Blocked operations raise PermissionError. This is a strong barrier for a
single-user data app; a hostile multi-tenant deployment should still run the
worker in a container (see "Security" in the README).
"""

from __future__ import annotations

import os
import site
import sys

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

# Starting programs and talking to the network: no snippet needs either.
_BLOCKED_PREFIXES = ("subprocess.", "socket.", "os.exec", "os.spawn", "os.posix_spawn", "ctypes.")
_BLOCKED_EVENTS = frozenset({
    "os.system", "os.startfile", "os.fork", "os.forkpty", "os.kill", "os.killpg",
    "pty.spawn", "_winapi.CreateProcess", "webbrowser.open",
})
# Events whose path arguments (by position) must stay inside the sandbox directory.
_MUTATING_EVENTS = {
    "os.remove": (0,), "os.rename": (0, 1), "os.mkdir": (0,), "os.rmdir": (0,),
    "os.chmod": (0,), "os.chown": (0,), "os.link": (0, 1), "os.symlink": (0, 1),
    "os.truncate": (0,), "os.utime": (0,), "shutil.copyfile": (0, 1), "shutil.copytree": (0, 1),
    "shutil.move": (0, 1), "shutil.rmtree": (0,), "shutil.chown": (0,),
}
# Read-only system files that the scientific stack may open while running.
_READABLE_FILES = ("/dev/null", "/dev/urandom", "/etc/localtime", "/etc/timezone")


def _normalize(path) -> str | None:
    """Absolute, case-normalized path text, or None for file descriptors."""
    if path is None or isinstance(path, int):
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    return os.path.normcase(os.path.abspath(text))


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _read_roots(workdir: str) -> tuple[str, ...]:
    """Where a snippet may read: its own folder, Python itself, fonts and time zones."""
    candidates = [workdir, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    try:
        candidates += site.getsitepackages() + [site.getusersitepackages()]
    except AttributeError:  # some embedded interpreters lack these
        pass
    candidates += [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
        "/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts", "/System/Library/Fonts",
        "/usr/share/zoneinfo",
    ]
    return tuple(dict.fromkeys(_normalize(c) for c in candidates if c))


def install_guard(workdir: str) -> None:
    """Install the audit hook. Call once, in the child, right before untrusted code runs."""
    write_roots = (_normalize(workdir),)
    read_roots = _read_roots(workdir)
    readable_files = tuple(_normalize(f) for f in _READABLE_FILES)

    def deny(what: str) -> None:
        raise PermissionError(f"Sandbox: {what} is not allowed (only the sandbox's own folder is usable).")

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = _normalize(args[0])
            if path is None:
                return
            mode, flags = args[1], args[2] if len(args) > 2 else 0
            writing = any(c in mode for c in "wax+") if isinstance(mode, str) else bool((flags or 0) & _WRITE_FLAGS)
            if writing:
                if not _inside(path, write_roots):
                    deny(f"writing '{path}' outside the sandbox")
            elif not (_inside(path, read_roots) or path in readable_files):
                deny(f"reading '{path}' outside the sandbox")
        elif event in _MUTATING_EVENTS:
            for position in _MUTATING_EVENTS[event]:
                path = _normalize(args[position]) if position < len(args) else None
                if path is not None and not _inside(path, write_roots):
                    deny(f"changing '{path}' outside the sandbox ({event})")
        elif event in _BLOCKED_EVENTS or event.startswith(_BLOCKED_PREFIXES):
            deny(f"'{event}' (network access or starting programs)")

    sys.addaudithook(hook)
