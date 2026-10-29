"""
Server configuration and per-client scopes.

Isolation model
---------------
A scope is everything one client can see: its own DatasetRegistry and its
own analyze_question runs (traces). Scopes never share a registry.

MCP 2026-07-28 requests are self-contained: no initialize handshake, no
Mcp-Session-Id, and the SDK builds a fresh Connection for every request —
even over stdio. The protocol therefore gives the server no client identity
to hang state on, so scopes are resolved explicitly:

- stdio: one server process serves exactly one client (the host launches a
  process per client), so the process owns one local scope. Two clients
  means two processes and two scopes.
- Streamable HTTP: many clients share one server process. A client calls
  open_session and passes the returned unguessable token as ``session`` on
  every call (``?session=`` on resource URIs); each token owns its own
  scope. Calls without a session are refused. This works identically for
  2025-era and 2026-era clients and does not depend on ``stateless_http``.

A ScopeStore belongs to one server instance (see create_server). Nothing in
this package keeps a module-level registry.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from core.datasets import DatasetRegistry

Transport = Literal["stdio", "streamable-http"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / "sample_data"
MB = 1024 * 1024

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


class SessionError(ValueError):
    """A request's session is missing, unknown, expired or over its limits."""


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class ServerConfig:
    """What the server exposes and how it scopes clients.

    ``allow_run_pandas`` and ``allow_analyze_question`` are the code-execution
    permission boundary: both run code in the sandbox (analyze_question runs
    LLM-written code and spends the server's Groq key). They default to on for
    local stdio and off for Streamable HTTP.
    """

    transport: Transport = "stdio"
    allow_run_pandas: bool = True
    allow_analyze_question: bool = True
    data_dir: Path | None = None  # approved directory for register_dataset(file_name=...)
    max_upload_bytes: int = 20 * MB
    max_sessions: int = 100  # Streamable HTTP sessions per server
    session_idle_seconds: float = 30 * 60
    max_datasets_per_scope: int = 20
    max_runs_per_scope: int = 20  # analyze_question traces kept per scope

    @classmethod
    def for_stdio(cls, **overrides: Any) -> ServerConfig:
        """Local defaults: code execution on, sample_data (or AGENTAI_DATA_DIR) approved."""
        data_dir = os.environ.get("AGENTAI_DATA_DIR")
        config = cls(
            transport="stdio",
            allow_run_pandas=_env_flag("AGENTAI_MCP_ALLOW_RUN_PANDAS") is not False,
            allow_analyze_question=_env_flag("AGENTAI_MCP_ALLOW_ANALYZE_QUESTION") is not False,
            data_dir=Path(data_dir) if data_dir else DEFAULT_LOCAL_DATA_DIR,
            max_upload_bytes=_env_mb("AGENTAI_MCP_MAX_UPLOAD_MB", 20),
        )
        return replace(config, **overrides)

    @classmethod
    def for_http(cls, **overrides: Any) -> ServerConfig:
        """Remote defaults: code execution off unless explicitly enabled, no server files."""
        data_dir = os.environ.get("AGENTAI_DATA_DIR")
        config = cls(
            transport="streamable-http",
            allow_run_pandas=_env_flag("AGENTAI_MCP_ALLOW_RUN_PANDAS") is True,
            allow_analyze_question=_env_flag("AGENTAI_MCP_ALLOW_ANALYZE_QUESTION") is True,
            data_dir=Path(data_dir) if data_dir else None,
            max_upload_bytes=_env_mb("AGENTAI_MCP_MAX_UPLOAD_MB", 5),
        )
        return replace(config, **overrides)


def _env_mb(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return (int(value) if value.isdigit() and int(value) > 0 else default) * MB


@dataclass(frozen=True)
class RunRecord:
    """One analyze_question run, kept for the trace:// resource."""

    run_id: str
    question: str
    dataset_id: str
    status: str
    created_at: str
    steps: list[dict[str, Any]]  # the graph's trace_log entries


@dataclass
class ClientScope:
    """One client's datasets and analyze_question runs."""

    registry: DatasetRegistry
    max_datasets: int
    max_runs: int
    runs: "OrderedDict[str, RunRecord]" = field(default_factory=OrderedDict)
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check_dataset_capacity(self) -> None:
        if len(self.registry.list_datasets()) >= self.max_datasets:
            raise SessionError(
                f"This session already has {self.max_datasets} datasets; "
                "open a new session to register more"
            )

    def add_run(self, record: RunRecord) -> None:
        with self.lock:
            self.runs[record.run_id] = record
            while len(self.runs) > self.max_runs:
                self.runs.popitem(last=False)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.lock:
            return self.runs.get(run_id)


class ScopeStore:
    """The scopes of one server instance (never shared between servers)."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._sessions: dict[str, ClientScope] = {}
        self._lock = threading.Lock()
        # stdio: the single client of this process
        self._local = self._new_scope() if config.transport == "stdio" else None

    def open_session(self) -> str:
        """Create a scope and return its token (a bearer secret)."""
        with self._lock:
            self._evict_idle()
            if len(self._sessions) >= self._config.max_sessions:
                raise SessionError("Too many open sessions on this server; try again later")
            token = secrets.token_urlsafe(32)
            self._sessions[token] = self._new_scope()
            return token

    def resolve(self, session: str | None) -> ClientScope:
        """The scope for a request: its session token's, or stdio's local one."""
        if session is None:
            if self._local is None:
                raise SessionError(
                    "This Streamable HTTP server needs a session: call open_session and pass "
                    "the returned token as `session` (MCP requests carry no client identity)"
                )
            self._local.last_used = time.monotonic()
            return self._local
        if not isinstance(session, str) or not _TOKEN_RE.match(session):
            raise SessionError("Invalid session token")
        with self._lock:
            self._evict_idle()
            scope = self._sessions.get(session)
            if scope is None:
                raise SessionError("Unknown or expired session; call open_session again")
            scope.last_used = time.monotonic()
            return scope

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _new_scope(self) -> ClientScope:
        return ClientScope(
            registry=DatasetRegistry(
                data_dir=self._config.data_dir, max_bytes=self._config.max_upload_bytes
            ),
            max_datasets=self._config.max_datasets_per_scope,
            max_runs=self._config.max_runs_per_scope,
        )

    def _evict_idle(self) -> None:
        """Drop idle sessions (lock held)."""
        cutoff = time.monotonic() - self._config.session_idle_seconds
        for token in [t for t, s in self._sessions.items() if s.last_used < cutoff]:
            del self._sessions[token]


@dataclass
class ServerState:
    """Per-server-instance state captured by the tool/resource handlers."""

    config: ServerConfig
    scopes: ScopeStore
    # The sandbox captures plots from matplotlib's global figure list, so
    # sandboxed runs (run_pandas, analyze_question) are serialized.
    sandbox_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, config: ServerConfig) -> ServerState:
        return cls(config=config, scopes=ScopeStore(config))
