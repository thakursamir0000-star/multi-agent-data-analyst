"""
Observability — decorator and utilities for tracing agent node execution.

Every node transition is logged with timestamp, latency, token count,
and a summary of inputs/outputs.  Logs are written to:
  1. state["trace_log"] — consumed by the Streamlit trace panel.
  2. logs/trace_{thread_id}.jsonl — persistent disk log.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Log directory (project root / logs)
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


@dataclass
class TraceEntry:
    """Single trace record for one node execution."""

    timestamp: str = ""
    node_name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text for log summaries."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _write_to_disk(entry: dict[str, Any], thread_id: str = "default") -> None:
    """Append a trace entry to a JSONL file."""
    log_file = _LOG_DIR / f"trace_{thread_id}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_node(node_name: str) -> Callable:
    """Decorator that wraps an agent node function to auto-log traces.

    Usage::

        @log_node("Planner")
        def planner_node(state: AgentState) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            start = time.perf_counter()
            timestamp = datetime.now(timezone.utc).isoformat()

            # Summarize input
            input_summary = _truncate(
                f"query={state.get('user_query', '')}, "
                f"task_idx={state.get('current_task_index', 0)}, "
                f"retry={state.get('retry_count', 0)}"
            )

            error_msg = None
            output_summary = ""
            result = {}

            try:
                result = func(state, *args, **kwargs)
                # Build output summary from returned state updates
                if isinstance(result, dict):
                    keys = list(result.keys())
                    output_summary = _truncate(
                        f"updated_keys={keys}, "
                        + ", ".join(
                            f"{k}={_truncate(str(v), 100)}"
                            for k, v in result.items()
                            if k != "trace_log"
                        )
                    )
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                entry = TraceEntry(
                    timestamp=timestamp,
                    node_name=node_name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    tokens_used=0,  # Updated by individual nodes if available
                    latency_ms=round(elapsed_ms, 2),
                    error=error_msg,
                )
                entry_dict = entry.to_dict()

                # Write to disk
                _write_to_disk(entry_dict)

                # Append to state trace_log via the return dict
                if isinstance(result, dict):
                    existing_trace = result.get("trace_log", [])
                    result["trace_log"] = existing_trace + [entry_dict]

            return result

        return wrapper

    return decorator
