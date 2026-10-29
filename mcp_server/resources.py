"""
MCP resources: dataset schemas and analyze_question traces.

Both are read-only views of the caller's scope (``?session=`` on Streamable
HTTP). Nothing here touches the filesystem: traces are the in-memory records
analyze_question keeps per scope — never the JSONL log files — so a trace URI
cannot become a file reader. The SDK also rejects path traversal, absolute
paths and NUL bytes in template parameters before these handlers run.
"""

from __future__ import annotations

import re

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from core.errors import CoreError
from mcp_server.context import ServerState, SessionError
from mcp_server.schemas import DatasetSchema, RunTrace

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def register_resources(server: MCPServer, state: ServerState) -> None:
    scopes = state.scopes

    @server.resource(
        "dataset://{dataset_id}/schema{?session}",
        name="dataset_schema",
        title="Dataset schema",
        description="ID, name, SQL table name, row count and columns (dtype, kind, nullability).",
        mime_type="application/json",
    )
    def dataset_schema(dataset_id: str, session: str | None = None) -> str:
        try:
            info = scopes.resolve(session).registry.get(dataset_id)
        except (CoreError, SessionError) as e:
            raise ResourceNotFoundError(str(e)) from None
        return DatasetSchema.from_info(info).model_dump_json(indent=2)

    @server.resource(
        "trace://{run_id}{?session}",
        name="run_trace",
        title="analyze_question trace",
        description="Per-node trace (latency, tokens, errors) of an analyze_question run.",
        mime_type="application/json",
    )
    def run_trace(run_id: str, session: str | None = None) -> str:
        if not _RUN_ID_RE.match(run_id):
            raise ResourceNotFoundError(
                f"Invalid run ID {run_id[:80]!r}: use the run_id returned by analyze_question"
            )
        try:
            record = scopes.resolve(session).get_run(run_id)
        except SessionError as e:
            raise ResourceNotFoundError(str(e)) from None
        if record is None:
            raise ResourceNotFoundError(f"No trace for run {run_id} in this session")
        return RunTrace.from_entries(
            record.steps,
            run_id=record.run_id,
            question=record.question,
            dataset_id=record.dataset_id,
            status=record.status,
            created_at=record.created_at,
        ).model_dump_json(indent=2)
