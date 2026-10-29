"""
MCP tools — thin adapters over core/, the existing sandbox and the graph.

Every tool resolves the caller's scope (see context.py), calls core, and
returns the core result model unchanged. Invalid requests (CoreError,
SessionError, sandbox SecurityError) become ToolError so the client sees why;
any other exception stays a generic error on the wire.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Annotated, Iterator

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from core import sql as core_sql
from core.errors import CoreError
from core.schemas import (
    AggFunc,
    AnomalyMethod,
    AnomalyResult,
    ChartMetadata,
    ChartResult,
    ChartType,
    CorrelationMethod,
    CorrelationResult,
    DatasetInfo,
    DatasetProfile,
    DescribeResult,
    GroupAggregateResult,
    SortOrder,
    SqlResult,
)
from core.tools import run_tool
from mcp_server.context import RunRecord, ServerState, SessionError
from mcp_server.schemas import (
    AnalysisStep,
    AnalyzeQuestionResult,
    DatasetListing,
    DatasetSchema,
    RunPandasResult,
    SessionInfo,
)

Session = Annotated[
    str | None,
    Field(description="Session token from open_session. Required on Streamable HTTP; omit on stdio."),
]
DatasetId = Annotated[
    str, Field(description="A dataset ID from register_dataset or list_datasets, e.g. ds_0123456789abcdef.")
]
Columns = Annotated[
    list[str] | None, Field(description="Column names; omit for all numeric columns.")
]

MAX_CODE_CHARS = 20_000
MAX_STDOUT_CHARS = 20_000
MAX_QUESTION_CHARS = 2_000

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Report invalid requests to the client; other exceptions stay generic."""
    try:
        yield
    except (CoreError, SessionError) as e:
        raise ToolError(str(e)) from None


def register_tools(server: MCPServer, state: ServerState) -> None:
    """Register every tool the server's configuration allows."""
    config = state.config
    scopes = state.scopes
    # Over HTTP the caller doesn't run the server, so SQL may not describe the server.
    sql_policy = "remote" if config.transport == "streamable-http" else "local"

    # ── Session & datasets ──────────────────────────────────────────

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def open_session() -> SessionInfo:
        """Start an isolated session and return its token.

        On Streamable HTTP, call this first and pass the token as `session` to
        every other tool: datasets are visible only inside their session. On
        stdio the process already has one local session, so this is optional.
        """
        with _tool_errors():
            token = scopes.open_session()
        return SessionInfo(
            session=token,
            expires_after_idle_seconds=int(config.session_idle_seconds),
            note="Keep this token private; anyone holding it can read this session's datasets.",
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def register_dataset(
        file_name: Annotated[str, Field(description=(
            "Bare file name (.csv, .xlsx or .xls). Without content: a file in the server's "
            "approved data directory (see list_datasets). With csv_text or content_base64: "
            "only a label that picks the parser - it is never opened as a path."
        ))],
        csv_text: Annotated[str | None, Field(description="CSV content to upload.")] = None,
        content_base64: Annotated[
            str | None, Field(description="Base64 file content to upload (e.g. an Excel file).")
        ] = None,
        session: Session = None,
    ) -> DatasetInfo:
        """Register a dataset and return its ID, table name and columns.

        Registering the same name and content again returns the same ID.
        """
        with _tool_errors():
            scope = scopes.resolve(session)
            if csv_text is not None and content_base64 is not None:
                raise ToolError("Pass csv_text or content_base64, not both")
            scope.check_dataset_capacity()
            if csv_text is not None:
                return scope.registry.register_upload(csv_text.encode("utf-8"), file_name)
            if content_base64 is not None:
                if len(content_base64) > config.max_upload_bytes * 4 // 3 + 4:
                    raise ToolError(f"Upload is larger than {config.max_upload_bytes} bytes")
                try:
                    data = base64.b64decode(content_base64, validate=True)
                except (binascii.Error, ValueError):
                    raise ToolError("content_base64 is not valid base64") from None
                return scope.registry.register_upload(data, file_name)
            if config.data_dir is None:
                raise ToolError(
                    "This server has no approved data directory; upload the data with "
                    "csv_text or content_base64"
                )
            return scope.registry.register_file(file_name)

    @server.tool(annotations=_READ_ONLY)
    def list_datasets(session: Session = None) -> DatasetListing:
        """List this session's datasets and the approved files that can be registered."""
        with _tool_errors():
            scope = scopes.resolve(session)
            return DatasetListing(
                datasets=scope.registry.list_datasets(),
                available_files=scope.registry.available_files(),
            )

    @server.tool(annotations=_READ_ONLY)
    def get_dataset_schema(dataset_id: DatasetId, session: Session = None) -> DatasetSchema:
        """Columns, dtypes, kinds and nullability of a dataset, plus its SQL table name."""
        with _tool_errors():
            return DatasetSchema.from_info(scopes.resolve(session).registry.get(dataset_id))

    @server.tool(annotations=_READ_ONLY)
    def get_dataset_profile(dataset_id: DatasetId, session: Session = None) -> DatasetProfile:
        """Per-column profile: missing values, cardinality, sample values, numeric stats."""
        with _tool_errors():
            return scopes.resolve(session).registry.profile(dataset_id)

    # ── Deterministic analysis (core.tools.run_tool) ────────────────

    def _run(session: str | None, dataset_id: str, tool: str, arguments: dict):
        with _tool_errors():
            return run_tool(scopes.resolve(session).registry, dataset_id, tool, arguments)

    @server.tool(annotations=_READ_ONLY)
    def describe(dataset_id: DatasetId, columns: Columns = None, session: Session = None) -> DescribeResult:
        """Exact descriptive statistics (count, nulls, mean, std, min, quartiles, max)."""
        return _run(session, dataset_id, "describe", {"columns": columns})

    @server.tool(annotations=_READ_ONLY)
    def group_aggregate(
        dataset_id: DatasetId,
        group_by: Annotated[str | list[str], Field(description="1-3 columns to group by.")],
        aggregation: AggFunc,
        value_column: Annotated[
            str | None, Field(description="Numeric column to aggregate; required unless aggregation is count.")
        ] = None,
        sort: SortOrder = "desc",
        limit: Annotated[int, Field(description="Maximum groups returned (1-1000).")] = 50,
        session: Session = None,
    ) -> GroupAggregateResult:
        """Aggregate a column per group, sorted by the aggregated value."""
        return _run(session, dataset_id, "group_aggregate", {
            "group_by": group_by, "agg": aggregation, "value_column": value_column,
            "sort": sort, "limit": limit,
        })

    @server.tool(annotations=_READ_ONLY)
    def correlate(
        dataset_id: DatasetId,
        columns: Columns = None,
        method: CorrelationMethod = "pearson",
        session: Session = None,
    ) -> CorrelationResult:
        """Pairwise correlations between numeric columns, strongest first."""
        return _run(session, dataset_id, "correlate", {"columns": columns, "method": method})

    @server.tool(annotations=_READ_ONLY)
    def detect_anomalies(
        dataset_id: DatasetId,
        column: str,
        method: AnomalyMethod = "iqr",
        threshold: Annotated[
            float | None, Field(description="IQR multiplier (default 1.5) or z-score cutoff (default 3).")
        ] = None,
        max_results: int = 50,
        session: Session = None,
    ) -> AnomalyResult:
        """Flag outliers in a numeric column (IQR by default, or z-score)."""
        return _run(session, dataset_id, "detect_anomalies", {
            "column": column, "method": method, "threshold": threshold, "max_results": max_results,
        })

    @server.tool(annotations=_READ_ONLY)
    def query_sql(
        sql: Annotated[str, Field(
            description="One read-only DuckDB SELECT; tables are the datasets' table_name values.",
        )],
        dataset_ids: Annotated[list[str] | None, Field(
            description="Datasets the query may read; omit for all of this session's datasets.",
        )] = None,
        max_rows: Annotated[int, Field(description="Maximum rows returned (1-10000).")] = 100,
        session: Session = None,
    ) -> SqlResult:
        """Run one read-only SQL SELECT over this session's datasets (DuckDB).

        Writes, DDL, ATTACH/COPY, PRAGMA/SET and file or network access are refused.
        """
        with _tool_errors():
            registry = scopes.resolve(session).registry
            return core_sql.query_sql(registry, sql, max_rows=max_rows, dataset_ids=dataset_ids,
                                      policy=sql_policy)

    @server.tool(annotations=_READ_ONLY)
    def make_chart(
        dataset_id: DatasetId,
        chart_type: ChartType,
        x: str,
        y: str | None = None,
        aggregation: Annotated[
            AggFunc | None, Field(description="bar/line only: sum by default with y, count without.")
        ] = None,
        title: str | None = None,
        limit: Annotated[int, Field(description="bar only: top groups plotted (1-50).")] = 20,
        bins: Annotated[int, Field(description="histogram only: number of bins (1-100).")] = 20,
        session: Session = None,
    ) -> Annotated[CallToolResult, ChartMetadata]:
        """Draw a bar, line, scatter or histogram chart; returns the PNG and the plotted values."""
        chart: ChartResult = _run(session, dataset_id, "make_chart", {
            "chart_type": chart_type, "x": x, "y": y, "agg": aggregation,
            "title": title, "limit": limit, "bins": bins,
        })
        metadata = chart.metadata()
        return CallToolResult(
            content=[
                ImageContent(type="image", data=chart.image_base64, mime_type=chart.mime_type),
                TextContent(type="text", text=metadata.model_dump_json()),
            ],
            structured_content=metadata.model_dump(mode="json"),
        )

    # ── Code execution (permission-gated) ───────────────────────────

    if config.allow_run_pandas:
        _register_run_pandas(server, state)
    if config.allow_analyze_question:
        _register_analyze_question(server, state)


def _register_run_pandas(server: MCPServer, state: ServerState) -> None:
    from tools.sandbox import SecurityError, run_code

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def run_pandas(
        dataset_id: DatasetId,
        code: Annotated[str, Field(description=(
            "Python using the pre-loaded DataFrame `df` (a copy), `pd`, `np` and `plt`; "
            "print() results. No file, network or system access."
        ))],
        session: Session = None,
    ) -> Annotated[CallToolResult, RunPandasResult]:
        """Fallback: run pandas code in the restricted sandbox when no deterministic tool fits."""
        if len(code) > MAX_CODE_CHARS:
            raise ToolError(f"code is longer than {MAX_CODE_CHARS} characters")
        with _tool_errors():
            df = state.scopes.resolve(session).registry.get_dataframe(dataset_id)
        try:
            with state.sandbox_lock:
                result = run_code(code, local_vars={"df": df})
        except SecurityError as e:
            raise ToolError(f"SecurityError: {e}") from None

        stdout = result.get("stdout") or ""
        output = RunPandasResult(
            dataset_id=dataset_id,
            stdout=stdout[:MAX_STDOUT_CHARS],
            stdout_truncated=len(stdout) > MAX_STDOUT_CHARS,
            error=result.get("error"),
            has_plot=bool(result.get("plot_base64")),
        )
        content = [TextContent(type="text", text=output.error or output.stdout or "(no output)")]
        if output.has_plot:
            content.append(ImageContent(type="image", data=result["plot_base64"], mime_type="image/png"))
        return CallToolResult(
            content=content,
            structured_content=output.model_dump(mode="json"),
            is_error=output.error is not None,
        )


def _register_analyze_question(server: MCPServer, state: ServerState) -> None:
    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def analyze_question(
        question: Annotated[str, Field(description="A natural-language question about the dataset.")],
        dataset_id: DatasetId,
        session: Session = None,
    ) -> Annotated[CallToolResult, AnalyzeQuestionResult]:
        """Answer a question with the full multi-agent pipeline (plan, compute, write, verify).

        Uses the server's Groq model. If the pipeline needs clarification it
        returns status "needs_clarification" with the question to answer; ask
        again with a clarified question (resuming is not supported yet).
        """
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ToolError(f"question must be 1-{MAX_QUESTION_CHARS} characters")
        from tools.config import get_env

        if not get_env("GROQ_API_KEY"):
            raise ToolError("analyze_question needs GROQ_API_KEY in the server's environment")
        with _tool_errors():
            scope = state.scopes.resolve(session)
            profile_text = scope.registry.get_profile_text(dataset_id)

        # The existing LangGraph pipeline, unchanged (direct core/sandbox calls)
        from agents.graph import build_graph, create_initial_state

        run_id = uuid.uuid4().hex
        try:
            with state.sandbox_lock:
                graph = build_graph(registry=scope.registry, dataset_id=dataset_id)
                final = graph.invoke(
                    create_initial_state(question, profile_text),
                    {"configurable": {"thread_id": run_id}},
                )
        except Exception as e:  # e.g. an LLM API error
            raise ToolError(f"Analysis failed: {type(e).__name__}: {e}") from None

        interrupts = final.get("__interrupt__") or []
        status = "needs_clarification" if interrupts else "completed"
        clarification = str(getattr(interrupts[0], "value", interrupts[0])) if interrupts else None
        trace_log = final.get("trace_log", [])
        scope.add_run(RunRecord(
            run_id=run_id,
            question=question,
            dataset_id=dataset_id,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            steps=trace_log,
        ))

        outputs = final.get("code_outputs", [])
        answer = "" if interrupts else (final.get("verified_insight") or final.get("draft_insight", ""))
        result = AnalyzeQuestionResult(
            run_id=run_id,
            status=status,
            dataset_id=dataset_id,
            question=question,
            answer=answer,
            verified=bool(final.get("verified_insight")),
            clarification_question=clarification,
            sub_tasks=final.get("sub_tasks", []),
            steps=[
                AnalysisStep(
                    task=o.get("task", ""),
                    task_index=o.get("task_index", i),
                    tool=o.get("tool"),
                    code=o.get("code", ""),
                    error=o.get("error"),
                    has_chart=bool(o.get("plot_base64")),
                )
                for i, o in enumerate(outputs)
            ],
            retry_count=final.get("retry_count", 0),
            total_tokens=sum(e.get("tokens_used", 0) for e in trace_log),
            trace_uri=f"trace://{run_id}" + (f"?session={session}" if session else ""),
        )
        content: list = [TextContent(type="text", text=answer or clarification or "")]
        content += [
            ImageContent(type="image", data=o["plot_base64"], mime_type="image/png")
            for o in outputs if o.get("plot_base64")
        ]
        return CallToolResult(content=content, structured_content=result.model_dump(mode="json"))
