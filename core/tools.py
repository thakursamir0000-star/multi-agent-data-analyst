"""
run_tool — the dataset-ID surface of the core layer.

One entry point for "run operation X on dataset Y with these arguments":

    run_tool(registry, dataset_id, "group_aggregate",
             {"group_by": "category", "agg": "sum", "value_column": "revenue"})

Arguments are validated against strict Pydantic models (core.schemas),
the dataset is looked up by ID in the registry (never by path), and the
result is a typed, JSON-serializable model. The LangGraph Coder calls this
when the LLM requests a tool; an MCP server can later expose each tool
with the same argument models.

query_sql runs against every dataset in the registry (so queries can join
them); the other tools run on ``dataset_id``. Tools always see the dataset
as registered — never changes made by run_pandas code in the sandbox.
"""

from __future__ import annotations

from typing import Any, Callable, Union, get_args

from pydantic import BaseModel, ValidationError

from core import analysis, charts, sql
from core.datasets import DatasetRegistry
from core.errors import CoreError
from core.schemas import (
    AnomalyResult,
    ChartResult,
    CorrelateArgs,
    CorrelationResult,
    DescribeArgs,
    DescribeResult,
    DetectAnomaliesArgs,
    GroupAggregateArgs,
    GroupAggregateResult,
    MakeChartArgs,
    QuerySqlArgs,
    SqlResult,
    ToolCall,
    ToolName,
)

ToolResult = Union[
    DescribeResult, GroupAggregateResult, CorrelationResult, AnomalyResult, ChartResult, SqlResult,
]

TOOL_NAMES: tuple[str, ...] = get_args(ToolName)

_ARG_MODELS: dict[str, type[BaseModel]] = {
    "describe": DescribeArgs,
    "group_aggregate": GroupAggregateArgs,
    "correlate": CorrelateArgs,
    "detect_anomalies": DetectAnomaliesArgs,
    "make_chart": MakeChartArgs,
    "query_sql": QuerySqlArgs,
}

# Tools that run on one dataset's DataFrame: (df, validated args) → result
_FRAME_TOOLS: dict[str, Callable[[Any, Any], ToolResult]] = {
    "describe": lambda df, a: analysis.describe(df, a.columns),
    "group_aggregate": lambda df, a: analysis.group_aggregate(
        df, a.group_by, a.agg, a.value_column, sort=a.sort, limit=a.limit
    ),
    "correlate": lambda df, a: analysis.correlate(df, a.columns, method=a.method),
    "detect_anomalies": lambda df, a: analysis.detect_anomalies(
        df, a.column, method=a.method, threshold=a.threshold, max_results=a.max_results
    ),
    "make_chart": lambda df, a: charts.make_chart(
        df, a.chart_type, a.x, a.y, agg=a.agg, title=a.title, limit=a.limit, bins=a.bins
    ),
}


def parse_tool_call(payload: Any) -> ToolCall:
    """Validate a ``{"tool": ..., "arguments": {...}}`` request.

    Raises CoreError with a message that says how to fix the request.
    """
    expected = '{"tool": "<name>", "arguments": {...}}'
    if not isinstance(payload, dict):
        raise CoreError(f"A tool call must be a JSON object: {expected}")
    if payload.get("tool") not in TOOL_NAMES:
        raise CoreError(
            f"Unknown tool: {payload.get('tool')!r}. Available tools: {', '.join(TOOL_NAMES)}"
        )
    try:
        return ToolCall.model_validate(payload)
    except ValidationError as e:
        raise CoreError(
            f"Invalid tool call: {_validation_message(e)}. Expected {expected}"
        ) from None


def run_tool(
    registry: DatasetRegistry,
    dataset_id: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    """Validate and run one core tool. Raises CoreError on any invalid request."""
    args_model = _ARG_MODELS.get(tool) if isinstance(tool, str) else None
    if args_model is None:
        raise CoreError(f"Unknown tool: {tool!r}. Available tools: {', '.join(TOOL_NAMES)}")

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise CoreError(f"Arguments for {tool} must be a JSON object")
    try:
        args = args_model.model_validate(arguments)
    except ValidationError as e:
        raise CoreError(f"Invalid arguments for {tool}: {_validation_message(e)}") from None

    df = registry._frame(dataset_id)  # validates the ID; tools never mutate it
    if tool == "query_sql":
        return sql.query_sql(registry, args.sql, max_rows=args.max_rows)
    return _FRAME_TOOLS[tool](df, args)


def result_json(result: ToolResult) -> str:
    """Compact JSON of a result for an LLM prompt; chart images are left out."""
    exclude = {"image_base64"} if isinstance(result, ChartResult) else None
    return result.model_dump_json(exclude=exclude)


def _validation_message(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:5]:
        where = ".".join(str(p) for p in item["loc"]) or "arguments"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)
