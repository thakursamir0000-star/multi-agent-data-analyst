"""
core — deterministic data-analysis layer.

The LLM chooses/requests an operation; a deterministic function here
executes it and returns a typed, JSON-serializable result. Shared by the
LangGraph application and (later) the MCP server. No LLM-specific logic
lives here.

Modules (import them directly; nothing is imported eagerly here so that
loading ``core.datasets`` does not pull in DuckDB or matplotlib):

- ``core.datasets`` — DatasetRegistry, the controlled entry point for data
- ``core.analysis`` — describe, group_aggregate, correlate, detect_anomalies
- ``core.sql``      — query_sql (read-only DuckDB over registered datasets)
- ``core.charts``   — make_chart (base64 PNG plus the plotted values)
- ``core.tools``    — run_tool: every operation by name, on a dataset ID
- ``core.schemas``  — Pydantic argument and result models
- ``core.errors``   — CoreError hierarchy
"""
