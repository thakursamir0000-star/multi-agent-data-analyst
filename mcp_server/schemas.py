"""
MCP-specific result models.

Numerical results are the core.schemas models, returned unchanged. The models
here only adapt what core does not model: session handles, the schema view
of a dataset (with nullability), sandbox output, and analyze_question runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from core.schemas import ColumnKind, DatasetInfo, DatasetSource
from tools.observability import TraceEntry


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SessionInfo(_Model):
    session: str  # bearer token: pass it as `session` on every call
    expires_after_idle_seconds: int
    note: str


class SchemaColumn(_Model):
    name: str
    dtype: str
    kind: ColumnKind
    nullable: bool  # true when the column contains missing values
    null_count: int
    unique_count: int


class DatasetSchema(_Model):
    dataset_id: str
    name: str
    table_name: str  # the table name to use in query_sql
    source: DatasetSource
    rows: int
    columns: list[SchemaColumn]

    @classmethod
    def from_info(cls, info: DatasetInfo) -> DatasetSchema:
        return cls(
            dataset_id=info.dataset_id,
            name=info.name,
            table_name=info.table_name,
            source=info.source,
            rows=info.rows,
            columns=[
                SchemaColumn(
                    name=c.name,
                    dtype=c.dtype,
                    kind=c.kind,
                    nullable=c.null_count > 0,
                    null_count=c.null_count,
                    unique_count=c.unique_count,
                )
                for c in info.columns
            ],
        )


class DatasetListing(_Model):
    datasets: list[DatasetInfo]
    available_files: list[str]  # files register_dataset(file_name=...) can load


class RunPandasResult(_Model):
    dataset_id: str
    stdout: str
    stdout_truncated: bool
    error: str | None
    has_plot: bool


class AnalysisStep(_Model):
    task: str
    task_index: int
    tool: str | None  # core tool the Coder chose, or None for sandbox code
    code: str  # the tool call JSON or the Python that ran
    error: str | None
    has_chart: bool


class AnalyzeQuestionResult(_Model):
    run_id: str
    status: Literal["completed", "needs_clarification"]
    dataset_id: str
    question: str
    answer: str  # Critic-verified insight, else the Analyst's draft
    verified: bool
    clarification_question: str | None
    sub_tasks: list[str]
    steps: list[AnalysisStep]
    retry_count: int
    total_tokens: int
    trace_uri: str


class RunTrace(_Model):
    run_id: str
    question: str
    dataset_id: str
    status: str
    created_at: str
    steps: list[TraceEntry]  # the graph's per-node records (tools.observability)

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]], **fields: Any) -> RunTrace:
        known = set(TraceEntry.__dataclass_fields__)
        steps = [TraceEntry(**{k: v for k, v in e.items() if k in known}) for e in entries]
        return cls(steps=steps, **fields)
