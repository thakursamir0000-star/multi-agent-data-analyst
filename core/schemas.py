"""
Typed inputs and JSON-serializable results for the core layer.

Every result model can be dumped with ``model_dump()`` /
``model_dump_json()``, so it can later be returned directly by an MCP tool.
Values inside results are always JSON-native (see ``to_json_value``).
"""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

# ── Input types ─────────────────────────────────────────────────────

AggFunc = Literal["count", "sum", "mean", "median", "min", "max"]
SortOrder = Literal["desc", "asc", "none"]
CorrelationMethod = Literal["pearson", "spearman"]
AnomalyMethod = Literal["iqr", "zscore"]
ChartType = Literal["bar", "line", "scatter", "histogram"]

ColumnKind = Literal["numeric", "datetime", "boolean", "text", "other"]
DatasetSource = Literal["file", "upload", "dataframe"]
ToolName = Literal[
    "describe", "group_aggregate", "correlate", "detect_anomalies", "make_chart", "query_sql",
]


# ── Tool arguments (see core.tools) ─────────────────────────────────

class _Args(BaseModel):
    """Strict: no coercion ("5" is not 5, true is not 1) and no unknown keys."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class DescribeArgs(_Args):
    columns: str | list[str] | None = None  # default: all numeric columns


class GroupAggregateArgs(_Args):
    group_by: str | list[str]
    agg: AggFunc
    value_column: str | None = None  # required unless agg is "count"
    sort: SortOrder = "desc"
    limit: int = 50


class CorrelateArgs(_Args):
    columns: list[str] | None = None  # default: all numeric columns
    method: CorrelationMethod = "pearson"


class DetectAnomaliesArgs(_Args):
    column: str
    method: AnomalyMethod = "iqr"
    threshold: float | None = None  # IQR k (1.5) or z-score cutoff (3.0)
    max_results: int = 50


class MakeChartArgs(_Args):
    chart_type: ChartType
    x: str
    y: str | None = None
    agg: AggFunc | None = None
    title: str | None = None
    limit: int = 20  # bar charts
    bins: int = 20  # histograms


class QuerySqlArgs(_Args):
    sql: str
    max_rows: int = 100  # smaller than query_sql's own default: results go to an LLM


class ToolCall(_Args):
    """A request to run one core tool, e.g. parsed from an LLM reply."""

    tool: ToolName
    arguments: dict[str, Any] = {}


# ── Results ─────────────────────────────────────────────────────────

class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnInfo(_Result):
    name: str
    dtype: str
    kind: ColumnKind
    null_count: int
    unique_count: int


class DatasetInfo(_Result):
    """Metadata for a registered dataset."""

    dataset_id: str
    name: str
    table_name: str  # name to use in query_sql
    source: DatasetSource
    rows: int
    columns: list[ColumnInfo]


class ColumnStats(_Result):
    column: str
    count: int  # non-null values
    null_count: int
    mean: float | None
    std: float | None  # sample standard deviation (ddof=1)
    min: float | None
    q25: float | None
    median: float | None
    q75: float | None
    max: float | None


class DescribeResult(_Result):
    rows: int
    columns: list[ColumnStats]


class GroupAggregateResult(_Result):
    group_by: list[str]
    agg: AggFunc
    value_column: str | None
    value_name: str  # key holding the aggregated value in each row
    total_groups: int
    truncated: bool
    rows: list[dict[str, Any]]


class CorrelationPair(_Result):
    x: str
    y: str
    r: float | None  # None when undefined (e.g. a constant column)
    n: int  # rows where both values are present


class CorrelationResult(_Result):
    method: CorrelationMethod
    columns: list[str]
    pairs: list[CorrelationPair]  # sorted by |r| descending


class Anomaly(_Result):
    index: Any  # row label in the dataset
    value: float
    direction: Literal["high", "low"]
    z_score: float | None = None  # zscore method only


class AnomalyResult(_Result):
    column: str
    method: AnomalyMethod
    threshold: float  # IQR multiplier k, or z-score cutoff
    parameters: dict[str, float | None]  # q1/q3/iqr or mean/std
    lower_bound: float | None
    upper_bound: float | None
    checked_count: int  # non-null values examined
    anomaly_count: int
    truncated: bool
    anomalies: list[Anomaly]


class SqlResult(_Result):
    columns: list[str]
    rows: list[list[Any]]  # row-major; SQL may repeat column names
    row_count: int
    truncated: bool


class ChartMetadata(_Result):
    """Everything about a chart except the image itself."""

    chart_type: ChartType
    title: str
    x: str
    y: str | None
    agg: AggFunc | None
    width_px: int
    height_px: int
    data_points: int
    data: list[dict[str, Any]]  # the plotted values
    data_truncated: bool


class ChartResult(ChartMetadata):
    mime_type: Literal["image/png"] = "image/png"
    image_base64: str

    def metadata(self) -> ChartMetadata:
        """The chart without its image (e.g. to send the image separately)."""
        return ChartMetadata(**self.model_dump(exclude={"mime_type", "image_base64"}))


class NumericSummary(_Result):
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None


class ColumnProfile(_Result):
    name: str
    dtype: str
    kind: ColumnKind
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]  # up to 5 distinct non-null values, JSON-native
    stats: NumericSummary | None = None  # numeric columns only


class DatasetProfile(_Result):
    """The dataset profile the agents see, as a typed model."""

    dataset_id: str
    name: str
    rows: int
    columns: list[ColumnProfile]
    summary_text: str  # the compact text form used in LLM prompts


# ── Helpers ─────────────────────────────────────────────────────────

def to_json_value(value: Any) -> Any:
    """Convert numpy / pandas / DuckDB scalars to JSON-native values.

    NaN, infinities and missing values become None.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, decimal.Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (dt.timedelta, np.timedelta64)):
        return str(pd.Timedelta(value))
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    return str(value)


def column_kind(series: pd.Series) -> ColumnKind:
    """Classify a column. Booleans are not treated as numeric."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        return "text"
    return "other"
