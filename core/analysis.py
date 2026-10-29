"""
Deterministic analysis operations on a DataFrame.

Each function validates its inputs (columns, types, options), computes an
exact result with pandas, and returns a typed, JSON-serializable model.
Invalid requests raise ColumnError / CoreError with a message an LLM can
act on.
"""

from __future__ import annotations

from typing import Any, Sequence, Union, get_args

import pandas as pd

from core.errors import ColumnError, CoreError
from core.schemas import (
    AggFunc,
    Anomaly,
    AnomalyMethod,
    AnomalyResult,
    ColumnStats,
    CorrelationMethod,
    CorrelationPair,
    CorrelationResult,
    DescribeResult,
    GroupAggregateResult,
    SortOrder,
    column_kind,
    to_json_value,
)

AGG_FUNCS: tuple[str, ...] = get_args(AggFunc)
SORT_ORDERS: tuple[str, ...] = get_args(SortOrder)
CORRELATION_METHODS: tuple[str, ...] = get_args(CorrelationMethod)
ANOMALY_METHODS: tuple[str, ...] = get_args(AnomalyMethod)

MAX_GROUP_KEYS = 3
MAX_ROWS = 1000
DEFAULT_IQR_K = 1.5
DEFAULT_Z_THRESHOLD = 3.0

Columns = Union[str, Sequence[str], None]


# ── Validation helpers ──────────────────────────────────────────────

def _as_list(columns: Columns) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))  # dedupe, keep order


def _available(df: pd.DataFrame) -> str:
    names = [str(c) for c in df.columns]
    shown = ", ".join(names[:50]) + (" ..." if len(names) > 50 else "")
    return f"Available columns: {shown}"


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ColumnError if any column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ColumnError(f"Unknown column(s): {', '.join(map(str, missing))}. {_available(df)}")


def require_numeric(df: pd.DataFrame, column: str) -> None:
    """Raise ColumnError if the column is missing or not numeric."""
    require_columns(df, [column])
    kind = column_kind(df[column])
    if kind != "numeric":
        raise ColumnError(f"Column '{column}' is not numeric (it is {kind})")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) == "numeric"]


def _resolve_numeric(df: pd.DataFrame, columns: Columns) -> list[str]:
    requested = _as_list(columns)
    if not requested:
        return numeric_columns(df)
    for col in requested:
        require_numeric(df, col)
    return requested


def _check_choice(value: Any, choices: tuple[str, ...], what: str) -> None:
    if value not in choices:
        raise CoreError(f"Unsupported {what}: {value!r}. Supported: {', '.join(choices)}")


def _check_limit(value: int, name: str, maximum: int = MAX_ROWS) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CoreError(f"{name} must be an integer between 1 and {maximum}")


def _float(value: Any) -> float | None:
    number = to_json_value(value)
    return None if number is None else float(number)


# ── describe ────────────────────────────────────────────────────────

def describe(df: pd.DataFrame, columns: Columns = None) -> DescribeResult:
    """Descriptive statistics for the given numeric columns (default: all).

    Statistics ignore missing values; ``std`` is the sample standard
    deviation (ddof=1), matching pandas.
    """
    stats = []
    for col in _resolve_numeric(df, columns):
        values = df[col].dropna()
        q25, median, q75 = (
            values.quantile([0.25, 0.5, 0.75]).tolist() if len(values) else (None,) * 3
        )
        stats.append(
            ColumnStats(
                column=str(col),
                count=int(values.count()),
                null_count=int(df[col].isna().sum()),
                mean=_float(values.mean()),
                std=_float(values.std()),
                min=_float(values.min()),
                q25=_float(q25),
                median=_float(median),
                q75=_float(q75),
                max=_float(values.max()),
            )
        )
    return DescribeResult(rows=len(df), columns=stats)


# ── group_aggregate ─────────────────────────────────────────────────

def check_aggregation(
    df: pd.DataFrame,
    keys: Sequence[str],
    agg: AggFunc,
    value_column: str | None,
    value_param: str = "value_column",
) -> None:
    """Validate an aggregation request (shared with core.charts).

    ``value_param`` is the caller's name for the value column, used in errors.
    """
    require_columns(df, keys)
    _check_choice(agg, AGG_FUNCS, "aggregation")
    if agg == "count":
        if value_column is not None:
            require_columns(df, [value_column])
    else:
        if value_column is None:
            raise CoreError(f"Aggregation '{agg}' needs a {value_param}")
        require_numeric(df, value_column)
    if value_column is not None and value_column in keys:
        raise CoreError(f"{value_param} must not also be a group_by column")


def aggregate(
    df: pd.DataFrame, keys: Sequence[str], agg: AggFunc, value_column: str | None
) -> pd.Series:
    """Aggregated values indexed by group key, in key order.

    Call ``check_aggregation`` first. Missing keys form their own group.
    """
    grouped = df.groupby(list(keys), dropna=False, sort=True)
    if value_column is None:
        return grouped.size()
    return getattr(grouped[value_column], agg)()


def group_aggregate(
    df: pd.DataFrame,
    group_by: Union[str, Sequence[str]],
    agg: AggFunc,
    value_column: str | None = None,
    sort: SortOrder = "desc",
    limit: int = 50,
) -> GroupAggregateResult:
    """Aggregate ``value_column`` per group.

    - ``count`` counts rows per group, or non-null ``value_column`` values
      when a column is given.
    - ``sum``/``mean``/``median``/``min``/``max`` need a numeric
      ``value_column`` and ignore missing values.
    - Missing group keys form their own group (key ``None``), so group
      totals always add up to the whole dataset.
    - ``sort`` orders by the aggregated value (``"none"`` keeps key order);
      ties keep key order.
    """
    keys = _as_list(group_by)
    if not 1 <= len(keys) <= MAX_GROUP_KEYS:
        raise CoreError(f"group_by must name 1 to {MAX_GROUP_KEYS} columns")
    require_columns(df, keys)
    _check_choice(agg, AGG_FUNCS, "aggregation")
    _check_choice(sort, SORT_ORDERS, "sort order")
    _check_limit(limit, "limit")
    check_aggregation(df, keys, agg, value_column)

    series = aggregate(df, keys, agg, value_column)

    if sort != "none":
        series = series.sort_values(
            ascending=(sort == "asc"), kind="mergesort", na_position="last"
        )

    value_name = "count" if value_column is None else f"{agg}_{value_column}"
    if value_name in keys:
        value_name = f"{value_name}_value"

    rows = []
    for key, value in series.iloc[:limit].items():
        key_values = key if isinstance(key, tuple) else (key,)
        row = {str(k): to_json_value(v) for k, v in zip(keys, key_values)}
        row[value_name] = to_json_value(value)
        rows.append(row)

    return GroupAggregateResult(
        group_by=[str(k) for k in keys],
        agg=agg,
        value_column=value_column,
        value_name=value_name,
        total_groups=len(series),
        truncated=len(series) > limit,
        rows=rows,
    )


# ── correlate ───────────────────────────────────────────────────────

def correlate(
    df: pd.DataFrame,
    columns: Columns = None,
    method: CorrelationMethod = "pearson",
) -> CorrelationResult:
    """Pairwise correlations between numeric columns (default: all).

    Uses pairwise-complete observations; ``n`` is the number of rows where
    both values are present. Pairs are sorted by |r| (undefined last).
    """
    _check_choice(method, CORRELATION_METHODS, "correlation method")
    cols = _resolve_numeric(df, columns)
    if len(cols) < 2:
        raise ColumnError("Correlation needs at least two numeric columns")

    matrix = df[cols].corr(method=method)
    present = df[cols].notna()
    pairs = [
        CorrelationPair(
            x=str(a),
            y=str(b),
            r=_float(matrix.loc[a, b]),
            n=int((present[a] & present[b]).sum()),
        )
        for i, a in enumerate(cols)
        for b in cols[i + 1:]
    ]
    pairs.sort(key=lambda p: (p.r is None, -abs(p.r) if p.r is not None else 0.0))
    return CorrelationResult(method=method, columns=[str(c) for c in cols], pairs=pairs)


# ── detect_anomalies ────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: AnomalyMethod = "iqr",
    threshold: float | None = None,
    max_results: int = 50,
) -> AnomalyResult:
    """Flag outliers in a numeric column. Missing values are ignored.

    - ``iqr`` (default): outside [Q1 - k·IQR, Q3 + k·IQR], k = 1.5.
    - ``zscore``: |value - mean| / std > threshold (default 3.0), using the
      sample standard deviation. A constant column has no anomalies.

    Anomalies are listed in row order.
    """
    _check_choice(method, ANOMALY_METHODS, "anomaly method")
    require_numeric(df, column)
    _check_limit(max_results, "max_results")
    default = DEFAULT_IQR_K if method == "iqr" else DEFAULT_Z_THRESHOLD
    threshold = default if threshold is None else threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not threshold > 0:
        raise CoreError("threshold must be a positive number")
    threshold = float(threshold)

    values = df[column].dropna().astype(float)
    lower = upper = None
    z_scores = None
    mask = pd.Series(False, index=values.index)

    if method == "iqr":
        q1, q3 = (values.quantile([0.25, 0.75]).tolist() if len(values) else (None, None))
        iqr = None if q1 is None else q3 - q1
        parameters = {"q1": _float(q1), "q3": _float(q3), "iqr": _float(iqr)}
        if iqr is not None:
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            mask = (values < lower) | (values > upper)
    else:
        mean, std = values.mean(), values.std()
        parameters = {"mean": _float(mean), "std": _float(std)}
        if _float(std):  # defined and non-zero
            lower, upper = mean - threshold * std, mean + threshold * std
            z_scores = (values - mean) / std
            mask = z_scores.abs() > threshold

    flagged = values[mask]
    flagged_z = [None] * len(flagged) if z_scores is None else z_scores[mask].tolist()
    anomalies = [
        Anomaly(
            index=to_json_value(idx),
            value=float(value),
            direction="high" if value > upper else "low",
            z_score=None if z is None else float(z),
        )
        for idx, value, z in list(zip(flagged.index, flagged.tolist(), flagged_z))[:max_results]
    ]
    return AnomalyResult(
        column=column,
        method=method,
        threshold=threshold,
        parameters=parameters,
        lower_bound=_float(lower),
        upper_bound=_float(upper),
        checked_count=len(values),
        anomaly_count=len(flagged),
        truncated=len(flagged) > max_results,
        anomalies=anomalies,
    )
