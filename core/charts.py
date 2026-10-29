"""
make_chart — deterministic charts from validated columns.

Returns a ChartResult: a base64 PNG plus the plotted values, so a caller
(the LangGraph app now, an MCP tool later) gets both the image and the
numbers behind it.

Charts are drawn with matplotlib's object API (``matplotlib.figure.Figure``),
never pyplot, so they do not touch pyplot's global figure list — which the
sandbox reads to capture plots made by generated code.

Chart types:
  - bar:       ``agg`` of ``y`` per ``x`` value (row count when ``y`` is
               omitted), largest first, top ``limit`` groups.
  - line:      ``agg`` of ``y`` per ``x`` value in ``x`` order (row count
               when ``y`` is omitted). Rows with a missing ``x`` are skipped.
  - scatter:   numeric ``y`` against numeric ``x``, one point per row with
               both values present.
  - histogram: distribution of numeric ``x`` in ``bins`` equal-width bins.
"""

from __future__ import annotations

import base64
import io
from typing import Any, get_args

import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.figure import Figure

from core.analysis import aggregate, check_aggregation, require_columns, require_numeric
from core.errors import CoreError
from core.schemas import AggFunc, ChartResult, ChartType, column_kind, to_json_value

CHART_TYPES: tuple[str, ...] = get_args(ChartType)

WIDTH_PX, HEIGHT_PX, DPI = 800, 480, 100
DEFAULT_BAR_LIMIT = 20
MAX_BAR_LIMIT = 50
DEFAULT_BINS = 20
MAX_BINS = 100
MAX_DATA_ROWS = 200  # plotted values returned in ChartResult.data
MAX_TITLE_CHARS = 120
MAX_TICK_LABEL_CHARS = 30

# Single-series palette: one hue for marks; text in neutral ink, never the
# series color; recessive hairline grid.
SERIES_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

# Column names and titles are shown literally ("$" is not mathtext)
_PLAIN = {"parse_math": False}

_AGG_LABELS = {
    "count": "Count", "sum": "Sum", "mean": "Mean",
    "median": "Median", "min": "Min", "max": "Max",
}


def make_chart(
    df: pd.DataFrame,
    chart_type: ChartType,
    x: str,
    y: str | None = None,
    agg: AggFunc | None = None,
    title: str | None = None,
    limit: int = DEFAULT_BAR_LIMIT,
    bins: int = DEFAULT_BINS,
) -> ChartResult:
    """Render a chart of ``df`` and return it with the plotted values.

    ``agg`` applies to bar and line charts only: it defaults to ``"sum"``
    when ``y`` is given and ``"count"`` (rows per ``x`` value) otherwise.
    ``limit`` applies to bar charts, ``bins`` to histograms.
    """
    if chart_type not in CHART_TYPES:
        raise CoreError(
            f"Unsupported chart type: {chart_type!r}. Supported: {', '.join(CHART_TYPES)}"
        )
    if not isinstance(x, str):
        raise CoreError("x must be a column name")
    require_columns(df, [x])
    if y is not None and not isinstance(y, str):
        raise CoreError("y must be a column name")
    title = _clean_title(title)

    if chart_type in ("bar", "line"):
        if agg is None:
            agg = "sum" if y is not None else "count"
        check_aggregation(df, [x], agg, y, value_param="y")
        if chart_type == "bar":
            _check_int(limit, "limit", MAX_BAR_LIMIT)
            return _bar(df, x, y, agg, title, limit)
        return _line(df, x, y, agg, title)

    if agg is not None:
        raise CoreError(f"agg is not used by {chart_type} charts; omit it")
    if chart_type == "scatter":
        if y is None:
            raise CoreError("A scatter chart needs a numeric y column")
        if y == x:
            raise CoreError("x and y must be different columns")
        require_numeric(df, x)
        require_numeric(df, y)
        return _scatter(df, x, y, title)

    if y is not None:
        raise CoreError("A histogram uses only x; omit y")
    require_numeric(df, x)
    _check_int(bins, "bins", MAX_BINS)
    return _histogram(df, x, title, bins)


# ── Chart builders ──────────────────────────────────────────────────

def _bar(df, x, y, agg, title, limit) -> ChartResult:
    series = aggregate(df, [x], agg, y)
    series = series.sort_values(ascending=False, kind="mergesort", na_position="last")
    shown = series.iloc[:limit]
    if shown.empty:
        raise CoreError("Nothing to plot: the dataset has no rows")

    value_label = _value_label(agg, y)
    truncated = len(series) > limit
    if title is None:
        title = f"{value_label} by {x}" + (f" (top {limit})" if truncated else "")

    fig, ax = _figure()
    n = len(shown)
    positions = np.arange(n)[::-1]  # largest bar at the top
    # Cap bar thickness at ~24px however few bars there are
    plot_height_px = HEIGHT_PX * 0.75
    thickness = min(0.7, 24 * n / plot_height_px)
    values = [_plot_number(v) for v in shown.tolist()]
    ax.barh(positions, values, height=thickness, color=SERIES_COLOR, edgecolor=SURFACE_COLOR,
            linewidth=2)
    ax.set_yticks(positions, [_tick_label(k) for k in shown.index], **_PLAIN)
    ax.set_xlabel(value_label, **_PLAIN)
    ax.set_ylabel(x, **_PLAIN)
    _style_value_axis(ax, "x")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)

    data = [{x: to_json_value(k), _value_key(agg, y): to_json_value(v)} for k, v in shown.items()]
    return _result(fig, "bar", title, x, y, agg, data_points=n, data=data, truncated=truncated)


def _line(df, x, y, agg, title) -> ChartResult:
    subset = df[df[x].notna()]
    series = aggregate(subset, [x], agg, y)
    if series.empty:
        raise CoreError(f"Nothing to plot: column '{x}' has no values")

    value_label = _value_label(agg, y)
    if title is None:
        title = f"{value_label} by {x}"

    fig, ax = _figure()
    keys = series.index
    kind = column_kind(subset[x])
    values = [_plot_number(v) for v in series.tolist()]
    if kind in ("numeric", "datetime"):
        xs: Any = keys.to_numpy()
    else:
        # Text keys (e.g. ISO dates stored as text): even spacing in key
        # order, with a readable subset of tick labels.
        xs = np.arange(len(keys))
        step = max(1, int(np.ceil(len(keys) / 10)))
        ticks = xs[::step]
        ax.set_xticks(ticks, [_tick_label(keys[i]) for i in ticks], rotation=30, ha="right",
                      **_PLAIN)
    ax.plot(xs, values, color=SERIES_COLOR, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", marker="o" if len(keys) <= 50 else None,
            markersize=6, markeredgecolor=SURFACE_COLOR, markeredgewidth=1.5)
    ax.set_xlabel(x, **_PLAIN)
    ax.set_ylabel(value_label, **_PLAIN)
    _style_value_axis(ax, "y")

    rows = list(series.items())
    data = [{x: to_json_value(k), _value_key(agg, y): to_json_value(v)} for k, v in rows[:MAX_DATA_ROWS]]
    return _result(fig, "line", title, x, y, agg, data_points=len(rows), data=data,
                   truncated=len(rows) > MAX_DATA_ROWS)


def _scatter(df, x, y, title) -> ChartResult:
    points = df[[x, y]].dropna()
    xs = points[x].astype(float).to_numpy()
    ys = points[y].astype(float).to_numpy()
    if len(xs) == 0:
        raise CoreError(f"Nothing to plot: no rows have both '{x}' and '{y}'")
    if title is None:
        title = f"{y} vs {x}"

    fig, ax = _figure()
    ax.scatter(xs, ys, s=36, color=SERIES_COLOR, alpha=0.75, edgecolors=SURFACE_COLOR,
               linewidths=0.8)
    ax.set_xlabel(x, **_PLAIN)
    ax.set_ylabel(y, **_PLAIN)
    _style_value_axis(ax, "both")

    data = [
        {x: to_json_value(a), y: to_json_value(b)}
        for a, b in zip(xs[:MAX_DATA_ROWS].tolist(), ys[:MAX_DATA_ROWS].tolist())
    ]
    return _result(fig, "scatter", title, x, y, None, data_points=len(xs), data=data,
                   truncated=len(xs) > MAX_DATA_ROWS)


def _histogram(df, x, title, bins) -> ChartResult:
    values = df[x].dropna().astype(float).to_numpy()
    if len(values) == 0:
        raise CoreError(f"Nothing to plot: column '{x}' has no values")
    counts, edges = np.histogram(values, bins=bins)
    if title is None:
        title = f"Distribution of {x}"

    fig, ax = _figure()
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=SERIES_COLOR,
           edgecolor=SURFACE_COLOR, linewidth=2)
    ax.set_xlabel(x, **_PLAIN)
    ax.set_ylabel("Count", **_PLAIN)
    _style_value_axis(ax, "y")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_tick))

    data = [
        {"bin_start": float(lo), "bin_end": float(hi), "count": int(c)}
        for lo, hi, c in zip(edges[:-1], edges[1:], counts)
    ]
    return _result(fig, "histogram", title, x, None, None, data_points=len(counts), data=data,
                   truncated=False)


# ── Helpers ─────────────────────────────────────────────────────────

def _figure() -> tuple[Figure, Any]:
    fig = Figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, layout="constrained",
                 facecolor=SURFACE_COLOR)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    return fig, ax


def _style_value_axis(ax, axis: str) -> None:
    """Hairline grid and comma-separated ticks on the value axis."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=1, linestyle="-")
    formatter = ticker.FuncFormatter(_format_tick)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def _result(fig, chart_type, title, x, y, agg, *, data_points, data, truncated) -> ChartResult:
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, fontweight="semibold", **_PLAIN)
    buf = io.BytesIO()
    # No "Software" metadata: the same input always gives the same bytes
    fig.savefig(buf, format="png", dpi=DPI, facecolor=SURFACE_COLOR, metadata={"Software": None})
    width, height = (int(round(v)) for v in fig.get_size_inches() * DPI)
    return ChartResult(
        chart_type=chart_type,
        title=title,
        x=x,
        y=y,
        agg=agg,
        image_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
        width_px=width,
        height_px=height,
        data_points=data_points,
        data=data,
        data_truncated=truncated,
    )


def _format_tick(value: float, _pos: Any = None) -> str:
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _plot_number(value: Any) -> float:
    number = to_json_value(value)
    return float("nan") if number is None else float(number)


def _tick_label(key: Any) -> str:
    label = "(missing)" if to_json_value(key) is None else str(to_json_value(key))
    if len(label) > MAX_TICK_LABEL_CHARS:
        label = label[: MAX_TICK_LABEL_CHARS - 1] + "…"
    return label


def _value_label(agg: str, y: str | None) -> str:
    return "Row count" if y is None else f"{_AGG_LABELS[agg]} of {y}"


def _value_key(agg: str, y: str | None) -> str:
    """Key of the plotted value in ChartResult.data (as in group_aggregate)."""
    return "count" if y is None else f"{agg}_{y}"


def _clean_title(title: Any) -> str | None:
    if title is None:
        return None
    if not isinstance(title, str):
        raise CoreError("title must be a string")
    cleaned = "".join(ch for ch in title if ch.isprintable()).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_TITLE_CHARS:
        raise CoreError(f"title must be at most {MAX_TITLE_CHARS} characters")
    return cleaned


def _check_int(value: Any, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CoreError(f"{name} must be an integer between 1 and {maximum}")
