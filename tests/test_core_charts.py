"""
Tests for core/charts.py — deterministic chart rendering.
"""

import base64
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.charts import MAX_DATA_ROWS, make_chart
from core.errors import ColumnError, CoreError
from core.schemas import ChartResult

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"


@pytest.fixture
def sales():
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def small():
    return pd.DataFrame({
        "region": ["N", "N", "S", "S", None, "E"],
        "sales": [10.0, 20.0, 5.0, None, 7.0, 1.0],
        "units": [1, 2, 3, 4, 5, 6],
        "label": list("abcdef"),
    })


def _png(result: ChartResult) -> bytes:
    return base64.b64decode(result.image_base64)


def _png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


# ── Output Format ───────────────────────────────────────────────────

class TestOutputFormat:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"chart_type": "bar", "x": "category", "y": "revenue"},
            {"chart_type": "line", "x": "date", "y": "revenue"},
            {"chart_type": "scatter", "x": "discount", "y": "profit"},
            {"chart_type": "histogram", "x": "revenue"},
        ],
        ids=["bar", "line", "scatter", "histogram"],
    )
    def test_valid_chart_is_png(self, sales, kwargs):
        result = make_chart(sales, **kwargs)
        png = _png(result)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert result.mime_type == "image/png"
        assert _png_size(png) == (result.width_px, result.height_px) == (800, 480)
        assert result.chart_type == kwargs["chart_type"]
        assert result.data_points > 0 and result.data

    def test_json_serializable(self, sales):
        result = make_chart(sales, "bar", "region", "profit", agg="mean")
        payload = json.loads(result.model_dump_json())
        assert payload["image_base64"] == result.image_base64
        assert payload["data"][0]["mean_profit"] == result.data[0]["mean_profit"]

    def test_deterministic(self, sales):
        first = make_chart(sales, "line", "date", "revenue")
        second = make_chart(sales, "line", "date", "revenue")
        assert first == second  # same image bytes and data

    def test_does_not_touch_pyplot_state(self, sales):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")
        make_chart(sales, "histogram", "revenue")
        assert plt.get_fignums() == []

    def test_sandbox_plot_capture_unaffected(self, sales):
        from tools.sandbox import run_code

        make_chart(sales, "bar", "category")
        assert run_code("print(1)")["plot_base64"] is None

    def test_does_not_modify_dataframe(self, small):
        before = small.copy()
        make_chart(small, "bar", "region", "sales")
        make_chart(small, "scatter", "units", "sales")
        pd.testing.assert_frame_equal(small, before)

    def test_dollar_signs_render_literally(self):
        df = pd.DataFrame({"cost ($)": [1.0, 2.0, 3.0], "$group$": ["a", "b", "a"]})
        result = make_chart(df, "bar", "$group$", "cost ($)", title="Cost in $ by $group$")
        assert result.title == "Cost in $ by $group$"
        assert result.data[0] == {"$group$": "a", "sum_cost ($)": 4.0}


# ── Bar ─────────────────────────────────────────────────────────────

class TestBar:
    def test_sum_by_default_matches_pandas(self, sales):
        result = make_chart(sales, "bar", "category", "revenue")
        assert result.agg == "sum"
        expected = sales.groupby("category")["revenue"].sum().sort_values(ascending=False)
        assert [row["category"] for row in result.data] == list(expected.index)
        assert [row["sum_revenue"] for row in result.data] == pytest.approx(expected.tolist())
        assert result.title == "Sum of revenue by category"

    def test_row_count_without_y(self, small):
        result = make_chart(small, "bar", "region")
        assert result.agg == "count" and result.y is None
        assert {row["region"]: row["count"] for row in result.data} == {"N": 2, "S": 2, "E": 1, None: 1}

    @pytest.mark.parametrize("agg", ["count", "sum", "mean", "median", "min", "max"])
    def test_each_aggregation(self, small, agg):
        result = make_chart(small, "bar", "region", "sales", agg=agg)
        expected = getattr(small.groupby("region", dropna=False)["sales"], agg)()
        got = {row["region"]: row[f"{agg}_sales"] for row in result.data}
        assert got == pytest.approx({(None if pd.isna(k) else k): v for k, v in expected.items()})

    def test_limit_keeps_top_groups(self, sales):
        result = make_chart(sales, "bar", "product", "revenue", limit=3)
        assert result.data_points == 3 and len(result.data) == 3
        assert result.data_truncated is True
        assert result.title.endswith("(top 3)")
        top = sales.groupby("product")["revenue"].sum().nlargest(3)
        assert [row["product"] for row in result.data] == list(top.index)

    def test_custom_title(self, sales):
        assert make_chart(sales, "bar", "region", title="  Orders  ").title == "Orders"


# ── Line ────────────────────────────────────────────────────────────

class TestLine:
    def test_in_x_order_with_missing_x_skipped(self, small):
        result = make_chart(small, "line", "region", "units")
        assert [row["region"] for row in result.data] == ["E", "N", "S"]
        assert [row["sum_units"] for row in result.data] == [6, 3, 7]

    def test_numeric_x(self):
        df = pd.DataFrame({"x": [3, 1, 2, 1], "y": [30.0, 10.0, 20.0, 5.0]})
        result = make_chart(df, "line", "x", "y", agg="mean")
        assert result.data == [{"x": 1, "mean_y": 7.5}, {"x": 2, "mean_y": 20.0}, {"x": 3, "mean_y": 30.0}]

    def test_datetime_x(self):
        df = pd.DataFrame({
            "when": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"]),
            "v": [1, 2, 3],
        })
        result = make_chart(df, "line", "when", "v")
        assert result.data == [
            {"when": "2024-01-01T00:00:00", "sum_v": 2},
            {"when": "2024-01-02T00:00:00", "sum_v": 4},
        ]

    def test_data_truncated_but_all_points_plotted(self, sales):
        result = make_chart(sales, "line", "date", "revenue")
        assert result.data_points == sales["date"].nunique() > MAX_DATA_ROWS
        assert len(result.data) == MAX_DATA_ROWS and result.data_truncated is True


# ── Scatter ─────────────────────────────────────────────────────────

class TestScatter:
    def test_points(self, small):
        result = make_chart(small, "scatter", "units", "sales")
        assert result.data_points == 5  # the row with missing sales is skipped
        assert result.data[0] == {"units": 1.0, "sales": 10.0}
        assert result.agg is None and result.title == "sales vs units"

    def test_needs_y(self, small):
        with pytest.raises(CoreError, match="needs a numeric y"):
            make_chart(small, "scatter", "units")

    def test_same_column_rejected(self, small):
        with pytest.raises(CoreError, match="different columns"):
            make_chart(small, "scatter", "units", "units")

    def test_non_numeric(self, small):
        with pytest.raises(ColumnError, match="not numeric"):
            make_chart(small, "scatter", "label", "sales")

    def test_agg_rejected(self, small):
        with pytest.raises(CoreError, match="agg is not used"):
            make_chart(small, "scatter", "units", "sales", agg="sum")


# ── Histogram ───────────────────────────────────────────────────────

class TestHistogram:
    def test_bins_match_numpy(self, sales):
        result = make_chart(sales, "histogram", "revenue", bins=10)
        counts, edges = np.histogram(sales["revenue"].dropna(), bins=10)
        assert [row["count"] for row in result.data] == counts.tolist()
        assert result.data[0]["bin_start"] == pytest.approx(edges[0])
        assert result.data[-1]["bin_end"] == pytest.approx(edges[-1])
        assert sum(row["count"] for row in result.data) == sales["revenue"].count()

    def test_y_rejected(self, small):
        with pytest.raises(CoreError, match="omit y"):
            make_chart(small, "histogram", "sales", "units")

    def test_non_numeric(self, small):
        with pytest.raises(ColumnError, match="not numeric"):
            make_chart(small, "histogram", "label")

    @pytest.mark.parametrize("bins", [0, 101, True, 2.5])
    def test_invalid_bins(self, small, bins):
        with pytest.raises(CoreError, match="bins"):
            make_chart(small, "histogram", "sales", bins=bins)


# ── Invalid Requests ────────────────────────────────────────────────

class TestInvalid:
    @pytest.mark.parametrize("chart_type", ["pie", "", None, "BAR"])
    def test_unsupported_chart_type(self, small, chart_type):
        with pytest.raises(CoreError, match="Unsupported chart type"):
            make_chart(small, chart_type, "region")

    def test_invalid_x_column(self, small):
        with pytest.raises(ColumnError, match="Unknown column.*nope"):
            make_chart(small, "bar", "nope")

    def test_invalid_y_column(self, small):
        with pytest.raises(ColumnError, match="Unknown column.*nope"):
            make_chart(small, "bar", "region", "nope")

    def test_non_numeric_y(self, small):
        with pytest.raises(ColumnError, match="not numeric"):
            make_chart(small, "line", "units", "label")

    def test_invalid_agg(self, small):
        with pytest.raises(CoreError, match="Unsupported aggregation"):
            make_chart(small, "bar", "region", "sales", agg="variance")

    def test_agg_needs_y(self, small):
        with pytest.raises(CoreError, match="needs a y"):
            make_chart(small, "bar", "region", agg="mean")

    def test_y_equal_to_x(self, small):
        with pytest.raises(CoreError, match="y must not also be"):
            make_chart(small, "bar", "units", "units")

    @pytest.mark.parametrize("limit", [0, 51, True])
    def test_invalid_limit(self, small, limit):
        with pytest.raises(CoreError, match="limit"):
            make_chart(small, "bar", "region", limit=limit)

    def test_title_too_long(self, small):
        with pytest.raises(CoreError, match="at most"):
            make_chart(small, "bar", "region", title="x" * 121)

    def test_title_must_be_string(self, small):
        with pytest.raises(CoreError, match="title must be a string"):
            make_chart(small, "bar", "region", title=5)

    def test_all_missing_values(self):
        df = pd.DataFrame({"v": [np.nan, np.nan]})
        with pytest.raises(CoreError, match="Nothing to plot"):
            make_chart(df, "histogram", "v")

    def test_empty_dataset(self):
        with pytest.raises(CoreError, match="Nothing to plot"):
            make_chart(pd.DataFrame({"g": pd.Series([], dtype=object)}), "bar", "g")


class TestChartMetadata:
    def test_metadata_excludes_only_the_image(self, sales):
        from core.schemas import ChartMetadata

        result = make_chart(sales, "bar", "category", "revenue")
        metadata = result.metadata()
        assert type(metadata) is ChartMetadata
        assert "image_base64" not in metadata.model_dump()
        assert metadata.model_dump() == result.model_dump(exclude={"image_base64", "mime_type"})
