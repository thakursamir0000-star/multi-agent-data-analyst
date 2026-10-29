"""
Tests for core/analysis.py — deterministic analysis functions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.analysis import correlate, describe, detect_anomalies, group_aggregate
from core.errors import ColumnError, CoreError

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"


@pytest.fixture
def sales():
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def small():
    return pd.DataFrame({
        "region": ["N", "N", "S", "S", None],
        "sales": [10.0, 20.0, 5.0, None, 7.0],
        "units": [1, 2, 3, 4, 5],
        "flag": [True, False, True, False, True],
    })


# ── describe ────────────────────────────────────────────────────────

class TestDescribe:
    def test_numeric_stats(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        stats = describe(df).columns[0]
        assert stats.column == "a"
        assert (stats.count, stats.null_count) == (4, 0)
        assert (stats.min, stats.q25, stats.median, stats.q75, stats.max) == (1, 1.75, 2.5, 3.25, 4)
        assert stats.mean == 2.5
        assert stats.std == pytest.approx(1.2909944, rel=1e-6)

    def test_default_uses_numeric_columns_only(self, small):
        result = describe(small)
        assert [c.column for c in result.columns] == ["sales", "units"]  # not text or bool
        assert result.rows == 5

    def test_missing_values_ignored(self, small):
        stats = describe(small, ["sales"]).columns[0]
        assert (stats.count, stats.null_count) == (4, 1)
        assert stats.mean == pytest.approx(10.5)

    def test_selected_columns_in_order(self, small):
        result = describe(small, ["units", "sales"])
        assert [c.column for c in result.columns] == ["units", "sales"]

    def test_single_column_as_string(self, small):
        assert [c.column for c in describe(small, "units").columns] == ["units"]

    def test_invalid_column(self, small):
        with pytest.raises(ColumnError, match="Unknown column.*nope"):
            describe(small, ["nope"])

    def test_non_numeric_column(self, small):
        with pytest.raises(ColumnError, match="not numeric"):
            describe(small, ["region"])

    def test_all_missing_column(self):
        stats = describe(pd.DataFrame({"x": [np.nan, np.nan]})).columns[0]
        assert stats.count == 0
        assert stats.mean is None and stats.median is None and stats.max is None

    def test_matches_pandas_on_sample(self, sales):
        stats = describe(sales, ["revenue"]).columns[0]
        assert stats.mean == pytest.approx(sales["revenue"].mean())
        assert stats.median == pytest.approx(sales["revenue"].median())
        assert stats.count == sales["revenue"].count()

    def test_json_serializable(self, sales):
        json.loads(describe(sales).model_dump_json())


# ── group_aggregate ─────────────────────────────────────────────────

def _by_key(result, key="region"):
    return {row[key]: row[result.value_name] for row in result.rows}


class TestGroupAggregate:
    @pytest.mark.parametrize(
        "agg, expected",
        [
            ("sum", {"N": 30.0, "S": 5.0, None: 7.0}),
            ("mean", {"N": 15.0, "S": 5.0, None: 7.0}),
            ("median", {"N": 15.0, "S": 5.0, None: 7.0}),
            ("min", {"N": 10.0, "S": 5.0, None: 7.0}),
            ("max", {"N": 20.0, "S": 5.0, None: 7.0}),
            ("count", {"N": 2, "S": 1, None: 1}),  # non-null 'sales' values
        ],
    )
    def test_each_aggregation(self, small, agg, expected):
        result = group_aggregate(small, "region", agg, "sales")
        assert _by_key(result) == expected
        assert result.value_name == f"{agg}_sales"

    def test_row_count_without_value_column(self, small):
        result = group_aggregate(small, "region", "count")
        assert result.value_name == "count"
        assert _by_key(result) == {"N": 2, "S": 2, None: 1}
        assert sum(_by_key(result).values()) == len(small)  # missing keys included

    def test_sorted_descending_by_default(self, small):
        result = group_aggregate(small, "region", "sum", "sales")
        assert [row["region"] for row in result.rows] == ["N", None, "S"]

    def test_sort_ascending_and_none(self, small):
        asc = group_aggregate(small, "region", "sum", "sales", sort="asc")
        assert [row["region"] for row in asc.rows] == ["S", None, "N"]
        none = group_aggregate(small, "region", "sum", "sales", sort="none")
        assert [row["region"] for row in none.rows] == ["N", "S", None]

    def test_multiple_group_keys(self, small):
        result = group_aggregate(small, ["region", "flag"], "sum", "units", sort="none")
        assert result.group_by == ["region", "flag"]
        assert {"region": "N", "flag": False, "sum_units": 2} in result.rows

    def test_limit_truncates(self, sales):
        result = group_aggregate(sales, "product", "sum", "revenue", limit=3)
        assert len(result.rows) == 3
        assert result.total_groups == sales["product"].nunique()
        assert result.truncated is True

    def test_matches_pandas_on_sample(self, sales):
        result = group_aggregate(sales, "category", "sum", "revenue")
        expected = sales.groupby("category")["revenue"].sum()
        for row in result.rows:
            assert row["sum_revenue"] == pytest.approx(expected[row["category"]])

    def test_invalid_aggregation(self, small):
        with pytest.raises(CoreError, match="Unsupported aggregation"):
            group_aggregate(small, "region", "variance", "sales")

    def test_invalid_group_column(self, small):
        with pytest.raises(ColumnError, match="Unknown column"):
            group_aggregate(small, "nope", "sum", "sales")

    def test_invalid_value_column(self, small):
        with pytest.raises(ColumnError, match="Unknown column"):
            group_aggregate(small, "region", "sum", "nope")

    def test_non_numeric_value_column(self, small):
        with pytest.raises(ColumnError, match="not numeric"):
            group_aggregate(small, "units", "mean", "region")

    def test_value_column_required(self, small):
        with pytest.raises(CoreError, match="needs a value_column"):
            group_aggregate(small, "region", "sum")

    def test_value_column_not_group_key(self, small):
        with pytest.raises(CoreError, match="must not also be"):
            group_aggregate(small, ["region", "units"], "sum", "units")

    @pytest.mark.parametrize("limit", [0, 5000, True, 2.5])
    def test_invalid_limit(self, small, limit):
        with pytest.raises(CoreError, match="limit"):
            group_aggregate(small, "region", "count", limit=limit)

    def test_invalid_sort(self, small):
        with pytest.raises(CoreError, match="Unsupported sort order"):
            group_aggregate(small, "region", "count", sort="random")

    def test_too_many_group_keys(self, small):
        with pytest.raises(CoreError, match="1 to 3"):
            group_aggregate(small, ["region", "flag", "units", "sales"], "count")


# ── correlate ───────────────────────────────────────────────────────

class TestCorrelate:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "x": [1, 2, 3, 4],
            "up": [2, 4, 6, 8],
            "down": [4, 3, 2, 1],
            "const": [5, 5, 5, 5],
            "label": ["a", "b", "c", "d"],
        })

    def test_valid_numeric_columns(self, df):
        result = correlate(df, ["x", "up", "down"])
        pairs = {(p.x, p.y): p.r for p in result.pairs}
        assert pairs[("x", "up")] == pytest.approx(1.0)
        assert pairs[("x", "down")] == pytest.approx(-1.0)
        assert all(p.n == 4 for p in result.pairs)

    def test_default_all_numeric_and_undefined_last(self, df):
        result = correlate(df)
        assert result.columns == ["x", "up", "down", "const"]
        assert result.pairs[-1].r is None  # constant column is undefined
        assert len(result.pairs) == 6

    def test_sorted_by_strength(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 3, 2, 5, 4], "c": [2, 4, 6, 8, 10]})
        rs = [abs(p.r) for p in correlate(df).pairs]
        assert rs == sorted(rs, reverse=True)

    def test_spearman(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 10, 100, 1000]})
        assert correlate(df, method="spearman").pairs[0].r == pytest.approx(1.0)

    def test_pairwise_n_with_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3, None], "b": [1, None, 3, 4]})
        assert correlate(df).pairs[0].n == 2

    def test_matches_pandas_on_sample(self, sales):
        result = correlate(sales, ["discount", "profit"])
        assert result.pairs[0].r == pytest.approx(sales["discount"].corr(sales["profit"]))

    def test_invalid_column(self, df):
        with pytest.raises(ColumnError, match="Unknown column"):
            correlate(df, ["x", "nope"])

    def test_non_numeric_column(self, df):
        with pytest.raises(ColumnError, match="not numeric"):
            correlate(df, ["x", "label"])

    def test_needs_two_columns(self, df):
        with pytest.raises(ColumnError, match="at least two"):
            correlate(df, ["x"])

    def test_invalid_method(self, df):
        with pytest.raises(CoreError, match="Unsupported correlation method"):
            correlate(df, method="kendall")


# ── detect_anomalies ────────────────────────────────────────────────

class TestDetectAnomalies:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"v": [10, 11, 12, 13, 12, 11, 10, 100], "t": list("abcdefgh")})

    def test_known_high_outlier_iqr(self, df):
        result = detect_anomalies(df, "v")
        assert result.method == "iqr" and result.threshold == 1.5
        assert result.anomaly_count == 1
        anomaly = result.anomalies[0]
        assert (anomaly.index, anomaly.value, anomaly.direction) == (7, 100.0, "high")
        assert result.parameters == {"q1": 10.75, "q3": 12.25, "iqr": 1.5}
        assert (result.lower_bound, result.upper_bound) == (8.5, 14.5)

    def test_known_low_outlier(self):
        df = pd.DataFrame({"v": [50, 51, 52, 53, 52, 51, 50, -40]})
        anomaly = detect_anomalies(df, "v").anomalies[0]
        assert (anomaly.value, anomaly.direction) == (-40.0, "low")

    def test_no_outlier(self):
        result = detect_anomalies(pd.DataFrame({"v": [1, 2, 3, 4, 5]}), "v")
        assert result.anomaly_count == 0 and result.anomalies == []
        assert result.checked_count == 5

    def test_zscore(self):
        values = [10.0] * 10 + [11.0] * 10 + [100.0]
        df = pd.DataFrame({"v": values})
        result = detect_anomalies(df, "v", method="zscore")
        s = pd.Series(values)
        expected_z = (100 - s.mean()) / s.std()
        assert result.anomaly_count == 1
        assert result.anomalies[0].z_score == pytest.approx(expected_z)
        assert result.parameters["mean"] == pytest.approx(s.mean())

    def test_zscore_constant_column(self):
        result = detect_anomalies(pd.DataFrame({"v": [3, 3, 3]}), "v", method="zscore")
        assert result.anomaly_count == 0
        assert result.lower_bound is None and result.upper_bound is None

    def test_custom_threshold(self, df):
        assert detect_anomalies(df, "v", threshold=100).anomaly_count == 0

    def test_missing_values_ignored(self):
        df = pd.DataFrame({"v": [10, 11, None, 12, 13, 12, 11, 10, 100]})
        result = detect_anomalies(df, "v")
        assert result.checked_count == 8
        assert result.anomalies[0].index == 8

    def test_max_results_truncates(self):
        df = pd.DataFrame({"v": [10] * 20 + [100, 200, 300]})
        result = detect_anomalies(df, "v", max_results=2)
        assert result.anomaly_count == 3 and len(result.anomalies) == 2
        assert result.truncated is True

    def test_duplicate_index_labels(self):
        df = pd.DataFrame({"v": [1.0] * 9 + [50.0]}, index=[0] * 10)
        result = detect_anomalies(df, "v", method="zscore", threshold=2)
        assert result.anomalies[0].value == 50.0

    def test_invalid_column(self, df):
        with pytest.raises(ColumnError, match="Unknown column"):
            detect_anomalies(df, "nope")

    def test_non_numeric_column(self, df):
        with pytest.raises(ColumnError, match="not numeric"):
            detect_anomalies(df, "t")

    def test_invalid_method(self, df):
        with pytest.raises(CoreError, match="Unsupported anomaly method"):
            detect_anomalies(df, "v", method="isolation_forest")

    @pytest.mark.parametrize("threshold", [0, -1, "3", True])
    def test_invalid_threshold(self, df, threshold):
        with pytest.raises(CoreError, match="threshold"):
            detect_anomalies(df, "v", threshold=threshold)

    def test_sample_revenue(self, sales):
        result = detect_anomalies(sales, "revenue")
        q1, q3 = sales["revenue"].quantile([0.25, 0.75])
        expected = ((sales["revenue"] < q1 - 1.5 * (q3 - q1)) | (sales["revenue"] > q3 + 1.5 * (q3 - q1))).sum()
        assert result.anomaly_count == expected
        json.loads(result.model_dump_json())
