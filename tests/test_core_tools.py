"""
Tests for core/tools.py — run_tool, the dataset-ID surface of the core layer.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from core import analysis, charts
from core.datasets import DatasetRegistry
from core.errors import ColumnError, CoreError, DatasetNotFoundError, QueryError
from core.schemas import (
    AnomalyResult,
    ChartResult,
    CorrelationResult,
    DescribeResult,
    GroupAggregateResult,
    SqlResult,
    ToolCall,
)
from core.tools import (
    TOOL_NAMES,
    _ARG_MODELS,
    _FRAME_TOOLS,
    parse_tool_call,
    result_json,
    run_tool,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def registry():
    return DatasetRegistry(data_dir=SAMPLE_DIR)


@pytest.fixture
def dataset_id(registry):
    return registry.register_file("sales_sample.csv").dataset_id


@pytest.fixture
def sales(registry, dataset_id):
    return registry.get_dataframe(dataset_id)


# ── Each Tool ───────────────────────────────────────────────────────

class TestTools:
    def test_describe(self, registry, dataset_id, sales):
        result = run_tool(registry, dataset_id, "describe", {"columns": ["revenue", "profit"]})
        assert isinstance(result, DescribeResult)
        assert result == analysis.describe(sales, ["revenue", "profit"])

    def test_describe_single_column_and_no_arguments(self, registry, dataset_id):
        assert [c.column for c in run_tool(registry, dataset_id, "describe", {"columns": "revenue"}).columns] == ["revenue"]
        assert len(run_tool(registry, dataset_id, "describe").columns) == 6  # all numeric

    def test_group_aggregate(self, registry, dataset_id, sales):
        args = {"group_by": "category", "agg": "sum", "value_column": "revenue"}
        result = run_tool(registry, dataset_id, "group_aggregate", args)
        assert isinstance(result, GroupAggregateResult)
        assert result == analysis.group_aggregate(sales, "category", "sum", "revenue")

    def test_group_aggregate_options(self, registry, dataset_id):
        args = {"group_by": ["region", "category"], "agg": "count", "sort": "asc", "limit": 3}
        result = run_tool(registry, dataset_id, "group_aggregate", args)
        assert len(result.rows) == 3 and result.truncated

    def test_correlate(self, registry, dataset_id, sales):
        result = run_tool(registry, dataset_id, "correlate",
                          {"columns": ["discount", "profit"], "method": "spearman"})
        assert isinstance(result, CorrelationResult)
        assert result == analysis.correlate(sales, ["discount", "profit"], method="spearman")

    def test_detect_anomalies(self, registry, dataset_id, sales):
        result = run_tool(registry, dataset_id, "detect_anomalies",
                          {"column": "revenue", "method": "zscore", "threshold": 3})
        assert isinstance(result, AnomalyResult)
        assert result == analysis.detect_anomalies(sales, "revenue", method="zscore", threshold=3.0)

    def test_make_chart(self, registry, dataset_id, sales):
        args = {"chart_type": "bar", "x": "region", "y": "profit", "agg": "mean"}
        result = run_tool(registry, dataset_id, "make_chart", args)
        assert isinstance(result, ChartResult)
        assert result == charts.make_chart(sales, "bar", "region", "profit", agg="mean")

    def test_query_sql(self, registry, dataset_id, sales):
        result = run_tool(registry, dataset_id, "query_sql",
                          {"sql": "SELECT COUNT(*) FROM sales_sample WHERE region = 'North'"})
        assert isinstance(result, SqlResult)
        assert result.rows == [[int((sales["region"] == "North").sum())]]

    def test_query_sql_default_row_cap(self, registry, dataset_id):
        result = run_tool(registry, dataset_id, "query_sql", {"sql": "SELECT * FROM sales_sample"})
        assert result.row_count == 100 and result.truncated
        result = run_tool(registry, dataset_id, "query_sql",
                          {"sql": "SELECT * FROM sales_sample", "max_rows": 500})
        assert result.row_count == 500 and not result.truncated

    def test_tool_tables_are_complete(self):
        assert set(_ARG_MODELS) == set(TOOL_NAMES)
        assert set(_FRAME_TOOLS) | {"query_sql"} == set(TOOL_NAMES)

    def test_registered_data_unchanged(self, registry, dataset_id, sales):
        calls = [
            ("describe", {}),
            ("group_aggregate", {"group_by": "region", "agg": "mean", "value_column": "profit"}),
            ("correlate", {}),
            ("detect_anomalies", {"column": "revenue"}),
            ("make_chart", {"chart_type": "scatter", "x": "discount", "y": "profit"}),
            ("make_chart", {"chart_type": "line", "x": "date", "y": "revenue"}),
            ("query_sql", {"sql": "SELECT * FROM sales_sample ORDER BY revenue"}),
        ]
        for tool, args in calls:
            run_tool(registry, dataset_id, tool, args)
        pd.testing.assert_frame_equal(registry.get_dataframe(dataset_id), sales)


# ── Invalid Requests ────────────────────────────────────────────────

class TestInvalidRequests:
    @pytest.mark.parametrize("tool", ["run_pandas", "exec", "", None, "DESCRIBE", "__import__"])
    def test_unknown_tool(self, registry, dataset_id, tool):
        with pytest.raises(CoreError, match="Unknown tool"):
            run_tool(registry, dataset_id, tool, {})

    @pytest.mark.parametrize("arguments", [["revenue"], "revenue", 5])
    def test_arguments_must_be_object(self, registry, dataset_id, arguments):
        with pytest.raises(CoreError, match="must be a JSON object"):
            run_tool(registry, dataset_id, "describe", arguments)

    def test_unknown_argument(self, registry, dataset_id):
        with pytest.raises(CoreError, match="path: Extra inputs are not permitted"):
            run_tool(registry, dataset_id, "describe", {"path": "../../secret.txt"})

    def test_missing_required_argument(self, registry, dataset_id):
        with pytest.raises(CoreError, match="Invalid arguments for group_aggregate: group_by"):
            run_tool(registry, dataset_id, "group_aggregate", {"agg": "count"})

    @pytest.mark.parametrize(
        "args",
        [
            {"group_by": "region", "agg": "count", "limit": "5"},
            {"group_by": "region", "agg": "count", "limit": True},
            {"group_by": "region", "agg": "count", "limit": 2.0},
            {"group_by": "region", "agg": "variance"},
            {"group_by": 3, "agg": "count"},
        ],
        ids=["string limit", "bool limit", "float limit", "bad agg", "non-string key"],
    )
    def test_strict_types(self, registry, dataset_id, args):
        with pytest.raises(CoreError, match="Invalid arguments"):
            run_tool(registry, dataset_id, "group_aggregate", args)

    def test_core_errors_propagate(self, registry, dataset_id):
        with pytest.raises(ColumnError, match="Unknown column"):
            run_tool(registry, dataset_id, "detect_anomalies", {"column": "nope"})
        with pytest.raises(ColumnError, match="not numeric"):
            run_tool(registry, dataset_id, "make_chart", {"chart_type": "histogram", "x": "product"})

    def test_sql_security_applies(self, registry, dataset_id):
        with pytest.raises(QueryError):
            run_tool(registry, dataset_id, "query_sql", {"sql": "DROP TABLE sales_sample"})
        with pytest.raises(QueryError):
            run_tool(registry, dataset_id, "query_sql", {"sql": "SELECT * FROM read_csv('../../secret.txt')"})

    @pytest.mark.parametrize(
        "bad_id",
        ["../../secret.txt", "C:\\Users\\secret.csv", "sales_sample.csv", "/etc/passwd", None],
    )
    def test_dataset_id_is_never_a_path(self, registry, bad_id):
        registry.register_file("sales_sample.csv")
        with pytest.raises(DatasetNotFoundError, match="Invalid dataset ID"):
            run_tool(registry, bad_id, "describe", {})

    def test_unknown_dataset_id(self, registry):
        with pytest.raises(DatasetNotFoundError, match="Unknown dataset ID"):
            run_tool(registry, "ds_0000000000000000", "query_sql", {"sql": "SELECT 1"})


# ── Serialization & ToolCall ────────────────────────────────────────

class TestSerialization:
    def test_result_json_omits_chart_image(self, registry, dataset_id):
        chart = run_tool(registry, dataset_id, "make_chart", {"chart_type": "bar", "x": "region"})
        payload = json.loads(result_json(chart))
        assert "image_base64" not in payload
        assert payload["data"] == chart.data

    def test_result_json_round_trips(self, registry, dataset_id):
        result = run_tool(registry, dataset_id, "group_aggregate",
                          {"group_by": "category", "agg": "mean", "value_column": "profit"})
        assert GroupAggregateResult.model_validate_json(result_json(result)) == result

    def test_tool_call_model(self):
        call = ToolCall.model_validate({"tool": "describe", "arguments": {"columns": ["a"]}})
        assert call.tool == "describe" and call.arguments == {"columns": ["a"]}
        assert ToolCall.model_validate({"tool": "correlate"}).arguments == {}

    @pytest.mark.parametrize(
        "payload",
        [{"tool": "run_pandas"}, {"tool": "describe", "extra": 1}, {"arguments": {}}, {"tool": "describe", "arguments": []}],
    )
    def test_tool_call_rejects_invalid(self, payload):
        with pytest.raises(ValidationError):
            ToolCall.model_validate(payload)


class TestParseToolCall:
    def test_valid(self):
        call = parse_tool_call({"tool": "correlate", "arguments": {"method": "spearman"}})
        assert (call.tool, call.arguments) == ("correlate", {"method": "spearman"})

    @pytest.mark.parametrize(
        "payload, message",
        [
            (["describe"], "must be a JSON object"),
            ({"tool": "shell"}, "Unknown tool: 'shell'"),
            ({"arguments": {}}, "Unknown tool: None"),
            ({"tool": "describe", "columns": ["a"]}, "columns: Extra inputs are not permitted"),
            ({"tool": "describe", "arguments": "a"}, "arguments: Input should be a valid dictionary"),
        ],
    )
    def test_invalid(self, payload, message):
        with pytest.raises(CoreError, match=message):
            parse_tool_call(payload)
