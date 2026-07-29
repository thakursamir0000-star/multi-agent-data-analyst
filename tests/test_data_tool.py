"""
Tests for tools/data_tool.py — CSV/Excel loading and profiling.
"""

import io
import pytest
import pandas as pd

from tools.data_tool import DataTool


# ── Fixtures ────────────────────────────────────────────────────────

SAMPLE_CSV = """\
name,age,salary,city
Alice,30,75000.50,New York
Bob,25,,London
Charlie,35,92000.00,
Diana,28,68000.25,Paris
Eve,,81000.00,Tokyo
"""


@pytest.fixture
def csv_buffer():
    """Create an in-memory CSV file-like object."""
    buf = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    buf.name = "test.csv"
    return buf


@pytest.fixture
def sample_df():
    """Create a sample DataFrame directly."""
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


# ── Load Tests ──────────────────────────────────────────────────────

class TestLoad:
    def test_load_csv_from_buffer(self, csv_buffer):
        df = DataTool.load(csv_buffer, file_name="test.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["name", "age", "salary", "city"]

    def test_load_csv_from_file(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
        df = DataTool.load(str(csv_file))
        assert len(df) == 5

    def test_load_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            DataTool.load("data.json")

    def test_load_buffer_without_name_raises(self):
        buf = io.BytesIO(b"a,b\n1,2\n")
        with pytest.raises(ValueError, match="Cannot determine file type"):
            DataTool.load(buf)

    def test_load_buffer_with_filename_param(self):
        buf = io.BytesIO(b"a,b\n1,2\n")
        df = DataTool.load(buf, file_name="data.csv")
        assert len(df) == 1


# ── Profile Tests ───────────────────────────────────────────────────

class TestProfile:
    def test_profile_row_count(self, sample_df):
        profile = DataTool.profile(sample_df)
        assert profile["_rows"] == 5
        assert profile["_columns"] == 4

    def test_profile_null_counts(self, sample_df):
        profile = DataTool.profile(sample_df)
        # 'age' has 1 null (Eve), 'salary' has 1 null (Bob), 'city' has 1 null (Charlie)
        assert profile["age"]["null_count"] == 1
        assert profile["salary"]["null_count"] == 1
        assert profile["city"]["null_count"] == 1
        assert profile["name"]["null_count"] == 0

    def test_profile_dtypes(self, sample_df):
        profile = DataTool.profile(sample_df)
        assert "float" in profile["age"]["dtype"] or "int" in profile["age"]["dtype"]
        assert "float" in profile["salary"]["dtype"]
        assert profile["name"]["dtype"] == "object"

    def test_profile_cardinality(self, sample_df):
        profile = DataTool.profile(sample_df)
        assert profile["name"]["cardinality"] == 5  # all unique
        assert profile["city"]["cardinality"] == 4  # 4 unique (1 null excluded)

    def test_profile_numeric_stats(self, sample_df):
        profile = DataTool.profile(sample_df)
        assert "stats" in profile["salary"]
        stats = profile["salary"]["stats"]
        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["mean"] is not None

    def test_profile_sample_values(self, sample_df):
        profile = DataTool.profile(sample_df)
        samples = profile["name"]["sample_values"]
        assert len(samples) <= DataTool.MAX_SAMPLE_VALUES
        assert all(isinstance(s, str) for s in samples)

    def test_profile_null_pct(self, sample_df):
        profile = DataTool.profile(sample_df)
        assert profile["age"]["null_pct"] == 20.0  # 1/5 = 20%


# ── Profile to Text Tests ──────────────────────────────────────────

class TestProfileToText:
    def test_to_text_contains_dimensions(self, sample_df):
        profile = DataTool.profile(sample_df)
        text = DataTool.profile_to_text(profile)
        assert "5 rows" in text
        assert "4 columns" in text

    def test_to_text_contains_column_names(self, sample_df):
        profile = DataTool.profile(sample_df)
        text = DataTool.profile_to_text(profile)
        for col in ["name", "age", "salary", "city"]:
            assert col in text

    def test_to_text_contains_stats(self, sample_df):
        profile = DataTool.profile(sample_df)
        text = DataTool.profile_to_text(profile)
        assert "range=" in text  # Numeric columns should have range


# ── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        profile = DataTool.profile(df)
        assert profile["_rows"] == 0
        assert profile["_columns"] == 0

    def test_single_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        profile = DataTool.profile(df)
        assert profile["_columns"] == 1
        assert profile["x"]["cardinality"] == 3

    def test_all_nulls_column(self):
        df = pd.DataFrame({"x": [None, None, None]})
        profile = DataTool.profile(df)
        assert profile["x"]["null_count"] == 3
        assert profile["x"]["null_pct"] == 100.0
