"""
Tests for core/datasets.py — the dataset registry.
"""

import io
import os
from pathlib import Path

import pandas as pd
import pytest

from core.datasets import DatasetRegistry
from core.errors import CoreError, DatasetError, DatasetNotFoundError

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

CSV_BYTES = b"name,age,score,active\nAlice,30,1.5,true\nBob,,2.5,false\nCara,41,,true\n"


@pytest.fixture
def data_dir(tmp_path):
    """An approved data directory with one CSV, plus a secret file outside it."""
    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / "people.csv").write_bytes(CSV_BYTES)
    (approved / "notes.txt").write_text("not a dataset")
    (tmp_path / "secret.csv").write_bytes(b"password\nhunter2\n")
    return approved


@pytest.fixture
def registry(data_dir):
    return DatasetRegistry(data_dir=data_dir)


# ── Registration ────────────────────────────────────────────────────

class TestRegistration:
    def test_register_file(self, registry):
        info = registry.register_file("people.csv")
        assert info.dataset_id.startswith("ds_") and len(info.dataset_id) == 19
        assert info.name == "people.csv"
        assert info.source == "file"
        assert info.rows == 3
        assert info.table_name == "people"

    def test_register_bundled_sample(self):
        registry = DatasetRegistry(data_dir=SAMPLE_DIR)
        assert "sales_sample.csv" in registry.available_files()
        info = registry.register_file("sales_sample.csv")
        assert info.rows == 500
        assert info.table_name == "sales_sample"

    def test_register_upload_bytes(self):
        info = DatasetRegistry().register_upload(CSV_BYTES, "people.csv")
        assert info.source == "upload"
        assert info.rows == 3

    def test_register_upload_file_like(self):
        buf = io.BytesIO(CSV_BYTES)
        buf.read()  # position at end: registry must rewind
        info = DatasetRegistry().register_upload(buf, "people.csv")
        assert info.rows == 3

    def test_upload_name_is_only_a_label(self, data_dir):
        # A path-like upload name must not be opened; only the bytes are used
        secret = str(data_dir.parent / "secret.csv")
        info = DatasetRegistry().register_upload(CSV_BYTES, secret)
        assert info.name == "secret.csv"
        assert info.rows == 3  # people data, not the secret file (1 row)

    def test_upload_name_with_spaces_and_brackets(self):
        info = DatasetRegistry().register_upload(CSV_BYTES, "Sales Report (2024).csv")
        assert info.name == "Sales Report (2024).csv"
        assert info.table_name == "sales_report_2024"

    def test_register_dataframe_stores_copy(self):
        df = pd.DataFrame({"a": [1, 2]})
        registry = DatasetRegistry()
        info = registry.register_dataframe(df, "frame")
        df.loc[0, "a"] = 99
        assert registry.get_dataframe(info.dataset_id)["a"].tolist() == [1, 2]

    def test_stable_id_and_idempotent(self):
        registry = DatasetRegistry()
        first = registry.register_upload(CSV_BYTES, "people.csv")
        second = registry.register_upload(CSV_BYTES, "people.csv")
        assert first.dataset_id == second.dataset_id
        assert len(registry.list_datasets()) == 1
        # Same name + content gives the same ID in a new registry too
        assert DatasetRegistry().register_upload(CSV_BYTES, "people.csv").dataset_id == first.dataset_id

    def test_different_content_different_id(self):
        registry = DatasetRegistry()
        a = registry.register_upload(CSV_BYTES, "people.csv")
        b = registry.register_upload(CSV_BYTES + b"Dan,50,3.5,false\n", "people.csv")
        assert a.dataset_id != b.dataset_id

    @pytest.mark.parametrize("name", ["data.json", "data.txt", "data.py", "noext"])
    def test_unsupported_upload_type(self, name):
        with pytest.raises(DatasetError, match="Unsupported file type"):
            DatasetRegistry().register_upload(CSV_BYTES, name)

    def test_empty_upload(self):
        with pytest.raises(DatasetError, match="empty"):
            DatasetRegistry().register_upload(b"", "empty.csv")

    def test_unreadable_upload(self):
        with pytest.raises(DatasetError, match="Could not read"):
            DatasetRegistry().register_upload(b"\x00\x01not a spreadsheet", "bad.xlsx")

    def test_upload_too_large(self):
        with pytest.raises(DatasetError, match="larger than"):
            DatasetRegistry(max_bytes=10).register_upload(CSV_BYTES, "people.csv")

    def test_file_too_large(self, data_dir):
        with pytest.raises(DatasetError, match="larger than"):
            DatasetRegistry(data_dir=data_dir, max_bytes=10).register_file("people.csv")


# ── Lookup ──────────────────────────────────────────────────────────

class TestLookup:
    def test_get_and_list(self, registry):
        info = registry.register_file("people.csv")
        assert registry.get(info.dataset_id) == info
        assert registry.list_datasets() == [info]

    def test_get_dataframe_returns_copy(self, registry):
        info = registry.register_file("people.csv")
        df = registry.get_dataframe(info.dataset_id)
        df["age"] = 0
        assert registry.get_dataframe(info.dataset_id)["age"].isna().sum() == 1

    def test_unknown_dataset_id(self, registry):
        with pytest.raises(DatasetNotFoundError, match="Unknown dataset ID"):
            registry.get("ds_0000000000000000")

    @pytest.mark.parametrize(
        "bad_id",
        ["../../secret.txt", "C:\\Users\\secret.csv", "/etc/passwd", "people.csv",
         "", "ds_", "DS_0000000000000000", "ds_000000000000000g", None, 123],
    )
    def test_invalid_dataset_ids(self, registry, bad_id):
        with pytest.raises(DatasetNotFoundError, match="Invalid dataset ID"):
            registry.get_dataframe(bad_id)

    def test_errors_are_core_errors(self, registry):
        with pytest.raises(CoreError):
            registry.get("nope")

    def test_remove(self, registry):
        info = registry.register_file("people.csv")
        registry.remove(info.dataset_id)
        assert registry.list_datasets() == []
        with pytest.raises(DatasetNotFoundError):
            registry.get(info.dataset_id)


# ── Metadata & Schema ───────────────────────────────────────────────

class TestMetadata:
    def test_schema(self, registry):
        info = registry.register_file("people.csv")
        columns = {c.name: c for c in registry.schema(info.dataset_id)}
        assert list(columns) == ["name", "age", "score", "active"]
        assert columns["name"].kind == "text"
        assert columns["age"].kind == "numeric"
        assert columns["age"].null_count == 1
        assert columns["score"].unique_count == 2
        assert columns["active"].kind == "boolean"

    def test_datetime_kind(self):
        df = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        info = DatasetRegistry().register_dataframe(df, "dates")
        assert info.columns[0].kind == "datetime"

    def test_profile_reuses_data_tool(self, registry):
        info = registry.register_file("people.csv")
        profile = registry.get_profile(info.dataset_id)
        assert profile["_rows"] == 3
        profile["_rows"] = 99  # a deep copy: registry unaffected
        assert registry.get_profile(info.dataset_id)["_rows"] == 3
        assert "3 rows" in registry.get_profile_text(info.dataset_id)

    def test_metadata_is_json_serializable(self, registry):
        info = registry.register_file("people.csv")
        assert '"dataset_id"' in info.model_dump_json()


# ── Multiple Datasets ───────────────────────────────────────────────

class TestMultipleDatasets:
    def test_multiple_registered_in_order(self, registry):
        a = registry.register_file("people.csv")
        b = registry.register_dataframe(pd.DataFrame({"x": [1]}), "other")
        assert [d.dataset_id for d in registry.list_datasets()] == [a.dataset_id, b.dataset_id]
        assert registry.get_dataframe(b.dataset_id)["x"].tolist() == [1]

    def test_table_names_are_unique(self):
        registry = DatasetRegistry()
        a = registry.register_upload(CSV_BYTES, "sales.csv")
        b = registry.register_upload(CSV_BYTES + b"Dan,50,3.5,false\n", "sales.csv")
        assert (a.table_name, b.table_name) == ("sales", "sales_2")

    @pytest.mark.parametrize(
        "name, table",
        [("2024 Sales (Q1).csv", "t_2024_sales_q1"), ("!!!.csv", "dataset"), ("Mixed-Case.CSV", "mixed_case")],
    )
    def test_table_name_sanitized(self, name, table):
        info = DatasetRegistry().register_upload(CSV_BYTES, name)
        assert info.table_name == table


# ── Path Traversal & Unauthorized Files ─────────────────────────────

class TestPathSecurity:
    @pytest.mark.parametrize(
        "file_name",
        [
            "../secret.csv",
            "..\\secret.csv",
            "../../secret.txt",
            "approved/../secret.csv",
            "sub/people.csv",
            "/etc/passwd",
            "C:\\Users\\secret.csv",
            "C:secret.csv",
            "\\\\server\\share\\x.csv",
            "people.csv:stream",
            ".env",
            ".hidden.csv",
            "people.csv\x00.txt",
            "",
            " people.csv",
        ],
    )
    def test_traversal_and_paths_rejected(self, registry, file_name):
        with pytest.raises(DatasetError):
            registry.register_file(file_name)
        assert registry.list_datasets() == []

    def test_absolute_path_to_approved_file_rejected(self, registry, data_dir):
        with pytest.raises(DatasetError, match="Invalid dataset file name"):
            registry.register_file(str(data_dir / "people.csv"))

    def test_file_outside_directory_not_loaded(self, registry):
        with pytest.raises(DatasetError):
            registry.register_file("secret.csv")  # exists one level up only

    def test_unsupported_file_in_directory(self, registry):
        with pytest.raises(DatasetError, match="Unsupported file type"):
            registry.register_file("notes.txt")

    def test_directory_named_like_csv(self, registry, data_dir):
        (data_dir / "folder.csv").mkdir()
        with pytest.raises(DatasetError, match="not a file"):
            registry.register_file("folder.csv")

    def test_symlink_escaping_directory(self, registry, data_dir):
        link = data_dir / "linked.csv"
        try:
            os.symlink(data_dir.parent / "secret.csv", link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this system")
        with pytest.raises(DatasetError, match="not a file in the approved"):
            registry.register_file("linked.csv")

    def test_no_data_dir_configured(self):
        with pytest.raises(DatasetError, match="No approved data directory"):
            DatasetRegistry().register_file("people.csv")

    def test_missing_data_dir(self, tmp_path):
        with pytest.raises(DatasetError, match="does not exist"):
            DatasetRegistry(data_dir=tmp_path / "missing")

    def test_available_files_only_lists_datasets(self, registry):
        assert registry.available_files() == ["people.csv"]


# ── Typed Profile ───────────────────────────────────────────────────

class TestTypedProfile:
    def test_profile_matches_data_tool(self, registry):
        info = registry.register_file("people.csv")
        profile = registry.profile(info.dataset_id)
        columns = {c.name: c for c in profile.columns}
        assert (profile.dataset_id, profile.name, profile.rows) == (info.dataset_id, "people.csv", 3)
        assert list(columns) == ["name", "age", "score", "active"]
        assert columns["age"].stats.min == 30.0 and columns["age"].stats.max == 41.0
        assert columns["age"].null_count == 1 and columns["age"].null_pct == pytest.approx(33.33)
        assert columns["name"].stats is None and columns["name"].sample_values == ["Alice", "Bob", "Cara"]
        assert columns["active"].kind == "boolean"
        assert profile.summary_text == registry.get_profile_text(info.dataset_id)

    def test_profile_is_json_native(self, registry):
        df = pd.DataFrame({
            "when": pd.to_datetime(["2024-01-01", None]),
            "flag": [True, False],
            "x": [1.5, float("nan")],
        })
        info = registry.register_dataframe(df, "mixed")
        payload = registry.profile(info.dataset_id).model_dump(mode="json")
        columns = {c["name"]: c for c in payload["columns"]}
        assert columns["when"]["sample_values"] == ["2024-01-01T00:00:00"]
        # Mirrors DataTool: pandas describe() has no mean for booleans, median() does
        assert columns["flag"]["stats"]["median"] == 0.5
        assert columns["x"]["stats"]["std"] is None  # one value: undefined

    def test_profile_unknown_id(self, registry):
        with pytest.raises(DatasetNotFoundError):
            registry.profile("../../secret.txt")
