"""
Tests for core/sql.py — read-only DuckDB queries over registered datasets.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import core.sql
from core.datasets import DatasetRegistry
from core.errors import QueryError
from core.sql import query_sql, validate_sql

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"


@pytest.fixture
def sales_df():
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def registry(sales_df):
    registry = DatasetRegistry()
    registry.register_dataframe(sales_df, "sales")
    registry.register_dataframe(
        pd.DataFrame({"region": ["North", "South", "East", "West"],
                      "manager": ["Ana", "Ben", "Cy", "Di"]}),
        "regions",
    )
    return registry


# ── Valid Queries ───────────────────────────────────────────────────

class TestValidQueries:
    def test_simple_select(self, registry):
        result = query_sql(registry, "SELECT product, revenue FROM sales LIMIT 3")
        assert result.columns == ["product", "revenue"]
        assert result.row_count == 3 and len(result.rows) == 3
        assert result.truncated is False

    def test_aggregation_matches_pandas(self, registry, sales_df):
        result = query_sql(
            registry,
            "SELECT category, SUM(revenue) AS total FROM sales GROUP BY category ORDER BY category",
        )
        expected = sales_df.groupby("category")["revenue"].sum()
        assert {r[0]: r[1] for r in result.rows} == pytest.approx(expected.to_dict())

    def test_filtering(self, registry, sales_df):
        result = query_sql(registry, "SELECT COUNT(*) FROM sales WHERE region = 'North' AND quantity > 10")
        expected = ((sales_df["region"] == "North") & (sales_df["quantity"] > 10)).sum()
        assert result.rows == [[int(expected)]]

    def test_join_between_datasets(self, registry, sales_df):
        result = query_sql(
            registry,
            "SELECT r.manager, COUNT(*) AS n FROM sales s JOIN regions r ON s.region = r.region "
            "GROUP BY r.manager ORDER BY r.manager",
        )
        assert [row[0] for row in result.rows] == ["Ana", "Ben", "Cy", "Di"]
        assert sum(row[1] for row in result.rows) == sales_df["region"].notna().sum()

    def test_cte_and_from_first_syntax(self, registry):
        cte = query_sql(registry, "WITH t AS (SELECT revenue FROM sales) SELECT COUNT(*) FROM t")
        from_first = query_sql(registry, "FROM sales SELECT COUNT(*)")
        assert cte.rows == from_first.rows == [[500]]

    def test_parenthesized_union(self, registry):
        result = query_sql(registry, "(SELECT 1 AS x) UNION ALL (SELECT 2)")
        assert sorted(r[0] for r in result.rows) == [1, 2]

    def test_comments_and_trailing_semicolon_allowed(self, registry):
        sql = "/* top */ SELECT COUNT(*) -- count rows\nFROM sales; -- DROP TABLE sales"
        assert query_sql(registry, sql).rows == [[500]]

    def test_semicolon_inside_string_is_one_statement(self, registry):
        assert query_sql(registry, "SELECT 'a;DROP TABLE sales' AS s").rows == [["a;DROP TABLE sales"]]

    def test_column_named_like_blocked_function(self):
        registry = DatasetRegistry()
        registry.register_dataframe(pd.DataFrame({"query": ["q1"], "read_csv": [1]}), "logs")
        assert query_sql(registry, "SELECT query, read_csv FROM logs").rows == [["q1", 1]]

    def test_max_rows_truncates(self, registry):
        result = query_sql(registry, "SELECT * FROM sales", max_rows=5)
        assert result.row_count == 5 and result.truncated is True

    def test_json_native_values(self, registry):
        result = query_sql(
            registry,
            "SELECT DATE '2024-01-02' AS d, 1.25::DECIMAL(5,2) AS dec, NULL AS n, SUM(quantity) AS q FROM sales",
        )
        assert result.rows[0][:3] == ["2024-01-02", 1.25, None]
        assert isinstance(result.rows[0][3], int)
        json.loads(result.model_dump_json())

    def test_engine_lockdown_is_active(self, registry):
        result = query_sql(registry, "SELECT current_setting('enable_external_access')")
        assert result.rows == [[False]]

    def test_no_temp_directory_or_allowed_directories(self, registry):
        # DuckDB would otherwise spill to ./.tmp and implicitly allow it
        result = query_sql(
            registry,
            "SELECT current_setting('temp_directory'), current_setting('allowed_directories')",
        )
        assert result.rows == [["", []]]


# ── Rejected Statements ─────────────────────────────────────────────

FORBIDDEN_STATEMENTS = {
    "insert": "INSERT INTO sales (quantity) VALUES (1)",
    "update": "UPDATE sales SET quantity = 0",
    "delete": "DELETE FROM sales",
    "drop": "DROP TABLE sales",
    "alter": "ALTER TABLE sales ADD COLUMN x INT",
    "create": "CREATE TABLE t AS SELECT * FROM sales",
    "create view": "CREATE VIEW v AS SELECT 1",
    "attach": "ATTACH 'other.db' AS other",
    "detach": "DETACH other",
    "copy to": "COPY sales TO 'out.csv'",
    "copy from": "COPY sales FROM 'data.csv'",
    "export": "EXPORT DATABASE 'dump'",
    "install": "INSTALL httpfs",
    "load": "LOAD httpfs",
    "set": "SET enable_external_access = true",
    "reset": "RESET threads",
    "pragma": "PRAGMA version",
    "call": "CALL pragma_version()",
    "show": "SHOW TABLES",
    "describe": "DESCRIBE sales",
    "summarize": "SUMMARIZE sales",
    "explain": "EXPLAIN SELECT 1",
    "checkpoint": "CHECKPOINT",
    "cte insert": "WITH x AS (SELECT 1) INSERT INTO sales (quantity) SELECT * FROM x",
    "lowercase": "delete from sales",
    "mixed case": "DeLeTe FrOm sales",
    "leading whitespace": "  \n\t DROP TABLE sales",
}

BYPASS_ATTEMPTS = {
    "second statement": "SELECT 1; DROP TABLE sales",
    "no space": "SELECT 1;DELETE FROM sales",
    "two selects": "SELECT 1; SELECT 2",
    "comment hides nothing": "SELECT 1; -- harmless?\nDELETE FROM sales",
    "block comment prefix": "/* SELECT */ DROP TABLE sales",
    "line comment prefix": "-- SELECT\nDELETE FROM sales",
    "comment between": "SELECT 1 /* ; */; DROP TABLE sales",
    "only a comment": "-- SELECT 1",
    "empty": "   ",
}


class TestRejectedStatements:
    @pytest.mark.parametrize("sql", FORBIDDEN_STATEMENTS.values(), ids=FORBIDDEN_STATEMENTS.keys())
    def test_forbidden_statement(self, registry, sql):
        with pytest.raises(QueryError):
            query_sql(registry, sql)

    @pytest.mark.parametrize("sql", BYPASS_ATTEMPTS.values(), ids=BYPASS_ATTEMPTS.keys())
    def test_bypass_attempt(self, registry, sql):
        with pytest.raises(QueryError):
            query_sql(registry, sql)

    def test_data_unchanged_after_rejections(self, registry):
        for sql in list(FORBIDDEN_STATEMENTS.values()) + list(BYPASS_ATTEMPTS.values()):
            with pytest.raises(QueryError):
                query_sql(registry, sql)
        assert query_sql(registry, "SELECT COUNT(*) FROM sales").rows == [[500]]

    def test_messages(self, registry):
        with pytest.raises(QueryError, match="Exactly one statement"):
            query_sql(registry, "SELECT 1; SELECT 2")
        with pytest.raises(QueryError, match="got DELETE"):
            query_sql(registry, "DELETE FROM sales")
        with pytest.raises(QueryError, match="start with SELECT"):
            query_sql(registry, "PRAGMA version")


# ── File-System & External Access ───────────────────────────────────

FILE_ACCESS = {
    "read_csv": "SELECT * FROM read_csv('requirements.txt')",
    "read_csv_auto": "SELECT * FROM read_csv_auto('requirements.txt')",
    "read_text": "SELECT * FROM read_text('requirements.txt')",
    "read_blob": "SELECT * FROM read_blob('requirements.txt')",
    "read_parquet": "SELECT * FROM read_parquet('data.parquet')",
    "read_json": "SELECT * FROM read_json('data.json')",
    "glob": "SELECT * FROM glob('*')",
    "sniff_csv": "SELECT * FROM sniff_csv('requirements.txt')",
    "quoted name": "SELECT * FROM \"read_csv\"('requirements.txt')",
    "schema-qualified": "SELECT * FROM main.read_text('requirements.txt')",
    "nested query()": "SELECT * FROM query('SELECT 1')",
    "getenv": "SELECT getenv('GROQ_API_KEY')",
    "string as table": "SELECT * FROM 'requirements.txt'",
    "url": "SELECT * FROM 'https://example.com/data.csv'",
    "in subquery": "SELECT * FROM sales WHERE product IN (SELECT * FROM read_csv('x.csv'))",
    "traversal read_csv": "SELECT * FROM read_csv('../../secret.txt')",
    "traversal string table": "SELECT * FROM '../../secret.csv'",
    "windows path": "SELECT * FROM 'C:\\Users\\secret.csv'",
    "traversal attach": "ATTACH '../../secret.db' AS s",
    "unregistered table": "SELECT * FROM not_registered",
    "absolute windows path forward slashes": "SELECT * FROM 'C:/Windows/win.ini'",
    "read_csv list of files": "SELECT * FROM read_csv(['requirements.txt'])",
    "comment before call paren": "SELECT * FROM read_csv /* x */ ('requirements.txt')",
    "lateral file read": "SELECT * FROM sales, LATERAL (SELECT * FROM read_csv('x.csv'))",
    "extension listing": "SELECT * FROM duckdb_extensions()",
    "checkpoint function": "SELECT * FROM checkpoint()",
}


class TestFileSystemAccess:
    @pytest.mark.parametrize("sql", FILE_ACCESS.values(), ids=FILE_ACCESS.keys())
    def test_file_and_external_access_blocked(self, registry, sql):
        with pytest.raises(QueryError):
            query_sql(registry, sql)

    def test_engine_blocks_files_even_without_denylist(self, registry, monkeypatch):
        # Defense in depth: with the static denylist removed, the locked-down
        # engine must still refuse file access.
        monkeypatch.setattr(core.sql, "BLOCKED_FUNCTIONS", frozenset())
        with pytest.raises(QueryError, match="File-system and network access are not allowed"):
            query_sql(registry, "SELECT * FROM read_csv('requirements.txt')")

    def test_engine_blocks_writes_via_nested_query(self, registry, monkeypatch, tmp_path):
        monkeypatch.setattr(core.sql, "BLOCKED_FUNCTIONS", frozenset())
        target = (tmp_path / "out.csv").as_posix()
        with pytest.raises(QueryError):
            query_sql(registry, f"SELECT * FROM query('COPY (SELECT 1) TO ''{target}''')")
        assert not (tmp_path / "out.csv").exists()

    def test_python_variables_not_reachable(self, registry):
        secret_frame = pd.DataFrame({"password": ["hunter2"]})  # noqa: F841
        with pytest.raises(QueryError, match="Available tables"):
            query_sql(registry, "SELECT * FROM secret_frame")

    def test_only_registered_tables_listed(self, registry):
        result = query_sql(registry, "SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1")
        assert [r[0] for r in result.rows] == ["regions", "sales"]


# ── Limits & Validation ─────────────────────────────────────────────

class TestLimits:
    def test_timeout(self, registry):
        with pytest.raises(QueryError, match="timed out"):
            query_sql(registry, "SELECT SUM(hash(i)) FROM range(100000000000) t(i)", timeout_s=0.3)

    def test_no_datasets_registered(self):
        with pytest.raises(QueryError, match="No datasets"):
            query_sql(DatasetRegistry(), "SELECT 1")

    def test_query_too_long(self, registry):
        with pytest.raises(QueryError, match="longer than"):
            query_sql(registry, "SELECT 1 " + " " * core.sql.MAX_SQL_CHARS)

    def test_nul_byte(self, registry):
        with pytest.raises(QueryError, match="NUL"):
            query_sql(registry, "SELECT 1\x00; DROP TABLE sales")

    def test_invalid_sql(self, registry):
        with pytest.raises(QueryError, match="Invalid SQL|SQL error"):
            query_sql(registry, "SELEC * FRM sales")

    @pytest.mark.parametrize("max_rows", [0, 10_001, True, 1.5])
    def test_invalid_max_rows(self, registry, max_rows):
        with pytest.raises(QueryError, match="max_rows"):
            query_sql(registry, "SELECT 1", max_rows=max_rows)

    @pytest.mark.parametrize("timeout_s", [0, -1, 61, True])
    def test_invalid_timeout(self, registry, timeout_s):
        with pytest.raises(QueryError, match="timeout_s"):
            query_sql(registry, "SELECT 1", timeout_s=timeout_s)

    def test_validate_sql_returns_query(self):
        assert validate_sql("SELECT 1") == "SELECT 1"


# ── Restricting Tables to Chosen Datasets ───────────────────────────

class TestDatasetIds:
    def test_only_chosen_datasets_are_tables(self, registry):
        sales_id = next(d.dataset_id for d in registry.list_datasets() if d.name == "sales")
        result = query_sql(registry, "SELECT COUNT(*) FROM sales", dataset_ids=[sales_id])
        assert result.rows == [[500]]
        with pytest.raises(QueryError, match="Available tables: sales$"):
            query_sql(registry, "SELECT * FROM regions", dataset_ids=[sales_id])

    def test_all_datasets_by_default(self, registry):
        result = query_sql(registry, "SELECT COUNT(*) FROM sales s JOIN regions r USING (region)")
        assert result.rows[0][0] > 0

    def test_unknown_dataset_id(self, registry):
        from core.errors import DatasetNotFoundError

        with pytest.raises(DatasetNotFoundError):
            query_sql(registry, "SELECT 1", dataset_ids=["ds_0000000000000000"])
        with pytest.raises(DatasetNotFoundError, match="Invalid dataset ID"):
            query_sql(registry, "SELECT 1", dataset_ids=["../../secret.csv"])

    @pytest.mark.parametrize("dataset_ids", ["ds_0000000000000000", [1]])
    def test_dataset_ids_must_be_a_list_of_strings(self, registry, dataset_ids):
        with pytest.raises(QueryError, match="dataset_ids must be a list"):
            query_sql(registry, "SELECT 1", dataset_ids=dataset_ids)
