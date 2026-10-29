"""
query_sql — read-only analytical SQL over registered datasets (DuckDB).

Layered defenses:

1. Static validation, before anything executes:
   - size limit and no NUL bytes;
   - exactly one statement, counted by DuckDB's own parser, so comments,
     string literals and whitespace cannot hide a second statement;
   - the statement type must be SELECT *and* its first keyword must be
     SELECT / WITH / FROM / VALUES / "(". This rejects INSERT, UPDATE,
     DELETE, DROP, ALTER, CREATE, ATTACH, COPY, EXPORT, INSTALL, LOAD, SET,
     CALL, ... as well as PRAGMA / SHOW / DESCRIBE / SUMMARIZE, which DuckDB
     rewrites into SELECTs;
   - identifiers naming file, network or nested-query functions
     (read_csv, glob, query, ...) are rejected.
2. Engine lockdown: a fresh in-memory database per query with external
   access disabled (no files, network or extension install/load), no
   temp directory (no spill files, no implicitly allowed directory), Python
   replacement scans disabled (no reaching Python variables by name), and
   configuration locked so the query cannot re-enable anything.
3. Only registered datasets exist in that database, as views over the
   registry's DataFrames.
4. A row cap and a timeout.

Phase 2 requirement — remote SQL hardening (not implemented yet):
This policy is sufficient for the local app, where the person querying
also runs the server. A remote MCP server must additionally prevent
information disclosure through environment/path introspection, which
these checks still allow, e.g.:
  - current_setting(...) and duckdb_settings() — settings such as
    allowed_paths reveal the server's working directory; TimeZone etc.;
  - version(), pragma_version(), pragma_platform(), pragma_user_agent()
    — software versions and OS;
  - other duckdb_*() / pragma_*() metadata functions.
Prefer a default-deny allowlist of analytical functions for the remote
policy over extending BLOCKED_FUNCTIONS. There is no policy object yet: the
denylist check in validate_sql() and _ENGINE_CONFIG are the places a
stricter, selectable policy would plug in.
"""

from __future__ import annotations

import re
import threading
from typing import Sequence

import duckdb

from core.datasets import DatasetRegistry
from core.errors import QueryError
from core.schemas import SqlResult, to_json_value

MAX_SQL_CHARS = 10_000
MAX_RESULT_ROWS = 10_000
MAX_TIMEOUT_SECONDS = 60.0

_ALLOWED_FIRST_KEYWORDS = frozenset({"select", "with", "from", "values"})

# Table/scalar functions that read files or URLs, run nested SQL, or expose
# secrets. Most are also blocked by the engine lockdown; this is defense in
# depth and gives a clearer error.
BLOCKED_FUNCTIONS = frozenset({
    "query", "query_table",
    "read_csv", "read_csv_auto", "sniff_csv", "read_text", "read_blob",
    "read_json", "read_json_auto", "read_json_objects", "read_json_objects_auto",
    "read_ndjson", "read_ndjson_auto", "read_ndjson_objects",
    "read_parquet", "parquet_scan", "parquet_metadata", "parquet_schema",
    "parquet_file_metadata", "parquet_kv_metadata", "read_xlsx",
    "glob", "iceberg_scan", "delta_scan", "st_read",
    "sqlite_scan", "sqlite_attach", "postgres_scan", "postgres_attach",
    "mysql_scan", "http_get", "http_post",
    "getenv", "load_aws_credentials", "duckdb_secrets", "which_secret",
    "checkpoint", "force_checkpoint",
})

_ENGINE_CONFIG = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "allow_community_extensions": False,
    "python_enable_replacements": False,
    # No spill-to-disk: otherwise DuckDB writes to ./.tmp and adds that
    # directory to allowed_directories even with external access disabled.
    "temp_directory": "",
    "threads": 2,
    "memory_limit": "512MB",
}

# A quoted ("...") or bare identifier/keyword starting at a token offset
_WORD_RE = re.compile(r'"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_$]*')


def validate_sql(sql: str) -> str:
    """Return the query if it is a single read-only SELECT; else raise QueryError."""
    if not isinstance(sql, str) or not sql.strip():
        raise QueryError("The query is empty")
    if len(sql) > MAX_SQL_CHARS:
        raise QueryError(f"The query is longer than {MAX_SQL_CHARS} characters")
    if "\x00" in sql:
        raise QueryError("The query contains a NUL byte")

    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise QueryError(f"Invalid SQL: {_first_line(e)}") from None
    if len(statements) != 1:
        raise QueryError(
            f"Exactly one statement is allowed (found {len(statements)})"
        )
    statement_type = statements[0].type.name
    if statement_type != "SELECT":
        raise QueryError(
            f"Only read-only SELECT queries are allowed (got {statement_type})"
        )

    tokens = [
        (offset, token_type)
        for offset, token_type in duckdb.tokenize(sql)
        if token_type != duckdb.token_type.comment
    ]
    first = sql[tokens[0][0]:] if tokens else ""
    first_match = _WORD_RE.match(first)
    if not first.startswith("(") and not (
        first_match and first_match.group(0).lower() in _ALLOWED_FIRST_KEYWORDS
    ):
        raise QueryError(
            "Only SELECT queries are allowed: start with SELECT, WITH, FROM or VALUES"
        )

    # A blocked name only matters when called, i.e. followed by "("
    for i, (offset, token_type) in enumerate(tokens[:-1]):
        if token_type not in (duckdb.token_type.identifier, duckdb.token_type.keyword):
            continue
        match = _WORD_RE.match(sql, offset)
        if not match or not sql.startswith("(", tokens[i + 1][0]):
            continue
        word = match.group(0)
        name = word[1:-1].replace('""', '"') if word.startswith('"') else word
        if name.lower() in BLOCKED_FUNCTIONS:
            raise QueryError(
                f"'{name}' is not allowed: queries may only read registered datasets"
            )
    return sql


def query_sql(
    registry: DatasetRegistry,
    sql: str,
    max_rows: int = 1000,
    timeout_s: float = 10.0,
    dataset_ids: Sequence[str] | None = None,
) -> SqlResult:
    """Run one read-only SELECT against the registered datasets.

    Each dataset is available as a table named ``DatasetInfo.table_name``.
    With ``dataset_ids``, only those datasets exist as tables.
    """
    if dataset_ids is not None and (
        isinstance(dataset_ids, str) or not all(isinstance(d, str) for d in dataset_ids)
    ):
        raise QueryError("dataset_ids must be a list of dataset IDs")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= MAX_RESULT_ROWS:
        raise QueryError(f"max_rows must be an integer between 1 and {MAX_RESULT_ROWS}")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= MAX_TIMEOUT_SECONDS:
        raise QueryError(f"timeout_s must be between 0 and {MAX_TIMEOUT_SECONDS} seconds")
    sql = validate_sql(sql)

    tables = registry._tables(dataset_ids)  # unknown IDs raise DatasetNotFoundError
    if not tables:
        raise QueryError("No datasets are registered")

    con = duckdb.connect(":memory:", config=_ENGINE_CONFIG)
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.daemon = True
    try:
        for table_name, df in tables.items():
            con.register(table_name, df)
        con.execute("SET lock_configuration = true")

        timer.start()
        cursor = con.execute(sql)
        columns = [str(d[0]) for d in cursor.description or []]
        fetched = cursor.fetchmany(max_rows + 1)
    except duckdb.PermissionException:
        raise QueryError(
            "File-system and network access are not allowed; query registered datasets only"
        ) from None
    except duckdb.InterruptException:
        raise QueryError(f"The query timed out after {timeout_s} seconds") from None
    except duckdb.CatalogException as e:
        raise QueryError(
            f"{_first_line(e)} Available tables: {', '.join(sorted(tables))}"
        ) from None
    except duckdb.Error as e:
        raise QueryError(f"SQL error: {_first_line(e)}") from None
    finally:
        timer.cancel()
        con.close()

    rows = [[to_json_value(v) for v in row] for row in fetched[:max_rows]]
    return SqlResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=len(fetched) > max_rows,
    )


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0][:300] if str(error).strip() else type(error).__name__
