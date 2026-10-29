"""
End-to-end tests for the MCP server: tools are called through MCPServer itself,
exactly as a client's requests would reach them.
"""

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.context import ServerConfig
from mcp_server.server import create_server, main

CSV = "region,product,sales\nNorth,A,10\nSouth,B,30\nNorth,B,5\nEast,A,25\n"


def call(server, name, **arguments):
    return asyncio.run(server.call_tool(name, arguments))


def tool_names(server):
    return {t.name for t in asyncio.run(server.list_tools())}


@pytest.fixture
def local():
    return create_server(ServerConfig.for_stdio(allow_analyze_question=False))


@pytest.fixture
def dataset(local):
    info = call(local, "register_dataset", file_name="sales.csv", csv_text=CSV).structured_content
    return info["dataset_id"], info["table_name"]


def test_stdio_exposes_code_tools_and_http_does_not_by_default(monkeypatch):
    for var in ("AGENTAI_MCP_ALLOW_RUN_PANDAS", "AGENTAI_MCP_ALLOW_ANALYZE_QUESTION"):
        monkeypatch.delenv(var, raising=False)
    stdio = tool_names(create_server(ServerConfig.for_stdio()))
    http = tool_names(create_server(ServerConfig.for_http()))
    assert {"run_pandas", "analyze_question", "query_sql", "group_aggregate"} <= stdio
    assert "query_sql" in http and "run_pandas" not in http and "analyze_question" not in http


def test_prompts_and_resources_are_registered(local):
    assert {p.name for p in asyncio.run(local.list_prompts())} >= {"analyze_dataset", "data_quality_report"}
    templates = {t.uri_template for t in asyncio.run(local.list_resource_templates())}
    assert "dataset://{dataset_id}/schema{?session}" in templates


def test_deterministic_tools_return_exact_numbers(local, dataset):
    ds, table = dataset
    grouped = call(local, "group_aggregate", dataset_id=ds, group_by="region",
                   aggregation="sum", value_column="sales").structured_content
    assert {r["region"]: r["sum_sales"] for r in grouped["rows"]} == {"South": 30, "East": 25, "North": 15}
    total = call(local, "query_sql", sql=f"SELECT SUM(sales) AS total FROM {table}").structured_content
    assert total["rows"] == [[70]]


@pytest.mark.parametrize("sql", [
    "COPY (SELECT 1) TO 'out.csv'",
    "SELECT * FROM read_csv('/etc/passwd')",
    "DROP TABLE sales",
])
def test_sql_refuses_writes_and_file_access(local, dataset, sql):
    with pytest.raises(ToolError):
        call(local, "query_sql", sql=sql)


def test_run_pandas_uses_the_sandbox(local, dataset):
    ds, _ = dataset
    ok = call(local, "run_pandas", dataset_id=ds, code="print(df['sales'].mean())")
    assert ok.content[0].text.strip() == "17.5" and not ok.is_error
    with pytest.raises(ToolError, match="Blocked import"):
        call(local, "run_pandas", dataset_id=ds, code="import os")


def test_dataset_schema_resource(local, dataset):
    ds, _ = dataset
    contents = list(asyncio.run(local.read_resource(f"dataset://{ds}/schema")))
    assert "sales" in contents[0].content


def test_http_requires_and_isolates_sessions():
    server = create_server(ServerConfig.for_http())
    with pytest.raises(ToolError, match="needs a session"):
        call(server, "list_datasets")

    alice = call(server, "open_session").structured_content["session"]
    bob = call(server, "open_session").structured_content["session"]
    ds = call(server, "register_dataset", file_name="a.csv", csv_text=CSV, session=alice).structured_content["dataset_id"]

    assert [d["dataset_id"] for d in call(server, "list_datasets", session=alice).structured_content["datasets"]] == [ds]
    assert call(server, "list_datasets", session=bob).structured_content["datasets"] == []
    with pytest.raises(ToolError):
        call(server, "describe", dataset_id=ds, session=bob)
    with pytest.raises(ToolError, match="Invalid session"):
        call(server, "list_datasets", session="not-a-token")


@pytest.mark.parametrize("sql", [
    "SELECT current_setting('TimeZone') AS v",
    "SELECT * FROM duckdb_settings()",
    "SELECT version() AS v",
    "SELECT * FROM pragma_platform()",
    "SELECT current_user AS u",
    "SELECT now() AS t",
    'SELECT "VERSION"() AS v',
])
def test_http_sql_cannot_describe_the_server(sql):
    server = create_server(ServerConfig.for_http())
    session = call(server, "open_session").structured_content["session"]
    call(server, "register_dataset", file_name="a.csv", csv_text=CSV, session=session)
    with pytest.raises(ToolError, match="not allowed on a remote server"):
        call(server, "query_sql", sql=sql, session=session)


def test_http_sql_still_answers_data_questions():
    server = create_server(ServerConfig.for_http())
    session = call(server, "open_session").structured_content["session"]
    table = call(server, "register_dataset", file_name="sales.csv", csv_text=CSV,
                 session=session).structured_content["table_name"]
    sql = (f'WITH t(r, s) AS (SELECT region, sales FROM {table}) '
           f'SELECT r, ROUND(AVG(s), 1) AS avg_sales, COUNT(*) AS n FROM t GROUP BY r ORDER BY r')
    rows = call(server, "query_sql", sql=sql, session=session).structured_content["rows"]
    assert rows == [["East", 25.0, 1], ["North", 7.5, 2], ["South", 30.0, 1]]


def test_local_sql_policy_is_unchanged(local, dataset):
    assert call(local, "query_sql", sql="SELECT version() AS v").structured_content["rows"]


def test_two_servers_never_share_datasets():
    first = create_server(ServerConfig.for_stdio(allow_analyze_question=False))
    second = create_server(ServerConfig.for_stdio(allow_analyze_question=False))
    call(first, "register_dataset", file_name="a.csv", csv_text=CSV)
    assert call(second, "list_datasets").structured_content["datasets"] == []


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--http" in capsys.readouterr().out
