"""
Talks to the real server process over stdio with the official MCP client —
the same path Claude Desktop and other MCP hosts use — so the entry point
(`python -m mcp_server`), process start-up and the wire format are covered,
not just the in-process tool functions.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
CSV = "region,sales\nNorth,10\nSouth,30\nNorth,5\n"


def server_params() -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY",)}
    env.update({"PYTHONPATH": str(ROOT), "AGENTAI_MCP_ALLOW_ANALYZE_QUESTION": "false"})
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], env=env, cwd=str(ROOT))


def structured(result):
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def session_walkthrough() -> dict:
    out = {}
    async with Client(server_params(), read_timeout_seconds=120) as client:
        out["server"] = client.server_info.name if client.server_info else None
        out["tools"] = sorted(t.name for t in (await client.list_tools()).tools)

        info = structured(await client.call_tool("register_dataset", {"file_name": "s.csv", "csv_text": CSV}))
        out["sql"] = structured(await client.call_tool(
            "query_sql", {"sql": f"SELECT SUM(sales) AS total FROM {info['table_name']}"}))
        chart = await client.call_tool("make_chart", {
            "dataset_id": info["dataset_id"], "chart_type": "bar", "x": "region", "y": "sales"})
        out["chart_types"] = sorted({c.type for c in chart.content})
        blocked = await client.call_tool("run_pandas", {"dataset_id": info["dataset_id"], "code": "import os"})
        out["blocked"] = (blocked.is_error, blocked.content[0].text)
        schema = await client.read_resource(f"dataset://{info['dataset_id']}/schema")
        out["schema_has_sales"] = "sales" in schema.contents[0].text
    return out


def test_real_client_over_stdio():
    out = asyncio.run(session_walkthrough())
    assert out["server"] == "agentai-data-analyst"
    assert {"register_dataset", "query_sql", "make_chart", "run_pandas"} <= set(out["tools"])
    assert "analyze_question" not in out["tools"]  # disabled through the environment
    assert out["sql"]["rows"] == [[45]]
    assert out["chart_types"] == ["image", "text"]
    is_error, message = out["blocked"]
    assert is_error and "Blocked import" in message
    assert out["schema_has_sales"]
