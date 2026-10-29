"""
Smoke-test a running MCP server over Streamable HTTP with the official client.

    python -m mcp_server.smoke http://127.0.0.1:8000/mcp

Waits for the server, then checks: tools are listed with code execution off,
calls without a session are refused, and a session can upload data and run
read-only SQL. Exits non-zero on any failure (used by CI against the Docker
image).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request

from mcp import Client

CSV = "region,sales\nNorth,10\nSouth,30\nNorth,5\n"


def wait_until_up(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)  # any HTTP answer means the server is listening
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            if time.monotonic() > deadline:
                raise SystemExit(f"Server at {url} did not start within {timeout:.0f}s")
            time.sleep(1)


def _data(result):
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def check(url: str) -> list[str]:
    report = []
    async with Client(url, read_timeout_seconds=60) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"open_session", "register_dataset", "query_sql"} <= tools, tools
        assert not {"run_pandas", "analyze_question"} & tools, "code execution must be off over HTTP by default"
        report.append(f"{len(tools)} tools, code execution off")

        refused = await client.call_tool("list_datasets", {})
        assert refused.is_error and "session" in refused.content[0].text, refused
        report.append("calls without a session are refused")

        session = _data(await client.call_tool("open_session", {}))["session"]
        info = _data(await client.call_tool("register_dataset",
                                            {"file_name": "s.csv", "csv_text": CSV, "session": session}))
        rows = _data(await client.call_tool("query_sql", {
            "sql": f"SELECT SUM(sales) AS total FROM {info['table_name']}", "session": session}))["rows"]
        assert rows == [[45]], rows
        leak = await client.call_tool("query_sql", {"sql": "SELECT version() AS v", "session": session})
        assert leak.is_error, "remote SQL policy must refuse server introspection"
        report.append("session upload + SQL work; server introspection refused")
    return report


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"
    wait_until_up(url)
    for line in asyncio.run(check(url)):
        print("ok:", line)


if __name__ == "__main__":
    main()
