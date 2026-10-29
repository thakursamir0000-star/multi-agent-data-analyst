"""
The Agentai MCP server: dataset tools, read-only SQL, charts, sandboxed pandas
and the full multi-agent pipeline, for any MCP client (Claude Desktop, IDEs,
other agents).

    python -m mcp_server                               # stdio: one local client per process
    python -m mcp_server --http --host 127.0.0.1 --port 8000

Code execution (run_pandas, analyze_question) is on by default for stdio and
off by default for Streamable HTTP; see ServerConfig and the README.
"""

from __future__ import annotations

import argparse

from mcp.server import MCPServer

from mcp_server.context import ServerConfig, ServerState
from mcp_server.prompts import register_prompts
from mcp_server.resources import register_resources
from mcp_server.tools import register_tools

SERVER_NAME = "agentai-data-analyst"
SERVER_VERSION = "0.2.0"
INSTRUCTIONS = (
    "Data-analysis tools over the datasets registered in this session. Register a dataset "
    "first (register_dataset), inspect it with get_dataset_schema / get_dataset_profile, then "
    "prefer the deterministic tools (describe, group_aggregate, correlate, detect_anomalies, "
    "query_sql, make_chart) because their numbers are exact. Use run_pandas only when no tool "
    "fits, and analyze_question to hand a whole question to the multi-agent pipeline."
)


def create_server(config: ServerConfig | None = None) -> MCPServer:
    """A fully registered server with its own scopes (never shared between servers)."""
    config = config or ServerConfig.for_stdio()
    state = ServerState.create(config)
    server = MCPServer(
        name=SERVER_NAME,
        title="Multi-Agent Data Analyst",
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
    )
    register_tools(server, state)
    register_resources(server, state)
    register_prompts(server, state)
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mcp_server", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000)")
    args = parser.parse_args(argv)

    if not args.http:
        create_server(ServerConfig.for_stdio()).run("stdio")
        return

    config = ServerConfig.for_http()
    create_server(config).run(
        "streamable-http",
        host=args.host,
        port=args.port,
        # Uploads arrive base64-encoded inside a JSON request: allow for ~4/3 growth.
        max_request_body_size=config.max_upload_bytes * 4 // 3 + 64 * 1024,
        session_idle_timeout=config.session_idle_seconds,
    )


if __name__ == "__main__":
    main()
