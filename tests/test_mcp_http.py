"""
Starts `python -m mcp_server --http` on a free port and runs the same smoke
checks CI runs against the Docker image (mcp_server/smoke.py), with the
official MCP client over Streamable HTTP.
"""

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

from mcp_server.smoke import check, wait_until_up

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_server_end_to_end():
    port = free_port()
    env = {k: v for k, v in os.environ.items()
           if k not in ("GROQ_API_KEY", "AGENTAI_MCP_ALLOW_RUN_PANDAS", "AGENTAI_MCP_ALLOW_ANALYZE_QUESTION")}
    env["PYTHONPATH"] = str(ROOT)
    server = subprocess.Popen(
        [sys.executable, "-m", "mcp_server", "--http", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        wait_until_up(url, timeout=90)
        report = asyncio.run(check(url))
        assert report == [
            "11 tools, code execution off",
            "calls without a session are refused",
            "session upload + SQL work; server introspection refused",
        ]
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
