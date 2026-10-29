# MCP server over Streamable HTTP (http://<host>:8000/mcp).
#
#   docker build -t agentai-mcp .
#   docker run -p 8000:8000 agentai-mcp
#
# Defaults are the remote-safe ones: code execution (run_pandas,
# analyze_question) is off and no server files are exposed. Enable code
# execution only for trusted clients, e.g.
#   docker run -p 8000:8000 -e AGENTAI_MCP_ALLOW_RUN_PANDAS=true agentai-mcp
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SANDBOX_ISOLATION=process \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agents/ agents/
COPY core/ core/
COPY tools/ tools/
COPY mcp_server/ mcp_server/
COPY sample_data/ sample_data/

# Unprivileged user; the code is read-only to it and logs go to a writable dir.
RUN useradd --create-home --uid 10001 analyst \
    && mkdir -p /app/logs && chown analyst /app/logs
USER analyst

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 3)"

CMD ["python", "-m", "mcp_server", "--http", "--host", "0.0.0.0", "--port", "8000"]
