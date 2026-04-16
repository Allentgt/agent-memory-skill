# Agent Memory MCP Server Dockerfile
# Build: docker build -t agent-memory .
# Run: docker run -p 8000:8000 agent-memory
# Or use docker-compose --profile mcp up

FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY agent_memory/ ./agent_memory/

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose MCP server port
EXPOSE 8000

# Run MCP server
CMD ["uv", "tool", "run", "agent-memory"]