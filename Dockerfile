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

# Pre-cache embedding model to avoid cold start
ENV TRANSFORMERS_CACHE=/app/model-cache
ENV HF_HOME=/app/model-cache
ENV TRANSFORMERS_OFFLINE=1

RUN uv sync --frozen --no-dev && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose MCP server port
EXPOSE 8000

# Run MCP server
CMD ["uv", "tool", "run", "agent-memory"]