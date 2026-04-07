# Agent Memory

Redis-based semantic memory for AI agents. Store and recall with natural language.

## Quick Start

```python
from agent_memory import remember, recall

remember("User prefers dark mode", "preferences")
recall("What theme does user prefer?")
# [('User prefers dark mode', 0.85)]
```

## Install

```bash
pip install agent-memory
# or
uv add git+https://github.com/Allentgt/opencode-memory-skill.git
```

## Redis

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

Or set `REDIS_HOST` / `REDIS_PORT` in `.env`.

## MCP Server

```bash
# CLI
uv run agent-memory

# OpenCode config
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory"
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `agent_memory_remember` | Store info with context |
| `agent_memory_recall` | Search with natural language |
| `agent_memory_count` | Get total memories |

### Parameters

`agent_memory_remember`: `content`, `context?`, `index_name?`

`agent_memory_recall`: `query`, `min_score?` (0.3), `limit?` (5), `index_name?`, `response_format?` ("markdown"/"json")

## API

```python
remember("content", "context")
recall("query", min_score=0.3, limit=5)
```

See `agent_memory/__init__.py` for full API.