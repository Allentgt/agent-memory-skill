# Agent Memory

Store and recall information using natural language. Uses Redis + sentence embeddings for semantic search.

## Installation

```bash
# Clone and install locally (recommended for development)
git clone https://github.com/Allentgt/opencode-memory-skill.git
cd opencode-memory-skill
uv sync

# Or install as a uv tool (CLI only, no dependency pollution)
uv tool install git+https://github.com/Allentgt/opencode-memory-skill.git

# Or with pip globally
pip install git+https://github.com/Allentgt/opencode-memory-skill.git
```

### Run

```bash
# From project directory
uv run agent-memory

# Or if installed as uv tool
uv tool run agent-memory

# Or if installed globally via pip
agent-memory
```

## Redis Setup

```bash
# Start Redis (required)
docker run -d -p 6379:6379 redis/redis-stack:latest

# Or use custom host/port via .env file:
# REDIS_HOST=your-host
# REDIS_PORT=6379
```

## Quick Start (Python API)

```python
from agent_memory import remember, recall

# Store memories with optional context
remember("User prefers dark mode", "preferences")
remember("Building a FastAPI project", "project")

# Search semantically - finds meaning, not just keywords
results = recall("What does the user like?")
# [('User prefers dark mode', 0.85), ...]
```

### Full API

```python
# Store info
remember(content: str, context: str = "default")

# Search
recall(query: str, min_score: float = 0.3, limit: int = 5)
# Returns: List[Tuple[str, float]] -> [(content, similarity_score), ...]

# Full control
with AgentMemory(index_name="my_index") as mem:
    mem.remember("content", "context")
    results = mem.recall("query")
```

## MCP Server (for AI agents)

Run as a subprocess to expose tools to AI agents:

```bash
# From project directory
uv run agent-memory

# Or install globally then run
uv sync
uv run agent-memory
```

### OpenCode Integration

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/opencode-memory-skill", "agent-memory"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `agent_memory_remember` | Store information with context label |
| `agent_memory_recall` | Search using natural language |
| `agent_memory_count` | Get total stored memories |

### Tool Parameters

**agent_memory_remember**
- `content` (required): Text to store
- `context` (optional): Category like "preferences", "project" (default: "default")
- `index_name` (optional): Custom memory index (default: "agent_memory")

**agent_memory_recall**
- `query` (required): Natural language search
- `min_score` (optional): Min similarity 0.0-1.0 (default: 0.3)
- `limit` (optional): Max results 1-100 (default: 5)
- `index_name` (optional): Which index to search
- `response_format` (optional): "markdown" or "json" (default: "markdown")

## Requirements

- Python 3.14+
- Redis server
- Dependencies: `redis`, `sentence-transformers`, `python-dotenv`, `mcp`