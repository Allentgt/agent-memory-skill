# Agent Memory Skill

A Redis-based semantic memory skill for AI agents - store and recall information using natural language.

## What It Does

**Store memories:**
```python
remember("The user prefers dark mode theme", "preferences")
remember("User is building a Python web app with FastAPI", "project")
```

**Recall with natural language:**
```python
results = recall("What does the user like?")
# Returns: [('The user prefers dark mode theme', 0.85), ...]

results = recall("What is being built?")
# Returns: [('User is building a Python web app with FastAPI', 0.78), ...]
```

## Installation

### Option 1: uv (Recommended for OpenCode)

```bash
# Clone and install
uv sync

# Or add to existing project
uv add git+https://github.com/Allentgt/opencode-memory-skill.git
```

### Option 2: pip

```bash
pip install redis sentence-transformers python-dotenv
```

### Option 3: Manual

```bash
# Clone repo
git clone https://github.com/Allentgt/opencode-memory-skill.git
cd opencode-memory-skill

# Install dependencies
pip install -e .
```

## Redis Setup

Start a Redis container:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

Or create `.env` file:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Quick Start

### Basic Usage

```python
from agent_memory import remember, recall

# Store information
remember("User prefers dark mode", "preferences")
remember("Working on a Python project", "context")

# Search - finds semantically similar content
results = recall("What theme does the user prefer?")
# [('User prefers dark mode', 0.85)]
```

### Natural Language Examples

```python
# Store various memories
remember("The user likes minimalist UI design", "preferences")
remember("Building an AI agent with semantic memory", "project")
remember("Meeting scheduled for 2pm tomorrow", "schedule")
remember("API docs at /docs/reference", "docs")

# Query naturally - it understands meaning, not just keywords
recall("What design preferences does user have?")
# → [('The user likes minimalist UI design', 0.82)]

recall("What is being built?")
# → [('Building an AI agent with semantic memory', 0.76)]

recall("When is the meeting?")
# → [('Meeting scheduled for 2pm tomorrow', 0.71)]

recall("Where is the documentation?")
# → [('API docs at /docs/reference', 0.68)]
```

### OpenCode Integration

Add to your agent workflow:

```python
import sys
sys.path.insert(0, "/path/to/opencode-memory-skill")

from agent_memory import remember, recall

# In your agent prompts, call these functions to store/recall info
```

## API Reference

### `remember(content: str, context: str = "default")`

Store text in memory with optional context label.

```python
remember("User preference info", "preferences")
remember("Code context", "codebase")
remember("Meeting notes", "meetings")
```

### `recall(query: str, min_score: float = 0.3, limit: int = 5)`

Search memory with natural language. Returns list of `(content, score)` tuples.

```python
# Basic search
results = recall("search query")

# With filters
results = recall("query", min_score=0.5, limit=10)
```

### `AgentMemory` Class

Full control over configuration:

```python
from agent_memory import AgentMemory

with AgentMemory(
    index_name="my_memory",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
) as mem:
    mem.remember("Note", "context")
    results = mem.recall("search query")
    print(f"Total: {mem.count}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Optional password |
| `REDIS_DB` | `0` | Database number |

## How It Works

1. **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text → 384-dim vectors
2. **Storage**: Redis hash with content + JSON embeddings
3. **Search**: Cosine similarity finds semantically similar memories

## Requirements

- Python 3.14+
- Redis server
- `redis`, `sentence-transformers`, `python-dotenv`

## License

MIT


## MCP Server (Auto-Discovery)

This package exposes `remember` and `recall` as MCP tools for automatic agent discovery. Uses FastMCP framework with full Pydantic validation and response format options.

### Run as MCP Server

```bash
# Install and run
uv run agent-memory

# Or directly
python mcp_server.py
```

### OpenCode Integration

Add to your OpenCode config:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "python",
      "args": ["path/to/mcp_server.py"]
    }
  }
}
```

### Available Tools

| Tool | Description | Annotations |
|------|-------------|-------------|
| `agent_memory_remember` | Store information in memory with context | readOnly: false, idempotent: true |
| `agent_memory_recall` | Search memory with natural language | readOnly: true, idempotent: true |
| `agent_memory_count` | Get total memories | readOnly: true, idempotent: true |

### Tool Parameters

**agent_memory_remember:**
- `content` (string, required): Content to store (max 50,000 chars)
- `context` (string, optional): Context label (default: "default")
- `index_name` (string, optional): Custom index name (default: "agent_memory")

**agent_memory_recall:**
- `query` (string, required): Natural language search query
- `min_score` (float, optional): Min similarity 0.0-1.0 (default: 0.3)
- `limit` (int, optional): Max results 1-100 (default: 5)
- `index_name` (string, optional): Custom index name
- `response_format` (string, optional): "markdown" or "json" (default: "markdown")

**agent_memory_count:**
- `index_name` (string, optional): Custom index name

### Response Format

Use `response_format` parameter in `agent_memory_recall`:

```python
# Markdown (default) - human readable
results = recall("query", response_format="markdown")
# Returns: "Found 2 memory(ies):\n- [0.85] Content here\n- [0.72] More content"

# JSON - machine readable
results = recall("query", response_format="json")
# Returns: {"total": 2, "results": [{"content": "...", "score": 0.85}, ...]}
```