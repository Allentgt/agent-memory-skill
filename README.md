<p align="center">
  <img src="logos/agent-memory-banner.svg" alt="Agent Memory Banner" width="100%">
</p>

> Semantic memory storage for AI agents. Store and recall information using natural language.

Agent Memory lets AI agents remember things using **semantic search** - it understands *meaning*, not just keywords. Built on **Redis** + sentence embeddings.

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why Agent Memory?

| Traditional Search | Agent Memory |
|-------------------|--------------|
| Exact keyword match | Finds "dark mode" when you search "theme preferences" |
| No context awareness | Understands meaning and relationships |
| Flat storage | Organize by context ("preferences", "project", "meetings") |

## Quick Start

```python
from agent_memory import remember, recall

# Store information
remember("User prefers dark mode", "preferences")
remember("Working on a FastAPI project", "project")

# Search semantically - finds meaning, not just keywords!
results = recall("What UI theme does the user like?")
# → [('User prefers dark mode', 0.85), ...]
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Allentgt/agent-memory-skill.git
cd agent-memory-skill

# Install dependencies
uv sync
```

### Prerequisites

**Redis is required** - run it via Docker:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

Or use a custom Redis instance and set environment variables:

```bash
export REDIS_HOST=your-host
export REDIS_PORT=6379
```

## Usage

### Basic API

```python
from agent_memory import remember, recall

# Store memories with context labels
memory_id = remember("User prefers dark mode", "preferences")
memory_id = remember("Meeting at 2pm tomorrow", "meetings")

# Search semantically
results = recall("What does the user like?")
# Returns: [('User prefers dark mode', 0.85), ...]
```

### Full API

```python
from agent_memory import (
    remember, recall, remember_batch, get_memory, list_memories,
    export_memories, import_memories,
    AgentMemory, AgentMemoryAsync
)

# Store single memory with TTL (auto-expire)
remember("Temporary note", "notes", ttl_days=7)

# Store multiple at once
ids = remember_batch([
    ("Memory 1", "context1"),
    ("Memory 2", "context2"),
])

# Search with filters
results = recall(
    "user preferences",
    min_score=0.3,      # Minimum similarity (0.0-1.0)
    limit=5,           # Max results
    context="preferences",  # Filter by context
    keyword_boost=0.3   # Hybrid search
)

# Get single memory with metadata
mem = get_memory(memory_id)
# {'memory_id': 'mem:...', 'content': '...', 
#  'context': '...', 'timestamp': '...', ...}

# List all memories with pagination
memories = list_memories(limit=50, offset=0, context="preferences")

# Export/Import for backup
export_memories("backup.json")
import_memories("backup.json", merge=True)
```

### Async API

```python
import asyncio
from agent_memory import remember_async, recall_async

async def main():
    await remember_async("Async storage", "notes")
    results = await recall_async("search query")
    
asyncio.run(main())
```

### Context Manager (Full Control)

```python
with AgentMemory(index_name="my_project") as mem:
    mem.remember("content", "context")
    results = mem.recall("query")
    memories = mem.list_memories()
    mem.delete(memory_id)
    count = mem.count
    mem.clear()  # Delete all
```

## Use Cases

### 🎯 User Preferences
```python
remember("User prefers dark mode", "preferences")
remember("User likes Python over JavaScript", "preferences")
# Later: recall("What does user like for UI?") → finds dark mode
```

### 📁 Project Context
```python
remember("Building a FastAPI REST API", "project")
remember("Using PostgreSQL for database", "project")
```

### 📅 Meeting Notes
```python
remember("Meeting with team about Q4 goals", "meetings")
remember("Action items: review PR #123", "meetings")
```

### 🔍 Time-Based Search
```python
from datetime import datetime, timedelta
last_week = datetime.now() - timedelta(days=7)
results = recall("project updates", since=last_week)
```

## MCP Server (for AI Agents)

Run as an MCP server to expose tools to AI agents:

```bash
uv run agent-memory
```

### OpenCode Configuration

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uv",
      "args": ["tool", "run", "agent-memory"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `agent_memory_remember` | Store information with context |
| `agent_memory_recall` | Search using natural language |
| `agent_memory_get` | Get memory by ID |
| `agent_memory_list` | List memories with filters |
| `agent_memory_delete` | Delete specific memory |
| `agent_memory_clear` | Clear all memories |
| `agent_memory_export` | Export to JSON |
| `agent_memory_import` | Import from JSON |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server host |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_PASSWORD` | - | Redis password (optional) |
| `REDIS_DB` | 0 | Redis database number |
| `AGENT_MEMORY_MODEL` | fast | Embedding model: "fast" or "accurate" |

### Embedding Models

- **fast** (default): `sentence-transformers/all-MiniLM-L6-v2` - Fast, 384 dimensions
- **accurate**: `sentence-transformers/all-mpnet-base-v2` - More accurate, 768 dimensions

## OpenCode Skill

This can also be loaded as a skill:

```bash
skill(agent-memory)
```

**Trigger phrases:**
- "remember that..."
- "store this information"
- "don't forget that..."
- "what do I know about..."
- "search my memories"

## Requirements

- Python 3.14+
- Redis server
- Dependencies: `redis`, `sentence-transformers`, `python-dotenv`, `mcp`

## Testing

```bash
# Run unit tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=agent_memory
```

## License

MIT License - see [LICENSE](LICENSE) file.
