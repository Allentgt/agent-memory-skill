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
from agent_memory import (
    remember, recall, remember_batch, get_memory, list_memories,
    export_memories, import_memories,
    AgentMemory, AgentMemoryAsync
)

# Store single memory
memory_id = remember("User prefers dark mode", "preferences")

# Store multiple memories at once
ids = remember_batch([
    ("Memory 1", "context1"),
    ("Memory 2", "context2"),
], ttl_days=30)

# Search with keyword boost
results = recall("user preferences", keyword_boost=0.3)

# Get single memory with metadata
mem = get_memory(memory_id)
# {'memory_id': 'mem:...', 'content': '...', 'context': '...', 
#  'timestamp': '...', 'access_count': 5, 'last_accessed': '...'}

# List all memories
memories = list_memories(limit=50, context="preferences")

# Export/Import
export_memories("backup.json", index_name="my_index")
import_memories("backup.json", index_name="my_index", merge=True)

# Full control
with AgentMemory(index_name="my_index") as mem:
    mem.remember("content", "context")
    results = mem.recall("query")
    memories = mem.list_memories()
```

## MCP Server (for AI agents)

Run as a subprocess to expose tools to AI agents:

```bash
# From project directory
uv run agent-memory

# Or if installed as uv tool
uv tool install .
uv tool run agent-memory

# Or if installed globally via pip
agent-memory
```

### OpenCode Integration

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory"
    }
  }
}
```

Or for uv tool run:

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
| `agent_memory_remember` | Store information with context label |
| `agent_memory_remember_batch` | Store multiple memories at once |
| `agent_memory_recall` | Search using natural language |
| `agent_memory_get` | Get single memory by ID with metadata |
| `agent_memory_list` | List all memories with optional context filter |
| `agent_memory_count` | Get total stored memories |
| `agent_memory_delete` | Delete specific memory by ID |
| `agent_memory_clear` | Clear all memories |
| `agent_memory_cleanup` | Remove expired memories |
| `agent_memory_export` | Export memories to JSON file |
| `agent_memory_import` | Import memories from JSON file |

### Tool Parameters

**agent_memory_remember**
- `content` (required): Text to store
- `context` (optional): Category like "preferences", "project" (default: "default")
- `index_name` (optional): Custom memory index (default: "agent_memory")
- `ttl_days` (optional): TTL in days 1-365. Memories auto-expire after this many days. None = no expiry (default: None)

**agent_memory_remember_batch**
- `items` (required): List of (content, context) tuples
- `index_name` (optional): Custom memory index (default: "agent_memory")
- `ttl_days` (optional): TTL in days for all items

**agent_memory_recall**
- `query` (required): Natural language search
- `min_score` (optional): Min similarity 0.0-1.0 (default: 0.3)
- `limit` (optional): Max results 1-100 (default: 5)
- `index_name` (optional): Which index to search
- `response_format` (optional): "markdown" or "json" (default: "markdown")
- `context` (optional): Filter by context label (e.g., "preferences", "project")
- `since` (optional): ISO8601 datetime - only return memories created after this time
- `until` (optional): ISO8601 datetime - only return memories created before this time
- `keyword_boost` (optional): 0.0 = semantic only, 1.0 = max keyword boost (default: 0.0)

**agent_memory_get**
- `memory_id` (required): The memory ID returned from remember
- `index_name` (optional): Which index (default: "agent_memory")

**agent_memory_list**
- `limit` (optional): Max memories to return 1-500 (default: 50)
- `context` (optional): Filter by context label
- `index_name` (optional): Which index (default: "agent_memory")

**agent_memory_delete**
- `memory_id` (required): The memory ID returned from remember
- `index_name` (optional): Which index (default: "agent_memory")

**agent_memory_cleanup**
- `index_name` (optional): Clean up expired memories from this index (default: "agent_memory")

**agent_memory_export**
- `filepath` (required): Path to export JSON file
- `index_name` (optional): Which index to export (default: "agent_memory")

**agent_memory_import**
- `filepath` (required): Path to import JSON file
- `index_name` (optional): Target index (default: "agent_memory")
- `merge` (optional): If false, clears existing first (default: true)

### Environment Variables

- `REDIS_HOST`: Redis host (default: "localhost")
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_DB`: Redis database number (default: 0)
- `AGENT_MEMORY_MODEL`: Embedding model - "fast" (default) or "accurate"

## Requirements

- Python 3.14+
- Redis server
- Dependencies: `redis`, `sentence-transformers`, `python-dotenv`, `mcp`

## OpenCode Skill Integration

### Option 1: Load as On-Demand Skill

This skill can be loaded when needed rather than running as a persistent MCP server:

```bash
# Load the skill when you need memory capabilities
skill(agent-memory)
```

**Trigger phrases:**
- "remember that..."
- "store this information"
- "don't forget that..."
- "what do I know about..."
- "search my memories"
- "find information about..."

Once loaded, use the Python API directly in your code.

### Option 2: MCP Server (Persistent)

For always-available memory tools, run as MCP server (see MCP Server section above).