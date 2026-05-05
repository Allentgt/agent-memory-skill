---
name: agent-memory
description: Semantic memory storage and retrieval for AI agents using Redis + sentence embeddings. Use this skill whenever users want to store information for later retrieval, remember user preferences, track conversation context across sessions, build persistent knowledge bases, or search memories using natural language queries - even when they don't explicitly say "memory" or "remember". This skill provides the agent_memory_remember, agent_memory_recall, agent_memory_count, agent_memory_clear, agent_memory_delete, agent_memory_cleanup, agent_memory_get, agent_memory_list, agent_memory_export, agent_memory_import, agent_memory_health MCP tools for AI agents.
---

# Agent Memory Skill

This skill provides semantic memory capabilities for AI agents. Store information once and retrieve it later using natural language - the system understands meaning, not just keywords.

## When to Use This Skill

Use this skill when the user wants to:
- Store information for later retrieval
- Remember user preferences, project context, or domain knowledge
- Search memories using natural language queries
- Track conversation context across sessions
- Create persistent knowledge bases

**Trigger phrases:**
- "remember that..."
- "store this information"
- "don't forget that..."
- "what do I know about..."
- "search my memories"
- "find information about..."
- "retrieve stored knowledge"

## Prerequisites

### Redis Connection Required

This skill requires a Redis server to be running:

```bash
# Start Redis (required)
docker run -d -p 6379:6379 redis/redis-stack:latest
```

## Usage

```python
from agent_memory import remember, recall, get_memory, list_memories

# Store information
memory_id = remember("User prefers dark mode", "preferences")

# Search semantically
results = recall("What does the user like?")
# Returns: [('User prefers dark mode', 0.85), ...]

# Get specific memory with metadata
memory = get_memory(memory_id)
# Returns: {'memory_id': '...', 'content': '...', 
#           'context': '...', 'timestamp': '...',
#           'access_count': 1, 'last_accessed': '...'}

# List all memories
memories = list_memories(limit=50, context="preferences")
```

## API Reference

### Core Functions

- `remember(content, context, index_name, ttl_days)` → memory_id
- `remember_batch(items, index_name, ttl_days)` → List[memory_id]
- `recall(query, min_score, limit, context, since, until, keyword_boost)` → List[Tuple[str, float]]
- `get_memory(memory_id, index_name)` → Optional[dict]
- `list_memories(limit, context, index_name)` → List[dict]
- `delete(memory_id, index_name)` → bool
- `export_memories(filepath, index_name)` → int
- `import_memories(filepath, index_name, merge)` → int

### Async API

```python
from agent_memory import remember_async, recall_async

await remember_async("Async storage", "notes")
results = await recall_async("search query")
```

## Examples

### Store User Preferences
```python
remember("User prefers dark mode", "preferences")
remember("User is working on FastAPI project", "project")
```

### Retrieve Context
```python
results = recall("user interface settings")
# Returns memories about dark mode, theme preferences, etc.
```

### Time-Based Queries
```python
from datetime import datetime, timedelta
last_week = datetime.utcnow() - timedelta(days=7)
results = recall("project updates", since=last_week)
```

### Hybrid Search
```python
results = recall("FastAPI REST API", keyword_boost=0.3)
```

## Environment Variables

- `REDIS_HOST` - Redis host (default: "localhost")
- `REDIS_PORT` - Redis port (default: 6379)
- `AGENT_MEMORY_MODEL` - "fast" (default) or "accurate"

## Troubleshooting

### Redis Connection Failed
```
Error: Redis connection failed
```
**Fix:** Ensure Redis is running:
```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```
Or check environment variables: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`

### Embedding Model Slow to Load
First-time model download is slow (~100MB for fast model, ~420MB for accurate).
**Tip:** Use `AGENT_MEMORY_MODEL=fast` for faster startup.

### MCP Server Not Starting
Ensure dependencies installed: `uv sync`
Run with: `uv run agent-memory`

## Performance Tips

- Use `remember_batch()` for storing multiple items at once
- Set appropriate `min_score` (higher = more precise, lower = more results)
- Use `context` parameter to filter memories - reduces scan time
- Use `AGENT_MEMORY_MODEL=fast` for lower latency (384 dims vs 768)

## Features

### Vector Similarity Search (VSS)
When using Redis Stack (FT.SEARCH available), uses RediSearch KNN for O(log n) search. Falls back to O(n) scan for standard Redis.

### Access Tracking
Memories track `access_count` and `last_accessed` - automatically updated on retrieval.

### UUID-Based IDs
Every memory gets a unique UUID4 identifier - no collision risk.

### Batch Optimization
Multiple memories stored efficiently using Redis pipeline.
