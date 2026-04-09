---
name: agent-memory
description: Provides semantic memory storage and retrieval using Redis + sentence embeddings. Store and recall information using natural language queries.
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
