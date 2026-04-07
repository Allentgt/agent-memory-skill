# Agent Memory Skill

A Redis-based semantic memory skill for AI agents. Store and retrieve information using natural language queries with semantic similarity search.

## Features

- **Semantic Search**: Uses sentence-transformers for embeddings to find semantically similar content
- **Redis Backend**: Stores memories in Redis with full-text search capabilities
- **Python Fallback**: Reliable Python-based similarity computation when RediSearch vector search has issues
- **Convenience Functions**: Simple `remember()` and `recall()` functions for easy use
- **Configurable**: Customizable index names, embedding models, and Redis connection

## Requirements

- Python 3.9+
- Redis server (with RediSearch module for best performance)
- `redis` Python package
- `sentence-transformers` Python package

## Installation

```bash
pip install redis sentence-transformers python-dotenv
```

## Quick Start

### 1. Configure Redis Connection

Create a `.env` file in your project:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

### 2. Use the Skill

```python
from agent_memory import remember, recall

# Store information in memory
remember("User prefers dark mode theme", "preferences")
remember("User is working on a Python project", "context")

# Search memory with natural language
results = recall("What theme does the user prefer?")
print(results)
# Output: [('User prefers dark mode theme', 0.85)]

# With custom parameters
results = recall("What is the user working on?", min_score=0.5, limit=10)
```

### 3. Using the AgentMemory Class

```python
from agent_memory import AgentMemory

# Create with custom settings
mem = AgentMemory(
    index_name="my_agent_memory",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

mem.connect()

# Store memories
mem.remember("Important information", "category")
mem.remember("Another note", "notes")

# Search
results = mem.recall("What information do I have?", min_score=0.3)

# Get memory count
print(f"Total memories: {mem.count}")

mem.close()

# Or use context manager
with AgentMemory() as mem:
    mem.remember("Temporary note", "temp")
    results = mem.recall("What notes exist?")
```

## Configuration Options

### AgentMemory Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_name` | str | `"agent_memory"` | Redis key prefix for memory storage |
| `embedding_model` | str | `"sentence-transformers/all-MiniLM-L6-v2"` | Model for generating embeddings |

### recall() Function

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | Search query |
| `min_score` | float | `0.3` | Minimum similarity score (0.0-1.0) |
| `limit` | int | `5` | Maximum results to return |

### remember() Function

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | str | Required | Text content to remember |
| `context` | str | `"default"` | Context label (e.g., 'conversation', 'codebase', 'docs') |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `"localhost"` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_PASSWORD` | `None` | Redis password (optional) |
| `REDIS_DB` | `0` | Redis database number |

## Redis Setup

### Using Docker

```bash
# Start Redis with RediSearch module
docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
volumes:
  redis_data:
```

## How It Works

1. **Embedding Generation**: When you call `remember()`, the content is converted to a 384-dimensional vector using sentence-transformers (all-MiniLM-L6-v2)
2. **Storage**: The embedding is stored in Redis along with the content and context
3. **Search**: When you call `recall()`, your query is converted to an embedding, and cosine similarity is computed against all stored memories
4. **Results**: Memories are ranked by similarity score and returned

## Performance Notes

- First call loads the embedding model (~2-3 seconds)
- Subsequent calls are faster due to model caching
- Redis connection is reused across calls
- For large memory banks, consider adding Redis indexing or using RediSearch vector capabilities

## Using with OpenCode

To use this skill with OpenCode, place the `agent_memory` folder in your project and import it:

```python
import sys
sys.path.insert(0, '/path/to/agent_memory')

from agent_memory import remember, recall
```

Or use as a reusable skill by copying to your OpenCode skills directory.

## Example: Full Usage

```python
import os
os.environ['REDIS_HOST'] = 'localhost'

from agent_memory import remember, recall, AgentMemory

# Store various types of information
remember("User prefers dark mode", "preferences")
remember("User is working on a Python web app", "project")
remember("Meeting with team at 3pm", "schedule")
remember("API documentation at /docs/api", "reference")

# Search with different queries
print(recall("What are user preferences?"))
# [('User prefers dark mode', 0.92), ...]

print(recall("What project is the user working on?"))
# [('User is working on a Python web app', 0.89), ...]

print(recall("When is the meeting?"))
# [('Meeting with team at 3pm', 0.85), ...]

# Using AgentMemory for more control
with AgentMemory(index_name="custom") as mem:
    mem.remember("Custom memory", "custom_context")
    results = mem.recall("Custom query")
```

## Troubleshooting

### Redis Connection Issues

If you get connection errors:
1. Check Redis is running: `redis-cli ping`
2. Verify REDIS_HOST and REDIS_PORT in .env
3. Check firewall settings

### Model Download Issues

If embedding model fails to load:
1. Set HF_TOKEN environment variable for faster downloads
2. Or download model manually first

### Memory Errors

If you get "WRONGTYPE" errors:
1. Clear old keys: `redis-cli KEYS "agent_memory:*" | xargs redis-cli DEL`
2. Or use a new index_name

## License

MIT License