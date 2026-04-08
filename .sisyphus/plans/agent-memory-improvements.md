# Agent Memory Skill - High Priority Improvements Plan

## Overview
Enhance the agent-memory skill with context filtering, individual memory management, time-based queries, and TTL support.

---

## 1. Context Filtering in Recall

**Objective**: Enable filtering memories by context/category during search.

### Changes Required

#### `agent_memory/__init__.py`
- Add `context` parameter to `recall()` method
- Filter results by context before returning

#### `agent_memory/server.py`
- Add `context` field to `RecallInput` Pydantic model
- Pass context filter to `_recall()` function

### Implementation Details
```python
# New signature
def recall(self, query: str, min_score: float = 0.3, limit: int = 5, 
           context: Optional[str] = None) -> List[Tuple[str, float]]:
```

### Files Modified
- `agent_memory/__init__.py` (~5 lines)
- `agent_memory/server.py` (~10 lines)

---

## 2. Update/Delete Individual Memories

**Objective**: Enable deletion of specific memories by ID, not just bulk clear.

### Changes Required

#### `agent_memory/__init__.py`
- Return `memory_id` from `remember()` method
- Add `delete(memory_id)` method
- Track memory IDs in stored data

#### `agent_memory/server.py`
- Add `DeleteMemoryInput` model
- Add `agent_memory_delete` MCP tool
- Update `remember` to return memory_id in response

### Implementation Details
```python
# remember() now returns memory_id
def remember(self, content: str, context: str = "default") -> str:
    """Returns the memory_id for later retrieval/deletion."""

# New delete method
def delete(self, memory_id: str) -> bool:
    """Delete a specific memory by ID. Returns True if deleted."""
```

### Files Modified
- `agent_memory/__init__.py` (~25 lines)
- `agent_memory/server.py` (~40 lines)

---

## 3. Time-Based Queries

**Objective**: Allow filtering memories by timestamp (e.g., "last week", "specific date range").

### Changes Required

#### `agent_memory/__init__.py`
- Store `created_at` timestamp with each memory
- Add `since` and `until` parameters to `recall()`
- Add `recent(limit, context)` convenience method

#### `agent_memory/server.py`
- Add `since` (ISO8601 datetime) and `until` to `RecallInput`
- Add `TimeFilterInput` for time-based queries

### Implementation Details
```python
# New recall parameters
def recall(self, query: str, min_score: float = 0.3, limit: int = 5,
           context: Optional[str] = None,
           since: Optional[datetime] = None,
           until: Optional[datetime] = None) -> List[Tuple[str, float]]:
```

### Files Modified
- `agent_memory/__init__.py` (~30 lines)
- `agent_memory/server.py` (~15 lines)

---

## 4. TTL (Time-To-Live) Support

**Objective**: Auto-expire memories after configurable duration.

### Changes Required

#### `agent_memory/__init__.py`
- Add `ttl_days` parameter to `remember()`
- Use Redis TTL or track expiry timestamp
- Add `cleanup()` method to remove expired memories

#### `agent_memory/server.py`
- Add `ttl_days` field to `RememberInput`
- Add `agent_memory_cleanup` tool

### Implementation Details
```python
# New remember signature
def remember(self, content: str, context: str = "default", 
             ttl_days: Optional[int] = None) -> str:
    """Store with optional TTL. None = no expiry."""
```

### Files Modified
- `agent_memory/__init__.py` (~20 lines)
- `agent_memory/server.py` (~20 lines)

---

## Summary

| Feature | Files Changed | Est. Lines |
|---------|---------------|------------|
| Context Filtering | 2 | ~15 |
| Individual Delete | 2 | ~65 |
| Time-Based Queries | 2 | ~45 |
| TTL Support | 2 | ~40 |
| Async Support | 2 | ~80 |
| **Total** | **2** | **~245** |

## Testing
- Add tests for each new feature in `test_basic.py`
- Verify backward compatibility with existing API

## New Tool Summary (MCP)

| New Tool | Description |
|----------|-------------|
| `agent_memory_delete` | Delete specific memory by ID |
| `agent_memory_cleanup` | Remove expired memories |
| `agent_memory_recall` (enhanced) | Now supports context, since, until filters |
| `agent_memory_remember` (enhanced) | Now supports TTL |

---

## 5. Async Support

**Objective**: Make all memory operations async-compatible for better concurrency in async contexts.

### Changes Required

#### `agent_memory/__init__.py`
- Add `AgentMemoryAsync` class with async methods
- Add `remember_async()`, `recall_async()`, `clear_async()` convenience functions
- Implement connection pooling with `redis.asyncio`
- Maintain backward compatibility with sync API

#### `agent_memory/server.py`
- Update MCP tools to use async methods
- Ensure no blocking calls in async context

### Implementation Details

```python
# New async class
class AgentMemoryAsync:
    """Async agent memory with semantic search."""
    
    async def __init__(self, index_name: str = "agent_memory"):
        ...
    
    async def remember(self, content: str, context: str = "default") -> str:
        ...
    
    async def recall(self, query: str, min_score: float = 0.3, 
                     limit: int = 5) -> List[Tuple[str, float]]:
        ...

# Convenience async functions
async def remember_async(content: str, context: str = "default") -> None:
    ...

async def recall_async(query: str, min_score: float = 0.3, 
                       limit: int = 5) -> List[Tuple[str, float]]:
    ...
```

### Key Benefits
- Non-blocking I/O for Redis operations
- Better performance under load
- Compatible with FastMCP async context
- Connection pooling reduces overhead

### Files Modified
- `agent_memory/__init__.py` (~50 lines)
- `agent_memory/server.py` (~30 lines)
