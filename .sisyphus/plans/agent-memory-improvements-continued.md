# Agent Memory Skill - Medium & Low Priority Improvements Plan

## Overview
Implement remaining features: batch operations, hybrid search, memory metadata, multiple embedding models, and persistence/export.

---

## Summary

| Feature | Priority | Est. Lines | Files |
|---------|----------|------------|-------|
| Batch Operations | Medium | ~40 | `__init__.py`, `server.py` |
| Memory Metadata | Medium | ~60 | `__init__.py`, `server.py` |
| Hybrid Search | Medium | ~50 | `__init__.py` |
| Multiple Embedding Models | Low | ~30 | `__init__.py` |
| Persistence/Export | Low | ~50 | `__init__.py`, `server.py` |
| **Total** | | **~230** | **2** |

---

## 1. Batch Operations

**Objective**: Store multiple memories in a single call to reduce connection overhead.

### Changes Required

#### `agent_memory/__init__.py`
- Add `remember_batch(items: List[Tuple[str, str]], index_name)` method to `AgentMemory`
- Add `remember_batch()` convenience function

#### `agent_memory/server.py`
- Add `RememberBatchInput` model
- Add `agent_memory_remember_batch` MCP tool

### Implementation Details
```python
def remember_batch(
    items: List[Tuple[str, str]],  # List of (content, context)
    index_name: str = "agent_memory",
    ttl_days: Optional[int] = None
) -> List[str]:
    """Store multiple memories. Returns list of memory_ids."""
```

### Files Modified
- `agent_memory/__init__.py` (~20 lines)
- `agent_memory/server.py` (~20 lines)

---

## 2. Memory Metadata

**Objective**: Track access patterns for analytics and smart eviction.

### Changes Required

#### `agent_memory/__init__.py`
- Add `access_count` and `last_accessed` fields to stored memories
- Update `recall()` to increment access count and update last_accessed
- Add `get_memory(memory_id)` method to retrieve single memory with metadata
- Add `list_memories(index_name, limit, context)` method for browsing

#### `agent_memory/server.py`
- Add `GetMemoryInput` model
- Add `ListMemoriesInput` model
- Add `agent_memory_get` MCP tool
- Add `agent_memory_list` MCP tool

### Storage Format
```python
{
    "content": str,
    "context": str,
    "embedding_json": str,
    "timestamp": str,        # ISO8601 - creation time
    "expires_at": str,      # ISO8601 or empty
    "access_count": int,    # Number of times recalled
    "last_accessed": str,   # ISO8601 - last recall time
}
```

### Files Modified
- `agent_memory/__init__.py` (~40 lines)
- `agent_memory/server.py` (~20 lines)

---

## 3. Hybrid Search

**Objective**: Combine semantic similarity with keyword matching for better results.

### Changes Required

#### `agent_memory/__init__.py`
- Add optional keyword boost parameter to `recall()`
- Use Redis FT.SEARCH for text matching
- Combine semantic score with keyword score

### Implementation Details
```python
def recall(
    self, 
    query: str, 
    min_score: float = 0.3, 
    limit: int = 5,
    context: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    keyword_boost: float = 0.0  # 0.0 = no boost, 1.0 = max keyword boost
) -> List[Tuple[str, float]]:
```

### Files Modified
- `agent_memory/__init__.py` (~50 lines)

---

## 4. Multiple Embedding Models

**Objective**: Allow choosing between fast/accurate models.

### Changes Required

#### `agent_memory/__init__.py`
- Add model selection via environment variable or parameter
- Support: "fast" (all-MiniLM-L6-v2), "accurate" (all-mpnet-base-v2)
- Add `list_models()` function

### Environment Variables
```
AGENT_MEMORY_MODEL=fast  # or "accurate"
```

### Files Modified
- `agent_memory/__init__.py` (~30 lines)

---

## 5. Persistence/Export

**Objective**: Backup and restore memories.

### Changes Required

#### `agent_memory/__init__.py`
- Add `export_memories(index_name, filepath)` method
- Add `import_memories(filepath, index_name, merge=True)` method

#### `agent_memory/server.py`
- Add `ExportMemoriesInput` model
- Add `ImportMemoriesInput` model
- Add `agent_memory_export` MCP tool
- Add `agent_memory_import` MCP tool

### Export Format (JSON)
```json
{
  "version": "1.0",
  "index_name": "agent_memory",
  "exported_at": "2024-01-15T10:30:00",
  "memories": [
    {
      "content": "...",
      "context": "...",
      "timestamp": "...",
      "expires_at": "...",
      "embedding": [...]
    }
  ]
}
```

### Files Modified
- `agent_memory/__init__.py` (~40 lines)
- `agent_memory/server.py` (~25 lines)

---

## New Tool Summary (MCP)

| New Tool | Description |
|----------|-------------|
| `agent_memory_remember_batch` | Store multiple memories at once |
| `agent_memory_get` | Get single memory by ID with metadata |
| `agent_memory_list` | List all memories with optional context filter |
| `agent_memory_export` | Export memories to JSON file |
| `agent_memory_import` | Import memories from JSON file |

---

## Testing
- Add tests for each new feature
- Verify backward compatibility
- Test edge cases (empty batch, invalid JSON import, etc.)
