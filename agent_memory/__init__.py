"""
Agent Memory Skill - Redis-based semantic memory for AI agents.

Uses Redis for storage with local embeddings via sentence-transformers.
Fully local, free to use, supports semantic search.

Module Structure:
- embeddings.py: Sentence embedding model loading and encoding
- storage.py: Redis connection and data persistence
- core.py: AgentMemory classes (sync/async orchestrators)
- server.py: MCP server tools
"""

import os
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

# Re-export for backward compatibility
from agent_memory.core import AgentMemory, AgentMemoryAsync
from agent_memory.embeddings import MODELS, get_model_name, list_models

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

if _HAS_DOTENV:
    load_dotenv()


# Convenience functions (backward compatible)
def remember(
    content: str,
    context: str = "default",
    index_name: str = "agent_memory",
    ttl_days: Optional[int] = None,
) -> str:
    """Store content in memory.

    Args:
        content: The content to store
        context: Context label for categorization
        index_name: Memory index name
        ttl_days: Optional TTL in days

    Returns:
        str: The memory_id for later retrieval/deletion
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.remember(content, context, ttl_days)


def recall(
    query: str,
    min_score: float = 0.3,
    limit: int = 5,
    index_name: str = "agent_memory",
    context: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    keyword_boost: float = 0.0,
) -> List[Tuple[str, float]]:
    """Search memory.

    Args:
        query: Natural language search query
        min_score: Minimum similarity score
        limit: Maximum results
        index_name: Memory index name
        context: Optional context filter
        since: Optional start datetime
        until: Optional end datetime
        keyword_boost: Keyword boost factor

    Returns:
        List[Tuple[str, float]]: List of (content, similarity_score) tuples
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.recall(query, min_score, limit, context, since, until, keyword_boost)


def delete(memory_id: str, index_name: str = "agent_memory") -> bool:
    """Delete a specific memory by ID."""
    with AgentMemory(index_name=index_name) as mem:
        return mem.delete(memory_id)


def clear(index_name: str = "agent_memory") -> int:
    """Clear all memories."""
    with AgentMemory(index_name=index_name) as mem:
        return mem.clear()


def cleanup(index_name: str = "agent_memory") -> int:
    """Remove expired memories."""
    from agent_memory.core import AgentMemory

    with AgentMemory(index_name=index_name) as mem:
        keys = mem.storage.get_all_keys()
        removed = 0
        for key in keys:
            data = mem.storage.conn.hgetall(key)
            expires_at = data.get("expires_at")
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expiry:
                        mem_id = key.split(":")[-1]
                        mem.delete(mem_id)
                        removed += 1
                except ValueError:
                    pass
        return removed


def get_memory(memory_id: str, index_name: str = "agent_memory") -> Optional[dict]:
    """Get memory with metadata."""
    with AgentMemory(index_name=index_name) as mem:
        return mem.get(memory_id)


def list_memories(
    limit: int = 50,
    offset: int = 0,
    context: Optional[str] = None,
    index_name: str = "agent_memory",
) -> List[dict]:
    """List all memories."""
    with AgentMemory(index_name=index_name) as mem:
        return mem.list_memories(limit, offset, context)


def remember_batch(
    items: List[Tuple[str, str]],
    index_name: str = "agent_memory",
    ttl_days: Optional[int] = None,
) -> List[str]:
    """Store multiple memories."""
    memory_ids = []
    with AgentMemory(index_name=index_name) as mem:
        for content, context in items:
            memory_ids.append(mem.remember(content, context, ttl_days))
    return memory_ids


def export_memories(filepath: str, index_name: str = "agent_memory") -> int:
    """Export memories to JSON file."""
    import json

    with AgentMemory(index_name=index_name) as mem:
        memories = mem.list_memories(limit=10000)
        with open(filepath, "w") as f:
            json.dump(memories, f, indent=2)
        return len(memories)


def import_memories(
    filepath: str, index_name: str = "agent_memory", merge: bool = True
) -> int:
    """Import memories from JSON file."""
    import json

    if not merge:
        with AgentMemory(index_name=index_name) as mem:
            mem.clear()

    with open(filepath, "r") as f:
        memories = json.load(f)

    count = 0
    with AgentMemory(index_name=index_name) as mem:
        for mem_data in memories:
            mem.remember(
                mem_data.get("content", ""), mem_data.get("context", "default")
            )
            count += 1
    return count


# Async convenience functions
async def remember_async(
    content: str,
    context: str = "default",
    index_name: str = "agent_memory",
    ttl_days: Optional[int] = None,
) -> str:
    """Store content in memory (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.remember(content, context, ttl_days)


async def recall_async(
    query: str,
    min_score: float = 0.3,
    limit: int = 5,
    index_name: str = "agent_memory",
    context: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    keyword_boost: float = 0.0,
) -> List[Tuple[str, float]]:
    """Search memory (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.recall(
            query, min_score, limit, context, since, until, keyword_boost
        )


async def delete_async(memory_id: str, index_name: str = "agent_memory") -> bool:
    """Delete a specific memory (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.delete(memory_id)


async def clear_async(index_name: str = "agent_memory") -> int:
    """Clear all memories (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.clear()


async def cleanup_async(index_name: str = "agent_memory") -> int:
    """Remove expired memories (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        keys = await (await mem.storage).get_all_keys()
        removed = 0
        for key in keys:
            conn = await (await mem.storage).conn
            data = await conn.hgetall(key)
            expires_at = data.get("expires_at")
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expiry:
                        mem_id = key.split(":")[-1]
                        await mem.delete(mem_id)
                        removed += 1
                except ValueError:
                    pass
        return removed


# Backward compatibility aliases
__all__ = [
    # Classes
    "AgentMemory",
    "AgentMemoryAsync",
    # Convenience sync functions
    "remember",
    "recall",
    "delete",
    "clear",
    "cleanup",
    "get_memory",
    "list_memories",
    "remember_batch",
    "export_memories",
    "import_memories",
    # Async functions
    "remember_async",
    "recall_async",
    "delete_async",
    "clear_async",
    "cleanup_async",
    # Utilities
    "list_models",
    "get_model_name",
    "MODELS",
]
