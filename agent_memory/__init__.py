"""
Agent Memory Skill - Redis-based semantic memory for AI agents.

Uses Redis for storage with local embeddings via sentence-transformers.
Fully local, free to use, supports semantic search.
"""

import os
import json
import struct
import asyncio
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

if _HAS_DOTENV:
    load_dotenv()


def _get_redis_connection():
    """Get Redis connection using environment variables."""
    import redis

    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        password=os.environ.get("REDIS_PASSWORD", None),
        db=int(os.environ.get("REDIS_DB", 0)),
        decode_responses=True,
    )


class AgentMemory:
    """Agent memory with semantic search capabilities using Redis."""

    # Model presets
    MODELS = {
        "fast": "sentence-transformers/all-MiniLM-L6-v2",
        "accurate": "sentence-transformers/all-mpnet-base-v2",
    }

    def __init__(
        self,
        index_name: str = "agent_memory",
        embedding_model: Optional[str] = None,
    ):
        self.index_name = index_name

        # Resolve model: allow explicit param, env var, or default
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            model_choice = os.environ.get("AGENT_MEMORY_MODEL", "fast")
            self.embedding_model = self.MODELS.get(model_choice, self.MODELS["fast"])

        self._conn = None
        self._model = None
        self._embedding_dim = 384

    @classmethod
    def list_models(cls) -> List[str]:
        """List available embedding models."""
        return list(cls.MODELS.keys())

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
            # Get actual embedding dimension
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def connect(self) -> None:
        """Connect to Redis."""
        if self._conn is not None:
            return
        self._conn = _get_redis_connection()
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure RediSearch index exists."""
        try:
            self._conn.execute_command("FT.INFO", self.index_name)
        except Exception:
            # Create simple text index (vector search falls back to Python)
            self._conn.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                f"{self.index_name}:mem:",
                "SCHEMA",
                "content",
                "TEXT",
                "context",
                "TEXT",
            )

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def remember(
        self, content: str, context: str = "default", ttl_days: Optional[int] = None
    ) -> str:
        """Store content in memory.

        Args:
            content: The content to store
            context: Context label for categorization
            ttl_days: Optional TTL in days. None = no expiry.

        Returns:
            str: The memory_id for later retrieval/deletion
        """
        if not self._conn:
            self.connect()

        import time

        embedding = self._embed(content)
        memory_id = f"mem:{int(time.time() * 1000000)}"
        key = f"{self.index_name}:{memory_id}"

        timestamp = datetime.utcnow().isoformat()
        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

        # Store as JSON in hash for Python fallback
        self._conn.hset(
            key,
            mapping={
                "content": content,
                "context": context,
                "embedding_json": json.dumps(embedding),
                "timestamp": timestamp,
                "expires_at": expires_at or "",
            },
        )

        return memory_id

    def recall(
        self,
        query: str,
        min_score: float = 0.3,
        limit: int = 5,
        context: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        keyword_boost: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search memory for relevant content.

        Args:
            query: Natural language search query
            min_score: Minimum similarity score (0.0-1.0)
            limit: Maximum results to return
            context: Optional context filter (e.g., "preferences", "project")
            since: Optional start datetime for filtering
            until: Optional end datetime for filtering
            keyword_boost: Keyword boost factor (0.0 = semantic only, 1.0 = max keyword boost)

        Returns:
            List[Tuple[str, float]]: List of (content, similarity_score) tuples
        """
        if not self._conn:
            self.connect()

        query_embedding = self._embed(query)
        query_terms = set(query.lower().split())

        # Python-based similarity search (more reliable than RediSearch vector)
        keys = self._conn.keys(f"{self.index_name}:mem:*")
        results = []
        for key in keys:
            try:
                data = self._conn.hgetall(key)
                if "embedding_json" in data:
                    try:
                        # Context filtering
                        if context and data.get("context") != context:
                            continue

                        # Time-based filtering
                        stored_timestamp = data.get("timestamp")
                        if stored_timestamp:
                            mem_time = datetime.fromisoformat(stored_timestamp)
                            if since and mem_time < since:
                                continue
                            if until and mem_time > until:
                                continue

                        # Check TTL expiry
                        expires_at = data.get("expires_at")
                        if expires_at:
                            expiry = datetime.fromisoformat(expires_at)
                            if datetime.utcnow() > expiry:
                                # Skip expired memories
                                continue

                        stored_embedding = json.loads(data["embedding_json"])
                        semantic_score = sum(
                            a * b for a, b in zip(query_embedding, stored_embedding)
                        )

                        # Hybrid: add keyword matching score
                        keyword_score = 0.0
                        if keyword_boost > 0:
                            content_lower = data.get("content", "").lower()
                            content_terms = set(content_lower.split())
                            matching_terms = query_terms & content_terms
                            if query_terms:
                                keyword_score = len(matching_terms) / len(query_terms)

                        # Combined score
                        similarity = semantic_score + (keyword_score * keyword_boost)

                        if similarity >= min_score:
                            results.append((data.get("content", ""), similarity))

                            # Update access metadata
                            mem_id = key.split(":")[-1]
                            access_count = int(data.get("access_count", 0))
                            self._conn.hset(
                                key,
                                mapping={
                                    "access_count": str(access_count + 1),
                                    "last_accessed": datetime.utcnow().isoformat(),
                                },
                            )
                    except json.JSONDecodeError, TypeError, ValueError:
                        continue
            except Exception:
                # Skip keys that have wrong type (e.g., string keys instead of hash)
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _fallback_recall(
        self, query_embedding: List[float], min_score: float, limit: int
    ) -> List[Tuple[str, float]]:
        """Fallback recall using Python-based similarity."""
        keys = self._conn.keys(f"{self.index_name}:mem:*")

        results = []
        for key in keys:
            try:
                data = self._conn.hgetall(key)
            except Exception:
                continue

            if "embedding_json" not in data:
                continue

            try:
                stored_embedding = json.loads(data["embedding_json"])
                similarity = sum(
                    a * b for a, b in zip(query_embedding, stored_embedding)
                )
                if similarity >= min_score:
                    results.append((data.get("content", ""), similarity))
            except json.JSONDecodeError, TypeError, ValueError:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _fallback_recall(
        self, query_embedding: List[float], min_score: float, limit: int
    ) -> List[Tuple[str, float]]:
        """Fallback recall using Python-based similarity."""
        keys = self._conn.keys(f"{self.index_name}:mem:*")

        results = []
        for key in keys:
            try:
                data = self._conn.hgetall(key)
            except Exception:
                continue

            if "embedding_json" not in data:
                continue

            try:
                stored_embedding = json.loads(data["embedding_json"])
                similarity = sum(
                    a * b for a, b in zip(query_embedding, stored_embedding)
                )
                if similarity >= min_score:
                    results.append((data.get("content", ""), similarity))
            except json.JSONDecodeError, TypeError, ValueError:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.

        Args:
            memory_id: The memory ID returned from remember()

        Returns:
            bool: True if deleted, False if not found
        """
        if not self._conn:
            self.connect()

        key = f"{self.index_name}:{memory_id}"
        return bool(self._conn.delete(key))

    def recent(
        self, limit: int = 10, context: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Get most recent memories.

        Args:
            limit: Maximum number of memories to return
            context: Optional context filter

        Returns:
            List[Tuple[str, str]]: List of (content, memory_id) tuples, most recent first
        """
        if not self._conn:
            self.connect()

        keys = self._conn.keys(f"{self.index_name}:mem:*")
        memories = []

        for key in keys:
            try:
                data = self._conn.hgetall(key)
                if "timestamp" in data:
                    # Context filtering
                    if context and data.get("context") != context:
                        continue

                    # Check TTL expiry
                    expires_at = data.get("expires_at")
                    if expires_at:
                        try:
                            expiry = datetime.fromisoformat(expires_at)
                            if datetime.utcnow() > expiry:
                                continue
                        except ValueError:
                            pass

                    memories.append(
                        (
                            data.get("content", ""),
                            key.split(":")[-1],
                            data.get("timestamp", ""),
                        )
                    )
            except Exception:
                continue

        # Sort by timestamp, most recent first
        memories.sort(key=lambda x: x[2], reverse=True)
        return [(content, mem_id) for content, mem_id, _ in memories[:limit]]

    def cleanup(self) -> int:
        """Remove all expired memories.

        Returns:
            int: Number of expired memories removed
        """
        if not self._conn:
            self.connect()

        keys = self._conn.keys(f"{self.index_name}:mem:*")
        removed = 0

        for key in keys:
            try:
                data = self._conn.hgetall(key)
                expires_at = data.get("expires_at")
                if expires_at:
                    try:
                        expiry = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expiry:
                            self._conn.delete(key)
                            removed += 1
                    except ValueError:
                        continue
            except Exception:
                continue

        return removed

    def close(self) -> None:
        """Close Redis connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._model = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def count(self) -> int:
        """Get total memory entries."""
        if not self._conn:
            self.connect()
        return len(self._conn.keys(f"{self.index_name}:mem:*"))

    def clear(self) -> int:
        """Delete all memories in this index. Returns count of deleted keys."""
        if not self._conn:
            self.connect()
        keys = self._conn.keys(f"{self.index_name}:mem:*")
        if keys:
            self._conn.delete(*keys)
        return len(keys)

    def remember_batch(
        self, items: List[Tuple[str, str]], ttl_days: Optional[int] = None
    ) -> List[str]:
        """Store multiple memories in a single call.

        Args:
            items: List of (content, context) tuples
            ttl_days: Optional TTL in days for all items

        Returns:
            List[str]: List of memory_ids
        """
        if not self._conn:
            self.connect()

        import time

        memory_ids = []
        for content, context in items:
            embedding = self._embed(content)
            memory_id = f"mem:{int(time.time() * 1000000)}"
            key = f"{self.index_name}:{memory_id}"

            timestamp = datetime.utcnow().isoformat()
            expires_at = None
            if ttl_days is not None:
                expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

            self._conn.hset(
                key,
                mapping={
                    "content": content,
                    "context": context,
                    "embedding_json": json.dumps(embedding),
                    "timestamp": timestamp,
                    "expires_at": expires_at or "",
                },
            )
            memory_ids.append(memory_id)

        return memory_ids

    def get_memory(self, memory_id: str) -> Optional[dict]:
        """Get a single memory by ID with metadata.

        Args:
            memory_id: The memory ID

        Returns:
            Optional[dict]: Memory data with metadata, or None if not found
        """
        if not self._conn:
            self.connect()

        key = f"{self.index_name}:{memory_id}"
        data = self._conn.hgetall(key)

        if not data or "content" not in data:
            return None

        return {
            "memory_id": memory_id,
            "content": data.get("content", ""),
            "context": data.get("context", "default"),
            "timestamp": data.get("timestamp", ""),
            "expires_at": data.get("expires_at", ""),
            "access_count": int(data.get("access_count", 0)),
            "last_accessed": data.get("last_accessed", ""),
        }

    def list_memories(
        self, limit: int = 50, offset: int = 0, context: Optional[str] = None
    ) -> List[dict]:
        """List all memories with optional context filter and pagination.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip (for pagination)
            context: Optional context filter

        Returns:
            List[dict]: List of memory data with metadata
        """
        if not self._conn:
            self.connect()

        keys = self._conn.keys(f"{self.index_name}:mem:*")
        memories = []

        for key in keys:
            try:
                data = self._conn.hgetall(key)
                if "content" not in data:
                    continue

                # Context filtering
                if context and data.get("context") != context:
                    continue

                # Check TTL expiry
                expires_at = data.get("expires_at")
                if expires_at:
                    try:
                        expiry = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expiry:
                            continue
                    except ValueError:
                        pass

                # Extract memory_id from key
                mem_id = key.split(":")[-1]
                memories.append(
                    {
                        "memory_id": mem_id,
                        "content": data.get("content", ""),
                        "context": data.get("context", "default"),
                        "timestamp": data.get("timestamp", ""),
                        "expires_at": data.get("expires_at", ""),
                        "access_count": int(data.get("access_count", 0)),
                        "last_accessed": data.get("last_accessed", ""),
                    }
                )
            except Exception:
                continue

        # Sort by timestamp, most recent first
        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories[offset : offset + limit]


# Convenience functions
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
    """Delete a specific memory by ID.

    Args:
        memory_id: The memory ID returned from remember()
        index_name: Memory index name

    Returns:
        bool: True if deleted
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.delete(memory_id)


def recent(
    limit: int = 10, context: Optional[str] = None, index_name: str = "agent_memory"
) -> List[Tuple[str, str]]:
    """Get most recent memories.

    Args:
        limit: Maximum memories to return
        context: Optional context filter
        index_name: Memory index name

    Returns:
        List[Tuple[str, str]]: List of (content, memory_id) tuples
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.recent(limit, context)


def cleanup(index_name: str = "agent_memory") -> int:
    """Remove all expired memories.

    Args:
        index_name: Memory index name

    Returns:
        int: Number of expired memories removed
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.cleanup()


def clear(index_name: str = "agent_memory") -> int:
    """Delete all memories in the index. Returns count of deleted entries."""
    with AgentMemory(index_name=index_name) as mem:
        return mem.clear()


def remember_batch(
    items: List[Tuple[str, str]],
    index_name: str = "agent_memory",
    ttl_days: Optional[int] = None,
) -> List[str]:
    """Store multiple memories in a single call.

    Args:
        items: List of (content, context) tuples
        index_name: Memory index name
        ttl_days: Optional TTL in days for all items

    Returns:
        List[str]: List of memory_ids
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.remember_batch(items, ttl_days)


def get_memory(memory_id: str, index_name: str = "agent_memory") -> Optional[dict]:
    """Get a single memory by ID with metadata.

    Args:
        memory_id: The memory ID
        index_name: Memory index name

    Returns:
        Optional[dict]: Memory data with metadata, or None if not found
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.get_memory(memory_id)


def list_memories(
    limit: int = 50,
    offset: int = 0,
    context: Optional[str] = None,
    index_name: str = "agent_memory",
) -> List[dict]:
    """List all memories with optional context filter and pagination.

    Args:
        limit: Maximum number of memories to return
        offset: Number of memories to skip (for pagination)
        context: Optional context filter
        index_name: Memory index name

    Returns:
        List[dict]: List of memory data with metadata
    """
    with AgentMemory(index_name=index_name) as mem:
        return mem.list_memories(limit, offset, context)


def export_memories(
    filepath: str,
    index_name: str = "agent_memory",
) -> int:
    """Export memories to JSON file.

    Args:
        filepath: Path to export file
        index_name: Memory index name

    Returns:
        int: Number of memories exported
    """
    with AgentMemory(index_name=index_name) as mem:
        memories = mem.list_memories(limit=10000)

        export_data = {
            "version": "1.0",
            "index_name": index_name,
            "exported_at": datetime.utcnow().isoformat(),
            "memories": [],
        }

        for mem_data in memories:
            mem_id = mem_data["memory_id"]
            key = f"{index_name}:{mem_id}"
            data = mem._conn.hgetall(key)
            if data and "embedding_json" in data:
                export_data["memories"].append(
                    {
                        "content": data.get("content", ""),
                        "context": data.get("context", "default"),
                        "timestamp": data.get("timestamp", ""),
                        "expires_at": data.get("expires_at", ""),
                        "embedding": json.loads(data["embedding_json"]),
                    }
                )

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        return len(export_data["memories"])


def import_memories(
    filepath: str,
    index_name: str = "agent_memory",
    merge: bool = True,
) -> int:
    """Import memories from JSON file.

    Args:
        filepath: Path to import file
        index_name: Target memory index name
        merge: If False, clear existing memories first

    Returns:
        int: Number of memories imported
    """
    with open(filepath, "r") as f:
        export_data = json.load(f)

    with AgentMemory(index_name=index_name) as mem:
        if not merge:
            mem.clear()

        import time

        imported = 0

        for mem_data in export_data.get("memories", []):
            embedding = mem_data.get("embedding")
            if not embedding:
                continue

            memory_id = f"mem:{int(time.time() * 1000000)}"
            key = f"{index_name}:{memory_id}"

            mem._conn.hset(
                key,
                mapping={
                    "content": mem_data.get("content", ""),
                    "context": mem_data.get("context", "default"),
                    "embedding_json": json.dumps(embedding),
                    "timestamp": mem_data.get(
                        "timestamp", datetime.utcnow().isoformat()
                    ),
                    "expires_at": mem_data.get("expires_at", ""),
                    "access_count": "0",
                    "last_accessed": "",
                },
            )
            imported += 1

        return imported


# Async implementation
class AgentMemoryAsync:
    """Async agent memory with semantic search capabilities using Redis."""

    def __init__(
        self,
        index_name: str = "agent_memory",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self._conn = None
        self._model = None
        self._pool = None

    async def _get_pool(self):
        """Get or create async connection pool."""
        if self._pool is None:
            import redis.asyncio as redis

            self._pool = redis.ConnectionPool(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD", None),
                db=int(os.environ.get("REDIS_DB", 0)),
                decode_responses=True,
            )
        return self._pool

    async def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    async def connect(self) -> None:
        """Connect to Redis."""
        pool = await self._get_pool()
        import redis.asyncio as redis

        self._conn = redis.Redis(connection_pool=pool)
        await self._ensure_index()

    async def _ensure_index(self) -> None:
        """Ensure RediSearch index exists."""
        try:
            await self._conn.execute_command("FT.INFO", self.index_name)
        except Exception:
            pass  # Index will be created on first use

    async def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = await self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    async def remember(
        self, content: str, context: str = "default", ttl_days: Optional[int] = None
    ) -> str:
        """Store content in memory.

        Args:
            content: The content to store
            context: Context label for categorization
            ttl_days: Optional TTL in days

        Returns:
            str: The memory_id
        """
        if not self._conn:
            await self.connect()

        import time

        embedding = await self._embed(content)
        memory_id = f"mem:{int(time.time() * 1000000)}"
        key = f"{self.index_name}:{memory_id}"

        timestamp = datetime.utcnow().isoformat()
        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

        await self._conn.hset(
            key,
            mapping={
                "content": content,
                "context": context,
                "embedding_json": json.dumps(embedding),
                "timestamp": timestamp,
                "expires_at": expires_at or "",
            },
        )

        return memory_id

    async def recall(
        self,
        query: str,
        min_score: float = 0.3,
        limit: int = 5,
        context: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        keyword_boost: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search memory for relevant content."""
        if not self._conn:
            await self.connect()

        query_embedding = await self._embed(query)
        query_terms = set(query.lower().split())

        keys = await self._conn.keys(f"{self.index_name}:mem:*")
        results = []

        for key in keys:
            try:
                data = await self._conn.hgetall(key)
                if "embedding_json" in data:
                    try:
                        # Context filtering
                        if context and data.get("context") != context:
                            continue

                        # Time-based filtering
                        stored_timestamp = data.get("timestamp")
                        if stored_timestamp:
                            mem_time = datetime.fromisoformat(stored_timestamp)
                            if since and mem_time < since:
                                continue
                            if until and mem_time > until:
                                continue

                        # Check TTL expiry
                        expires_at = data.get("expires_at")
                        if expires_at:
                            try:
                                expiry = datetime.fromisoformat(expires_at)
                                if datetime.utcnow() > expiry:
                                    continue
                            except ValueError:
                                pass

                        stored_embedding = json.loads(data["embedding_json"])
                        similarity = sum(
                            a * b for a, b in zip(query_embedding, stored_embedding)
                        )
                        if similarity >= min_score:
                            results.append((data.get("content", ""), similarity))
                    except json.JSONDecodeError, TypeError, ValueError:
                        continue
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        if not self._conn:
            await self.connect()

        key = f"{self.index_name}:{memory_id}"
        return bool(await self._conn.delete(key))

    async def recent(
        self, limit: int = 10, context: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Get most recent memories."""
        if not self._conn:
            await self.connect()

        keys = await self._conn.keys(f"{self.index_name}:mem:*")
        memories = []

        for key in keys:
            try:
                data = await self._conn.hgetall(key)
                if "timestamp" in data:
                    if context and data.get("context") != context:
                        continue

                    expires_at = data.get("expires_at")
                    if expires_at:
                        try:
                            expiry = datetime.fromisoformat(expires_at)
                            if datetime.utcnow() > expiry:
                                continue
                        except ValueError:
                            pass

                    memories.append(
                        (
                            data.get("content", ""),
                            key.split(":")[-1],
                            data.get("timestamp", ""),
                        )
                    )
            except Exception:
                continue

        memories.sort(key=lambda x: x[2], reverse=True)
        return [(content, mem_id) for content, mem_id, _ in memories[:limit]]

    async def cleanup(self) -> int:
        """Remove all expired memories."""
        if not self._conn:
            await self.connect()

        keys = await self._conn.keys(f"{self.index_name}:mem:*")
        removed = 0

        for key in keys:
            try:
                data = await self._conn.hgetall(key)
                expires_at = data.get("expires_at")
                if expires_at:
                    try:
                        expiry = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expiry:
                            await self._conn.delete(key)
                            removed += 1
                    except ValueError:
                        continue
            except Exception:
                continue

        return removed

    async def close(self) -> None:
        """Close Redis connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        self._model = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    async def count(self) -> int:
        """Get total memory entries."""
        if not self._conn:
            await self.connect()
        return len(await self._conn.keys(f"{self.index_name}:mem:*"))

    async def clear(self) -> int:
        """Delete all memories in this index."""
        if not self._conn:
            await self.connect()
        keys = await self._conn.keys(f"{self.index_name}:mem:*")
        if keys:
            await self._conn.delete(*keys)
        return len(keys)


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
) -> List[Tuple[str, float]]:
    """Search memory (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.recall(query, min_score, limit, context, since, until)


async def delete_async(memory_id: str, index_name: str = "agent_memory") -> bool:
    """Delete specific memory (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.delete(memory_id)


async def clear_async(index_name: str = "agent_memory") -> int:
    """Delete all memories (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.clear()


async def cleanup_async(index_name: str = "agent_memory") -> int:
    """Remove expired memories (async)."""
    async with AgentMemoryAsync(index_name=index_name) as mem:
        return await mem.cleanup()


__all__ = [
    "AgentMemory",
    "AgentMemoryAsync",
    "remember",
    "recall",
    "remember_async",
    "recall_async",
    "delete",
    "delete_async",
    "recent",
    "remember_batch",
    "get_memory",
    "list_memories",
    "export_memories",
    "import_memories",
    "cleanup",
    "cleanup_async",
    "clear",
    "clear_async",
]
