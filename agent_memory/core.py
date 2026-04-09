"""
Core memory module - Pure business logic for Agent Memory.

Contains the AgentMemory class that orchestrates embeddings and storage
to provide semantic memory operations. No I/O dependencies here.
"""

import json
import time
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

from agent_memory.embeddings import (
    EmbeddingEngine,
    AsyncEmbeddingEngine,
    get_model_name,
)
from agent_memory.storage import RedisStorage, AsyncRedisStorage


class AgentMemory:
    """Sync agent memory orchestrator.

    Coordinates embedding generation and storage to provide semantic memory.
    Uses context manager for proper resource cleanup.
    """

    def __init__(
        self, index_name: str = "agent_memory", embedding_model: Optional[str] = None
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model or get_model_name()
        self._storage = None
        self._embedder = None

    @property
    def storage(self) -> RedisStorage:
        """Lazy Redis storage."""
        if self._storage is None:
            self._storage = RedisStorage(self.index_name)
        return self._storage

    @property
    def embedder(self) -> EmbeddingEngine:
        """Lazy embedding engine."""
        if self._embedder is None:
            self._embedder = EmbeddingEngine(self.embedding_model)
        return self._embedder

    def connect(self):
        """Connect storage (called before operations)."""
        self.storage.connect()

    def close(self):
        """Close storage connection."""
        if self._storage:
            self._storage.close()
            self._storage = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        return f"mem:{int(time.time() * 1000000)}"

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
        embedding = self.embedder.encode(content)
        memory_id = self._generate_id()

        timestamp = datetime.utcnow().isoformat()
        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

        self.storage.set(memory_id, content, context, embedding, timestamp, expires_at)
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
            context: Optional context filter
            since: Optional start datetime for filtering
            until: Optional end datetime for filtering
            keyword_boost: Keyword boost factor (0.0 = semantic only, 1.0 = max keyword)

        Returns:
            List[Tuple[str, float]]: List of (content, similarity_score) tuples
        """
        query_embedding = self.embedder.encode(query)
        query_terms = set(query.lower().split())

        keys = self.storage.get_all_keys()
        results = []

        for key in keys:
            data = self.storage.conn.hgetall(key)
            if "content" not in data:
                continue

            # Context filtering
            if context and data.get("context") != context:
                continue

            # Time filtering
            if since or until:
                ts = data.get("timestamp")
                if ts:
                    mem_time = datetime.fromisoformat(ts)
                    if since and mem_time < since:
                        continue
                    if until and mem_time > until:
                        continue

            # TTL check
            expires_at = data.get("expires_at")
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expiry:
                        continue
                except ValueError:
                    pass

            # Calculate similarity
            try:
                stored_embedding = json.loads(data["embedding_json"])
                similarity = sum(
                    a * b for a, b in zip(query_embedding, stored_embedding)
                )
            except json.JSONDecodeError, TypeError, ValueError, KeyError:
                similarity = 0.0

            # Keyword boost
            if keyword_boost > 0:
                content_lower = data.get("content", "").lower()
                matches = sum(1 for term in query_terms if term in content_lower)
                keyword_score = matches / max(len(query_terms), 1)
                similarity = (
                    similarity * (1 - keyword_boost) + keyword_score * keyword_boost
                )

            if similarity >= min_score:
                results.append((data.get("content", ""), similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        return self.storage.delete(memory_id)

    def get(self, memory_id: str) -> Optional[dict]:
        """Get memory with metadata."""
        data = self.storage.get(memory_id)
        if not data:
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
        """List all memories with optional context filter."""
        keys = self.storage.get_all_keys()
        memories = []

        for key in keys:
            data = self.storage.conn.hgetall(key)
            if "content" not in data:
                continue

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

        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories[offset : offset + limit]

    @property
    def count(self) -> int:
        """Get total memory count."""
        return self.storage.count()

    def clear(self) -> int:
        """Clear all memories, return count deleted."""
        return self.storage.clear()


class AgentMemoryAsync:
    """Async agent memory orchestrator."""

    def __init__(
        self, index_name: str = "agent_memory", embedding_model: Optional[str] = None
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model or get_model_name()
        self._storage = None
        self._embedder = None

    @property
    async def storage(self) -> AsyncRedisStorage:
        """Lazy async Redis storage."""
        if self._storage is None:
            self._storage = AsyncRedisStorage(self.index_name)
        return self._storage

    @property
    async def embedder(self) -> AsyncEmbeddingEngine:
        """Lazy async embedding engine."""
        if self._embedder is None:
            self._embedder = AsyncEmbeddingEngine(self.embedding_model)
        return self._embedder

    async def connect(self):
        """Connect storage."""
        await (await self.storage).connect()

    async def close(self):
        """Close storage connection."""
        if self._storage:
            await self._storage.close()
            self._storage = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _generate_id(self) -> str:
        """Generate unique memory ID."""
        return f"mem:{int(time.time() * 1000000)}"

    async def remember(
        self, content: str, context: str = "default", ttl_days: Optional[int] = None
    ) -> str:
        """Store content in memory (async)."""
        embedder = await self.embedder
        storage = await self.storage

        embedding = await embedder.encode(content)
        memory_id = await self._generate_id()

        timestamp = datetime.utcnow().isoformat()
        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

        await storage.set(memory_id, content, context, embedding, timestamp, expires_at)
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
        """Search memory for relevant content (async)."""
        embedder = await self.embedder
        storage = await self.storage

        query_embedding = await embedder.encode(query)
        query_terms = set(query.lower().split())

        keys = await storage.get_all_keys()
        results = []

        conn = await storage.conn
        for key in keys:
            data = await conn.hgetall(key)
            if "content" not in data:
                continue

            if context and data.get("context") != context:
                continue

            if since or until:
                ts = data.get("timestamp")
                if ts:
                    mem_time = datetime.fromisoformat(ts)
                    if since and mem_time < since:
                        continue
                    if until and mem_time > until:
                        continue

            expires_at = data.get("expires_at")
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expiry:
                        continue
                except ValueError:
                    pass

            try:
                stored_embedding = json.loads(data["embedding_json"])
                similarity = sum(
                    a * b for a, b in zip(query_embedding, stored_embedding)
                )
            except json.JSONDecodeError, TypeError, ValueError, KeyError:
                similarity = 0.0

            if keyword_boost > 0:
                content_lower = data.get("content", "").lower()
                matches = sum(1 for term in query_terms if term in content_lower)
                keyword_score = matches / max(len(query_terms), 1)
                similarity = (
                    similarity * (1 - keyword_boost) + keyword_score * keyword_boost
                )

            if similarity >= min_score:
                results.append((data.get("content", ""), similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID (async)."""
        storage = await self.storage
        return await storage.delete(memory_id)

    async def get(self, memory_id: str) -> Optional[dict]:
        """Get memory with metadata (async)."""
        storage = await self.storage
        data = await storage.get(memory_id)
        if not data:
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

    async def list_memories(
        self, limit: int = 50, offset: int = 0, context: Optional[str] = None
    ) -> List[dict]:
        """List all memories (async)."""
        storage = await self.storage
        conn = await storage.conn

        keys = await storage.get_all_keys()
        memories = []

        for key in keys:
            data = await conn.hgetall(key)
            if "content" not in data:
                continue

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

        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories[offset : offset + limit]

    async def count(self) -> int:
        """Get total memory count (async)."""
        storage = await self.storage
        return await storage.count()

    async def clear(self) -> int:
        """Clear all memories (async)."""
        storage = await self.storage
        return await storage.clear()
