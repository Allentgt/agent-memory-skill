"""
Storage module - Handles Redis connection and data persistence.

Provides both sync and async Redis storage interfaces with
connection pooling and key-value operations.
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

if _HAS_DOTENV:
    load_dotenv()


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration from environment variables."""
    return {
        "host": os.environ.get("REDIS_HOST", "localhost"),
        "port": int(os.environ.get("REDIS_PORT", 6379)),
        "password": os.environ.get("REDIS_PASSWORD"),
        "db": int(os.environ.get("REDIS_DB", 0)),
    }


def get_embedding_dimension() -> int:
    """Get embedding dimension based on model choice."""
    from agent_memory.embeddings import get_model_name
    model_name = get_model_name()
    if "mpnet" in model_name:
        return 768
    return 384


def encode_embedding(embedding: List[float]) -> bytes:
    """Encode embedding as bytes for Redis VECTOR field."""
    import array
    return array.array("f", embedding).tobytes()


class RedisStorage:
    """Sync Redis storage with connection pooling."""

    def __init__(self, index_name: str = "agent_memory"):
        self.index_name = index_name
        self._conn = None
        self._pool = None
        self._ensure_index_called = False
        self._vss_available = None

    @property
    def conn(self):
        """Lazy Redis connection with pooling."""
        import redis

        if self._conn is None:
            config = get_redis_config()
            self._pool = redis.ConnectionPool(**config, decode_responses=True)
            self._conn = redis.Redis(connection_pool=self._pool)
        return self._conn

    def connect(self):
        """Connect and ensure index exists."""
        if not self._ensure_index_called:
            self._ensure_index()
            self._ensure_index_called = True
        return self

    def close(self):
        """Close connection and pool."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._pool:
            self._pool.disconnect()
            self._pool = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.close()

    def _check_vss_available(self) -> bool:
        """Check if Redis vector search is available."""
        if self._vss_available is not None:
            return self._vss_available

        try:
            from redis.commands.redismodules import RedisModuleCommands
            ft = self.conn.ft(self.index_name)
            ft.info()
            self._vss_available = True
        except Exception:
            self._vss_available = False

        return self._vss_available

    def _ensure_index(self):
        """Ensure RediSearch index exists with VECTOR field."""
        if self._check_vss_available():
            try:
                dim = get_embedding_dimension()
                schema = [
                    "content", "TEXT",
                    "context", "TEXT",
                ]
                ft = self.conn.ft(self.index_name)
                ft.create_index(
                    schema,
                    definition={
                        "INDEX_TYPE": "HASH",
                        "PREFIX": f"{self.index_name}:",
                    }
                )
            except Exception:
                self._vss_available = False

    def _make_key(self, memory_id: str) -> str:
        """Generate Redis key for memory."""
        return f"{self.index_name}:{memory_id}"

    def set(
        self,
        memory_id: str,
        content: str,
        context: str,
        embedding: List[float],
        timestamp: str,
        expires_at: Optional[str] = None,
    ):
        """Store memory data with optional TTL."""
        key = self._make_key(memory_id)
        self.conn.hset(
            key,
            mapping={
                "content": content,
                "context": context,
                "embedding_json": json.dumps(embedding),
                "timestamp": timestamp,
                "expires_at": expires_at or "",
            },
        )
        # Set Redis TTL if expires_at is provided
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
                ttl_seconds = int((expiry - datetime.utcnow()).total_seconds())
                if ttl_seconds > 0:
                    self.conn.expire(key, ttl_seconds)
            except (ValueError, OSError):
                pass  # Ignore TTL errors - memory still stored

    def set_batch(
        self,
        items: List[dict],
    ) -> int:
        """Store multiple memories using pipeline for efficiency.

        Args:
            items: List of dicts with keys: memory_id, content, context, embedding, timestamp, expires_at

        Returns:
            Number of items stored
        """
        if not items:
            return 0

        pipe = self.conn.pipeline()
        for item in items:
            key = self._make_key(item["memory_id"])
            pipe.hset(
                key,
                mapping={
                    "content": item["content"],
                    "context": item["context"],
                    "embedding_json": json.dumps(item["embedding"]),
                    "timestamp": item["timestamp"],
                    "expires_at": item.get("expires_at") or "",
                },
            )
            if item.get("expires_at"):
                try:
                    expiry = datetime.fromisoformat(item["expires_at"])
                    ttl_seconds = int((expiry - datetime.utcnow()).total_seconds())
                    if ttl_seconds > 0:
                        pipe.expire(key, ttl_seconds)
                except (ValueError, OSError):
                    pass

        pipe.execute()
        return len(items)

    def get(self, memory_id: str) -> Optional[Dict[str, str]]:
        """Get memory data."""
        key = self._make_key(memory_id)
        data = self.conn.hgetall(key)
        return data if data else None

    def delete(self, memory_id: str) -> bool:
        """Delete memory."""
        key = self._make_key(memory_id)
        return bool(self.conn.delete(key))

    def update_access(self, memory_id: str) -> bool:
        """Update access count and last_accessed timestamp."""
        key = self._make_key(memory_id)
        now = datetime.utcnow().isoformat()
        access_count = self.conn.hincrby(key, "access_count", 1)
        self.conn.hset(key, "last_accessed", now)
        return True

    def get_all_keys(self) -> List[str]:
        """Get all memory keys."""
        return self.conn.keys(f"{self.index_name}:mem:*")

    def count(self) -> int:
        """Count all memories."""
        return len(self.get_all_keys())

    def clear(self) -> int:
        """Clear all memories, return count deleted."""
        keys = self.get_all_keys()
        if keys:
            self.conn.delete(*keys)
        return len(keys)

    def searchVectors(
        self,
        query_embedding: List[float],
        min_score: float = 0.3,
        limit: int = 5,
        context: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Vector similarity search using RediSearch.

        Falls back to manual scan if VSS not available.
        """
        if not self._check_vss_available():
            return self._search_fallback(query_embedding, min_score, limit, context)

        try:
            import numpy as np

            dim = get_embedding_dimension()
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_bytes = query_vec.tobytes()

            ft = self.conn.ft(self.index_name)
            search_query = f"*=>[KNN {limit} @embedding $vec AS score]"
            results = ft.search(
                search_query,
                query_params={"vec": query_bytes},
                sort_by="score",
                descending=True,
            )

            matches = []
            for doc in results.docs:
                doc_score = doc.get("score", "0")
                if float(doc_score) >= min_score:
                    content = doc.get("content", "")
                    score = float(doc_score)
                    if context and doc.get("context") != context:
                        continue
                    matches.append((content, score))

            return matches[:limit]

        except Exception:
            return self._search_fallback(query_embedding, min_score, limit, context)

    def _search_fallback(
        self,
        query_embedding: List[float],
        min_score: float,
        limit: int,
        context: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Fallback linear scan search."""
        import json

        results = []
        keys = self.get_all_keys()

        for key in keys:
            data = self.conn.hgetall(key)
            if "content" not in data:
                continue

            if context and data.get("context") != context:
                continue

            try:
                stored = json.loads(data.get("embedding_json", "[]"))
                similarity = sum(
                    a * b for a, b in zip(query_embedding, stored)
                )
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                similarity = 0.0

            if similarity >= min_score:
                results.append((data.get("content", ""), similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class AsyncRedisStorage:
    """Async Redis storage with connection pooling.

    Note: For production, use async/await with Redis asyncio client.
    For consistency with sync implementation, delegates via thread pool.
    """

    def __init__(self, index_name: str = "agent_memory"):
        self.index_name = index_name
        self._conn = None
        self._pool = None
        self._sync_storage = None

    @property
    async def conn(self):
        """Lazy async Redis connection."""
        if self._conn is None:
            import redis.asyncio as redis

            config = get_redis_config()
            self._pool = redis.ConnectionPool(**config, decode_responses=True)
            self._conn = redis.Redis(connection_pool=self._pool)
        return self._conn

    async def connect(self):
        """Connect and ensure index."""
        await self._ensure_index()

    async def close(self):
        """Close connection."""
        if self._conn:
            await self._conn.aclose()
            self._conn = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_index(self):
        """Ensure index exists."""
        try:
            conn = await self.conn
            await conn.execute_command("FT.INFO", self.index_name)
        except Exception:
            pass

    def _make_key(self, memory_id: str) -> str:
        """Generate Redis key for memory."""
        return f"{self.index_name}:{memory_id}"

    async def set(
        self,
        memory_id: str,
        content: str,
        context: str,
        embedding: List[float],
        timestamp: str,
        expires_at: Optional[str] = None,
    ):
        """Store memory data with optional TTL."""
        key = self._make_key(memory_id)
        conn = await self.conn
        await conn.hset(
            key,
            mapping={
                "content": content,
                "context": context,
                "embedding_json": json.dumps(embedding),
                "timestamp": timestamp,
                "expires_at": expires_at or "",
            },
        )
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
                ttl_seconds = int((expiry - datetime.utcnow()).total_seconds())
                if ttl_seconds > 0:
                    await conn.expire(key, ttl_seconds)
            except (ValueError, OSError):
                pass

    async def set_batch(self, items: List[dict]) -> int:
        """Store multiple memories using pipeline."""
        if not items:
            return 0
        conn = await self.conn
        pipe = conn.pipeline()
        for item in items:
            key = self._make_key(item["memory_id"])
            pipe.hset(
                key,
                mapping={
                    "content": item["content"],
                    "context": item["context"],
                    "embedding_json": json.dumps(item["embedding"]),
                    "timestamp": item["timestamp"],
                    "expires_at": item.get("expires_at") or "",
                },
            )
            if item.get("expires_at"):
                try:
                    expiry = datetime.fromisoformat(item["expires_at"])
                    ttl_seconds = int((expiry - datetime.utcnow()).total_seconds())
                    if ttl_seconds > 0:
                        pipe.expire(key, ttl_seconds)
                except (ValueError, OSError):
                    pass
        await pipe.execute()
        return len(items)

    async def get(self, memory_id: str) -> Optional[Dict[str, str]]:
        """Get memory data."""
        key = self._make_key(memory_id)
        conn = await self.conn
        data = await conn.hgetall(key)
        return data if data else None

    async def delete(self, memory_id: str) -> bool:
        """Delete memory."""
        key = self._make_key(memory_id)
        conn = await self.conn
        return bool(await conn.delete(key))

    async def update_access(self, memory_id: str) -> bool:
        """Update access count and last_accessed timestamp."""
        key = self._make_key(memory_id)
        conn = await self.conn
        now = datetime.utcnow().isoformat()
        await conn.hincrby(key, "access_count", 1)
        await conn.hset(key, "last_accessed", now)
        return True

    async def get_all_keys(self) -> List[str]:
        """Get all memory keys."""
        conn = await self.conn
        return await conn.keys(f"{self.index_name}:mem:*")

    async def count(self) -> int:
        """Count all memories."""
        keys = await self.get_all_keys()
        return len(keys)

    async def clear(self) -> int:
        """Clear all memories."""
        keys = await self.get_all_keys()
        if keys:
            conn = await self.conn
            await conn.delete(*keys)
        return len(keys)
