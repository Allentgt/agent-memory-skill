"""
Storage module - Handles Redis connection and data persistence.

Provides both sync and async Redis storage interfaces with
connection pooling and key-value operations.
"""

import os
import json
from typing import Optional, List, Dict, Any
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


class RedisStorage:
    """Sync Redis storage with connection management."""

    def __init__(self, index_name: str = "agent_memory"):
        self.index_name = index_name
        self._conn = None
        self._ensure_index_called = False

    @property
    def conn(self):
        """Lazy Redis connection."""
        if self._conn is None:
            import redis

            config = get_redis_config()
            self._conn = redis.Redis(**config, decode_responses=True)
        return self._conn

    def connect(self):
        """Connect and ensure index exists."""
        if not self._ensure_index_called:
            self._ensure_index()
            self._ensure_index_called = True
        return self

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.close()

    def _ensure_index(self):
        """Ensure RediSearch index exists (no-op for Python fallback)."""
        try:
            self.conn.execute_command("FT.INFO", self.index_name)
        except Exception:
            pass  # Index created on first memory store

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
        """Store memory data."""
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

    def get(self, memory_id: str) -> Optional[Dict[str, str]]:
        """Get memory data."""
        key = self._make_key(memory_id)
        data = self.conn.hgetall(key)
        return data if data else None

    def delete(self, memory_id: str) -> bool:
        """Delete memory."""
        key = self._make_key(memory_id)
        return bool(self.conn.delete(key))

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


class AsyncRedisStorage:
    """Async Redis storage with connection pooling."""

    def __init__(self, index_name: str = "agent_memory"):
        self.index_name = index_name
        self._conn = None
        self._pool = None

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
        """Store memory data."""
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

    async def get_all_keys(self) -> List[str]:
        """Get all memory keys."""
        conn = await self.conn
        return await conn.keys(f"{self.index_name}:mem:*")

    async def count(self) -> int:
        """Count all memories."""
        keys = await self.get_all_keys()
        return len(keys)

    async def clear(self) -> int:
        """Clear all memories, return count deleted."""
        keys = await self.get_all_keys()
        if keys:
            conn = await self.conn
            await conn.delete(*keys)
        return len(keys)
