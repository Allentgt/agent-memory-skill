"""
Agent Memory Skill - Redis-based semantic memory for AI agents.

Uses Redis for storage with local embeddings via sentence-transformers.
Fully local, free to use, supports semantic search.
"""

import os
import json
import struct
from pathlib import Path
from typing import Optional, List, Tuple

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

    def __init__(
        self,
        index_name: str = "agent_memory",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self._conn = None
        self._model = None
        self._embedding_dim = 384

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

    def remember(self, content: str, context: str = "default") -> None:
        """Store content in memory."""
        if not self._conn:
            self.connect()

        import time

        embedding = self._embed(content)
        memory_id = f"mem:{int(time.time() * 1000000)}"
        key = f"{self.index_name}:{memory_id}"

        # Store as JSON in hash for Python fallback
        self._conn.hset(
            key,
            mapping={
                "content": content,
                "context": context,
                "embedding_json": json.dumps(embedding),
            },
        )

    def recall(
        self, query: str, min_score: float = 0.3, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Search memory for relevant content."""
        if not self._conn:
            self.connect()

        query_embedding = self._embed(query)

        # Python-based similarity search (more reliable than RediSearch vector)
        keys = self._conn.keys(f"{self.index_name}:mem:*")
        results = []
        for key in keys:
            try:
                data = self._conn.hgetall(key)
                if "embedding_json" in data:
                    try:
                        stored_embedding = json.loads(data["embedding_json"])
                        similarity = sum(
                            a * b for a, b in zip(query_embedding, stored_embedding)
                        )
                        if similarity >= min_score:
                            results.append((data.get("content", ""), similarity))
                    except (json.JSONDecodeError, TypeError):
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
            data = self._conn.hgetall(key)
            if "embedding_json" in data:
                try:
                    stored_embedding = json.loads(data["embedding_json"])
                    similarity = sum(
                        a * b for a, b in zip(query_embedding, stored_embedding)
                    )
                    if similarity >= min_score:
                        results.append((data.get("content", ""), similarity))
                except Exception:
                    continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

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


# Convenience functions
def remember(content: str, context: str = "default", **kwargs) -> None:
    """Store content in memory."""
    with AgentMemory(**kwargs) as mem:
        mem.remember(content, context)


def recall(query: str, min_score: float = 0.3, **kwargs) -> List[Tuple[str, float]]:
    """Search memory."""
    with AgentMemory(**kwargs) as mem:
        return mem.recall(query, min_score)


__all__ = ["AgentMemory", "remember", "recall"]
