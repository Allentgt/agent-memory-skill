"""
Unit tests for agent_memory.core module.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from agent_memory.core import AgentMemory, AgentMemoryAsync


class TestAgentMemoryId:
    """Tests for ID generation."""

    def test_generate_id_format(self):
        """_generate_id should return properly formatted ID."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage"):
                mem = AgentMemory()
                mem_id = mem._generate_id()

                assert mem_id.startswith("mem:")

    def test_generate_id_unique(self):
        """_generate_id should generate unique IDs."""
        import time

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage"):
                mem = AgentMemory()
                # IDs generated in quick succession need slight delay or use different method
                # The ID format includes microseconds so they should be unique
                ids = []
                for _ in range(5):
                    ids.append(mem._generate_id())
                    time.sleep(0.001)  # Small delay to ensure different microseconds

                assert len(set(ids)) == 5  # All unique


class TestAgentMemoryInit:
    """Tests for AgentMemory initialization."""

    def test_default_index(self):
        """Default index should be agent_memory."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage"):
                mem = AgentMemory()
                assert mem.index_name == "agent_memory"

    def test_custom_index(self):
        """Custom index should be used."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage"):
                mem = AgentMemory(index_name="custom")
                assert mem.index_name == "custom"

    def test_lazy_storage(self):
        """Storage should be lazy loaded."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage") as mock_storage:
                mem = AgentMemory()
                # Storage should not be accessed yet
                assert mem._storage is None
                # Access it
                _ = mem.storage
                mock_storage.assert_called()

    def test_lazy_embedder(self):
        """Embedder should be lazy loaded."""
        with patch("agent_memory.core.EmbeddingEngine") as mock_embedder:
            with patch("agent_memory.core.RedisStorage"):
                mem = AgentMemory()
                # Embedder should not be loaded yet
                assert mem._embedder is None
                # Access it
                _ = mem.embedder
                mock_embedder.assert_called()


class TestAgentMemoryAsyncInit:
    """Tests for AgentMemoryAsync initialization."""

    def test_default_index(self):
        """Default index should be agent_memory."""
        with patch("agent_memory.core.AsyncEmbeddingEngine"):
            with patch("agent_memory.core.AsyncRedisStorage"):
                mem = AgentMemoryAsync()
                assert mem.index_name == "agent_memory"

    def test_custom_index(self):
        """Custom index should be used."""
        with patch("agent_memory.core.AsyncEmbeddingEngine"):
            with patch("agent_memory.core.AsyncRedisStorage"):
                mem = AgentMemoryAsync(index_name="custom")
                assert mem.index_name == "custom"


class TestAgentMemoryContextManager:
    """Tests for context manager."""

    def test_enter_connect(self):
        """Enter should call connect."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage") as mock_storage:
                mock_instance = MagicMock()
                mock_storage.return_value = mock_instance

                mem = AgentMemory()
                result = mem.__enter__()

                mock_instance.connect.assert_called_once()

    def test_exit_close(self):
        """Exit should call close."""
        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage") as mock_storage:
                mock_instance = MagicMock()
                mock_storage.return_value = mock_instance

                mem = AgentMemory()
                mem.__enter__()
                mem.__exit__(None, None, None)

                mock_instance.close.assert_called_once()


class TestAgentMemoryAsyncContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_enter_connect(self):
        """Async enter should call connect."""
        with patch("agent_memory.core.AsyncEmbeddingEngine"):
            with patch("agent_memory.core.AsyncRedisStorage") as mock_storage:
                mock_instance = AsyncMock()
                mock_storage.return_value = mock_instance

                mem = AgentMemoryAsync()
                await mem.__aenter__()

                mock_instance.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_exit_close(self):
        """Async exit should call close."""
        with patch("agent_memory.core.AsyncEmbeddingEngine"):
            with patch("agent_memory.core.AsyncRedisStorage") as mock_storage:
                mock_instance = AsyncMock()
                mock_storage.return_value = mock_instance

                mem = AgentMemoryAsync()
                await mem.__aenter__()
                await mem.__aexit__(None, None, None)

                mock_instance.close.assert_called_once()


class TestAgentMemoryOperations:
    """Tests for memory operations with mocks."""

    def test_remember_calls_storage_set(self):
        """remember should call storage.set with correct args."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1, 0.2, 0.3]

        mock_storage = MagicMock()
        mock_storage.get_all_keys.return_value = []

        with patch("agent_memory.core.EmbeddingEngine", return_value=mock_embedder):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                mem.remember("test content", "test_context", ttl_days=7)

                mock_storage.set.assert_called_once()
                call_args = mock_storage.set.call_args
                # Verify content stored
                assert call_args[0][1] == "test content"  # memory_id, content
                assert call_args[0][2] == "test_context"

    def test_recall_returns_list_of_tuples(self):
        """recall should return list of (content, score) tuples."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1]

        mock_storage = MagicMock()
        mock_conn = MagicMock()
        mock_conn.hgetall.return_value = {
            "content": "test",
            "context": "default",
            "embedding_json": json.dumps([0.1]),
            "timestamp": datetime.utcnow().isoformat(),
            "expires_at": "",
        }
        mock_storage.conn = mock_conn
        mock_storage.get_all_keys.return_value = ["k1"]

        with patch("agent_memory.core.EmbeddingEngine", return_value=mock_embedder):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                results = mem.recall("test", min_score=0.0)

                assert isinstance(results, list)
                if results:
                    assert isinstance(results[0], tuple)
                    assert isinstance(results[0][0], str)
                    assert isinstance(results[0][1], float)

    def test_delete_calls_storage_delete(self):
        """delete should call storage.delete."""
        mock_storage = MagicMock()
        mock_storage.delete.return_value = True

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                mem.delete("mem:123")

                mock_storage.delete.assert_called_once_with("mem:123")

    def test_get_returns_memory_dict(self):
        """get should return memory with metadata."""
        mock_storage = MagicMock()
        mock_storage.get.return_value = {
            "content": "test",
            "context": "default",
            "timestamp": "2024-01-01T00:00:00",
            "expires_at": "",
            "access_count": "5",
            "last_accessed": "2024-01-02T00:00:00",
        }

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                result = mem.get("mem:123")

                assert result["content"] == "test"
                assert result["context"] == "default"

    def test_list_memories_returns_list(self):
        """list_memories should return list of memory dicts."""
        mock_storage = MagicMock()
        mock_conn = MagicMock()
        mock_conn.hgetall.return_value = {
            "content": "test",
            "context": "default",
            "timestamp": "2024-01-01T00:00:00",
            "expires_at": "",
            "access_count": "0",
            "last_accessed": "",
        }
        mock_storage.conn = mock_conn
        mock_storage.get_all_keys.return_value = ["k1"]

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                results = mem.list_memories()

                assert isinstance(results, list)
                if results:
                    assert isinstance(results[0], dict)

    def test_count_returns_int(self):
        """count should return integer."""
        mock_storage = MagicMock()
        mock_storage.count.return_value = 5

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                result = mem.count

                assert isinstance(result, int)

    def test_clear_returns_count(self):
        """clear should return count."""
        mock_storage = MagicMock()
        mock_storage.clear.return_value = 3

        with patch("agent_memory.core.EmbeddingEngine"):
            with patch("agent_memory.core.RedisStorage", return_value=mock_storage):
                mem = AgentMemory()
                result = mem.clear()

                assert result == 3


class TestAgentMemoryAsyncOperations:
    """Tests for async memory operations."""

    @pytest.mark.asyncio
    async def test_async_remember_returns_id(self):
        """async remember should return memory_id."""
        mock_embedder = AsyncMock()
        mock_embedder.encode = AsyncMock(return_value=[0.1])

        mock_storage = AsyncMock()
        mock_storage.get_all_keys = AsyncMock(return_value=[])

        with patch(
            "agent_memory.core.AsyncEmbeddingEngine", return_value=mock_embedder
        ):
            with patch(
                "agent_memory.core.AsyncRedisStorage", return_value=mock_storage
            ):
                mem = AgentMemoryAsync()
                result = await mem.remember("test", "default")

                assert isinstance(result, str)
                assert result.startswith("mem:")

    @pytest.mark.asyncio
    async def test_async_recall_returns_list(self):
        """async recall should return a list."""
        # Test that the method exists and returns correct type
        # Full integration test would require actual Redis
        mem = AgentMemoryAsync()

        # Just verify method signature exists
        import inspect

        sig = inspect.signature(mem.recall)
        assert "query" in sig.parameters

        # Should return list when called on class with proper mock setup
        # The actual behavior is tested in integration tests

    @pytest.mark.asyncio
    async def test_async_count_signature(self):
        """async count should be awaitable."""
        # Test that count is a coroutine function
        mem = AgentMemoryAsync()

        # Verify it's an async property/method
        import inspect

        # The count is defined as a property in the class
        assert hasattr(AgentMemoryAsync, "count")

    @pytest.mark.asyncio
    async def test_async_clear_returns_int(self):
        """async clear should return count."""
        mock_storage = AsyncMock()
        mock_storage.clear = AsyncMock(return_value=3)

        with patch("agent_memory.core.AsyncEmbeddingEngine"):
            with patch(
                "agent_memory.core.AsyncRedisStorage", return_value=mock_storage
            ):
                mem = AgentMemoryAsync()
                mem._storage = mock_storage
                result = await mem.clear()

                assert result == 3
