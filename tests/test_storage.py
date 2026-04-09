"""
Unit tests for agent_memory.storage module.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agent_memory.storage import (
    get_redis_config,
    RedisStorage,
    AsyncRedisStorage,
)


class TestRedisConfig:
    """Tests for Redis configuration."""

    def test_get_redis_config_defaults(self):
        """get_redis_config should return defaults when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = get_redis_config()

            assert config["host"] == "localhost"
            assert config["port"] == 6379
            assert config["db"] == 0
            assert config["password"] is None

    def test_get_redis_config_from_env(self):
        """get_redis_config should read from environment."""
        with patch.dict(
            "os.environ",
            {
                "REDIS_HOST": "redis.example.com",
                "REDIS_PORT": "6380",
                "REDIS_DB": "1",
                "REDIS_PASSWORD": "secret",
            },
        ):
            config = get_redis_config()

            assert config["host"] == "redis.example.com"
            assert config["port"] == 6380
            assert config["db"] == 1
            assert config["password"] == "secret"

    def test_get_redis_config_port_conversion(self):
        """get_redis_config should convert port to int."""
        with patch.dict("os.environ", {"REDIS_PORT": "6380"}):
            config = get_redis_config()
            assert isinstance(config["port"], int)
            assert config["port"] == 6380


class TestRedisStorageKeyGeneration:
    """Tests for RedisStorage key generation (no Redis needed)."""

    def test_make_key(self):
        """_make_key should generate correct Redis key."""
        storage = RedisStorage(index_name="test_index")
        key = storage._make_key("mem:123")

        assert key == "test_index:mem:123"

    def test_make_key_different_index(self):
        """_make_key should use index_name."""
        storage = RedisStorage(index_name="custom")
        key = storage._make_key("mem:456")

        assert key == "custom:mem:456"

    def test_default_index_name(self):
        """Default index name should be agent_memory."""
        storage = RedisStorage()
        key = storage._make_key("mem:789")

        assert key == "agent_memory:mem:789"


class TestAsyncRedisStorageKeyGeneration:
    """Tests for AsyncRedisStorage key generation."""

    def test_make_key(self):
        """_make_key should generate correct Redis key."""
        storage = AsyncRedisStorage(index_name="test_index")
        key = storage._make_key("mem:123")

        assert key == "test_index:mem:123"

    def test_make_key_different_index(self):
        """_make_key should use index_name."""
        storage = AsyncRedisStorage(index_name="custom")
        key = storage._make_key("mem:456")

        assert key == "custom:mem:456"


class TestRedisStorageInit:
    """Tests for RedisStorage initialization."""

    def test_default_index(self):
        """Default index should be agent_memory."""
        storage = RedisStorage()
        assert storage.index_name == "agent_memory"

    def test_custom_index(self):
        """Custom index should be used."""
        storage = RedisStorage(index_name="my_index")
        assert storage.index_name == "my_index"

    def test_initial_state(self):
        """Initial state should have no connection."""
        storage = RedisStorage()
        assert storage._conn is None
        assert storage._ensure_index_called is False


class TestAsyncRedisStorageInit:
    """Tests for AsyncRedisStorage initialization."""

    def test_default_index(self):
        """Default index should be agent_memory."""
        storage = AsyncRedisStorage()
        assert storage.index_name == "agent_memory"

    def test_custom_index(self):
        """Custom index should be used."""
        storage = AsyncRedisStorage(index_name="my_index")
        assert storage.index_name == "my_index"

    def test_initial_state(self):
        """Initial state should have no connection."""
        storage = AsyncRedisStorage()
        assert storage._conn is None
        assert storage._pool is None


class TestRedisStorageWithMock:
    """Tests for RedisStorage with mocked connection."""

    def test_set_stores_data_correctly(self):
        """set should store data with correct keys."""
        storage = RedisStorage()

        # Create a mock connection
        mock_conn = MagicMock()
        storage._conn = mock_conn

        storage.set(
            memory_id="mem:123",
            content="test content",
            context="test_context",
            embedding=[0.1, 0.2],
            timestamp="2024-01-01T00:00:00",
            expires_at=None,
        )

        mock_conn.hset.assert_called_once()
        call_kwargs = mock_conn.hset.call_args[1]
        assert call_kwargs["mapping"]["content"] == "test content"
        assert call_kwargs["mapping"]["context"] == "test_context"
        assert json.loads(call_kwargs["mapping"]["embedding_json"]) == [0.1, 0.2]

    def test_get_returns_data(self):
        """get should return stored data."""
        storage = RedisStorage()
        mock_conn = MagicMock()
        mock_conn.hgetall.return_value = {"content": "test", "context": "default"}
        storage._conn = mock_conn

        result = storage.get("mem:123")

        assert result["content"] == "test"
        assert result["context"] == "default"

    def test_get_returns_none_for_empty(self):
        """get should return None for empty result."""
        storage = RedisStorage()
        mock_conn = MagicMock()
        mock_conn.hgetall.return_value = {}
        storage._conn = mock_conn

        result = storage.get("mem:missing")

        assert result is None

    def test_delete_calls_redis_delete(self):
        """delete should call Redis delete."""
        storage = RedisStorage()
        mock_conn = MagicMock()
        mock_conn.delete.return_value = 1
        storage._conn = mock_conn

        result = storage.delete("mem:123")

        mock_conn.delete.assert_called_once_with("agent_memory:mem:123")
        assert result is True

    def test_get_all_keys_pattern(self):
        """get_all_keys should use correct pattern."""
        storage = RedisStorage()
        mock_conn = MagicMock()
        mock_conn.keys.return_value = ["agent_memory:mem:1", "agent_memory:mem:2"]
        storage._conn = mock_conn

        result = storage.get_all_keys()

        mock_conn.keys.assert_called_once_with("agent_memory:mem:*")
        assert len(result) == 2

    def test_clear_deletes_all_keys(self):
        """clear should delete all memory keys."""
        storage = RedisStorage()
        mock_conn = MagicMock()
        mock_conn.keys.return_value = ["k1", "k2"]
        mock_conn.delete.return_value = 2
        storage._conn = mock_conn

        result = storage.clear()

        mock_conn.delete.assert_called_once_with("k1", "k2")
        assert result == 2


class TestAsyncRedisStorageWithMock:
    """Tests for AsyncRedisStorage with mocked connection."""

    @pytest.mark.asyncio
    async def test_set_stores_data_correctly(self):
        """set should store data with correct keys."""
        storage = AsyncRedisStorage()
        mock_conn = AsyncMock()
        storage._conn = mock_conn

        await storage.set(
            memory_id="mem:123",
            content="test content",
            context="test_context",
            embedding=[0.1, 0.2],
            timestamp="2024-01-01T00:00:00",
            expires_at=None,
        )

        mock_conn.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_returns_data(self):
        """get should return stored data."""
        storage = AsyncRedisStorage()
        mock_conn = AsyncMock()
        mock_conn.hgetall.return_value = {"content": "test", "context": "default"}
        storage._conn = mock_conn

        result = await storage.get("mem:123")

        assert result["content"] == "test"

    @pytest.mark.asyncio
    async def test_delete(self):
        """delete should work."""
        storage = AsyncRedisStorage()
        mock_conn = AsyncMock()
        mock_conn.delete.return_value = 1
        storage._conn = mock_conn

        result = await storage.delete("mem:123")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_all_keys(self):
        """get_all_keys should return keys."""
        storage = AsyncRedisStorage()
        mock_conn = AsyncMock()
        mock_conn.keys.return_value = ["k1", "k2"]
        storage._conn = mock_conn

        result = await storage.get_all_keys()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        """clear should delete all keys."""
        storage = AsyncRedisStorage()
        mock_conn = AsyncMock()
        mock_conn.keys.return_value = ["k1", "k2"]
        mock_conn.delete.return_value = 2
        storage._conn = mock_conn

        result = await storage.clear()

        assert result == 2
