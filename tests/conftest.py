"""
Pytest configuration and fixtures.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
import sys


# Set test environment variables before any imports
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_DB", "15")


@pytest.fixture
def mock_redis():
    """Create a mock Redis connection."""
    with patch("agent_memory.storage.redis.Redis") as mock:
        conn = MagicMock()
        mock.return_value = conn
        yield conn


@pytest.fixture
def mock_async_redis():
    """Create a mock async Redis connection."""
    with patch("agent_memory.storage.redis.asyncio.Redis") as mock:
        conn = AsyncMock()
        mock.return_value = conn
        yield conn


@pytest.fixture
def mock_embedding_model():
    """Create a mock sentence transformer model."""
    with patch("agent_memory.embeddings.SentenceTransformer") as mock:
        model = MagicMock()
        model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        model.get_sentence_embedding_dimension.return_value = 384
        mock.return_value = model
        yield model


@pytest.fixture
def test_index_name():
    """Return a test index name."""
    return "test_agent_memory"


@pytest.fixture
def patch_redis_module():
    """Patch redis module to use fakeredis for both sync and async."""
    import fakeredis
    import fakeredis.aioredis

    # Create shared fakeredis instances
    fake_sync = fakeredis.FakeRedis(version=(7,), decode_responses=True)
    fake_async = fakeredis.aioredis.FakeRedis(async_redis_fake=fake_sync)

    # Create mock module
    mock = MagicMock()
    mock.Redis.return_value = fake_sync

    # Async setup
    mock.asyncio = MagicMock()

    # Connection pool returns fake async redis
    mock_pool = MagicMock()
    mock_pool.disconnect = AsyncMock()

    def get_redis(*args, connection_pool=None, **kwargs):
        return fake_async

    mock.asyncio.Redis = MagicMock(side_effect=get_redis)
    mock.asyncio.ConnectionPool = MagicMock(return_value=mock_pool)

    # Save and replace
    original = sys.modules.get("redis")
    sys.modules["redis"] = mock

    yield fake_sync

    # Restore
    if original:
        sys.modules["redis"] = original
