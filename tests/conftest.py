"""
Pytest configuration and fixtures.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock


# Set test environment variables before any imports
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_DB", "15")  # Use separate DB for tests


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
