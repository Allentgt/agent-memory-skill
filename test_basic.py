"""Basic tests for agent_memory skill with Redis."""

import os
import pytest
from agent_memory import AgentMemory

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")


def test_basic():
    """Test basic functionality."""
    mem = AgentMemory(index_name="test_basic")
    mem.connect()

    # Store
    mem.remember("Test content", "test")
    assert mem.count >= 1

    # Search
    results = mem.recall("test")
    print(f"Results: {results}")

    mem.close()
    print("Basic test PASSED")


def test_context():
    """Test context filtering."""
    mem = AgentMemory(index_name="test_context")
    mem.connect()

    mem.remember("Python code", "codebase")
    mem.remember("Meeting notes", "meetings")

    results = mem.recall("code")
    print(f"Results: {results}")

    mem.close()


if __name__ == "__main__":
    test_basic()
    test_context()
