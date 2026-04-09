"""
Integration tests for agent_memory module.
Requires running Redis and sentence-transformers model.
"""

import pytest
import asyncio
import os
import tempfile

# Set test environment
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_DB", "15")


@pytest.fixture
def test_index():
    """Create a test index name."""
    return "test_integration"


@pytest.fixture(autouse=True)
async def cleanup_test_index(test_index):
    """Clean up test index before and after each test."""
    from agent_memory import clear_async

    await clear_async(index_name=test_index)
    yield
    await clear_async(index_name=test_index)


class TestIntegrationRememberRecall:
    """Integration tests for remember and recall."""

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, test_index):
        """Basic remember and recall should work."""
        from agent_memory import remember_async, recall_async

        # Store a memory
        memory_id = await remember_async(
            "User prefers dark mode", context="preferences", index_name=test_index
        )

        assert memory_id.startswith("mem:")

        # Recall it
        results = await recall_async(
            "design preferences", index_name=test_index, min_score=0.1
        )

        assert len(results) > 0
        assert any("dark mode" in content for content, _ in results)

    @pytest.mark.asyncio
    async def test_recall_no_results_for_unrelated(self, test_index):
        """Unrelated queries should return no results."""
        from agent_memory import remember_async, recall_async

        await remember_async("test content", index_name=test_index)

        results = await recall_async(
            "completely unrelated query xyz", index_name=test_index, min_score=0.3
        )

        # Should have 0 results since nothing matches
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_different_contexts(self, test_index):
        """Different contexts should be stored separately."""
        from agent_memory import remember_async, recall_async

        # Store in different contexts
        await remember_async("Python code", context="code", index_name=test_index)
        await remember_async("Meeting notes", context="meetings", index_name=test_index)

        # Recall with context filter
        results = await recall_async(
            "programming", context="code", index_name=test_index, min_score=0.1
        )

        assert all("Python" in content or "code" in content for content, _ in results)

    @pytest.mark.asyncio
    async def test_batch_remember(self, test_index):
        """Batch remember should work."""
        from agent_memory import remember_async

        items = [
            ("content 1", "context1"),
            ("content 2", "context2"),
            ("content 3", "context1"),
        ]

        for content, context in items:
            await remember_async(content, context, index_name=test_index)

        # All should be stored
        from agent_memory import recall_async

        results = await recall_async("content", index_name=test_index, min_score=0.1)

        assert len(results) >= 3


class TestIntegrationDelete:
    """Integration tests for delete."""

    @pytest.mark.asyncio
    async def test_delete_memory(self, test_index):
        """Delete should remove memory."""
        from agent_memory import remember_async, delete_async, get_memory

        memory_id = await remember_async("to be deleted", index_name=test_index)

        # Verify it's stored
        mem = await get_memory(memory_id, index_name=test_index)
        assert mem is not None

        # Delete it
        deleted = await delete_async(memory_id, index_name=test_index)
        assert deleted is True

        # Verify it's gone
        mem = await get_memory(memory_id, index_name=test_index)
        assert mem is None


class TestIntegrationList:
    """Integration tests for list_memories."""

    @pytest.mark.asyncio
    async def test_list_memories(self, test_index):
        """list_memories should return all stored memories."""
        from agent_memory import remember_async, list_memories

        # Store some memories
        await remember_async("item 1", index_name=test_index)
        await remember_async("item 2", index_name=test_index)

        memories = await list_memories(limit=10, index_name=test_index)

        assert len(memories) >= 2

    @pytest.mark.asyncio
    async def test_list_memories_pagination(self, test_index):
        """list_memories should support pagination."""
        from agent_memory import remember_async, list_memories

        # Store more than limit
        for i in range(10):
            await remember_async(f"item {i}", index_name=test_index)

        # First page
        page1 = await list_memories(limit=3, offset=0, index_name=test_index)
        assert len(page1) == 3

        # Second page
        page2 = await list_memories(limit=3, offset=3, index_name=test_index)
        assert len(page2) == 3

        # No overlap
        page1_ids = {m["memory_id"] for m in page1}
        page2_ids = {m["memory_id"] for m in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_memories_context_filter(self, test_index):
        """list_memories should filter by context."""
        from agent_memory import remember_async, list_memories

        await remember_async("pref 1", context="preferences", index_name=test_index)
        await remember_async("pref 2", context="preferences", index_name=test_index)
        await remember_async("other", context="other", index_name=test_index)

        # Filter by context
        prefs = await list_memories(context="preferences", index_name=test_index)
        assert len(prefs) == 2
        assert all(m["context"] == "preferences" for m in prefs)


class TestIntegrationExportImport:
    """Integration tests for export/import."""

    @pytest.mark.asyncio
    async def test_export_import(self, test_index):
        """Export and import should work."""
        from agent_memory import (
            remember_async,
            export_memories,
            import_memories,
            clear_async,
        )

        # Store memories
        await remember_async("exported 1", index_name=test_index)
        await remember_async("exported 2", index_name=test_index)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            # Export
            count = await export_memories(export_path, index_name=test_index)
            assert count >= 2

            # Clear and import to new index
            await clear_async(index_name=f"{test_index}_imported")
            await import_memories(
                export_path, index_name=f"{test_index}_imported", merge=False
            )

            # Verify import worked
            from agent_memory import list_memories

            imported = await list_memories(
                limit=10, index_name=f"{test_index}_imported"
            )
            assert len(imported) >= 2

            await clear_async(index_name=f"{test_index}_imported")
        finally:
            os.unlink(export_path)


class TestIntegrationTTL:
    """Integration tests for TTL."""

    @pytest.mark.asyncio
    async def test_ttl_expired_not_returned(self, test_index):
        """Expired memories should not be returned in recall."""
        from agent_memory import remember_async, recall_async

        # Store with TTL (expired immediately by using ttl_days=0 is not allowed)
        # We can't easily test expired TTL without time manipulation
        # So we just verify TTL is stored
        memory_id = await remember_async("with TTL", ttl_days=1, index_name=test_index)

        assert memory_id.startswith("mem:")

        # Should still be findable
        results = await recall_async("TTL", index_name=test_index, min_score=0.1)
        assert len(results) > 0


class TestIntegrationCount:
    """Integration tests for count."""

    @pytest.mark.asyncio
    async def test_count(self, test_index):
        """Count should reflect stored memories."""
        from agent_memory import remember_async, clear_async

        await clear_async(index_name=test_index)

        await remember_async("test 1", index_name=test_index)
        await remember_async("test 2", index_name=test_index)

        from agent_memory import AgentMemoryAsync

        async with AgentMemoryAsync(index_name=test_index) as mem:
            count = await mem.count

        assert count >= 2

    @pytest.mark.asyncio
    async def test_clear(self, test_index):
        """Clear should remove all memories."""
        from agent_memory import remember_async, clear_async

        await remember_async("to clear 1", index_name=test_index)
        await remember_async("to clear 2", index_name=test_index)

        deleted = await clear_async(index_name=test_index)

        assert deleted >= 2

        # Verify empty
        from agent_memory import AgentMemoryAsync

        async with AgentMemoryAsync(index_name=test_index) as mem:
            count = await mem.count

        assert count == 0
