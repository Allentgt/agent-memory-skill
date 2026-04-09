"""
Unit tests for agent_memory.server module.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from agent_memory.server import (
    truncate_response,
    ResponseFormat,
    RememberInput,
    RecallInput,
    ListMemoriesInput,
    DeleteMemoryInput,
    ClearMemoryInput,
    CleanupMemoryInput,
    GetMemoryInput,
    MemoryCountInput,
    RememberBatchInput,
    ExportMemoriesInput,
    ImportMemoriesInput,
)


class TestTruncateResponse:
    """Tests for truncate_response helper."""

    def test_under_limit_returns_unchanged(self):
        """Text under limit should be returned unchanged."""
        text = "short text"
        result = truncate_response(text, limit=100)

        assert result == text

    def test_over_limit_truncates(self):
        """Text over limit should be truncated with indicator."""
        text = "a" * 100
        result = truncate_response(text, limit=50)

        assert len(result) <= 50
        assert "[... truncated]" in result

    def test_default_limit(self):
        """Default CHARACTER_LIMIT should be used."""
        text = "a" * 30000
        result = truncate_response(text)

        assert "[... truncated]" in result


class TestResponseFormat:
    """Tests for ResponseFormat enum."""

    def test_has_markdown_and_json(self):
        """ResponseFormat should have markdown and json values."""
        assert hasattr(ResponseFormat, "MARKDOWN")
        assert hasattr(ResponseFormat, "JSON")
        assert ResponseFormat.MARKDOWN.value == "markdown"
        assert ResponseFormat.JSON.value == "json"


class TestRememberInput:
    """Tests for RememberInput validation."""

    def test_valid_input(self):
        """Valid input should pass validation."""
        input_obj = RememberInput(content="test content")

        assert input_obj.content == "test content"
        assert input_obj.context == "default"
        assert input_obj.index_name == "agent_memory"
        assert input_obj.response_format == ResponseFormat.MARKDOWN

    def test_with_ttl_days(self):
        """TTL days should be validated."""
        input_obj = RememberInput(content="test", ttl_days=30)

        assert input_obj.ttl_days == 30

    def test_ttl_days_validation_rejects_negative(self):
        """Negative TTL should fail validation."""
        with pytest.raises(Exception):
            RememberInput(content="test", ttl_days=-1)

    def test_ttl_days_validation_rejects_over_365(self):
        """TTL over 365 should fail validation."""
        with pytest.raises(Exception):
            RememberInput(content="test", ttl_days=400)

    def test_empty_content_rejected(self):
        """Empty content should fail validation."""
        with pytest.raises(Exception):
            RememberInput(content="")

    def test_response_format_json(self):
        """JSON response format should work."""
        input_obj = RememberInput(content="test", response_format=ResponseFormat.JSON)

        assert input_obj.response_format == ResponseFormat.JSON


class TestRecallInput:
    """Tests for RecallInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = RecallInput(query="search term")

        assert input_obj.query == "search term"
        assert input_obj.min_score == 0.3
        assert input_obj.limit == 5

    def test_min_score_bounds(self):
        """min_score should be bounded 0-1."""
        RecallInput(query="test", min_score=0.0)
        RecallInput(query="test", min_score=1.0)

        with pytest.raises(Exception):
            RecallInput(query="test", min_score=-0.1)

        with pytest.raises(Exception):
            RecallInput(query="test", min_score=1.1)

    def test_limit_bounds(self):
        """limit should be bounded 1-100."""
        RecallInput(query="test", limit=1)
        RecallInput(query="test", limit=100)

        with pytest.raises(Exception):
            RecallInput(query="test", limit=0)

        with pytest.raises(Exception):
            RecallInput(query="test", limit=101)

    def test_context_filter(self):
        """context filter should work."""
        input_obj = RecallInput(query="test", context="preferences")

        assert input_obj.context == "preferences"

    def test_query_whitespace_stripped(self):
        """Query whitespace should be stripped."""
        input_obj = RecallInput(query="  test query  ")

        assert input_obj.query == "test query"

    def test_query_empty_rejected(self):
        """Empty or whitespace-only query should fail."""
        with pytest.raises(Exception):
            RecallInput(query="")

        with pytest.raises(Exception):
            RecallInput(query="   ")


class TestListMemoriesInput:
    """Tests for ListMemoriesInput validation."""

    def test_valid_with_pagination(self):
        """Pagination parameters should work."""
        input_obj = ListMemoriesInput(limit=10, offset=5)

        assert input_obj.limit == 10
        assert input_obj.offset == 5

    def test_offset_must_be_non_negative(self):
        """offset cannot be negative."""
        with pytest.raises(Exception):
            ListMemoriesInput(offset=-1)

    def test_limit_bounds(self):
        """limit should be bounded 1-500."""
        ListMemoriesInput(limit=1)
        ListMemoriesInput(limit=500)

        with pytest.raises(Exception):
            ListMemoriesInput(limit=0)

        with pytest.raises(Exception):
            ListMemoriesInput(limit=501)


class TestDeleteMemoryInput:
    """Tests for DeleteMemoryInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = DeleteMemoryInput(memory_id="mem:123")

        assert input_obj.memory_id == "mem:123"
        assert input_obj.index_name == "agent_memory"

    def test_memory_id_required(self):
        """memory_id is required."""
        with pytest.raises(Exception):
            DeleteMemoryInput()


class TestClearMemoryInput:
    """Tests for ClearMemoryInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = ClearMemoryInput()

        assert input_obj.index_name == "agent_memory"
        assert input_obj.response_format == ResponseFormat.MARKDOWN


class TestCleanupMemoryInput:
    """Tests for CleanupMemoryInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = CleanupMemoryInput()

        assert input_obj.index_name == "agent_memory"


class TestGetMemoryInput:
    """Tests for GetMemoryInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = GetMemoryInput(memory_id="mem:123")

        assert input_obj.memory_id == "mem:123"

    def test_memory_id_required(self):
        """memory_id is required."""
        with pytest.raises(Exception):
            GetMemoryInput()


class TestMemoryCountInput:
    """Tests for MemoryCountInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = MemoryCountInput(index_name="custom")

        assert input_obj.index_name == "custom"


class TestRememberBatchInput:
    """Tests for RememberBatchInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        items = [("content1", "context1"), ("content2", "context2")]
        input_obj = RememberBatchInput(items=items)

        assert len(input_obj.items) == 2

    def test_items_required(self):
        """items is required."""
        with pytest.raises(Exception):
            RememberBatchInput()


class TestExportMemoriesInput:
    """Tests for ExportMemoriesInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = ExportMemoriesInput(filepath="/tmp/export.json")

        assert input_obj.filepath == "/tmp/export.json"

    def test_filepath_required(self):
        """filepath is required."""
        with pytest.raises(Exception):
            ExportMemoriesInput()


class TestImportMemoriesInput:
    """Tests for ImportMemoriesInput validation."""

    def test_valid_input(self):
        """Valid input should pass."""
        input_obj = ImportMemoriesInput(filepath="/tmp/import.json")

        assert input_obj.filepath == "/tmp/import.json"
        assert input_obj.merge is True

    def test_merge_false(self):
        """merge=False should work."""
        input_obj = ImportMemoriesInput(filepath="/tmp/import.json", merge=False)

        assert input_obj.merge is False

    def test_filepath_required(self):
        """filepath is required."""
        with pytest.raises(Exception):
            ImportMemoriesInput()
