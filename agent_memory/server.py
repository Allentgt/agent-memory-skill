"""
Agent Memory MCP Server

Exposes remember() and recall() as MCP tools for automatic agent use.
Uses FastMCP framework with Pydantic validation.
"""

import os
import json
from typing import List, Tuple, Optional, Dict
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("agent_memory_mcp")

# Constants
CHARACTER_LIMIT = 25000


def truncate_response(text: str, limit: int = CHARACTER_LIMIT) -> str:
    """Truncate response text to character limit.

    Args:
        text: Text to truncate
        limit: Maximum characters (default: CHARACTER_LIMIT)

    Returns:
        Truncated text with indicator if truncated
    """
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n\n[... truncated]"


# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import agent memory
from agent_memory import (
    remember_async,
    recall_async,
    delete_async,
    clear_async,
    cleanup_async,
    list_memories,
    get_memory,
    export_memories,
    import_memories,
)


# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


# Pydantic Models for Input Validation
class RememberInput(BaseModel):
    """Input model for storing information in agent memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    content: str = Field(
        ...,
        description="The content to remember. This is the information that will be stored and can later be recalled.",
        min_length=1,
        max_length=50000,
    )
    context: str = Field(
        default="default",
        description="Context label for organizing memories (e.g., 'preferences', 'project', 'codebase', 'meetings'). Use consistent context values to enable filtering memories by category.",
        min_length=1,
        max_length=100,
    )
    index_name: str = Field(
        default="agent_memory",
        description="Optional custom index name for memory storage. Defaults to 'agent_memory'.",
        max_length=100,
    )
    ttl_days: Optional[int] = Field(
        default=None,
        description="Optional TTL in days. If set, memory will auto-expire after this many days. None = no expiry.",
        ge=1,
        le=365,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable, 'json' for machine-readable.",
    )


class RecallInput(BaseModel):
    """Input model for searching agent memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(
        ...,
        description="Natural language search query. The query is embedded and compared against stored memories using semantic similarity. Use natural language rather than keywords for best results.",
        min_length=1,
        max_length=500,
    )
    min_score: float = Field(
        default=0.3,
        description="Minimum similarity score threshold (0.0-1.0). Only memories with a similarity score at or above this value will be returned. Higher values return more relevant but fewer results.",
        ge=0.0,
        le=1.0,
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return. Must be between 1 and 100.",
        ge=1,
        le=100,
    )
    index_name: str = Field(
        default="agent_memory",
        description="Optional custom index name to search. Must match the index used when storing memories.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable text, 'json' for machine-readable structured data.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context filter. If set, only memories with matching context will be returned.",
        max_length=100,
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Optional start datetime for filtering memories (ISO8601 format). Only memories created after this time will be returned.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Optional end datetime for filtering memories (ISO8601 format). Only memories created before this time will be returned.",
    )
    keyword_boost: float = Field(
        default=0.0,
        description="Keyword boost factor (0.0 = semantic only, 1.0 = max keyword boost).",
        ge=0.0,
        le=1.0,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class MemoryCountInput(BaseModel):
    """Input model for getting memory count."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    index_name: str = Field(
        default="agent_memory",
        description="Optional custom index name. Must match the index used when storing memories.",
        max_length=100,
    )


class ClearMemoryInput(BaseModel):
    """Input model for clearing memory index."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    index_name: str = Field(
        default="agent_memory",
        description="The index name to clear. Use 'agent_memory' for default index.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class DeleteMemoryInput(BaseModel):
    """Input model for deleting a specific memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    memory_id: str = Field(
        ...,
        description="The memory ID to delete. This is returned from the remember tool.",
        min_length=1,
    )
    index_name: str = Field(
        default="agent_memory",
        description="The index name the memory belongs to.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class CleanupMemoryInput(BaseModel):
    """Input model for cleaning up expired memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    index_name: str = Field(
        default="agent_memory",
        description="The index name to clean up.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class RememberBatchInput(BaseModel):
    """Input model for batch storing memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    items: List[Tuple[str, str]] = Field(
        ...,
        description="List of (content, context) tuples to store.",
    )
    index_name: str = Field(
        default="agent_memory",
        description="Optional custom index name.",
        max_length=100,
    )
    ttl_days: Optional[int] = Field(
        default=None,
        description="Optional TTL in days for all items.",
        ge=1,
        le=365,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class GetMemoryInput(BaseModel):
    """Input model for getting a single memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    memory_id: str = Field(
        ...,
        description="The memory ID to retrieve.",
        min_length=1,
    )
    index_name: str = Field(
        default="agent_memory",
        description="The index name.",
        max_length=100,
    )


class ListMemoriesInput(BaseModel):
    """Input model for listing memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    limit: int = Field(
        default=50,
        description="Maximum number of memories to return.",
        ge=1,
        le=500,
    )
    offset: int = Field(
        default=0,
        description="Number of memories to skip (for pagination).",
        ge=0,
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context filter.",
        max_length=100,
    )
    index_name: str = Field(
        default="agent_memory",
        description="The index name.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class ExportMemoriesInput(BaseModel):
    """Input model for exporting memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    filepath: str = Field(
        ...,
        description="Path to export file.",
        min_length=1,
    )
    index_name: str = Field(
        default="agent_memory",
        description="The index name.",
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class ImportMemoriesInput(BaseModel):
    """Input model for importing memories."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    filepath: str = Field(
        ...,
        description="Path to import file.",
        min_length=1,
    )
    index_name: str = Field(
        default="agent_memory",
        description="Target index name.",
        max_length=100,
    )
    merge: bool = Field(
        default=True,
        description="If False, clear existing memories first.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class ListIndexesInput(BaseModel):
    """Input model for listing indexes."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )


class DeleteIndexInput(BaseModel):
    """Input model for deleting an index."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    index_name: str = Field(
        ...,
        description="The index name to delete.",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


# MCP Tools
@mcp.tool(
    name="agent_memory_remember",
    annotations={
        "title": "Store Information in Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_remember(params: RememberInput) -> str:
    """Store information in agent memory with optional context.

    This tool persists content into semantic memory that can be later retrieved
    using natural language queries. It uses sentence embeddings to enable
    semantic similarity matching, not just keyword matching.

    The tool creates new memory entries with timestamps, allowing for
    chronological organization and time-based filtering if needed.

    Args:
        params (RememberInput): Validated input parameters containing:
            - content (str): The text content to store in memory
            - context (str): Optional context label for categorization
            - index_name (str): Optional custom index name

    Returns:
        str: Confirmation message with stored content preview

    Examples:
        - Use when: "Remember that the user prefers dark mode" -> params with content="User prefers dark mode", context="preferences"
        - Use when: "Store this project context" -> params with content="Building a Python web app with FastAPI", context="project"
        - Use when: "Remember meeting notes" -> params with content="Meeting with team at 2pm", context="meetings"
        - Don't use when: You need to search memories (use agent_memory_recall instead)
        - Don't use when: You need to delete or update memories (not supported)

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to generate embedding" if embedding model fails
        - Returns "Error: Failed to store memory" on other storage failures
    """
    try:
        memory_id = await remember_async(
            content=params.content,
            context=params.context,
            index_name=params.index_name,
            ttl_days=params.ttl_days,
        )

        # Truncate content for display if too long
        display_content = params.content[:50]
        if len(params.content) > 50:
            display_content += "..."

        return f"Remembered [{memory_id}]: {display_content}"

    except Exception as e:
        return f"Error: Failed to store memory: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_recall",
    annotations={
        "title": "Search Memory with Natural Language",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_recall(params: RecallInput) -> str:
    """Search agent memory with natural language.

    This tool finds semantically similar memories even without exact keyword matches.
    It uses cosine similarity between query embeddings and stored embeddings to
    rank and filter results.

    The search is particularly effective when:
    - Using natural language queries rather than keywords
    - Setting appropriate min_score thresholds (higher = more precise)
    - Using consistent context labels when storing memories

    Args:
        params (RecallInput): Validated input parameters containing:
            - query (str): Natural language search query
            - min_score (float): Minimum similarity score (0.0-1.0)
            - limit (int): Maximum results to return (1-100)
            - index_name (str): Custom index name to search
            - response_format (ResponseFormat): Output format (markdown/json)

    Returns:
        str: Formatted search results in requested format

    Success response (Markdown):
        "Found N memory(ies):\\n- [0.85] Memory content here\\n- [0.72] Another memory"

    Success response (JSON):
        {
            "total": 2,
            "results": [
                {"content": "Memory content", "score": 0.85},
                {"content": "Another memory", "score": 0.72}
            ]
        }

    Error response:
        "Error: <error message>" or "No matching memories found."

    Examples:
        - Use when: "What design preferences does user have?" -> params with query="design preferences", min_score=0.3
        - Use when: "What project is being worked on?" -> params with query="project context", min_score=0.5, limit=3
        - Use when: "Find recent meetings" -> params with query="meeting notes", limit=10
        - Don't use when: You need to store new information (use agent_memory_remember instead)
        - Don't use when: You have exact memory IDs (this uses semantic search)

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to generate query embedding" if embedding fails
        - Returns "No matching memories found." when no results meet the threshold
    """
    try:
        results = await recall_async(
            query=params.query,
            min_score=params.min_score,
            limit=params.limit,
            index_name=params.index_name,
            context=params.context,
            since=params.since,
            until=params.until,
            keyword_boost=params.keyword_boost,
        )

        if not results:
            return "No matching memories found."

        # Format response based on requested format
        if params.response_format == ResponseFormat.JSON:
            response = {
                "total": len(results),
                "results": [
                    {"content": content, "score": round(score, 2)}
                    for content, score in results
                ],
            }
            response_text = json.dumps(response, indent=2)
        else:
            # Markdown format
            formatted = []
            for content, score in results:
                formatted.append(f"- [{score:.2f}] {content}")

            response_text = f"Found {len(results)} memory(ies):\n" + "\n".join(
                formatted
            )

        return truncate_response(response_text)

    except Exception as e:
        return f"Error: Failed to recall memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_count",
    annotations={
        "title": "Get Memory Count",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_count(params: MemoryCountInput) -> str:
    """Get the number of memories stored in agent memory.

    This tool returns the total count of memory entries stored under the
    specified index. It is useful for understanding memory usage and
    for debugging purposes.

    Args:
        params (MemoryCountInput): Validated input parameters containing:
            - index_name (str): Optional custom index name

    Returns:
        str: Confirmation message with total memory count

    Examples:
        - Use when: "How many memories have been stored?" -> params with default index_name
        - Use when: "Check memory usage for project index" -> params with index_name="project"

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to get memory count" on other failures
    """
    try:
        with AgentMemory(index_name=params.index_name) as mem:
            count = mem.count

        return f"Total memories: {count}"

    except Exception as e:
        return f"Error: Failed to get memory count: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_clear",
    annotations={
        "title": "Clear Memory Index",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_clear(params: ClearMemoryInput) -> str:
    """Clear all memories from a specific index.

    This tool permanently deletes all memory entries in the specified index.
    Use with caution - this cannot be undone.

    Args:
        params (ClearMemoryInput): Validated input parameters containing:
            - index_name (str): The index name to clear

    Returns:
        str: Confirmation message with count of deleted memories

    Examples:
        - Use when: "Clear all food preferences" -> params with index_name="agent_memory"
        - Use when: "Reset the project memory" -> params with index_name="project"

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to clear memory" on other failures
    """
    try:
        deleted = await clear_async(index_name=params.index_name)
        return f"Cleared {deleted} memory(ies) from index '{params.index_name}'"

    except Exception as e:
        return f"Error: Failed to clear memory: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_delete",
    annotations={
        "title": "Delete Specific Memory",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def agent_memory_delete(params: DeleteMemoryInput) -> str:
    """Delete a specific memory by ID.

    This tool deletes a single memory entry using the memory_id returned
    from the remember tool. Use this to remove specific memories.

    Args:
        params (DeleteMemoryInput): Validated input parameters containing:
            - memory_id (str): The memory ID to delete
            - index_name (str): The index name the memory belongs to

    Returns:
        str: Confirmation message with count of deleted memories

    Examples:
        - Use when: "Delete the memory about user preferences" -> params with memory_id from remember response
        - Don't use when: You want to clear all memories (use agent_memory_clear instead)

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to delete memory" on other failures
    """
    try:
        deleted = await delete_async(
            memory_id=params.memory_id, index_name=params.index_name
        )
        if deleted:
            return (
                f"Deleted memory '{params.memory_id}' from index '{params.index_name}'"
            )
        else:
            return (
                f"Memory '{params.memory_id}' not found in index '{params.index_name}'"
            )

    except Exception as e:
        return f"Error: Failed to delete memory: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_cleanup",
    annotations={
        "title": "Clean Up Expired Memories",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_cleanup(params: CleanupMemoryInput) -> str:
    """Remove all expired memories from an index.

    This tool removes all memories that have exceeded their TTL (time-to-live).
    Memories with TTL set via remember() will be removed.

    Args:
        params (CleanupMemoryInput): Validated input parameters containing:
            - index_name (str): The index name to clean up

    Returns:
        str: Confirmation message with count of removed memories

    Examples:
        - Use when: "Clean up old temporary memories" -> params with index_name="agent_memory"
        - Use when: "Remove expired cache data" -> params with index_name="cache"

    Error Handling:
        - Returns "Error: Redis connection failed" if Redis is unavailable
        - Returns "Error: Failed to cleanup memories" on other failures
    """
    try:
        removed = await cleanup_async(index_name=params.index_name)
        return (
            f"Cleaned up {removed} expired memory(ies) from index '{params.index_name}'"
        )

    except Exception as e:
        return f"Error: Failed to cleanup memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_remember_batch",
    annotations={
        "title": "Batch Store Memories",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_remember_batch(params: RememberBatchInput) -> str:
    """Store multiple memories in a single call.

    Args:
        params (RememberBatchInput): Validated input parameters

    Returns:
        str: Confirmation message with count of stored memories
    """
    try:
        memory_ids = _remember_batch(
            items=params.items,
            index_name=params.index_name,
            ttl_days=params.ttl_days,
        )
        return f"Stored {len(memory_ids)} memory(ies) in index '{params.index_name}'"

    except Exception as e:
        return f"Error: Failed to store memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_get",
    annotations={
        "title": "Get Single Memory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_get(params: GetMemoryInput) -> str:
    """Get a single memory by ID with metadata.

    Args:
        params (GetMemoryInput): Validated input parameters

    Returns:
        str: Memory data in JSON format
    """
    try:
        memory = _get_memory(
            memory_id=params.memory_id,
            index_name=params.index_name,
        )
        if memory:
            return json.dumps(memory, indent=2)
        else:
            return f"Memory '{params.memory_id}' not found"

    except Exception as e:
        return f"Error: Failed to get memory: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_list",
    annotations={
        "title": "List All Memories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_list(params: ListMemoriesInput) -> str:
    """List all memories with optional context filter and pagination.

    Args:
        params (ListMemoriesInput): Validated input parameters

    Returns:
        str: List of memories in requested format
    """
    try:
        memories = _list_memories(
            limit=params.limit,
            offset=params.offset,
            context=params.context,
            index_name=params.index_name,
        )

        if params.response_format == ResponseFormat.JSON:
            response_text = json.dumps(
                {"total": len(memories), "memories": memories}, indent=2
            )
        else:
            # Markdown format
            if not memories:
                return "No memories stored."
            lines = [f"Found {len(memories)} memory(ies):"]
            for mem in memories:
                lines.append(
                    f"- [{mem['memory_id']}] {mem['content'][:100]}"
                    + ("" if len(mem["content"]) <= 100 else "...")
                )
            response_text = "\n".join(lines)

        return truncate_response(response_text)

    except Exception as e:
        return f"Error: Failed to list memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_export",
    annotations={
        "title": "Export Memories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_export(params: ExportMemoriesInput) -> str:
    """Export memories to JSON file.

    Args:
        params (ExportMemoriesInput): Validated input parameters

    Returns:
        str: Confirmation message with count of exported memories
    """
    try:
        count = _export_memories(
            filepath=params.filepath,
            index_name=params.index_name,
        )
        return f"Exported {count} memory(ies) to '{params.filepath}'"

    except Exception as e:
        return f"Error: Failed to export memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_import",
    annotations={
        "title": "Import Memories",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_import(params: ImportMemoriesInput) -> str:
    """Import memories from JSON file.

    Args:
        params (ImportMemoriesInput): Validated input parameters

    Returns:
        str: Confirmation message with count of imported memories
    """
    try:
        count = _import_memories(
            filepath=params.filepath,
            index_name=params.index_name,
            merge=params.merge,
        )
        return f"Imported {count} memory(ies) into index '{params.index_name}'"

    except Exception as e:
        return f"Error: Failed to import memories: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_list_indexes",
    annotations={
        "title": "List Memory Indexes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_list_indexes(params: ListIndexesInput) -> str:
    """List all memory indexes in Redis.

    Returns JSON with list of index names and their memory counts.
    """
    try:
        from agent_memory.storage import RedisStorage

        storage = RedisStorage()
        storage.connect()
        keys = storage.conn.keys("*:mem:*")
        storage.close()

        # Group by index name
        indexes: Dict[str, int] = {}
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 2:
                idx = parts[0]
                indexes[idx] = indexes.get(idx, 0) + 1

        return json.dumps(
            {"indexes": [{"name": k, "count": v} for k, v in indexes.items()]}
        )

    except Exception as e:
        return f"Error: Failed to list indexes: {type(e).__name__}: {str(e)}"


@mcp.tool(
    name="agent_memory_delete_index",
    annotations={
        "title": "Delete Memory Index",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def agent_memory_delete_index(params: DeleteIndexInput) -> str:
    """Delete all memories in an index.

    Args:
        params (DeleteIndexInput): Validated input with index_name

    Returns:
        str: Confirmation message
    """
    try:
        from agent_memory.storage import RedisStorage

        storage = RedisStorage(params.index_name)
        storage.connect()
        deleted = storage.clear()
        storage.close()

        return f"Deleted {deleted} memory(ies) from index '{params.index_name}'"

    except Exception as e:
        return f"Error: Failed to delete index: {type(e).__name__}: {str(e)}"


if __name__ == "__main__":
    mcp.run()


def main():
    """Entry point for CLI."""
    mcp.run()


# Tool: Health check for container orchestration
class HealthCheckInput(BaseModel):
    """Input model for health check."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )


async def _health_check_impl() -> dict:
    """Check service health."""
    health = {"status": "healthy", "checks": {}}

    # Check Redis connection
    try:
        from agent_memory.storage import RedisStorage

        storage = RedisStorage()
        storage.connect()
        storage.conn.ping()
        storage.close()
        health["checks"]["redis"] = "ok"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["redis"] = f"error: {type(e).__name__}"

    # Check embedding model
    try:
        from agent_memory.embeddings import get_embedding_model

        get_embedding_model()
        health["checks"]["embedding_model"] = "ok"
    except Exception as e:
        health["checks"]["embedding_model"] = f"error: {type(e).__name__}"

    return health


@mcp.tool(
    name="agent_memory_health",
    annotations={
        "title": "Health Check",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
async def agent_memory_health(params: HealthCheckInput) -> str:
    """Check service health for container orchestration.

    Returns JSON with status, Redis connection, and embedding model availability.
    Use this for container health checks (e.g., Docker HEALTHCHECK, Kubernetes liveness).
    """
    try:
        health = await _health_check_impl()
        return json.dumps(health, indent=2)
    except Exception as e:
        return json.dumps({"status": "unhealthy", "error": str(e)}, indent=2)
