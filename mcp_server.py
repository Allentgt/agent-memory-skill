"""
Agent Memory MCP Server

Exposes remember() and recall() as MCP tools for automatic agent use.
Uses FastMCP framework with Pydantic validation.
"""

import os
import json
from typing import List, Tuple, Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("agent_memory_mcp")

# Constants
CHARACTER_LIMIT = 25000

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import agent memory
from agent_memory import AgentMemory, remember as _remember, recall as _recall


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
        _remember(
            content=params.content, context=params.context, index_name=params.index_name
        )

        # Truncate content for display if too long
        display_content = params.content[:50]
        if len(params.content) > 50:
            display_content += "..."

        return f"Remembered: {display_content}"

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
        results = _recall(
            query=params.query,
            min_score=params.min_score,
            limit=params.limit,
            index_name=params.index_name,
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
            return json.dumps(response, indent=2)
        else:
            # Markdown format
            formatted = []
            for content, score in results:
                formatted.append(f"- [{score:.2f}] {content}")

            return f"Found {len(results)} memory(ies):\n" + "\n".join(formatted)

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


if __name__ == "__main__":
    mcp.run()
