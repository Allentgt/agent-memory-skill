"""
Agent Memory MCP Server

Exposes remember() and recall() as MCP tools for automatic agent use.
"""

import os
import json
from typing import List, Tuple, Optional
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import agent memory
from agent_memory import AgentMemory, remember as _remember, recall as _recall


# Create MCP server
app = Server("agent-memory")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="remember",
            description="Store information in agent memory with optional context. Use this to remember user preferences, project context, or any important information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "context": {
                        "type": "string",
                        "description": "Context label (e.g., 'preferences', 'project', 'codebase', 'meetings')",
                        "default": "default",
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Optional custom index name",
                        "default": "agent_memory",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="recall",
            description="Search agent memory with natural language. Finds semantically similar memories even without exact keyword matches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.3,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 5,
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Optional custom index name",
                        "default": "agent_memory",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_count",
            description="Get the number of memories stored.",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_name": {
                        "type": "string",
                        "description": "Optional custom index name",
                        "default": "agent_memory",
                    }
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "remember":
        content = arguments.get("content")
        context = arguments.get("context", "default")
        index_name = arguments.get("index_name", "agent_memory")

        _remember(content, context, index_name=index_name)

        return [
            TextContent(
                type="text",
                text=f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
            )
        ]

    elif name == "recall":
        query = arguments.get("query")
        min_score = arguments.get("min_score", 0.3)
        limit = arguments.get("limit", 5)
        index_name = arguments.get("index_name", "agent_memory")

        results = _recall(query, min_score, index_name=index_name)

        if not results:
            return [TextContent(type="text", text="No matching memories found.")]

        # Format results
        formatted = []
        for content, score in results:
            formatted.append(f"- [{score:.2f}] {content}")

        return [
            TextContent(
                type="text",
                text=f"Found {len(results)} memory(ies):\n" + "\n".join(formatted),
            )
        ]

    elif name == "memory_count":
        index_name = arguments.get("index_name", "agent_memory")

        with AgentMemory(index_name=index_name) as mem:
            count = mem.count

        return [TextContent(type="text", text=f"Total memories: {count}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
