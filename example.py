"""
Example usage of agent_memory skill with Redis.
"""

import os
from agent_memory import AgentMemory, remember, recall

# Example 1: Using context manager
print("=" * 50)
print("Example 1: Context Manager")
print("=" * 50)

with AgentMemory(index_name="test_memory") as mem:
    # Store some memories
    mem.remember("User prefers concise responses", "preferences")
    mem.remember("Project deadline is March 15th", "project")
    mem.remember("Using Python with asyncio for concurrency", "codebase")
    mem.remember("The user is building an agent memory skill", "context")

    print(f"Stored {mem.count} memories")

    # Search
    results = mem.recall("What's the project timeline?")
    for content, score in results:
        print(f"  [{score:.2f}] {content[:60]}...")

print()

# Example 2: Convenience functions
print("=" * 50)
print("Example 2: Convenience Functions")
print("=" * 50)

remember("Python asyncio is used for async operations", "codebase")
results = recall("async")

print(f"Found {len(results)} results:")
for content, score in results:
    print(f"  [{score:.2f}] {content[:60]}...")

print()

# Example 3: Custom configuration
print("=" * 50)
print("Example 3: Custom Index and Context")
print("=" * 50)

with AgentMemory(index_name="custom") as mem:
    mem.remember("Meeting scheduled for 2pm", "meetings")
    mem.remember("Code review needed for PR #42", "reviews")

    results = mem.recall("meeting")
    print(f"Meeting results: {results}")

print("\nDone!")
