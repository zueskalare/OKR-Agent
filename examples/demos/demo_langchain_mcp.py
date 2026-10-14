#!/usr/bin/env python3
"""
Demo: Use LangChain MCP adapter to drive the AtomAgents MCP service.

Prerequisites:
  - Start the AtomAgents MCP service (e.g., via `uvicorn AtomAgents.mcp_service.mcp_service:app --reload`).
  - Ensure OPENAI_API_KEY is set in your environment.
  - Install dependencies: `pip install langchain-mcp-adapters langgraph "langchain[openai]"`
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def main():
    # Configure the MCP server endpoint (streamable HTTP transport)
    client = MultiServerMCPClient(
        {
            "atomagents": {
                "url": "http://localhost:9999/mcp/",
                "transport": "streamable_http",
            }
        }
    )

    # Retrieve all tools exposed by the server
    tools = await client.get_tools()
    print("Available MCP tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Create a LangGraph agent using OpenAI GPT and the MCP tools
    agent = create_react_agent("openai:gpt-4.1", tools)

    # Ask a sample question via the agent
    query = "What tools are available in the MCP service?"
    result = await agent.ainvoke({"messages": query})
    print("Agent response:")
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
