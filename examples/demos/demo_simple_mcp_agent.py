#!/usr/bin/env python3
"""
demo_simple_mcp_agent.py

Demo: A minimal script demonstrating how to interact with the AtomAgents MCP HTTP API
directly from Python, without any external agent frameworks.

Prerequisites:
  - Start the AtomAgents MCP service, e.g.:
      uvicorn AtomAgents.mcp_service.mcp_service:app --reload
  - Install dependencies:
      pip install requests
  - (Optional) Set OPENAI_API_KEY in your environment if using tools that require it.
"""

import os
import requests

# Base URL of the MCP service; set MCP_HTTP_URL to override (e.g. http://localhost:8000 or http://localhost:9999/mcp)
MCP_HTTP_URL = os.getenv("MCP_HTTP_URL", "http://localhost:8000")
API_BASE = f"{MCP_HTTP_URL.rstrip('/')}/tools"


def list_tools():
    resp = requests.get(f"{API_BASE}")
    resp.raise_for_status()
    return resp.json()


def run_code_runner(code: str):
    payload = {"code": code}
    resp = requests.post(f"{API_BASE}/code_runner", json=payload)
    resp.raise_for_status()
    return resp.json()


def run_web_search(query: str, max_results: int = 3):
    payload = {"query": query, "max_results": max_results}
    resp = requests.post(f"{API_BASE}/web_search", json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    print("Available MCP tools:")
    tools = list_tools()
    print(tools)

    print("\nRunning code_runner tool:")
    code_result = run_code_runner("print('Hello from MCP!')")
    print(code_result)

    print("\nRunning web_search tool:")
    search_result = run_web_search("OpenAI latest research", max_results=2)
    print(search_result)


if __name__ == "__main__":
    main()
