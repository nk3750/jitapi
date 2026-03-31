"""Main entry point for JitAPI MCP server."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    """Run the JitAPI MCP server.

    Configuration via environment variables:
    - JITAPI_STORAGE_DIR: Optional. Directory for data storage (default: ~/.jitapi)
    - JITAPI_LOG_LEVEL: Optional. Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - JITAPI_LOG_FILE: Optional. File path for logs (default: stderr)
    - JITAPI_EMBEDDING_PROVIDER: Optional. Force embedding provider: voyage|openai|cohere|local

    Embedding provider auto-detection (if JITAPI_EMBEDDING_PROVIDER not set):
    - VOYAGE_API_KEY set → Voyage AI embeddings
    - OPENAI_API_KEY set → OpenAI embeddings
    - COHERE_API_KEY set → Cohere embeddings
    - None set → local fastembed (zero-config, no API key needed)

    Reranker (workflow planning):
    - Uses MCP sampling (host LLM) by default — no API key needed
    - Falls back to OpenAI if OPENAI_API_KEY is set and sampling unavailable
    """
    # Load .env files (in order of priority)
    load_dotenv()
    load_dotenv(Path.home() / ".jitapi" / ".env")
    load_dotenv(Path.home() / ".env")

    # Get configuration from environment
    storage_dir = os.environ.get(
        "JITAPI_STORAGE_DIR",
        str(Path.home() / ".jitapi"),
    )
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    log_level = os.environ.get("JITAPI_LOG_LEVEL", "INFO")
    log_file = os.environ.get("JITAPI_LOG_FILE")

    # Create and run server — no API keys required
    from .mcp.server import create_server

    server = create_server(
        storage_dir=storage_dir,
        openai_api_key=openai_api_key,
        log_level=log_level,
        log_file=log_file,
    )

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
