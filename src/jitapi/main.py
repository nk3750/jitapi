"""Main entry point for Samvaad MCP server."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    """Run the Samvaad MCP server.

    Configuration via environment variables:
    - OPENAI_API_KEY: Required. OpenAI API key for embeddings and reranking.
    - JITAPI_STORAGE_DIR: Optional. Directory for data storage (default: ~/.jitapi)
    - JITAPI_LOG_LEVEL: Optional. Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - JITAPI_LOG_FILE: Optional. File path for logs (default: stderr)

    The API key can also be set in a .env file in:
    - Current working directory
    - ~/.jitapi/.env
    - Home directory
    """
    # Load .env files (in order of priority)
    # 1. Current directory
    load_dotenv()
    # 2. Samvaad storage directory
    load_dotenv(Path.home() / ".jitapi" / ".env")
    # 3. Home directory
    load_dotenv(Path.home() / ".env")

    # Get configuration from environment
    storage_dir = os.environ.get(
        "JITAPI_STORAGE_DIR",
        str(Path.home() / ".jitapi"),
    )
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    log_level = os.environ.get("JITAPI_LOG_LEVEL", "INFO")
    log_file = os.environ.get("JITAPI_LOG_FILE")

    # Validate required keys
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
        print("", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY=your-key", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or configure it in your Claude Code MCP settings:", file=sys.stderr)
        print('  "env": { "OPENAI_API_KEY": "sk-..." }', file=sys.stderr)
        sys.exit(1)

    # Create and run server
    from .mcp.server import create_server

    server = create_server(
        storage_dir=storage_dir,
        openai_api_key=openai_api_key,
        log_level=log_level,
        log_file=log_file,
    )

    # Run the server
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
