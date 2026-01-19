"""MCP Server for Samvaad.

Creates and configures the MCP server that exposes API orchestration
capabilities to Claude.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

from ..execution.auth_handler import AuthHandler
from ..ingestion.embedder import EndpointEmbedder
from ..ingestion.indexer import APIIndexer
from ..retrieval.reranker import LLMReranker
from ..stores.graph_store import GraphStore
from ..stores.spec_store import SpecStore
from ..stores.vector_store import VectorStore
from .resources import ResourceRegistry
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the Samvaad MCP server.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for logging. If None, logs to stderr.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger for jitapi
    jitapi_logger = logging.getLogger("jitapi")
    jitapi_logger.setLevel(log_level)

    # Remove existing handlers
    jitapi_logger.handlers.clear()

    # Add handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        # Use stderr to avoid interfering with MCP's stdout/stdin communication
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    jitapi_logger.addHandler(handler)

    # Also configure the mcp logger to reduce noise
    mcp_logger = logging.getLogger("mcp")
    mcp_logger.setLevel(logging.WARNING)


class SamvaadServer:
    """MCP Server for API orchestration.

    Provides tools for:
    - Registering and managing APIs
    - Searching endpoints semantically
    - Getting complete workflows with dependencies
    - Executing API calls
    """

    def __init__(
        self,
        storage_dir: str | Path,
        openai_api_key: str | None = None,
        log_level: str = "INFO",
        log_file: str | None = None,
    ):
        """Initialize the Samvaad server.

        Args:
            storage_dir: Directory for data storage.
            openai_api_key: OpenAI API key for embeddings and reranking.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_file: Optional file path for logging.
        """
        # Setup logging first
        setup_logging(log_level, log_file)

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing Samvaad server with storage: {self.storage_dir}")

        # Initialize components
        logger.debug("Initializing indexer...")
        self.indexer = APIIndexer(self.storage_dir, openai_api_key)

        logger.debug("Initializing stores...")
        self.spec_store = SpecStore(self.storage_dir)
        self.vector_store = VectorStore(self.storage_dir)
        self.graph_store = GraphStore(self.storage_dir)

        logger.debug("Initializing embedder...")
        self.embedder = EndpointEmbedder(api_key=openai_api_key)

        logger.debug("Initializing auth handler...")
        self.auth_handler = AuthHandler(self.storage_dir)

        # Reranker (uses OpenAI)
        logger.debug("Initializing reranker...")
        self.reranker = LLMReranker(api_key=openai_api_key)

        # Initialize registries
        logger.debug("Initializing tool registry...")
        self.tool_registry = ToolRegistry(
            indexer=self.indexer,
            spec_store=self.spec_store,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            auth_handler=self.auth_handler,
            reranker=self.reranker,
        )
        self.resource_registry = ResourceRegistry(self.spec_store)

        # Create MCP server
        self.server = Server("jitapi")
        self._setup_handlers()
        logger.info("Samvaad server initialized successfully")

    def _setup_handlers(self):
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            tool_defs = self.tool_registry.get_tool_definitions()
            return [
                Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in tool_defs
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a tool."""
            logger.info(f"Tool call: {name}")
            logger.debug(f"Tool arguments: {arguments}")

            result = await self.tool_registry.execute_tool(name, arguments)

            if result.success:
                logger.info(f"Tool {name} completed successfully")
                content = json.dumps(result.data, indent=2)
            else:
                logger.warning(f"Tool {name} failed: {result.error}")
                content = json.dumps(
                    {"error": result.error, "success": False}, indent=2
                )

            return [TextContent(type="text", text=content)]

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List available resources."""
            resource_defs = self.resource_registry.list_resources()
            return [
                Resource(
                    uri=r["uri"],
                    name=r["name"],
                    description=r.get("description"),
                    mimeType=r.get("mimeType"),
                )
                for r in resource_defs
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource."""
            resource = self.resource_registry.get_resource(uri)
            if resource is None:
                raise ValueError(f"Resource not found: {uri}")

            return json.dumps(resource["content"], indent=2)

        @self.server.list_prompts()
        async def list_prompts() -> list[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="explore_api",
                    description="Explore what an API can do",
                    arguments=[
                        PromptArgument(
                            name="api_id",
                            description="The API to explore",
                            required=True,
                        ),
                        PromptArgument(
                            name="task",
                            description="What you want to accomplish",
                            required=True,
                        ),
                    ],
                ),
                Prompt(
                    name="register_and_explore",
                    description="Register a new API and explore its capabilities",
                    arguments=[
                        PromptArgument(
                            name="api_id",
                            description="Identifier for the API",
                            required=True,
                        ),
                        PromptArgument(
                            name="spec_url",
                            description="URL to the OpenAPI spec",
                            required=True,
                        ),
                        PromptArgument(
                            name="task",
                            description="What you want to accomplish",
                            required=True,
                        ),
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
            """Get a prompt."""
            arguments = arguments or {}

            if name == "explore_api":
                api_id = arguments.get("api_id", "")
                task = arguments.get("task", "")

                return GetPromptResult(
                    description=f"Explore {api_id} API",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""I want to use the {api_id} API to: {task}

Please:
1. Search for relevant endpoints using search_endpoints
2. Get the workflow using get_workflow
3. Explain what endpoints are needed and in what order
4. Show me the required parameters for each step""",
                            ),
                        ),
                    ],
                )

            elif name == "register_and_explore":
                api_id = arguments.get("api_id", "")
                spec_url = arguments.get("spec_url", "")
                task = arguments.get("task", "")

                return GetPromptResult(
                    description=f"Register and explore {api_id}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""Please:
1. Register the API using register_api with api_id="{api_id}" and spec_url="{spec_url}"
2. Then help me accomplish: {task}

Search for relevant endpoints and show me the workflow needed.""",
                            ),
                        ),
                    ],
                )

            raise ValueError(f"Unknown prompt: {name}")

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Samvaad MCP server on stdio...")
        async with stdio_server() as (read_stream, write_stream):
            logger.debug("Server connected, processing requests...")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
        logger.info("Samvaad MCP server stopped")


def create_server(
    storage_dir: str | Path,
    openai_api_key: str | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
) -> SamvaadServer:
    """Create a Samvaad MCP server instance.

    Args:
        storage_dir: Directory for data storage.
        openai_api_key: OpenAI API key for embeddings and reranking.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for logging.

    Returns:
        Configured SamvaadServer instance.
    """
    return SamvaadServer(
        storage_dir=storage_dir,
        openai_api_key=openai_api_key,
        log_level=log_level,
        log_file=log_file,
    )
