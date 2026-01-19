"""Tests for the MCP server components."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jitapi.mcp.tools import ToolRegistry, ToolResult


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_components():
    """Create mock components for tool registry."""
    return {
        "indexer": MagicMock(),
        "spec_store": MagicMock(),
        "vector_store": MagicMock(),
        "graph_store": MagicMock(),
        "embedder": MagicMock(),
        "auth_handler": MagicMock(),
        "reranker": MagicMock(),
    }


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_get_tool_definitions(self, mock_components):
        """Test that tool definitions are properly generated."""
        registry = ToolRegistry(**mock_components)
        tools = registry.get_tool_definitions()

        assert len(tools) > 0

        # Check for expected tools
        tool_names = [t["name"] for t in tools]
        assert "register_api" in tool_names
        assert "list_apis" in tool_names
        assert "search_endpoints" in tool_names
        assert "get_workflow" in tool_names
        assert "call_api" in tool_names
        assert "set_api_auth" in tool_names

    def test_tool_definitions_have_schemas(self, mock_components):
        """Test that tool definitions have proper input schemas."""
        registry = ToolRegistry(**mock_components)
        tools = registry.get_tool_definitions()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_list_apis_tool(self, mock_components):
        """Test the list_apis tool execution."""
        mock_components["indexer"].list_apis.return_value = [
            {
                "api_id": "petstore",
                "title": "Petstore API",
                "version": "1.0.0",
                "description": "A pet store",
                "endpoint_count": 10,
                "source_url": "https://example.com/petstore.yaml",
            }
        ]

        registry = ToolRegistry(**mock_components)
        result = await registry.execute_tool("list_apis", {})

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["apis"][0]["api_id"] == "petstore"

    @pytest.mark.asyncio
    async def test_search_endpoints_tool(self, mock_components):
        """Test the search_endpoints tool execution."""
        # Setup mock vector searcher behavior
        mock_search_result = MagicMock()
        mock_search_result.endpoint_id = "GET /pets"
        mock_search_result.api_id = "petstore"
        mock_search_result.path = "/pets"
        mock_search_result.method = "GET"
        mock_search_result.summary = "List pets"
        mock_search_result.score = 0.9

        registry = ToolRegistry(**mock_components)
        registry.vector_searcher = MagicMock()
        registry.vector_searcher.search.return_value = [mock_search_result]

        result = await registry.execute_tool(
            "search_endpoints",
            {"query": "list all pets", "api_id": "petstore"},
        )

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["results"][0]["endpoint_id"] == "GET /pets"

    @pytest.mark.asyncio
    async def test_set_api_auth_tool(self, mock_components):
        """Test the set_api_auth tool execution."""
        registry = ToolRegistry(**mock_components)

        result = await registry.execute_tool(
            "set_api_auth",
            {
                "api_id": "petstore",
                "auth_type": "api_key",
                "credential": "test-api-key",
                "header_name": "X-API-Key",
            },
        )

        assert result.success is True
        assert result.data["api_id"] == "petstore"
        mock_components["auth_handler"].set_api_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mock_components):
        """Test that unknown tools return an error."""
        registry = ToolRegistry(**mock_components)

        result = await registry.execute_tool("unknown_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_tool_exception_handling(self, mock_components):
        """Test that tool exceptions are properly handled."""
        mock_components["indexer"].list_apis.side_effect = Exception("Test error")

        registry = ToolRegistry(**mock_components)
        result = await registry.execute_tool("list_apis", {})

        assert result.success is False
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_call_api_requires_auth(self, mock_components):
        """Test that call_api checks for authentication."""
        mock_components["spec_store"].get_endpoint.return_value = MagicMock()
        mock_components["auth_handler"].has_auth.return_value = False

        registry = ToolRegistry(**mock_components)

        result = await registry.execute_tool(
            "call_api",
            {
                "api_id": "petstore",
                "endpoint_id": "GET /pets",
            },
        )

        assert result.success is False
        assert "authentication" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_endpoint_schema_tool(self, mock_components):
        """Test the get_endpoint_schema tool execution."""
        mock_endpoint = MagicMock()
        mock_endpoint.endpoint_id = "GET /pets"
        mock_endpoint.path = "/pets"
        mock_endpoint.method = "GET"
        mock_endpoint.summary = "List pets"
        mock_endpoint.description = "List all pets"
        mock_endpoint.parameters = []
        mock_endpoint.request_body = None
        mock_endpoint.responses = []
        mock_endpoint.security = []
        mock_endpoint.servers = ["https://api.example.com"]
        mock_endpoint.tags = ["pets"]

        mock_components["spec_store"].get_endpoint.return_value = mock_endpoint
        mock_components["graph_store"].get_dependencies.return_value = []

        registry = ToolRegistry(**mock_components)

        result = await registry.execute_tool(
            "get_endpoint_schema",
            {"api_id": "petstore", "endpoint_id": "GET /pets"},
        )

        assert result.success is True
        assert result.data["endpoint_id"] == "GET /pets"
        assert "schema" in result.data
        assert "call_details" in result.data
