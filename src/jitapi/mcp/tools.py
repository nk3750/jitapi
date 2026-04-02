"""MCP tool definitions for Samvaad.

Defines the tools exposed to Claude through MCP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..execution.auth_handler import AuthHandler
from ..execution.http_executor import HTTPExecutor
from ..execution.schema_formatter import SchemaFormatter
from ..ingestion.embedder import EndpointEmbedder
from ..ingestion.indexer import APIIndexer
from ..retrieval.graph_expander import GraphExpander
from ..retrieval.vector_search import VectorSearcher
from ..stores.graph_store import GraphStore
from ..stores.spec_store import SpecStore
from ..stores.vector_store import VectorStore
from .models import validate_tool_input

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any
    error: str | None = None


class ToolRegistry:
    """Registry and executor for Samvaad MCP tools.

    Provides the following tools:
    - register_api: Ingest an OpenAPI spec
    - list_apis: Show all registered APIs
    - search_endpoints: Semantic search for endpoints
    - get_workflow: Full retrieval pipeline
    - call_api: Execute an API call
    - set_api_auth: Configure authentication
    - execute_workflow: Execute a planned workflow
    - delete_api: Remove an API and all its data
    """

    def __init__(
        self,
        indexer: APIIndexer,
        spec_store: SpecStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: EndpointEmbedder,
        auth_handler: AuthHandler,
    ):
        """Initialize the tool registry.

        Args:
            indexer: API indexer for ingestion.
            spec_store: Spec store for endpoint data.
            vector_store: Vector store for search.
            graph_store: Graph store for dependencies.
            embedder: Embedder for query embedding.
            auth_handler: Auth handler for API credentials.
        """
        self.indexer = indexer
        self.spec_store = spec_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.auth_handler = auth_handler

        # Initialize helper components
        self.vector_searcher = VectorSearcher(vector_store, embedder)
        self.graph_expander = GraphExpander(graph_store, spec_store)
        self.schema_formatter = SchemaFormatter()
        self.http_executor = HTTPExecutor(auth_handler)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get MCP tool definitions for all tools.

        Returns:
            List of tool definition dicts.
        """
        return [
            {
                "name": "register_api",
                "description": "Register a new API by ingesting its OpenAPI specification. "
                "This parses the spec, builds a dependency graph, and creates searchable embeddings.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_id": {
                            "type": "string",
                            "description": "Unique identifier for this API (e.g., 'stripe', 'github')",
                        },
                        "spec_url": {
                            "type": "string",
                            "description": "URL to the OpenAPI specification (JSON or YAML)",
                        },
                    },
                    "required": ["api_id", "spec_url"],
                },
            },
            {
                "name": "list_apis",
                "description": "List all registered APIs with their basic information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "search_endpoints",
                "description": "Search for API endpoints using natural language. "
                "Returns semantically similar endpoints based on the query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you're looking for",
                        },
                        "api_id": {
                            "type": "string",
                            "description": "Optional: limit search to a specific API",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_workflow",
                "description": "Get relevant endpoints with dependency resolution and full schemas "
                "for accomplishing a task. Returns search results expanded with their dependencies "
                "so you can plan and execute the right API calls in the right order. "
                "After reviewing the results, use call_api to execute each step.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What you want to accomplish (e.g., 'create a user and place an order')",
                        },
                        "api_id": {
                            "type": "string",
                            "description": "The API to use",
                        },
                        "max_steps": {
                            "type": "integer",
                            "description": "Maximum number of endpoints to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query", "api_id"],
                },
            },
            {
                "name": "get_endpoint_schema",
                "description": "Get the full schema for a specific endpoint. "
                "Use this to get detailed parameter and response information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_id": {
                            "type": "string",
                            "description": "The API identifier",
                        },
                        "endpoint_id": {
                            "type": "string",
                            "description": "The endpoint identifier (e.g., 'GET /users/{id}')",
                        },
                    },
                    "required": ["api_id", "endpoint_id"],
                },
            },
            {
                "name": "call_api",
                "description": "Execute an API call. Make sure authentication is configured first.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_id": {
                            "type": "string",
                            "description": "The API identifier",
                        },
                        "endpoint_id": {
                            "type": "string",
                            "description": "The endpoint to call (e.g., 'GET /users/{id}')",
                        },
                        "path_params": {
                            "type": "object",
                            "description": "Path parameter values",
                        },
                        "query_params": {
                            "type": "object",
                            "description": "Query parameter values",
                        },
                        "body": {
                            "type": "object",
                            "description": "Request body for POST/PUT/PATCH",
                        },
                    },
                    "required": ["api_id", "endpoint_id"],
                },
            },
            {
                "name": "set_api_auth",
                "description": "Configure authentication for an API. "
                "Supports API key (header or query param) and bearer token auth.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_id": {
                            "type": "string",
                            "description": "The API identifier",
                        },
                        "auth_type": {
                            "type": "string",
                            "enum": ["api_key", "api_key_query", "bearer"],
                            "description": "Type of authentication",
                        },
                        "credential": {
                            "type": "string",
                            "description": "The API key or bearer token",
                        },
                        "header_name": {
                            "type": "string",
                            "description": "Header name for API key (default: X-API-Key)",
                        },
                        "param_name": {
                            "type": "string",
                            "description": "Query param name for API key auth",
                        },
                    },
                    "required": ["api_id", "auth_type", "credential"],
                },
            },
            {
                "name": "delete_api",
                "description": "Delete a registered API and all its data including endpoints, "
                "embeddings, dependency graph, and authentication credentials.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_id": {
                            "type": "string",
                            "description": "The API identifier to delete",
                        },
                    },
                    "required": ["api_id"],
                },
            },
        ]

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: The tool to execute.
            arguments: Tool arguments.

        Returns:
            ToolResult with execution results.
        """
        handlers = {
            "register_api": self._register_api,
            "list_apis": self._list_apis,
            "search_endpoints": self._search_endpoints,
            "get_workflow": self._get_workflow,
            "get_endpoint_schema": self._get_endpoint_schema,
            "call_api": self._call_api,
            "set_api_auth": self._set_api_auth,
            "delete_api": self._delete_api,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
            )

        # Validate input using Pydantic models
        try:
            validated = validate_tool_input(tool_name, arguments)
            # Convert back to dict for handlers (preserves validated/defaulted values)
            validated_args = validated.model_dump()
            logger.debug(f"Validated input for {tool_name}: {validated_args}")
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                error_messages.append(f"{field}: {msg}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Input validation failed: {'; '.join(error_messages)}",
            )
        except ValueError as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )

        try:
            return await handler(validated_args)
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _register_api(self, args: dict[str, Any]) -> ToolResult:
        """Register a new API."""
        api_id = args["api_id"]
        spec_url = args["spec_url"]

        result = await self.indexer.index_from_url(api_id, spec_url)

        if result.success:
            return ToolResult(
                success=True,
                data={
                    "api_id": result.api_id,
                    "title": result.title,
                    "version": result.version,
                    "endpoint_count": result.endpoint_count,
                    "dependency_count": result.dependency_count,
                    "message": f"Successfully registered {result.title} with {result.endpoint_count} endpoints",
                },
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error=result.error_message,
            )

    async def _list_apis(self, args: dict[str, Any]) -> ToolResult:
        """List all registered APIs."""
        apis = self.indexer.list_apis()
        return ToolResult(
            success=True,
            data={
                "apis": apis,
                "count": len(apis),
            },
        )

    async def _search_endpoints(self, args: dict[str, Any]) -> ToolResult:
        """Search for endpoints."""
        query = args["query"]
        api_id = args.get("api_id")
        top_k = args.get("top_k", 5)

        results = self.vector_searcher.search(query, api_id=api_id, top_k=top_k)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": [
                    {
                        "endpoint_id": r.endpoint_id,
                        "api_id": r.api_id,
                        "path": r.path,
                        "method": r.method,
                        "summary": r.summary,
                        "score": round(r.score, 3),
                    }
                    for r in results
                ],
                "count": len(results),
            },
        )

    async def _get_workflow(self, args: dict[str, Any]) -> ToolResult:
        """Get relevant endpoints with dependencies and schemas for a task."""
        query = args["query"]
        api_id = args["api_id"]
        max_steps = args.get("max_steps", 5)

        # Step 1: Vector search
        search_results = self.vector_searcher.search(query, api_id=api_id, top_k=10)

        if not search_results:
            return ToolResult(
                success=False,
                data=None,
                error=f"No endpoints found for query: {query}",
            )

        # Step 2: Graph expansion (adds dependencies the search might have missed)
        expansion = self.graph_expander.expand(
            search_results, api_id, max_depth=2, max_total=10
        )

        # Step 3: Collect endpoint data
        step_data = []
        step_endpoint_ids = set()
        for ep in expansion.endpoints[:max_steps]:
            endpoint = self.spec_store.get_endpoint(api_id, ep.endpoint_id)
            if endpoint:
                schema = self.schema_formatter.format_endpoint_for_call(endpoint)
                dependencies = self.graph_store.get_dependencies(api_id, ep.endpoint_id)
                step_endpoint_ids.add(ep.endpoint_id)
                step_data.append(
                    {
                        "endpoint_id": ep.endpoint_id,
                        "path": ep.path,
                        "method": ep.method,
                        "summary": ep.summary,
                        "relevance_score": round(ep.score, 3),
                        "is_dependency": ep.is_dependency,
                        "dependencies": dependencies,
                        "schema": schema,
                    }
                )

        # Step 4: Order steps — dependencies (providers) before dependents (consumers)
        # Simple heuristic: endpoints marked as dependencies come first,
        # then sort by relevance score within each group.
        # This handles cycles gracefully (common in auto-generated dependency graphs).
        dep_steps = [s for s in step_data if s["is_dependency"]]
        direct_steps = [s for s in step_data if not s["is_dependency"]]
        dep_steps.sort(key=lambda s: s["relevance_score"], reverse=True)
        direct_steps.sort(key=lambda s: s["relevance_score"], reverse=True)
        step_data = dep_steps + direct_steps

        # Number the steps
        steps = []
        for i, step in enumerate(step_data):
            step["step"] = i + 1
            steps.append(step)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "api_id": api_id,
                "steps": steps,
                "total_endpoints": len(steps),
                "message": "Steps are ordered by dependency (prerequisites first). Use call_api to execute each step.",
            },
        )

    async def _get_endpoint_schema(self, args: dict[str, Any]) -> ToolResult:
        """Get schema for a specific endpoint."""
        api_id = args["api_id"]
        endpoint_id = args["endpoint_id"]

        endpoint = self.spec_store.get_endpoint(api_id, endpoint_id)
        if not endpoint:
            return ToolResult(
                success=False,
                data=None,
                error=f"Endpoint not found: {endpoint_id}",
            )

        schema = self.schema_formatter.format_endpoint(endpoint)
        call_details = self.schema_formatter.format_endpoint_for_call(endpoint)

        # Get dependencies
        dependencies = self.graph_store.get_dependencies(api_id, endpoint_id)

        return ToolResult(
            success=True,
            data={
                "endpoint_id": endpoint_id,
                "schema": schema,
                "call_details": call_details,
                "dependencies": dependencies,
            },
        )

    async def _call_api(self, args: dict[str, Any]) -> ToolResult:
        """Execute an API call."""
        api_id = args["api_id"]
        endpoint_id = args["endpoint_id"]
        path_params = args.get("path_params", {})
        query_params = args.get("query_params", {})
        body = args.get("body")

        # Get the endpoint
        endpoint = self.spec_store.get_endpoint(api_id, endpoint_id)
        if not endpoint:
            return ToolResult(
                success=False,
                data=None,
                error=f"Endpoint not found: {endpoint_id}",
            )

        # Check auth
        if not self.auth_handler.has_auth(api_id):
            return ToolResult(
                success=False,
                data=None,
                error=f"No authentication configured for API: {api_id}. Use set_api_auth first.",
            )

        # Execute the call
        result = await self.http_executor.call_endpoint(
            endpoint=endpoint,
            api_id=api_id,
            path_params=path_params,
            query_params=query_params,
            body=body,
        )

        return ToolResult(
            success=result.success,
            data={
                "status_code": result.status_code,
                "body": result.body,
                "headers": dict(result.headers),
            },
            error=result.error_message,
        )

    async def _set_api_auth(self, args: dict[str, Any]) -> ToolResult:
        """Configure API authentication."""
        api_id = args["api_id"]
        auth_type = args["auth_type"]
        credential = args["credential"]

        if auth_type == "api_key":
            header_name = args.get("header_name", "X-API-Key")
            self.auth_handler.set_api_key(api_id, credential, header_name)
        elif auth_type == "api_key_query":
            param_name = args.get("param_name", "apikey")
            self.auth_handler.set_api_key_query_param(api_id, credential, param_name)
        elif auth_type == "bearer":
            self.auth_handler.set_bearer_token(api_id, credential)
        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown auth type: {auth_type}",
            )

        return ToolResult(
            success=True,
            data={
                "api_id": api_id,
                "auth_type": auth_type,
                "message": f"Authentication configured for {api_id}",
            },
        )

    async def _delete_api(self, args: dict[str, Any]) -> ToolResult:
        """Delete an API and all its data."""
        api_id = args["api_id"]

        # Check if API exists
        if not self.spec_store.api_exists(api_id):
            return ToolResult(
                success=False,
                data=None,
                error=f"API not found: {api_id}",
            )

        # Track what was deleted
        deleted = {
            "spec": False,
            "graph": False,
            "embeddings": 0,
            "auth": False,
        }

        # Delete from spec store (includes endpoints)
        deleted["spec"] = self.spec_store.delete_api(api_id)

        # Delete from graph store
        deleted["graph"] = self.graph_store.delete_graph(api_id)

        # Delete from vector store
        deleted["embeddings"] = self.vector_store.delete_api(api_id)

        # Delete auth credentials
        deleted["auth"] = self.auth_handler.remove_auth(api_id)

        logger.info(f"Deleted API {api_id}: {deleted}")

        return ToolResult(
            success=True,
            data={
                "api_id": api_id,
                "deleted": deleted,
                "message": f"Successfully deleted API '{api_id}' and all associated data",
            },
        )
