"""MCP resources for Samvaad.

Exposes API specifications and endpoint information as MCP resources.
"""

from __future__ import annotations

from typing import Any

from ..stores.spec_store import SpecStore


class ResourceRegistry:
    """Registry for MCP resources.

    Provides resources:
    - api://{api_id}/spec - Full OpenAPI specification
    - api://{api_id}/endpoints - List of all endpoints
    - api://{api_id}/endpoint/{endpoint_id} - Single endpoint details
    """

    def __init__(self, spec_store: SpecStore):
        """Initialize the resource registry.

        Args:
            spec_store: The spec store for API data.
        """
        self.spec_store = spec_store

    def list_resources(self) -> list[dict[str, Any]]:
        """List all available resources.

        Returns:
            List of resource definitions.
        """
        resources = []

        # Add resources for each registered API
        for api_meta in self.spec_store.list_apis():
            api_id = api_meta.api_id

            # Spec resource
            resources.append(
                {
                    "uri": f"api://{api_id}/spec",
                    "name": f"{api_meta.title} - OpenAPI Specification",
                    "description": f"Full OpenAPI specification for {api_meta.title}",
                    "mimeType": "application/json",
                }
            )

            # Endpoints list resource
            resources.append(
                {
                    "uri": f"api://{api_id}/endpoints",
                    "name": f"{api_meta.title} - Endpoints",
                    "description": f"List of all {api_meta.endpoint_count} endpoints",
                    "mimeType": "application/json",
                }
            )

        return resources

    def get_resource(self, uri: str) -> dict[str, Any] | None:
        """Get a resource by URI.

        Args:
            uri: The resource URI.

        Returns:
            Resource content dict, or None if not found.
        """
        # Parse URI
        if not uri.startswith("api://"):
            return None

        parts = uri[6:].split("/")
        if len(parts) < 2:
            return None

        api_id = parts[0]
        resource_type = parts[1]

        if resource_type == "spec":
            return self._get_spec_resource(api_id)
        elif resource_type == "endpoints":
            return self._get_endpoints_resource(api_id)
        elif resource_type == "endpoint" and len(parts) >= 3:
            endpoint_id = "/".join(parts[2:])  # Handle paths with slashes
            return self._get_endpoint_resource(api_id, endpoint_id)

        return None

    def _get_spec_resource(self, api_id: str) -> dict[str, Any] | None:
        """Get the full spec resource."""
        spec = self.spec_store.get_raw_spec(api_id)
        if not spec:
            return None

        return {
            "uri": f"api://{api_id}/spec",
            "mimeType": "application/json",
            "content": spec,
        }

    def _get_endpoints_resource(self, api_id: str) -> dict[str, Any] | None:
        """Get the endpoints list resource."""
        endpoints = self.spec_store.get_endpoints(api_id)
        if not endpoints:
            return None

        endpoint_list = [
            {
                "endpoint_id": ep.endpoint_id,
                "path": ep.path,
                "method": ep.method,
                "summary": ep.summary,
                "tags": ep.tags,
                "deprecated": ep.deprecated,
            }
            for ep in endpoints
        ]

        return {
            "uri": f"api://{api_id}/endpoints",
            "mimeType": "application/json",
            "content": {
                "api_id": api_id,
                "endpoints": endpoint_list,
                "count": len(endpoint_list),
            },
        }

    def _get_endpoint_resource(
        self, api_id: str, endpoint_id: str
    ) -> dict[str, Any] | None:
        """Get a single endpoint resource."""
        endpoint = self.spec_store.get_endpoint(api_id, endpoint_id)
        if not endpoint:
            return None

        return {
            "uri": f"api://{api_id}/endpoint/{endpoint_id}",
            "mimeType": "application/json",
            "content": {
                "endpoint_id": endpoint.endpoint_id,
                "path": endpoint.path,
                "method": endpoint.method,
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": [
                    {
                        "name": p.name,
                        "in": p.location,
                        "required": p.required,
                        "type": p.schema_type,
                        "description": p.description,
                    }
                    for p in endpoint.parameters
                ],
                "request_body": (
                    {
                        "content_type": endpoint.request_body.content_type,
                        "required": endpoint.request_body.required,
                        "schema": endpoint.request_body.schema,
                    }
                    if endpoint.request_body
                    else None
                ),
                "responses": [
                    {
                        "status_code": r.status_code,
                        "description": r.description,
                        "schema": r.schema,
                    }
                    for r in endpoint.responses
                ],
            },
        }
