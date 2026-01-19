"""Specification store for raw OpenAPI specs and parsed data."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..ingestion.parser import Endpoint, ParsedSpec, Parameter, RequestBody, Response, SpecVersion


@dataclass
class APIMetadata:
    """Metadata about a registered API."""

    api_id: str
    title: str
    version: str
    description: str
    base_url: str
    endpoint_count: int
    spec_version: str
    source_url: str | None = None


class SpecStore:
    """Store for OpenAPI specifications and parsed endpoint data.

    Provides persistence for:
    - Raw OpenAPI specs (JSON)
    - Parsed endpoint data
    - API metadata for quick listing
    """

    def __init__(self, storage_dir: str | Path):
        """Initialize the spec store.

        Args:
            storage_dir: Directory to store specification data.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.specs_dir = self.storage_dir / "specs"
        self.endpoints_dir = self.storage_dir / "endpoints"
        self.metadata_file = self.storage_dir / "apis.json"

        self.specs_dir.mkdir(exist_ok=True)
        self.endpoints_dir.mkdir(exist_ok=True)

        # Load metadata index
        self._metadata: dict[str, APIMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load the API metadata index from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                for api_id, meta in data.items():
                    self._metadata[api_id] = APIMetadata(**meta)

    def _save_metadata(self) -> None:
        """Save the API metadata index to disk."""
        data = {api_id: asdict(meta) for api_id, meta in self._metadata.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def store_spec(
        self,
        api_id: str,
        parsed_spec: ParsedSpec,
        source_url: str | None = None,
    ) -> APIMetadata:
        """Store a parsed OpenAPI specification.

        Args:
            api_id: Unique identifier for this API.
            parsed_spec: The parsed OpenAPI specification.
            source_url: Original URL where the spec was fetched from.

        Returns:
            Metadata about the stored API.
        """
        # Store raw spec
        spec_file = self.specs_dir / f"{api_id}.json"
        with open(spec_file, "w") as f:
            json.dump(parsed_spec.raw_spec, f, indent=2)

        # Store parsed endpoints
        endpoints_file = self.endpoints_dir / f"{api_id}.json"
        endpoints_data = [self._endpoint_to_dict(ep) for ep in parsed_spec.endpoints]
        with open(endpoints_file, "w") as f:
            json.dump(endpoints_data, f, indent=2)

        # Update metadata
        metadata = APIMetadata(
            api_id=api_id,
            title=parsed_spec.title,
            version=parsed_spec.version,
            description=parsed_spec.description,
            base_url=parsed_spec.base_url,
            endpoint_count=len(parsed_spec.endpoints),
            spec_version=parsed_spec.spec_version.value,
            source_url=source_url,
        )
        self._metadata[api_id] = metadata
        self._save_metadata()

        return metadata

    def get_raw_spec(self, api_id: str) -> dict[str, Any] | None:
        """Get the raw OpenAPI spec for an API.

        Args:
            api_id: The API identifier.

        Returns:
            The raw spec dictionary, or None if not found.
        """
        spec_file = self.specs_dir / f"{api_id}.json"
        if not spec_file.exists():
            return None

        with open(spec_file) as f:
            return json.load(f)

    def get_endpoints(self, api_id: str) -> list[Endpoint] | None:
        """Get all parsed endpoints for an API.

        Args:
            api_id: The API identifier.

        Returns:
            List of parsed endpoints, or None if not found.
        """
        endpoints_file = self.endpoints_dir / f"{api_id}.json"
        if not endpoints_file.exists():
            return None

        with open(endpoints_file) as f:
            data = json.load(f)
            return [self._dict_to_endpoint(ep) for ep in data]

    def get_endpoint(self, api_id: str, endpoint_id: str) -> Endpoint | None:
        """Get a specific endpoint by ID.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier (e.g., "GET /users").

        Returns:
            The endpoint if found, None otherwise.
        """
        endpoints = self.get_endpoints(api_id)
        if not endpoints:
            return None

        for ep in endpoints:
            if ep.endpoint_id == endpoint_id:
                return ep

        return None

    def get_metadata(self, api_id: str) -> APIMetadata | None:
        """Get metadata for an API.

        Args:
            api_id: The API identifier.

        Returns:
            API metadata, or None if not found.
        """
        return self._metadata.get(api_id)

    def list_apis(self) -> list[APIMetadata]:
        """List all registered APIs.

        Returns:
            List of API metadata objects.
        """
        return list(self._metadata.values())

    def delete_api(self, api_id: str) -> bool:
        """Delete an API and all its data.

        Args:
            api_id: The API identifier.

        Returns:
            True if deleted, False if not found.
        """
        if api_id not in self._metadata:
            return False

        # Remove files
        spec_file = self.specs_dir / f"{api_id}.json"
        endpoints_file = self.endpoints_dir / f"{api_id}.json"

        if spec_file.exists():
            spec_file.unlink()
        if endpoints_file.exists():
            endpoints_file.unlink()

        # Update metadata
        del self._metadata[api_id]
        self._save_metadata()

        return True

    def api_exists(self, api_id: str) -> bool:
        """Check if an API is registered.

        Args:
            api_id: The API identifier.

        Returns:
            True if the API exists.
        """
        return api_id in self._metadata

    def _endpoint_to_dict(self, endpoint: Endpoint) -> dict[str, Any]:
        """Convert an Endpoint to a dictionary for JSON storage."""
        return {
            "endpoint_id": endpoint.endpoint_id,
            "path": endpoint.path,
            "method": endpoint.method,
            "summary": endpoint.summary,
            "description": endpoint.description,
            "operation_id": endpoint.operation_id,
            "tags": endpoint.tags,
            "parameters": [
                {
                    "name": p.name,
                    "location": p.location,
                    "required": p.required,
                    "description": p.description,
                    "schema_type": p.schema_type,
                    "schema_format": p.schema_format,
                    "enum_values": p.enum_values,
                    "default": p.default,
                    "example": p.example,
                }
                for p in endpoint.parameters
            ],
            "request_body": (
                {
                    "content_type": endpoint.request_body.content_type,
                    "schema": endpoint.request_body.schema,
                    "required": endpoint.request_body.required,
                    "description": endpoint.request_body.description,
                }
                if endpoint.request_body
                else None
            ),
            "responses": [
                {
                    "status_code": r.status_code,
                    "description": r.description,
                    "schema": r.schema,
                    "content_type": r.content_type,
                }
                for r in endpoint.responses
            ],
            "security": endpoint.security,
            "deprecated": endpoint.deprecated,
            "servers": endpoint.servers,
            "required_params": endpoint.required_params,
            "returned_fields": endpoint.returned_fields,
        }

    def _dict_to_endpoint(self, data: dict[str, Any]) -> Endpoint:
        """Convert a dictionary back to an Endpoint."""
        parameters = [
            Parameter(
                name=p["name"],
                location=p["location"],
                required=p["required"],
                description=p["description"],
                schema_type=p["schema_type"],
                schema_format=p.get("schema_format"),
                enum_values=p.get("enum_values"),
                default=p.get("default"),
                example=p.get("example"),
            )
            for p in data.get("parameters", [])
        ]

        request_body = None
        if data.get("request_body"):
            rb = data["request_body"]
            request_body = RequestBody(
                content_type=rb["content_type"],
                schema=rb["schema"],
                required=rb["required"],
                description=rb["description"],
            )

        responses = [
            Response(
                status_code=r["status_code"],
                description=r["description"],
                schema=r.get("schema"),
                content_type=r.get("content_type"),
            )
            for r in data.get("responses", [])
        ]

        return Endpoint(
            endpoint_id=data["endpoint_id"],
            path=data["path"],
            method=data["method"],
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            operation_id=data.get("operation_id"),
            tags=data.get("tags", []),
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            security=data.get("security", []),
            deprecated=data.get("deprecated", False),
            servers=data.get("servers", []),
            required_params=data.get("required_params", []),
            returned_fields=data.get("returned_fields", []),
        )
