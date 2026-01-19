"""OpenAPI specification parser.

Parses OpenAPI 3.x and Swagger 2.0 specs, extracting endpoints, schemas, and parameters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import yaml


class SpecVersion(Enum):
    """OpenAPI specification version."""

    OPENAPI_3 = "3.x"
    SWAGGER_2 = "2.0"


@dataclass
class Parameter:
    """API endpoint parameter."""

    name: str
    location: str  # path, query, header, cookie, body
    required: bool = False
    description: str = ""
    schema_type: str = "string"
    schema_format: str | None = None
    enum_values: list[str] | None = None
    default: Any = None
    example: Any = None


@dataclass
class RequestBody:
    """API request body definition."""

    content_type: str
    schema: dict[str, Any]
    required: bool = False
    description: str = ""


@dataclass
class Response:
    """API response definition."""

    status_code: str
    description: str
    schema: dict[str, Any] | None = None
    content_type: str | None = None


@dataclass
class Endpoint:
    """Parsed API endpoint with all metadata."""

    endpoint_id: str  # unique identifier: "METHOD /path"
    path: str
    method: str
    summary: str = ""
    description: str = ""
    operation_id: str | None = None
    tags: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    request_body: RequestBody | None = None
    responses: list[Response] = field(default_factory=list)
    security: list[dict[str, list[str]]] = field(default_factory=list)
    deprecated: bool = False
    servers: list[str] = field(default_factory=list)

    # Extracted for dependency analysis
    required_params: list[str] = field(default_factory=list)
    returned_fields: list[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding."""
        parts = [
            f"{self.method.upper()} {self.path}",
            self.summary,
            self.description,
        ]

        # Add parameter names
        param_names = [p.name for p in self.parameters]
        if param_names:
            parts.append(f"Parameters: {', '.join(param_names)}")

        # Add tags
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        return " ".join(filter(None, parts))


@dataclass
class ParsedSpec:
    """Complete parsed OpenAPI specification."""

    title: str
    version: str
    description: str
    spec_version: SpecVersion
    base_url: str
    endpoints: list[Endpoint]
    schemas: dict[str, dict[str, Any]]
    security_schemes: dict[str, dict[str, Any]]
    tags: list[dict[str, str]]
    raw_spec: dict[str, Any]


class OpenAPIParser:
    """Parser for OpenAPI 3.x and Swagger 2.0 specifications."""

    def __init__(self):
        self._resolved_refs: dict[str, Any] = {}

    async def parse_from_url(self, url: str) -> ParsedSpec:
        """Fetch and parse an OpenAPI spec from a URL."""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content = response.text

        # Determine format
        if url.endswith(".json") or content.strip().startswith("{"):
            spec_dict = json.loads(content)
        else:
            spec_dict = yaml.safe_load(content)

        return self.parse(spec_dict)

    def parse_from_file(self, file_path: str) -> ParsedSpec:
        """Parse an OpenAPI spec from a local file."""
        with open(file_path) as f:
            content = f.read()

        if file_path.endswith(".json"):
            spec_dict = json.loads(content)
        else:
            spec_dict = yaml.safe_load(content)

        return self.parse(spec_dict)

    def parse(self, spec: dict[str, Any]) -> ParsedSpec:
        """Parse an OpenAPI specification dictionary."""
        self._resolved_refs = {}

        # Detect version
        if "openapi" in spec:
            spec_version = SpecVersion.OPENAPI_3
            return self._parse_openapi3(spec)
        elif "swagger" in spec:
            spec_version = SpecVersion.SWAGGER_2
            return self._parse_swagger2(spec)
        else:
            raise ValueError("Unknown OpenAPI specification format")

    def _parse_openapi3(self, spec: dict[str, Any]) -> ParsedSpec:
        """Parse OpenAPI 3.x specification."""
        info = spec.get("info", {})

        # Extract base URL from servers
        servers = spec.get("servers", [])
        base_url = servers[0].get("url", "") if servers else ""

        # Parse schemas
        schemas = spec.get("components", {}).get("schemas", {})

        # Parse security schemes
        security_schemes = spec.get("components", {}).get("securitySchemes", {})

        # Parse endpoints
        endpoints = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            # Get path-level parameters
            path_params = path_item.get("parameters", [])

            for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                if method not in path_item:
                    continue

                operation = path_item[method]
                endpoint = self._parse_operation_openapi3(
                    path=path,
                    method=method,
                    operation=operation,
                    path_params=path_params,
                    spec=spec,
                    base_url=base_url,
                )
                endpoints.append(endpoint)

        return ParsedSpec(
            title=info.get("title", "Untitled API"),
            version=info.get("version", "1.0.0"),
            description=info.get("description", ""),
            spec_version=SpecVersion.OPENAPI_3,
            base_url=base_url,
            endpoints=endpoints,
            schemas=schemas,
            security_schemes=security_schemes,
            tags=spec.get("tags", []),
            raw_spec=spec,
        )

    def _parse_operation_openapi3(
        self,
        path: str,
        method: str,
        operation: dict[str, Any],
        path_params: list[dict],
        spec: dict[str, Any],
        base_url: str,
    ) -> Endpoint:
        """Parse a single OpenAPI 3.x operation."""
        endpoint_id = f"{method.upper()} {path}"

        # Parse parameters
        all_params = path_params + operation.get("parameters", [])
        parameters = []
        required_params = []

        for param in all_params:
            # Resolve $ref if needed
            param = self._resolve_ref(param, spec)
            if not param:
                continue

            p = Parameter(
                name=param.get("name", ""),
                location=param.get("in", "query"),
                required=param.get("required", False),
                description=param.get("description", ""),
                schema_type=param.get("schema", {}).get("type", "string"),
                schema_format=param.get("schema", {}).get("format"),
                enum_values=param.get("schema", {}).get("enum"),
                default=param.get("schema", {}).get("default"),
                example=param.get("example"),
            )
            parameters.append(p)
            if p.required:
                required_params.append(p.name)

        # Parse request body
        request_body = None
        if "requestBody" in operation:
            rb = operation["requestBody"]
            rb = self._resolve_ref(rb, spec)
            if rb:
                content = rb.get("content", {})
                # Prefer JSON content type
                for ct in ["application/json", "application/x-www-form-urlencoded"]:
                    if ct in content:
                        schema = content[ct].get("schema", {})
                        schema = self._resolve_ref(schema, spec)
                        request_body = RequestBody(
                            content_type=ct,
                            schema=schema or {},
                            required=rb.get("required", False),
                            description=rb.get("description", ""),
                        )
                        # Extract required params from body schema
                        if schema:
                            required_params.extend(schema.get("required", []))
                        break

        # Parse responses
        responses = []
        returned_fields = []
        for status_code, resp in operation.get("responses", {}).items():
            resp = self._resolve_ref(resp, spec)
            if not resp:
                continue

            response_schema = None
            content_type = None
            content = resp.get("content", {})
            if "application/json" in content:
                content_type = "application/json"
                response_schema = content["application/json"].get("schema", {})
                response_schema = self._resolve_ref(response_schema, spec)

                # Extract returned fields for dependency analysis
                if response_schema:
                    returned_fields.extend(self._extract_field_names(response_schema))

            responses.append(
                Response(
                    status_code=str(status_code),
                    description=resp.get("description", ""),
                    schema=response_schema,
                    content_type=content_type,
                )
            )

        # Get servers (operation-level override or path-level)
        servers = [s.get("url", "") for s in operation.get("servers", [])]
        if not servers:
            servers = [base_url]

        return Endpoint(
            endpoint_id=endpoint_id,
            path=path,
            method=method.upper(),
            summary=operation.get("summary", ""),
            description=operation.get("description", ""),
            operation_id=operation.get("operationId"),
            tags=operation.get("tags", []),
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            security=operation.get("security", []),
            deprecated=operation.get("deprecated", False),
            servers=servers,
            required_params=required_params,
            returned_fields=list(set(returned_fields)),
        )

    def _parse_swagger2(self, spec: dict[str, Any]) -> ParsedSpec:
        """Parse Swagger 2.0 specification."""
        info = spec.get("info", {})

        # Build base URL from host, basePath, schemes
        schemes = spec.get("schemes", ["https"])
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        base_url = f"{schemes[0]}://{host}{base_path}" if host else ""

        # Parse definitions (schemas in Swagger 2)
        schemas = spec.get("definitions", {})

        # Parse security definitions
        security_schemes = spec.get("securityDefinitions", {})

        # Parse endpoints
        endpoints = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            path_params = path_item.get("parameters", [])

            for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                if method not in path_item:
                    continue

                operation = path_item[method]
                endpoint = self._parse_operation_swagger2(
                    path=path,
                    method=method,
                    operation=operation,
                    path_params=path_params,
                    spec=spec,
                    base_url=base_url,
                )
                endpoints.append(endpoint)

        return ParsedSpec(
            title=info.get("title", "Untitled API"),
            version=info.get("version", "1.0.0"),
            description=info.get("description", ""),
            spec_version=SpecVersion.SWAGGER_2,
            base_url=base_url,
            endpoints=endpoints,
            schemas=schemas,
            security_schemes=security_schemes,
            tags=spec.get("tags", []),
            raw_spec=spec,
        )

    def _parse_operation_swagger2(
        self,
        path: str,
        method: str,
        operation: dict[str, Any],
        path_params: list[dict],
        spec: dict[str, Any],
        base_url: str,
    ) -> Endpoint:
        """Parse a single Swagger 2.0 operation."""
        endpoint_id = f"{method.upper()} {path}"

        # Parse parameters (includes body params in Swagger 2)
        all_params = path_params + operation.get("parameters", [])
        parameters = []
        required_params = []
        request_body = None

        for param in all_params:
            param = self._resolve_ref(param, spec)
            if not param:
                continue

            if param.get("in") == "body":
                # Body parameter in Swagger 2
                schema = param.get("schema", {})
                schema = self._resolve_ref(schema, spec)
                request_body = RequestBody(
                    content_type="application/json",
                    schema=schema or {},
                    required=param.get("required", False),
                    description=param.get("description", ""),
                )
                if schema:
                    required_params.extend(schema.get("required", []))
            else:
                p = Parameter(
                    name=param.get("name", ""),
                    location=param.get("in", "query"),
                    required=param.get("required", False),
                    description=param.get("description", ""),
                    schema_type=param.get("type", "string"),
                    schema_format=param.get("format"),
                    enum_values=param.get("enum"),
                    default=param.get("default"),
                    example=param.get("x-example"),
                )
                parameters.append(p)
                if p.required:
                    required_params.append(p.name)

        # Parse responses
        responses = []
        returned_fields = []
        for status_code, resp in operation.get("responses", {}).items():
            resp = self._resolve_ref(resp, spec)
            if not resp:
                continue

            response_schema = resp.get("schema")
            if response_schema:
                response_schema = self._resolve_ref(response_schema, spec)
                returned_fields.extend(self._extract_field_names(response_schema))

            responses.append(
                Response(
                    status_code=str(status_code),
                    description=resp.get("description", ""),
                    schema=response_schema,
                    content_type="application/json" if response_schema else None,
                )
            )

        return Endpoint(
            endpoint_id=endpoint_id,
            path=path,
            method=method.upper(),
            summary=operation.get("summary", ""),
            description=operation.get("description", ""),
            operation_id=operation.get("operationId"),
            tags=operation.get("tags", []),
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            security=operation.get("security", []),
            deprecated=operation.get("deprecated", False),
            servers=[base_url],
            required_params=required_params,
            returned_fields=list(set(returned_fields)),
        )

    def _resolve_ref(self, obj: Any, spec: dict[str, Any]) -> Any:
        """Resolve a $ref reference in the spec."""
        if not isinstance(obj, dict):
            return obj

        if "$ref" not in obj:
            return obj

        ref = obj["$ref"]

        # Check cache
        if ref in self._resolved_refs:
            return self._resolved_refs[ref]

        # Parse reference path
        if ref.startswith("#/"):
            # Local reference
            parts = ref[2:].split("/")
            resolved = spec
            for part in parts:
                # Handle JSON pointer escaping
                part = part.replace("~1", "/").replace("~0", "~")
                if isinstance(resolved, dict) and part in resolved:
                    resolved = resolved[part]
                else:
                    return None

            # Cache and return
            self._resolved_refs[ref] = resolved
            return resolved

        # External references not supported yet
        return None

    def _extract_field_names(self, schema: dict[str, Any]) -> list[str]:
        """Extract field names from a schema for dependency analysis."""
        if not schema:
            return []

        fields = []

        # Direct properties
        if "properties" in schema:
            fields.extend(schema["properties"].keys())

        # Array items
        if schema.get("type") == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict) and "properties" in items:
                fields.extend(items["properties"].keys())

        # AllOf composition
        if "allOf" in schema:
            for sub_schema in schema["allOf"]:
                fields.extend(self._extract_field_names(sub_schema))

        return fields
