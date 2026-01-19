"""Schema formatter for LLM consumption.

Formats OpenAPI schemas into a clear, concise format that LLMs can
easily understand and use to generate API calls.
"""

from __future__ import annotations

from typing import Any

from ..ingestion.parser import Endpoint, Parameter, RequestBody


class SchemaFormatter:
    """Formats API schemas for LLM consumption.

    Converts complex OpenAPI schemas into a simplified format that's
    easier for LLMs to understand and use.
    """

    def format_endpoint(self, endpoint: Endpoint) -> dict[str, Any]:
        """Format an endpoint for LLM consumption.

        Args:
            endpoint: The endpoint to format.

        Returns:
            A simplified dict representation of the endpoint.
        """
        formatted = {
            "endpoint_id": endpoint.endpoint_id,
            "method": endpoint.method,
            "path": endpoint.path,
            "summary": endpoint.summary,
            "description": endpoint.description or endpoint.summary,
        }

        # Format parameters
        if endpoint.parameters:
            formatted["parameters"] = self._format_parameters(endpoint.parameters)

        # Format request body
        if endpoint.request_body:
            formatted["request_body"] = self._format_request_body(endpoint.request_body)

        # Format responses
        if endpoint.responses:
            formatted["responses"] = self._format_responses(endpoint.responses)

        # Add auth requirements
        if endpoint.security:
            formatted["authentication"] = self._format_security(endpoint.security)

        return formatted

    def format_endpoint_for_call(self, endpoint: Endpoint) -> dict[str, Any]:
        """Format an endpoint with details needed to make an API call.

        Args:
            endpoint: The endpoint to format.

        Returns:
            Dict with call-ready information.
        """
        formatted = {
            "method": endpoint.method,
            "path": endpoint.path,
            "base_url": endpoint.servers[0] if endpoint.servers else "",
        }

        # Separate parameters by location
        path_params = []
        query_params = []
        header_params = []

        for param in endpoint.parameters:
            param_info = {
                "name": param.name,
                "type": param.schema_type,
                "required": param.required,
                "description": param.description,
            }
            if param.enum_values:
                param_info["allowed_values"] = param.enum_values
            if param.default is not None:
                param_info["default"] = param.default
            if param.example is not None:
                param_info["example"] = param.example

            if param.location == "path":
                path_params.append(param_info)
            elif param.location == "query":
                query_params.append(param_info)
            elif param.location == "header":
                header_params.append(param_info)

        if path_params:
            formatted["path_parameters"] = path_params
        if query_params:
            formatted["query_parameters"] = query_params
        if header_params:
            formatted["header_parameters"] = header_params

        # Request body schema
        if endpoint.request_body:
            formatted["request_body"] = {
                "content_type": endpoint.request_body.content_type,
                "required": endpoint.request_body.required,
                "schema": self._simplify_schema(endpoint.request_body.schema),
            }

        return formatted

    def format_workflow_step(
        self,
        step_number: int,
        endpoint: Endpoint,
        purpose: str,
        requires: list[str],
        provides: list[str],
    ) -> dict[str, Any]:
        """Format a workflow step for LLM execution.

        Args:
            step_number: The step number in the workflow.
            endpoint: The endpoint for this step.
            purpose: Why this step is needed.
            requires: What this step needs from previous steps.
            provides: What this step provides for later steps.

        Returns:
            Formatted workflow step.
        """
        return {
            "step": step_number,
            "endpoint": endpoint.endpoint_id,
            "purpose": purpose,
            "requires": requires,
            "provides": provides,
            "call_details": self.format_endpoint_for_call(endpoint),
        }

    def _format_parameters(self, parameters: list[Parameter]) -> list[dict[str, Any]]:
        """Format parameter list."""
        formatted = []
        for param in parameters:
            p = {
                "name": param.name,
                "in": param.location,
                "type": param.schema_type,
                "required": param.required,
            }
            if param.description:
                p["description"] = param.description
            if param.enum_values:
                p["allowed_values"] = param.enum_values
            if param.default is not None:
                p["default"] = param.default
            if param.example is not None:
                p["example"] = param.example
            formatted.append(p)
        return formatted

    def _format_request_body(self, body: RequestBody) -> dict[str, Any]:
        """Format request body."""
        return {
            "content_type": body.content_type,
            "required": body.required,
            "schema": self._simplify_schema(body.schema),
        }

    def _format_responses(self, responses) -> dict[str, Any]:
        """Format response definitions."""
        formatted = {}
        for resp in responses:
            formatted[resp.status_code] = {
                "description": resp.description,
            }
            if resp.schema:
                formatted[resp.status_code]["schema"] = self._simplify_schema(resp.schema)
        return formatted

    def _format_security(self, security: list[dict]) -> list[str]:
        """Format security requirements."""
        auth_methods = []
        for sec in security:
            auth_methods.extend(sec.keys())
        return list(set(auth_methods))

    def _simplify_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Simplify a JSON schema for LLM consumption.

        Removes unnecessary complexity while preserving essential information.
        """
        if not schema:
            return {}

        simplified = {}

        # Type
        if "type" in schema:
            simplified["type"] = schema["type"]

        # For objects, simplify properties
        if schema.get("type") == "object" and "properties" in schema:
            simplified["properties"] = {}
            for name, prop in schema["properties"].items():
                simplified["properties"][name] = self._simplify_property(prop)

            # Required fields
            if "required" in schema:
                simplified["required"] = schema["required"]

        # For arrays, simplify items
        elif schema.get("type") == "array" and "items" in schema:
            simplified["items"] = self._simplify_schema(schema["items"])

        # Enum values
        if "enum" in schema:
            simplified["enum"] = schema["enum"]

        # Format
        if "format" in schema:
            simplified["format"] = schema["format"]

        return simplified

    def _simplify_property(self, prop: dict[str, Any]) -> dict[str, Any]:
        """Simplify a single property definition."""
        simplified = {}

        if "type" in prop:
            simplified["type"] = prop["type"]
        if "description" in prop:
            simplified["description"] = prop["description"]
        if "enum" in prop:
            simplified["enum"] = prop["enum"]
        if "format" in prop:
            simplified["format"] = prop["format"]
        if "example" in prop:
            simplified["example"] = prop["example"]
        if "default" in prop:
            simplified["default"] = prop["default"]

        # Nested object
        if prop.get("type") == "object" and "properties" in prop:
            simplified["properties"] = {}
            for name, nested in prop["properties"].items():
                simplified["properties"][name] = self._simplify_property(nested)

        # Array items
        if prop.get("type") == "array" and "items" in prop:
            simplified["items"] = self._simplify_property(prop["items"])

        return simplified

    def format_for_tool_definition(
        self, endpoint: Endpoint, api_id: str
    ) -> dict[str, Any]:
        """Format an endpoint as an MCP tool definition.

        Args:
            endpoint: The endpoint to format.
            api_id: The API this endpoint belongs to.

        Returns:
            Tool definition dict for MCP.
        """
        # Build input schema
        properties = {}
        required = []

        # Path parameters
        for param in endpoint.parameters:
            if param.location == "path":
                properties[param.name] = self._param_to_json_schema(param)
                if param.required:
                    required.append(param.name)

        # Query parameters
        for param in endpoint.parameters:
            if param.location == "query":
                properties[param.name] = self._param_to_json_schema(param)
                if param.required:
                    required.append(param.name)

        # Request body
        if endpoint.request_body and endpoint.request_body.schema:
            body_schema = endpoint.request_body.schema
            if body_schema.get("type") == "object" and "properties" in body_schema:
                for name, prop in body_schema["properties"].items():
                    properties[name] = self._simplify_property(prop)
                if endpoint.request_body.required and "required" in body_schema:
                    required.extend(body_schema["required"])

        # Build tool name from endpoint
        tool_name = self._make_tool_name(endpoint, api_id)

        return {
            "name": tool_name,
            "description": endpoint.summary or f"{endpoint.method} {endpoint.path}",
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _param_to_json_schema(self, param: Parameter) -> dict[str, Any]:
        """Convert a Parameter to JSON Schema property."""
        schema = {"type": param.schema_type}

        if param.description:
            schema["description"] = param.description
        if param.enum_values:
            schema["enum"] = param.enum_values
        if param.default is not None:
            schema["default"] = param.default

        return schema

    def _make_tool_name(self, endpoint: Endpoint, api_id: str) -> str:
        """Generate a tool name from an endpoint."""
        if endpoint.operation_id:
            # Use operation ID if available
            name = endpoint.operation_id
        else:
            # Generate from method and path
            path_parts = endpoint.path.strip("/").split("/")
            # Remove path parameters
            path_parts = [p for p in path_parts if not p.startswith("{")]
            name = f"{endpoint.method.lower()}_{'_'.join(path_parts)}"

        # Prefix with API ID
        return f"{api_id}_{name}"
