"""Pydantic models for MCP tool input validation.

Provides type-safe validation for all tool inputs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RegisterApiInput(BaseModel):
    """Input for register_api tool."""

    api_id: str = Field(
        ...,
        description="Unique identifier for this API",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    spec_url: str = Field(
        ...,
        description="URL to the OpenAPI specification",
        min_length=1,
    )

    @field_validator("spec_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "file://")):
            raise ValueError("spec_url must be a valid URL starting with http://, https://, or file://")
        return v


class ListApisInput(BaseModel):
    """Input for list_apis tool."""

    pass  # No required inputs


class SearchEndpointsInput(BaseModel):
    """Input for search_endpoints tool."""

    query: str = Field(
        ...,
        description="Natural language search query",
        min_length=1,
        max_length=500,
    )
    api_id: str | None = Field(
        None,
        description="Optional: limit search to a specific API",
    )
    top_k: int = Field(
        5,
        description="Number of results to return",
        ge=1,
        le=50,
    )


class GetWorkflowInput(BaseModel):
    """Input for get_workflow tool."""

    query: str = Field(
        ...,
        description="What you want to accomplish",
        min_length=1,
        max_length=1000,
    )
    api_id: str = Field(
        ...,
        description="The API to use",
        min_length=1,
    )
    max_steps: int = Field(
        5,
        description="Maximum steps in the workflow",
        ge=1,
        le=20,
    )


class GetEndpointSchemaInput(BaseModel):
    """Input for get_endpoint_schema tool."""

    api_id: str = Field(
        ...,
        description="The API identifier",
        min_length=1,
    )
    endpoint_id: str = Field(
        ...,
        description="The endpoint identifier (e.g., 'GET /users/{id}')",
        min_length=1,
    )


class CallApiInput(BaseModel):
    """Input for call_api tool."""

    api_id: str = Field(
        ...,
        description="The API identifier",
        min_length=1,
    )
    endpoint_id: str = Field(
        ...,
        description="The endpoint to call",
        min_length=1,
    )
    path_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Path parameter values",
    )
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameter values",
    )
    body: dict[str, Any] | None = Field(
        None,
        description="Request body for POST/PUT/PATCH",
    )


class SetApiAuthInput(BaseModel):
    """Input for set_api_auth tool."""

    api_id: str = Field(
        ...,
        description="The API identifier",
        min_length=1,
    )
    auth_type: Literal["api_key", "api_key_query", "bearer"] = Field(
        ...,
        description="Type of authentication",
    )
    credential: str = Field(
        ...,
        description="The API key or bearer token",
        min_length=1,
    )
    header_name: str = Field(
        "X-API-Key",
        description="Header name for API key auth",
    )
    param_name: str = Field(
        "apikey",
        description="Query param name for API key auth",
    )


class ExecuteWorkflowInput(BaseModel):
    """Input for execute_workflow tool."""

    workflow_id: str = Field(
        ...,
        description="The workflow ID returned by get_workflow",
        min_length=1,
    )
    api_id: str = Field(
        ...,
        description="The API identifier",
        min_length=1,
    )
    override_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters to override",
    )


class DeleteApiInput(BaseModel):
    """Input for delete_api tool."""

    api_id: str = Field(
        ...,
        description="The API identifier to delete",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )


# Mapping of tool names to their input models
TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "register_api": RegisterApiInput,
    "list_apis": ListApisInput,
    "search_endpoints": SearchEndpointsInput,
    "get_workflow": GetWorkflowInput,
    "get_endpoint_schema": GetEndpointSchemaInput,
    "call_api": CallApiInput,
    "set_api_auth": SetApiAuthInput,
    "execute_workflow": ExecuteWorkflowInput,
    "delete_api": DeleteApiInput,
}


def validate_tool_input(tool_name: str, arguments: dict[str, Any]) -> BaseModel:
    """Validate tool input using the appropriate Pydantic model.

    Args:
        tool_name: The name of the tool.
        arguments: The input arguments.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError: If tool_name is unknown.
        ValidationError: If input validation fails.
    """
    model_class = TOOL_INPUT_MODELS.get(tool_name)
    if not model_class:
        raise ValueError(f"Unknown tool: {tool_name}")

    return model_class(**arguments)
