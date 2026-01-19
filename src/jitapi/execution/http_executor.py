"""HTTP executor for making API calls.

Executes API calls with proper authentication, parameter handling,
and response processing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ..ingestion.parser import Endpoint
from .auth_handler import AuthHandler


@dataclass
class APICallResult:
    """Result of an API call."""

    success: bool
    status_code: int
    headers: dict[str, str]
    body: Any  # Parsed JSON or raw text
    raw_body: str
    error_message: str | None = None
    request_url: str = ""
    request_method: str = ""


class HTTPExecutor:
    """Executes HTTP requests to APIs.

    Handles:
    - Parameter substitution
    - Authentication injection
    - Request/response formatting
    - Error handling
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        auth_handler: AuthHandler | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the HTTP executor.

        Args:
            auth_handler: Optional auth handler for credential injection.
            timeout: Request timeout in seconds.
        """
        self.auth_handler = auth_handler
        self.timeout = timeout

    async def call_endpoint(
        self,
        endpoint: Endpoint,
        api_id: str,
        path_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        base_url_override: str | None = None,
    ) -> APICallResult:
        """Execute an API call to an endpoint.

        Args:
            endpoint: The endpoint to call.
            api_id: The API identifier (for auth lookup).
            path_params: Path parameter values.
            query_params: Query parameter values.
            body: Request body (for POST/PUT/PATCH).
            headers: Additional headers.
            base_url_override: Override the endpoint's base URL.

        Returns:
            APICallResult with response data.
        """
        # Build the URL
        base_url = base_url_override or (endpoint.servers[0] if endpoint.servers else "")
        path = self._substitute_path_params(endpoint.path, path_params or {})
        url = f"{base_url.rstrip('/')}{path}"

        # Prepare headers
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)

        # Add content-type for body
        if body and endpoint.request_body:
            content_type = endpoint.request_body.content_type
            req_headers["Content-Type"] = content_type

        # Apply authentication
        if self.auth_handler:
            req_headers, query_params = self.auth_handler.apply_auth(
                api_id, req_headers, query_params
            )

        # Make the request
        return await self._execute_request(
            method=endpoint.method,
            url=url,
            headers=req_headers,
            query_params=query_params,
            body=body,
        )

    async def call_raw(
        self,
        method: str,
        url: str,
        api_id: str | None = None,
        path_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APICallResult:
        """Execute a raw API call.

        Args:
            method: HTTP method.
            url: Full URL (may contain path params like {id}).
            api_id: Optional API ID for auth lookup.
            path_params: Path parameter values.
            query_params: Query parameter values.
            body: Request body.
            headers: Request headers.

        Returns:
            APICallResult with response data.
        """
        # Substitute path params in URL
        if path_params:
            url = self._substitute_path_params(url, path_params)

        # Prepare headers
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        if body:
            req_headers["Content-Type"] = "application/json"

        # Apply auth if available
        if api_id and self.auth_handler:
            req_headers, query_params = self.auth_handler.apply_auth(
                api_id, req_headers, query_params
            )

        return await self._execute_request(
            method=method,
            url=url,
            headers=req_headers,
            query_params=query_params,
            body=body,
        )

    async def _execute_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> APICallResult:
        """Execute the HTTP request."""
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            try:
                # Prepare request kwargs
                kwargs: dict[str, Any] = {
                    "headers": headers,
                }

                if query_params:
                    kwargs["params"] = query_params

                if body:
                    kwargs["json"] = body

                # Make request
                response = await client.request(method, url, **kwargs)

                # Parse response body
                raw_body = response.text
                try:
                    parsed_body = response.json()
                except json.JSONDecodeError:
                    parsed_body = raw_body

                # Determine success
                success = 200 <= response.status_code < 300

                return APICallResult(
                    success=success,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=parsed_body,
                    raw_body=raw_body,
                    error_message=None if success else self._extract_error(parsed_body),
                    request_url=str(response.url),
                    request_method=method,
                )

            except httpx.TimeoutException:
                return APICallResult(
                    success=False,
                    status_code=0,
                    headers={},
                    body=None,
                    raw_body="",
                    error_message="Request timed out",
                    request_url=url,
                    request_method=method,
                )

            except httpx.RequestError as e:
                return APICallResult(
                    success=False,
                    status_code=0,
                    headers={},
                    body=None,
                    raw_body="",
                    error_message=f"Request error: {str(e)}",
                    request_url=url,
                    request_method=method,
                )

    def _substitute_path_params(
        self, path: str, params: dict[str, Any]
    ) -> str:
        """Substitute path parameters in a URL path.

        Args:
            path: URL path with {param} placeholders.
            params: Dict of param name -> value.

        Returns:
            Path with substituted values.
        """
        result = path
        for name, value in params.items():
            # Handle both {name} and {name} patterns
            result = result.replace(f"{{{name}}}", str(value))
        return result

    def _extract_error(self, body: Any) -> str | None:
        """Extract error message from response body."""
        if isinstance(body, dict):
            # Common error field names
            for key in ["error", "message", "error_message", "detail", "errors"]:
                if key in body:
                    error = body[key]
                    if isinstance(error, str):
                        return error
                    elif isinstance(error, dict) and "message" in error:
                        return error["message"]
                    elif isinstance(error, list) and error:
                        return str(error[0])
                    else:
                        return str(error)
        elif isinstance(body, str):
            return body[:500]  # Truncate long error messages

        return None

    def build_curl_command(
        self,
        endpoint: Endpoint,
        api_id: str,
        path_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        base_url_override: str | None = None,
    ) -> str:
        """Generate a curl command for an API call.

        Useful for debugging and documentation.

        Args:
            endpoint: The endpoint.
            api_id: The API identifier.
            path_params: Path parameters.
            query_params: Query parameters.
            body: Request body.
            base_url_override: Base URL override.

        Returns:
            curl command string.
        """
        base_url = base_url_override or (endpoint.servers[0] if endpoint.servers else "")
        path = self._substitute_path_params(endpoint.path, path_params or {})
        url = f"{base_url.rstrip('/')}{path}"

        # Add query params to URL
        if query_params:
            params_str = "&".join(f"{k}={v}" for k, v in query_params.items())
            url = f"{url}?{params_str}"

        parts = [f"curl -X {endpoint.method}"]
        parts.append(f"'{url}'")

        # Headers
        parts.append("-H 'Accept: application/json'")
        if body:
            parts.append("-H 'Content-Type: application/json'")

        # Auth placeholder
        if self.auth_handler and self.auth_handler.has_auth(api_id):
            parts.append("-H 'Authorization: <YOUR_API_KEY>'")

        # Body
        if body:
            body_json = json.dumps(body)
            parts.append(f"-d '{body_json}'")

        return " \\\n  ".join(parts)
