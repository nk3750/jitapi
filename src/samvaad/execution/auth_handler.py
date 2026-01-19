"""Authentication handler for API calls.

Manages authentication credentials and injects them into API requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class AuthType(Enum):
    """Supported authentication types."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"


@dataclass
class AuthConfig:
    """Configuration for API authentication."""

    auth_type: AuthType
    api_id: str

    # For API key auth
    api_key: str | None = None
    api_key_header: str = "X-API-Key"  # or 'Authorization'
    api_key_prefix: str = ""  # e.g., "Bearer " or "ApiKey "

    # For query param auth
    api_key_param: str | None = None  # Use query param instead of header

    # For basic auth
    username: str | None = None
    password: str | None = None

    # For OAuth2
    access_token: str | None = None
    refresh_token: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    # For custom headers
    custom_headers: dict[str, str] | None = None


class AuthHandler:
    """Handles authentication for API calls.

    Stores credentials per API and injects them into requests.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        """Initialize the auth handler.

        Args:
            storage_dir: Optional directory for credential storage.
                        If None, credentials are only stored in memory.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._configs: dict[str, AuthConfig] = {}

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._auth_file = self.storage_dir / "auth.json"
            self._load_configs()

    def _load_configs(self) -> None:
        """Load saved auth configs from disk."""
        if not self._auth_file.exists():
            return

        try:
            with open(self._auth_file) as f:
                data = json.load(f)

            for api_id, config_data in data.items():
                self._configs[api_id] = AuthConfig(
                    auth_type=AuthType(config_data["auth_type"]),
                    api_id=api_id,
                    api_key=config_data.get("api_key"),
                    api_key_header=config_data.get("api_key_header", "X-API-Key"),
                    api_key_prefix=config_data.get("api_key_prefix", ""),
                    api_key_param=config_data.get("api_key_param"),
                    username=config_data.get("username"),
                    password=config_data.get("password"),
                    access_token=config_data.get("access_token"),
                    refresh_token=config_data.get("refresh_token"),
                    token_url=config_data.get("token_url"),
                    client_id=config_data.get("client_id"),
                    client_secret=config_data.get("client_secret"),
                    custom_headers=config_data.get("custom_headers"),
                )
        except (json.JSONDecodeError, KeyError):
            pass

    def _save_configs(self) -> None:
        """Save auth configs to disk."""
        if not self.storage_dir:
            return

        data = {}
        for api_id, config in self._configs.items():
            data[api_id] = {
                "auth_type": config.auth_type.value,
                "api_key": config.api_key,
                "api_key_header": config.api_key_header,
                "api_key_prefix": config.api_key_prefix,
                "api_key_param": config.api_key_param,
                "username": config.username,
                "password": config.password,
                "access_token": config.access_token,
                "refresh_token": config.refresh_token,
                "token_url": config.token_url,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "custom_headers": config.custom_headers,
            }

        with open(self._auth_file, "w") as f:
            json.dump(data, f, indent=2)

    def set_api_key(
        self,
        api_id: str,
        api_key: str,
        header_name: str = "X-API-Key",
        prefix: str = "",
    ) -> None:
        """Set API key authentication for an API.

        Args:
            api_id: The API identifier.
            api_key: The API key value.
            header_name: The header to use (default: X-API-Key).
            prefix: Optional prefix for the key (e.g., "Bearer ").
        """
        self._configs[api_id] = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_id=api_id,
            api_key=api_key,
            api_key_header=header_name,
            api_key_prefix=prefix,
        )
        self._save_configs()

    def set_api_key_query_param(
        self,
        api_id: str,
        api_key: str,
        param_name: str = "apikey",
    ) -> None:
        """Set API key authentication via query parameter.

        Args:
            api_id: The API identifier.
            api_key: The API key value.
            param_name: The query parameter name.
        """
        self._configs[api_id] = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_id=api_id,
            api_key=api_key,
            api_key_param=param_name,
        )
        self._save_configs()

    def set_bearer_token(self, api_id: str, token: str) -> None:
        """Set Bearer token authentication.

        Args:
            api_id: The API identifier.
            token: The bearer token.
        """
        self._configs[api_id] = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            api_id=api_id,
            access_token=token,
        )
        self._save_configs()

    def set_basic_auth(self, api_id: str, username: str, password: str) -> None:
        """Set basic authentication.

        Args:
            api_id: The API identifier.
            username: The username.
            password: The password.
        """
        self._configs[api_id] = AuthConfig(
            auth_type=AuthType.BASIC,
            api_id=api_id,
            username=username,
            password=password,
        )
        self._save_configs()

    def set_custom_headers(self, api_id: str, headers: dict[str, str]) -> None:
        """Set custom header authentication.

        Args:
            api_id: The API identifier.
            headers: Dict of header name -> value.
        """
        self._configs[api_id] = AuthConfig(
            auth_type=AuthType.CUSTOM_HEADER,
            api_id=api_id,
            custom_headers=headers,
        )
        self._save_configs()

    def get_auth_config(self, api_id: str) -> AuthConfig | None:
        """Get auth config for an API.

        Args:
            api_id: The API identifier.

        Returns:
            AuthConfig if configured, None otherwise.
        """
        return self._configs.get(api_id)

    def apply_auth(
        self,
        api_id: str,
        headers: dict[str, str],
        query_params: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, str] | None]:
        """Apply authentication to request headers/params.

        Args:
            api_id: The API identifier.
            headers: Existing headers dict (will be modified).
            query_params: Existing query params dict (may be modified).

        Returns:
            Tuple of (modified headers, modified query params).
        """
        config = self._configs.get(api_id)
        if not config:
            return headers, query_params

        if config.auth_type == AuthType.API_KEY:
            if config.api_key_param:
                # Add to query params
                if query_params is None:
                    query_params = {}
                query_params[config.api_key_param] = config.api_key
            else:
                # Add to headers
                value = f"{config.api_key_prefix}{config.api_key}"
                headers[config.api_key_header] = value

        elif config.auth_type == AuthType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {config.access_token}"

        elif config.auth_type == AuthType.BASIC:
            import base64

            credentials = f"{config.username}:{config.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif config.auth_type == AuthType.CUSTOM_HEADER:
            if config.custom_headers:
                headers.update(config.custom_headers)

        return headers, query_params

    def remove_auth(self, api_id: str) -> bool:
        """Remove authentication for an API.

        Args:
            api_id: The API identifier.

        Returns:
            True if removed, False if not found.
        """
        if api_id in self._configs:
            del self._configs[api_id]
            self._save_configs()
            return True
        return False

    def has_auth(self, api_id: str) -> bool:
        """Check if auth is configured for an API.

        Args:
            api_id: The API identifier.

        Returns:
            True if auth is configured.
        """
        return api_id in self._configs

    def list_configured_apis(self) -> list[str]:
        """List APIs with configured authentication.

        Returns:
            List of API IDs with auth configured.
        """
        return list(self._configs.keys())
