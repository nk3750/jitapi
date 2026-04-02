"""Tests for the authentication handler."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from jitapi.execution.auth_handler import AuthConfig, AuthHandler, AuthType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for auth storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def handler(temp_dir):
    """Create an AuthHandler with temporary storage."""
    return AuthHandler(temp_dir)


class TestAuthHandlerBasic:
    """Tests for basic credential storage and retrieval."""

    def test_set_api_key_header(self, handler):
        handler.set_api_key("myapi", "secret123", header_name="X-Custom-Key")
        headers, params = handler.apply_auth("myapi", {})
        assert headers["X-Custom-Key"] == "secret123"

    def test_set_api_key_query_param(self, handler):
        handler.set_api_key_query_param("weather", "abc", param_name="appid")
        headers, params = handler.apply_auth("weather", {})
        assert params["appid"] == "abc"

    def test_set_bearer_token(self, handler):
        handler.set_bearer_token("github", "ghp_token123")
        headers, params = handler.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer ghp_token123"

    def test_set_basic_auth(self, handler):
        handler.set_basic_auth("myapi", "user", "pass")
        headers, params = handler.apply_auth("myapi", {})
        assert headers["Authorization"].startswith("Basic ")

    def test_set_custom_headers(self, handler):
        handler.set_custom_headers("myapi", {"X-Foo": "bar", "X-Baz": "qux"})
        headers, params = handler.apply_auth("myapi", {})
        assert headers["X-Foo"] == "bar"
        assert headers["X-Baz"] == "qux"

    def test_no_auth_configured(self, handler):
        headers, params = handler.apply_auth("unknown", {"Accept": "json"})
        assert headers == {"Accept": "json"}
        assert params is None

    def test_has_auth(self, handler):
        assert handler.has_auth("myapi") is False
        handler.set_bearer_token("myapi", "token")
        assert handler.has_auth("myapi") is True

    def test_remove_auth(self, handler):
        handler.set_bearer_token("myapi", "token")
        assert handler.remove_auth("myapi") is True
        assert handler.has_auth("myapi") is False
        assert handler.remove_auth("myapi") is False

    def test_list_configured_apis(self, handler):
        handler.set_bearer_token("github", "token1")
        handler.set_api_key("stripe", "key1")
        apis = handler.list_configured_apis()
        assert set(apis) == {"github", "stripe"}


class TestAuthHandlerPersistence:
    """Tests for credential persistence to disk."""

    def test_persists_and_reloads(self, temp_dir):
        handler1 = AuthHandler(temp_dir)
        handler1.set_bearer_token("github", "ghp_token")

        handler2 = AuthHandler(temp_dir)
        assert handler2.has_auth("github")
        headers, _ = handler2.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer ghp_token"

    def test_file_permissions(self, temp_dir):
        handler = AuthHandler(temp_dir)
        handler.set_bearer_token("myapi", "token")
        auth_file = temp_dir / "auth.json"
        mode = oct(auth_file.stat().st_mode & 0o777)
        assert mode == "0o600"

    def test_dir_permissions(self, temp_dir):
        # AuthHandler sets 0700 on the storage dir
        handler = AuthHandler(temp_dir)
        mode = oct(temp_dir.stat().st_mode & 0o777)
        assert mode == "0o700"

    def test_in_memory_only(self):
        handler = AuthHandler(storage_dir=None)
        handler.set_bearer_token("myapi", "token")
        headers, _ = handler.apply_auth("myapi", {})
        assert headers["Authorization"] == "Bearer token"


class TestAuthHandlerEnvVar:
    """Tests for env var-based credential resolution."""

    def test_bearer_from_env_var(self, handler):
        handler.set_bearer_token("github", "", env_var="GITHUB_TOKEN")
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_from_env"}):
            headers, _ = handler.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer ghp_from_env"

    def test_api_key_header_from_env_var(self, handler):
        handler.set_api_key("stripe", "", header_name="X-API-Key", env_var="STRIPE_KEY")
        with patch.dict(os.environ, {"STRIPE_KEY": "sk_live_123"}):
            headers, _ = handler.apply_auth("stripe", {})
        assert headers["X-API-Key"] == "sk_live_123"

    def test_api_key_query_from_env_var(self, handler):
        handler.set_api_key_query_param("weather", "", param_name="appid", env_var="OWM_KEY")
        with patch.dict(os.environ, {"OWM_KEY": "weather_secret"}):
            _, params = handler.apply_auth("weather", {})
        assert params["appid"] == "weather_secret"

    def test_env_var_missing_logs_warning(self, handler):
        handler.set_bearer_token("github", "", env_var="NONEXISTENT_VAR")
        # Env var not set — should not inject auth
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_VAR", None)
            headers, _ = handler.apply_auth("github", {})
        assert "Authorization" not in headers

    def test_env_var_not_persisted_to_disk(self, temp_dir):
        handler = AuthHandler(temp_dir)
        handler.set_bearer_token("github", "should_not_appear", env_var="GH_TOKEN")

        auth_file = temp_dir / "auth.json"
        data = json.loads(auth_file.read_text())
        # The actual secret should NOT be in the file
        assert data["github"]["access_token"] is None
        # But the env var name should be
        assert data["github"]["credential_env_var"] == "GH_TOKEN"

    def test_env_var_survives_reload(self, temp_dir):
        handler1 = AuthHandler(temp_dir)
        handler1.set_bearer_token("github", "", env_var="GH_TOKEN")

        handler2 = AuthHandler(temp_dir)
        with patch.dict(os.environ, {"GH_TOKEN": "reloaded_token"}):
            headers, _ = handler2.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer reloaded_token"

    def test_env_var_takes_priority_over_stale_credential(self, handler):
        """When env_var is set, even if a credential was also passed, env var wins."""
        handler.set_bearer_token("github", "stale_token", env_var="GH_TOKEN")
        with patch.dict(os.environ, {"GH_TOKEN": "fresh_token"}):
            headers, _ = handler.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer fresh_token"

    def test_direct_credential_still_works(self, handler):
        """Without env_var, direct credential is used (backward compat)."""
        handler.set_bearer_token("github", "direct_token")
        headers, _ = handler.apply_auth("github", {})
        assert headers["Authorization"] == "Bearer direct_token"

    def test_direct_credential_persisted_to_disk(self, temp_dir):
        handler = AuthHandler(temp_dir)
        handler.set_bearer_token("github", "direct_token")

        data = json.loads((temp_dir / "auth.json").read_text())
        assert data["github"]["access_token"] == "direct_token"
        assert data["github"]["credential_env_var"] is None


class TestSetApiAuthTool:
    """Tests for the set_api_auth MCP tool with env_var support."""

    @pytest.fixture
    def registry(self):
        from unittest.mock import MagicMock
        from jitapi.mcp.tools import ToolRegistry

        return ToolRegistry(
            indexer=MagicMock(),
            spec_store=MagicMock(),
            vector_store=MagicMock(),
            graph_store=MagicMock(),
            embedder=MagicMock(),
            auth_handler=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_set_auth_with_env_var(self, registry):
        result = await registry.execute_tool(
            "set_api_auth",
            {"api_id": "github", "auth_type": "bearer", "env_var": "GITHUB_TOKEN"},
        )
        assert result.success is True
        assert result.data["credential_source"] == "env_var"
        registry.auth_handler.set_bearer_token.assert_called_once_with(
            "github", "", env_var="GITHUB_TOKEN",
        )

    @pytest.mark.asyncio
    async def test_set_auth_with_credential(self, registry):
        result = await registry.execute_tool(
            "set_api_auth",
            {"api_id": "github", "auth_type": "bearer", "credential": "ghp_123"},
        )
        assert result.success is True
        assert result.data["credential_source"] == "direct"

    @pytest.mark.asyncio
    async def test_set_auth_requires_credential_or_env_var(self, registry):
        result = await registry.execute_tool(
            "set_api_auth",
            {"api_id": "github", "auth_type": "bearer"},
        )
        assert result.success is False
        assert "credential" in result.error.lower() or "env_var" in result.error.lower()
