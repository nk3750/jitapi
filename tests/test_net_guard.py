"""Tests for the outbound-request SSRF guard."""

from __future__ import annotations

import pytest

from jitapi.execution.http_executor import HTTPExecutor
from jitapi.net_guard import BlockedRequestError, allow_private_hosts, validate_url


class TestValidateUrl:
    def test_allows_public_ip(self):
        # Numeric IP -> no DNS lookup; 8.8.8.8 is globally routable.
        assert validate_url("https://8.8.8.8/openapi.json") == "https://8.8.8.8/openapi.json"

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/latest",  # loopback
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
            "http://10.0.0.1/",  # RFC1918
            "http://192.168.1.1/",  # RFC1918
            "http://172.16.0.1/",  # RFC1918
            "http://[::1]/",  # IPv6 loopback
            "http://100.64.0.1/",  # CGNAT / Tailscale (RFC6598)
            "http://[::ffff:127.0.0.1]/",  # IPv4-mapped IPv6 loopback
            "http://2130706433/",  # decimal-encoded 127.0.0.1
            "http://0.0.0.0/",  # unspecified
        ],
    )
    def test_blocks_internal_addresses(self, url):
        with pytest.raises(BlockedRequestError):
            validate_url(url)

    def test_malformed_port_raises_blocked_request_error(self):
        # Contract: validate_url raises BlockedRequestError, not a bare ValueError.
        with pytest.raises(BlockedRequestError):
            validate_url("http://example.com:99999/x")

    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://8.8.8.8/x", "gopher://8.8.8.8/", "data:text/plain,x"],
    )
    def test_blocks_non_http_schemes(self, url):
        with pytest.raises(BlockedRequestError):
            validate_url(url)

    def test_allow_private_arg_bypasses(self):
        assert validate_url("http://127.0.0.1/x", allow_private=True) == "http://127.0.0.1/x"

    def test_env_opt_in_allows_private(self, monkeypatch):
        monkeypatch.setenv("JITAPI_ALLOW_PRIVATE_HOSTS", "1")
        assert allow_private_hosts() is True
        # Should not raise once the operator opts in.
        assert validate_url("http://10.0.0.1/x") == "http://10.0.0.1/x"

    def test_env_default_blocks_private(self, monkeypatch):
        monkeypatch.delenv("JITAPI_ALLOW_PRIVATE_HOSTS", raising=False)
        assert allow_private_hosts() is False


class TestMaxBytesParsing:
    def test_invalid_value_falls_back_without_crashing(self, monkeypatch):
        from jitapi.net_guard import _parse_max_bytes

        monkeypatch.setenv("JITAPI_MAX_RESPONSE_BYTES", "10MB")
        assert _parse_max_bytes(default=123) == 123  # falls back, does not raise

    def test_valid_value_is_used(self, monkeypatch):
        from jitapi.net_guard import _parse_max_bytes

        monkeypatch.setenv("JITAPI_MAX_RESPONSE_BYTES", "5000")
        assert _parse_max_bytes() == 5000


class TestExecutorSSRF:
    async def test_call_raw_blocks_metadata_endpoint(self):
        executor = HTTPExecutor()
        result = await executor.call_raw("GET", "http://169.254.169.254/latest/meta-data/")
        assert result.success is False
        assert result.status_code == 0
        assert "internal address" in (result.error_message or "").lower()
