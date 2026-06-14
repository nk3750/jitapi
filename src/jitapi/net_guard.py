"""Outbound-request guard: SSRF protection for an LLM-driven HTTP client.

JitAPI fetches arbitrary OpenAPI spec URLs and calls arbitrary API hosts on the
user's behalf, driven by a model. This module blocks requests to private,
loopback, link-local, and cloud-metadata addresses (e.g. 169.254.169.254)
unless the operator explicitly opts in via ``JITAPI_ALLOW_PRIVATE_HOSTS``.

The host is resolved and every returned address is checked, so a public
hostname that resolves to an internal IP is still blocked. (Note: this validates
at request time; it does not pin the resolved IP into the connection, so it is
not a complete defense against DNS rebinding — a reasonable trade-off for a
local, single-user MCP server.)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https")


def _parse_max_bytes(default: int = 10 * 1024 * 1024) -> int:
    """Read the response size cap from the environment, defensively.

    A bad value must not crash the package at import time, so anything
    non-numeric falls back to the default with a warning.
    """
    raw = os.environ.get("JITAPI_MAX_RESPONSE_BYTES")
    if raw is None:
        return default
    raw = raw.strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    logger.warning(
        "Invalid JITAPI_MAX_RESPONSE_BYTES=%r; using default %d bytes", raw, default
    )
    return default


DEFAULT_MAX_BYTES = _parse_max_bytes()


class BlockedRequestError(ValueError):
    """Raised when an outbound request targets a disallowed scheme or host."""


def allow_private_hosts() -> bool:
    """Whether private/loopback targets are permitted (opt-in env flag)."""
    return os.environ.get("JITAPI_ALLOW_PRIVATE_HOSTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _ip_is_internal(ip: ipaddress._BaseAddress) -> bool:
    # Block anything that is not a globally-routable address. This is stricter
    # than an explicit category list and covers private (RFC1918), loopback,
    # link-local (incl. 169.254.169.254 cloud metadata), CGNAT/Tailscale
    # (100.64.0.0/10), IPv6 ULA, reserved, multicast, and unspecified ranges.
    return not ip.is_global


def validate_url(url: str, *, allow_private: bool | None = None) -> str:
    """Validate an outbound URL, raising BlockedRequestError if disallowed.

    Rejects non-http(s) schemes and hosts that resolve to internal addresses
    unless private hosts are explicitly allowed.
    """
    if allow_private is None:
        allow_private = allow_private_hosts()

    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise BlockedRequestError(
            f"Blocked URL scheme '{parsed.scheme or '(none)'}': only http/https are allowed"
        )

    host = parsed.hostname
    if not host:
        raise BlockedRequestError(f"URL has no host: {url}")

    if allow_private:
        return url

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as e:
        raise BlockedRequestError(f"Invalid port in URL: {url}") from e

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise BlockedRequestError(f"Could not resolve host '{host}': {e}") from e

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _ip_is_internal(ip):
            raise BlockedRequestError(
                f"Blocked request to internal address {ip} (host '{host}'). "
                "Set JITAPI_ALLOW_PRIVATE_HOSTS=1 to allow private/loopback targets."
            )
    return url


async def fetch_text(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = 5,
    allow_private: bool | None = None,
) -> str:
    """Fetch a text body with SSRF validation, manual (re-validated) redirects,
    and a hard size cap.

    Each hop — including redirect targets — is validated before connecting, and
    the body is streamed and aborted if it exceeds ``max_bytes`` (defends against
    decompression bombs / multi-GB responses).
    """
    current = url
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        for _ in range(max_redirects + 1):
            validate_url(current, allow_private=allow_private)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        response.raise_for_status()
                        break
                    current = str(httpx.URL(response.url).join(location))
                    continue

                response.raise_for_status()

                total = 0
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise BlockedRequestError(
                            f"Response exceeded the {max_bytes}-byte limit while fetching {url}"
                        )
                    chunks.append(chunk)
                encoding = response.encoding or "utf-8"
                return b"".join(chunks).decode(encoding, errors="replace")

    raise BlockedRequestError(f"Too many redirects (> {max_redirects}) fetching {url}")
