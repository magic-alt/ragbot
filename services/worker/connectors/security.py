from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def csv_values(name: str) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in os.getenv(name, "").split(",") if value.strip())


def validate_remote_url(
    url: str,
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    allow_private: Optional[bool] = None,
    schemes: tuple[str, ...] = ("http", "https"),
) -> str:
    """Validate a remotely fetched URL against SSRF-sensitive destinations."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in schemes:
        raise ValueError(f"Unsupported source URL scheme: {parsed.scheme or '<missing>'}")
    if not parsed.hostname:
        raise ValueError("Source URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials embedded in source URLs are not allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    host_allowlist = tuple((allowed_hosts or ()))
    if host_allowlist and not any(_host_matches(hostname, pattern) for pattern in host_allowlist):
        raise ValueError(f"Source host is not allowlisted: {hostname}")

    private_ok = env_flag("RAGBOT_ALLOW_PRIVATE_SOURCE_NETWORKS") if allow_private is None else allow_private
    if not private_ok:
        for address in _resolve_addresses(hostname, parsed.port):
            if _is_non_public(address):
                raise ValueError(f"Source host resolves to a non-public address: {address}")
    return url


def validate_local_repo_path(path: str) -> str:
    """Ensure production local Git ingestion stays inside configured roots."""
    resolved = Path(path).expanduser().resolve()
    roots_raw = os.getenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "")
    roots = [Path(value).expanduser().resolve() for value in roots_raw.split(os.pathsep) if value.strip()]

    # Local source access remains convenient in development. In production an
    # explicit root list is required so a Source cannot request arbitrary files.
    environment = os.getenv("RAGBOT_ENV", "development").strip().lower()
    if not roots:
        if environment in {"production", "prod"}:
            raise ValueError("Production local Git ingestion requires RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS")
        return str(resolved)

    if not any(_is_within(resolved, root) for root in roots):
        raise ValueError("Local Git source is outside RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS")
    return str(resolved)


def _resolve_addresses(hostname: str, port: Optional[int]) -> set[str]:
    try:
        infos = socket.getaddrinfo(hostname, port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve source host: {hostname}") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise ValueError(f"Unable to resolve source host: {hostname}")
    return addresses


def _is_non_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_matches(hostname: str, pattern: str) -> bool:
    normalized = pattern.strip().lower().rstrip(".")
    if normalized.startswith("*."):
        suffix = normalized[1:]
        return hostname.endswith(suffix) and hostname != suffix.lstrip(".")
    return hostname == normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
