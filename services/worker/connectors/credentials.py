"""Secret-reference helpers for cloud/SaaS connectors.

Connector configuration stores references such as ``env:RAGBOT_NOTION_TOKEN``
rather than credential material. The worker resolves them at execution time so
PostgreSQL Source/Job snapshots never persist access tokens, OAuth refresh
tokens or service-account private keys.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

_ENV_REF = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")


def validate_secret_ref(ref: str) -> str:
    """Validate a secret reference without requiring the secret in the API pod."""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("credential_ref must be a non-empty secret reference")
    value = ref.strip()
    if not _ENV_REF.fullmatch(value):
        raise ValueError("Only env:VARIABLE credential references are supported")
    return value


def resolve_secret(ref: str) -> str:
    """Resolve a supported secret reference without allowing arbitrary file reads."""
    value = validate_secret_ref(ref)
    match = _ENV_REF.fullmatch(value)
    assert match is not None
    name = match.group(1)
    secret = os.getenv(name)
    if secret is None or not secret.strip():
        raise ValueError(f"Credential environment variable is missing or empty: {name}")
    return secret


def resolve_json_secret(ref: str) -> dict[str, Any]:
    raw = resolve_secret(ref)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Credential reference does not contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Credential JSON must be an object")
    return value


def redacted_ref(ref: str | None) -> str | None:
    """Return a non-secret identifier suitable for logs/diagnostics."""
    if not ref:
        return None
    match = _ENV_REF.fullmatch(ref.strip())
    return f"env:{match.group(1)}" if match else "unsupported-secret-ref"
