from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from fastapi import HTTPException


@dataclass(frozen=True)
class ApiPrincipal:
    """Trusted identity and tenant scope associated with one API key.

    The legacy ``RAGBOT_API_KEYS`` setting only proves that a caller knows a
    service credential. ``RAGBOT_API_KEY_PRINCIPALS`` optionally binds that
    credential to tenants, a stable user identity, groups and roles so request
    payloads cannot expand the caller's authorization scope.
    """

    tenant_ids: frozenset[str]
    user_id: Optional[str] = None
    groups: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    admin: bool = False


def load_api_key_principals(raw: Optional[str] = None) -> Dict[str, ApiPrincipal]:
    value = os.getenv("RAGBOT_API_KEY_PRINCIPALS", "") if raw is None else raw
    if not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("RAGBOT_API_KEY_PRINCIPALS must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("RAGBOT_API_KEY_PRINCIPALS must be a JSON object keyed by API key")

    principals: Dict[str, ApiPrincipal] = {}
    for api_key, config in decoded.items():
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Principal API keys must be non-empty strings")
        if not isinstance(config, dict):
            raise ValueError(f"Principal definition for {api_key!r} must be an object")

        tenant_ids = _string_set(config.get("tenant_ids"), field="tenant_ids", api_key=api_key)
        groups = tuple(sorted(_string_set(config.get("groups"), field="groups", api_key=api_key)))
        roles = tuple(sorted(_string_set(config.get("roles"), field="roles", api_key=api_key)))
        user_id = config.get("user_id")
        if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
            raise ValueError(f"Principal user_id for {api_key!r} must be a non-empty string")
        admin = config.get("admin", False)
        if not isinstance(admin, bool):
            raise ValueError(f"Principal admin for {api_key!r} must be boolean")
        if not admin and not tenant_ids:
            raise ValueError(f"Principal {api_key!r} must declare tenant_ids or admin=true")

        principals[api_key] = ApiPrincipal(
            tenant_ids=frozenset(tenant_ids),
            user_id=user_id.strip() if isinstance(user_id, str) else None,
            groups=groups,
            roles=roles,
            admin=admin,
        )
    return principals


def get_api_principal(api_key: Optional[str]) -> Optional[ApiPrincipal]:
    principals = load_api_key_principals()
    if not principals:
        # Backward-compatible development mode: an unscoped API key retains the
        # historical request-supplied tenant/user semantics. Production startup
        # validation forbids this mode.
        return None
    if not api_key or api_key not in principals:
        raise HTTPException(status_code=403, detail="API key has no authorized principal")
    return principals[api_key]


def authorize_tenant(api_key: Optional[str], tenant_id: str) -> Optional[ApiPrincipal]:
    principal = get_api_principal(api_key)
    if principal is None:
        return None
    if not principal.admin and tenant_id not in principal.tenant_ids:
        raise HTTPException(status_code=403, detail="API key is not authorized for this tenant")
    return principal


def authorize_identity(
    api_key: Optional[str],
    tenant_id: str,
    requested_user_id: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    principal = authorize_tenant(api_key, tenant_id)
    if principal is None:
        return requested_user_id, (), ()
    if principal.user_id and requested_user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="Requested user_id does not match API key principal")
    return principal.user_id or requested_user_id, principal.groups, principal.roles


def allowed_tenants(api_key: Optional[str]) -> Optional[frozenset[str]]:
    principal = get_api_principal(api_key)
    if principal is None or principal.admin:
        return None
    return principal.tenant_ids


def validate_principal_coverage(api_keys: Iterable[str]) -> None:
    principals = load_api_key_principals()
    missing = sorted(key for key in api_keys if key not in principals)
    if missing:
        raise ValueError(
            "Every production RAGBOT_API_KEYS entry must have a principal mapping; "
            f"missing {len(missing)} key(s)"
        )
    for api_key in api_keys:
        principal = principals[api_key]
        if not principal.user_id:
            raise ValueError(
                "Production API principals require a stable user_id for ACL evaluation"
            )


def _string_set(value: object, *, field: str, api_key: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Principal {field} for {api_key!r} must be a list of strings")
    return {item.strip() for item in value if item.strip()}
