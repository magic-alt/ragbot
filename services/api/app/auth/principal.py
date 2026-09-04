from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from fastapi import HTTPException


# Tenant-scoped product capabilities. Global operational endpoints remain behind
# admin=true and are intentionally not granted by any tenant role.
CAP_KNOWLEDGE_QUERY = "knowledge.query"
CAP_CATALOG_READ = "catalog.read"
CAP_FEEDBACK_WRITE = "feedback.write"
CAP_SOURCE_CREATE = "source.create"
CAP_SOURCE_UPDATE = "source.update"
CAP_SOURCE_SYNC = "source.sync"
CAP_SOURCE_DELETE = "source.delete"
CAP_INGEST_RUN = "ingestion.run"
CAP_INGEST_RETRY = "ingestion.retry"

READER_CAPABILITIES = frozenset(
    {
        CAP_KNOWLEDGE_QUERY,
        CAP_CATALOG_READ,
        CAP_FEEDBACK_WRITE,
    }
)
OPERATOR_CAPABILITIES = READER_CAPABILITIES | frozenset(
    {
        CAP_SOURCE_CREATE,
        CAP_SOURCE_UPDATE,
        CAP_SOURCE_SYNC,
        CAP_INGEST_RUN,
        CAP_INGEST_RETRY,
    }
)
OWNER_CAPABILITIES = OPERATOR_CAPABILITIES | frozenset({CAP_SOURCE_DELETE})

ROLE_CAPABILITIES = {
    "reader": READER_CAPABILITIES,
    "operator": OPERATOR_CAPABILITIES,
    "owner": OWNER_CAPABILITIES,
}
ALL_TENANT_CAPABILITIES = frozenset().union(*ROLE_CAPABILITIES.values())


@dataclass(frozen=True)
class ApiPrincipal:
    """Trusted identity and tenant scope associated with one API key."""

    tenant_ids: frozenset[str]
    user_id: Optional[str] = None
    groups: Tuple[str, ...] = ()
    # roles may contain application/ACL roles in addition to the platform RBAC
    # roles reader/operator/owner. Only recognized RBAC roles grant capabilities.
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
        # Backward-compatible development mode. Production startup validation
        # requires scoped principals and therefore never enters this branch.
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


def capabilities_for_principal(principal: Optional[ApiPrincipal]) -> frozenset[str]:
    """Return the effective tenant capability set for one principal.

    Development mode (no principal mapping) preserves historical unrestricted
    behavior. admin=true is global and therefore also receives every tenant
    capability. Custom ACL roles never implicitly grant platform capabilities.
    """
    if principal is None or principal.admin:
        return ALL_TENANT_CAPABILITIES
    capabilities: set[str] = set()
    for role in principal.roles:
        capabilities.update(ROLE_CAPABILITIES.get(role.strip().lower(), ()))
    return frozenset(capabilities)


def require_capability(api_key: Optional[str], capability: str) -> Optional[ApiPrincipal]:
    """Require a named tenant capability when scoped principals are enabled."""
    if capability not in ALL_TENANT_CAPABILITIES:
        raise ValueError(f"Unknown Ragbot capability: {capability}")
    principal = get_api_principal(api_key)
    if capability in capabilities_for_principal(principal):
        return principal
    raise HTTPException(status_code=403, detail=f"API principal requires capability: {capability}")


def require_role(api_key: Optional[str], *allowed_roles: str) -> Optional[ApiPrincipal]:
    """Compatibility helper for callers that still reason in role names.

    New API authorization should use require_capability(). Role inheritance is
    resolved through ROLE_CAPABILITIES rather than special-casing owner.
    """
    principal = get_api_principal(api_key)
    if principal is None or principal.admin:
        return principal
    wanted = {role.strip().lower() for role in allowed_roles if role.strip()}
    actual = {role.strip().lower() for role in principal.roles}
    if actual.intersection(wanted):
        return principal
    # Preserve historical hierarchy for downstream imports of this helper.
    if "owner" in actual and wanted.intersection({"reader", "operator", "owner"}):
        return principal
    if "operator" in actual and "reader" in wanted:
        return principal
    expected = ", ".join(sorted(wanted)) or "required role"
    raise HTTPException(status_code=403, detail=f"API principal requires one of roles: {expected}")


def require_operator(api_key: Optional[str]) -> Optional[ApiPrincipal]:
    """Backward-compatible alias for the operator ingestion capability."""
    return require_capability(api_key, CAP_INGEST_RUN)


def require_admin(api_key: Optional[str]) -> None:
    """Protect global operational endpoints when scoped principals are enabled."""
    principal = get_api_principal(api_key)
    if principal is not None and not principal.admin:
        raise HTTPException(status_code=403, detail="Admin API principal required")


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
            raise ValueError("Production API principals require a stable user_id for ACL evaluation")
        if not principal.admin and not capabilities_for_principal(principal):
            raise ValueError(
                "Production tenant principals require at least one platform RBAC role: "
                "reader, operator, or owner"
            )


def _string_set(value: object, *, field: str, api_key: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Principal {field} for {api_key!r} must be a list of strings")
    return {item.strip() for item in value if item.strip()}
