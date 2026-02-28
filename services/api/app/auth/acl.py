from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .policy import compute_policy_hash
from ..storage.models import ACLPolicy

PUBLIC_SCOPE = "public"


class UserContext:
    """Encapsulates a user's identity including groups and roles."""

    def __init__(
        self,
        user_id: str,
        groups: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
    ) -> None:
        self.user_id = user_id
        self.groups: Set[str] = set(groups or [])
        self.roles: Set[str] = set(roles or [])


def compute_security_scope(
    user_id: str,
    policies: Iterable[ACLPolicy],
    groups: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
) -> List[str]:
    """Compute the security scope (list of allowed policy hashes) for a user.

    Supports:
    - ``allow_all``: public access
    - ``allow_users``: direct user-level access
    - ``allow_groups``: group membership
    - ``allow_roles``: role-based access
    """
    allowed: List[str] = [PUBLIC_SCOPE]
    user_groups = set(groups or [])
    user_roles = set(roles or [])

    for policy in policies:
        rules = policy.rules or {}

        if rules.get("allow_all") is True:
            allowed.append(policy.policy_hash)
            continue

        # Check user-level access
        allowed_users = set(rules.get("allow_users") or [])
        if user_id in allowed_users:
            allowed.append(policy.policy_hash)
            continue

        # Check group-level access
        allowed_groups = set(rules.get("allow_groups") or [])
        if user_groups & allowed_groups:
            allowed.append(policy.policy_hash)
            continue

        # Check role-level access
        allowed_roles = set(rules.get("allow_roles") or [])
        if user_roles & allowed_roles:
            allowed.append(policy.policy_hash)
            continue

    return allowed


def compute_security_scope_from_context(
    ctx: UserContext,
    policies: Iterable[ACLPolicy],
) -> List[str]:
    """Convenience wrapper taking a UserContext."""
    return compute_security_scope(
        user_id=ctx.user_id,
        policies=policies,
        groups=list(ctx.groups),
        roles=list(ctx.roles),
    )


def build_policy(acl_policy_id: str, tenant_id: str, rules: dict) -> ACLPolicy:
    policy_hash = compute_policy_hash(rules)
    return ACLPolicy(
        acl_policy_id=acl_policy_id,
        tenant_id=tenant_id,
        rules=rules,
        policy_hash=policy_hash,
    )
