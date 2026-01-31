from __future__ import annotations

from typing import Iterable, List

from .policy import compute_policy_hash
from ..storage.models import ACLPolicy

PUBLIC_SCOPE = "public"


def compute_security_scope(user_id: str, policies: Iterable[ACLPolicy]) -> List[str]:
    allowed: List[str] = [PUBLIC_SCOPE]
    for policy in policies:
        rules = policy.rules or {}
        if rules.get("allow_all") is True:
            allowed.append(policy.policy_hash)
            continue
        allowed_users = set(rules.get("allow_users") or [])
        if user_id in allowed_users:
            allowed.append(policy.policy_hash)
    return allowed


def build_policy(acl_policy_id: str, tenant_id: str, rules: dict) -> ACLPolicy:
    policy_hash = compute_policy_hash(rules)
    return ACLPolicy(
        acl_policy_id=acl_policy_id,
        tenant_id=tenant_id,
        rules=rules,
        policy_hash=policy_hash,
    )

