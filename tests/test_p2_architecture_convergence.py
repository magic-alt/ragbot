from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from services.api.app.auth.principal import (
    CAP_CATALOG_READ,
    CAP_INGEST_RUN,
    CAP_KNOWLEDGE_QUERY,
    CAP_SOURCE_CREATE,
    CAP_SOURCE_DELETE,
    OPERATOR_CAPABILITIES,
    OWNER_CAPABILITIES,
    READER_CAPABILITIES,
    ApiPrincipal,
    capabilities_for_principal,
    require_capability,
    validate_principal_coverage,
)
from services.api.app.observability.metrics import MetricsCollector, RequestMetrics
from services.api.app.observability.prometheus import render_prometheus
from services.api.app.storage.repo import InMemoryRepo


def _principal_env(role: str) -> dict[str, str]:
    return {
        "RAGBOT_API_KEY_PRINCIPALS": json.dumps(
            {
                "key": {
                    "tenant_ids": ["tenant-a"],
                    "user_id": "svc-a",
                    "groups": [],
                    "roles": [role],
                    "admin": False,
                }
            }
        )
    }


def test_rbac_matrix_is_monotonic_and_owner_only_delete() -> None:
    assert READER_CAPABILITIES < OPERATOR_CAPABILITIES < OWNER_CAPABILITIES
    assert CAP_KNOWLEDGE_QUERY in READER_CAPABILITIES
    assert CAP_CATALOG_READ in READER_CAPABILITIES
    assert CAP_SOURCE_CREATE not in READER_CAPABILITIES
    assert CAP_INGEST_RUN in OPERATOR_CAPABILITIES
    assert CAP_SOURCE_DELETE not in OPERATOR_CAPABILITIES
    assert CAP_SOURCE_DELETE in OWNER_CAPABILITIES


def test_custom_acl_role_does_not_grant_platform_capabilities() -> None:
    principal = ApiPrincipal(
        tenant_ids=frozenset({"tenant-a"}),
        user_id="svc-a",
        roles=("finance-approver",),
    )
    assert capabilities_for_principal(principal) == frozenset()


def test_reader_operator_owner_capability_enforcement() -> None:
    with patch.dict(os.environ, _principal_env("reader"), clear=True):
        require_capability("key", CAP_KNOWLEDGE_QUERY)
        with pytest.raises(HTTPException) as exc:
            require_capability("key", CAP_SOURCE_CREATE)
        assert exc.value.status_code == 403

    with patch.dict(os.environ, _principal_env("operator"), clear=True):
        require_capability("key", CAP_SOURCE_CREATE)
        require_capability("key", CAP_INGEST_RUN)
        with pytest.raises(HTTPException) as exc:
            require_capability("key", CAP_SOURCE_DELETE)
        assert exc.value.status_code == 403

    with patch.dict(os.environ, _principal_env("owner"), clear=True):
        require_capability("key", CAP_SOURCE_DELETE)


def test_production_principal_requires_platform_role() -> None:
    raw = json.dumps(
        {
            "key": {
                "tenant_ids": ["tenant-a"],
                "user_id": "svc-a",
                "groups": [],
                "roles": ["custom-acl-role"],
                "admin": False,
            }
        }
    )
    with patch.dict(os.environ, {"RAGBOT_API_KEY_PRINCIPALS": raw}, clear=True):
        with pytest.raises(ValueError, match="platform RBAC role"):
            validate_principal_coverage(["key"])


def test_agent_events_are_exposed_as_prometheus_counters_and_histograms() -> None:
    collector = MetricsCollector(max_history=10)
    collector.record(
        RequestMetrics(
            request_id="p2-metrics-request",
            tenant_id="tenant-a",
            user_id="svc-a",
            route="doc_rag",
            total_duration_ms=250,
            retrieval_duration_ms=80,
            citation_count=2,
            evidence_count=3,
            confidence="high",
            has_citations=True,
            iterations=1,
            tool_calls=[{"name": "retrieve", "ok": True, "duration_ms": 75}],
        )
    )
    assert collector.record_feedback("p2-metrics-request", "positive") is True

    payload, _ = render_prometheus(InMemoryRepo())
    text = payload.decode("utf-8")
    assert "ragbot_agent_requests_total" in text
    assert 'route="doc_rag"' in text
    assert "ragbot_agent_request_duration_seconds" in text
    assert "ragbot_agent_tool_calls_total" in text
    assert 'tool="retrieve"' in text
    assert "ragbot_agent_feedback_total" in text


def test_admin_cache_surface_is_removed() -> None:
    from services.api.app.api import app

    paths = {route.path for route in app.routes}
    assert "/admin/cache" not in paths
