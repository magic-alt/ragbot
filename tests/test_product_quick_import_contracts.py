from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from services.api.app.routes.quick_import import QuickSourceSpec, _run_quick_import
from services.api.app.storage.repo import InMemoryRepo


def test_idempotency_key_requires_stable_source_identity(monkeypatch):
    monkeypatch.setenv("RAGBOT_INGESTION_MODE", "worker")
    services = SimpleNamespace(repo=InMemoryRepo(), qdrant=object(), embedder=object())
    spec = QuickSourceSpec(
        location="/data/manuals",
        reuse_source=False,
        idempotency_key="nightly-build",
    )

    with pytest.raises(HTTPException) as exc_info:
        _run_quick_import(tenant_id="engineering", spec=spec, services=services)

    assert exc_info.value.status_code == 422
    assert "reuse_source=true" in str(exc_info.value.detail)
    assert services.repo.list_sources("engineering") == []
    assert services.repo.list_jobs(tenant_id="engineering") == []
