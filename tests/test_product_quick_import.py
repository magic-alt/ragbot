from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cli.rag import _manifest_source_spec, _normalize_manifest
from services.api.app.routes.quick_import import (
    QuickSourceSpec,
    _run_quick_import,
    canonical_location,
    create_quick_import_router,
    deterministic_source_id,
    infer_source_type,
)
from services.api.app.storage.repo import InMemoryRepo


@pytest.fixture(autouse=True)
def _durable_worker_mode(monkeypatch):
    # Quick-import API tests validate queue semantics only; connector execution
    # belongs to the existing ingestion integration suite.
    monkeypatch.setenv("RAGBOT_INGESTION_MODE", "worker")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


def _services():
    return SimpleNamespace(repo=InMemoryRepo(), qdrant=object(), embedder=object())


def test_source_type_inference_for_product_locations():
    assert infer_source_type("/data/manual.pdf") == "pdf"
    assert infer_source_type("/data/docs") == "local_fs"
    assert infer_source_type("https://example.com/manual.pdf") == "pdf"
    assert infer_source_type("https://example.com/docs/start") == "web"
    assert infer_source_type("https://github.com/magic-alt/ragbot") == "repo"
    assert infer_source_type("https://example.com/repo.git") == "repo"


def test_canonical_location_produces_stable_source_identity():
    assert canonical_location("HTTPS://EXAMPLE.COM/docs/") == "https://example.com/docs"
    left = deterministic_source_id("tenant-a", "web", "https://example.com/docs/")
    right = deterministic_source_id("tenant-a", "web", "HTTPS://EXAMPLE.COM/docs")
    assert left == right


def test_quick_import_reuses_source_and_active_job():
    services = _services()
    spec = QuickSourceSpec(location="/data/engineering", name="Engineering", tags=["internal"])

    first = _run_quick_import(tenant_id="t1", spec=spec, services=services)
    second = _run_quick_import(tenant_id="t1", spec=spec, services=services)

    assert first["status"] == "accepted"
    assert first["source_reused"] is False
    assert first["job_reused"] is False
    assert second["status"] == "already_queued"
    assert second["source_reused"] is True
    assert second["job_reused"] is True
    assert second["source_id"] == first["source_id"]
    assert second["job_id"] == first["job_id"]
    assert len(services.repo.list_sources("t1")) == 1
    assert len(services.repo.list_jobs(tenant_id="t1")) == 1


def test_quick_import_idempotency_key_replays_exact_job():
    services = _services()
    spec = QuickSourceSpec(
        location="https://example.com/manual.pdf",
        idempotency_key="release-2026-09-02",
        dedupe_active_job=False,
    )

    first = _run_quick_import(tenant_id="t1", spec=spec, services=services)
    second = _run_quick_import(tenant_id="t1", spec=spec, services=services)

    assert first["status"] == "accepted"
    assert second["status"] == "idempotent_replay"
    assert second["job_id"] == first["job_id"]
    assert len(services.repo.list_jobs(tenant_id="t1")) == 1


def test_quick_import_syncs_metadata_for_reused_source():
    services = _services()
    first = QuickSourceSpec(
        location="https://example.com/team/repo.git",
        source_type="repo",
        name="Team repo",
        tags=["v1"],
        config={"ref": "main"},
    )
    second = QuickSourceSpec(
        location="https://example.com/team/repo.git/",
        source_type="repo",
        name="Team repo v2",
        tags=["v2"],
        config={"ref": "release"},
    )

    one = _run_quick_import(tenant_id="t1", spec=first, services=services)
    two = _run_quick_import(tenant_id="t1", spec=second, services=services)
    source = services.repo.get_source(one["source_id"])

    assert two["source_reused"] is True
    assert source is not None
    assert source.name == "Team repo v2"
    assert source.tags == ["v2"]
    assert source.config["ref"] == "release"


def test_quick_import_can_force_distinct_sources():
    services = _services()
    spec = QuickSourceSpec(location="/data/docs", reuse_source=False)

    first = _run_quick_import(tenant_id="t1", spec=spec, services=services)
    second = _run_quick_import(tenant_id="t1", spec=spec, services=services)

    assert first["source_id"] != second["source_id"]
    assert len(services.repo.list_sources("t1")) == 2


def test_batch_endpoint_submits_multiple_source_types():
    services = _services()
    app = FastAPI()

    async def auth_dep():
        return None

    app.include_router(create_quick_import_router(lambda: services, auth_dep))
    client = TestClient(app)
    response = client.post(
        "/ingest/batch",
        json={
            "tenant_id": "t1",
            "sources": [
                {"location": "/data/docs", "name": "Docs"},
                {"location": "https://example.com/guide.pdf", "name": "Guide"},
                {"location": "https://example.com/wiki", "name": "Wiki"},
            ],
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["total"] == 3
    assert body["accepted"] == 3
    assert body["failed"] == 0
    assert [item["source_type"] for item in body["items"]] == ["local_fs", "pdf", "web"]
    assert len(services.repo.list_jobs(tenant_id="t1")) == 3


def test_manifest_accepts_object_or_array_and_normalizes_aliases():
    tenant, sources = _normalize_manifest(
        {"tenant_id": "engineering", "sources": [{"path": "/data/docs"}]},
        "default",
    )
    assert tenant == "engineering"
    assert _manifest_source_spec(sources[0]) == {
        "location": "/data/docs",
        "source_type": "local_fs",
    }

    tenant, sources = _normalize_manifest([{"url": "https://example.com/guide.pdf"}], "default")
    assert tenant == "default"
    assert _manifest_source_spec(sources[0])["source_type"] == "pdf"


def test_manifest_rejects_empty_source_list():
    with pytest.raises(ValueError, match="non-empty 'sources'"):
        _normalize_manifest({"sources": []}, "default")
