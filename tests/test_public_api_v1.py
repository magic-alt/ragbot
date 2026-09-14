from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.app import api
from services.api.app.agent.graph import build_default_services


def _client(monkeypatch):
    monkeypatch.delenv("RAGBOT_API_KEYS", raising=False)
    monkeypatch.delenv("RAGBOT_API_KEY_PRINCIPALS", raising=False)
    api._VALID_API_KEYS = None
    api._services = build_default_services()
    return TestClient(api.app)


def test_v1_surface_reuses_core_routers(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/v1/sources", params={"tenant_id": "tenant-a", "limit": 10})
    assert response.status_code == 200
    assert response.json() == {"total": 0, "next_cursor": None, "sources": []}
    assert response.headers["X-Ragbot-API-Version"] == "v1"
    assert response.headers.get("X-Request-ID")

    legacy = client.get("/sources", params={"tenant_id": "tenant-a", "limit": 10})
    assert legacy.status_code == 200
    assert "X-Ragbot-API-Version" not in legacy.headers


def test_v1_errors_use_typed_envelope_without_breaking_legacy(monkeypatch) -> None:
    client = _client(monkeypatch)
    versioned = client.get("/v1/sources/missing-source")
    assert versioned.status_code == 404
    payload = versioned.json()
    assert payload["error"]["code"] == "not_found"
    assert payload["error"]["message"] == "Source not found"
    assert payload["error"]["request_id"] == versioned.headers["X-Request-ID"]
    assert payload["error"]["retryable"] is False

    legacy = client.get("/sources/missing-source")
    assert legacy.status_code == 404
    assert legacy.json() == {"detail": "Source not found"}


def test_v1_validation_error_is_machine_readable(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.post("/v1/search", json={"query": "x"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert isinstance(error["details"]["errors"], list)


def test_openapi_contains_frozen_v1_core_surface() -> None:
    schema = api.app.openapi()
    for path, method in (
        ("/v1/search", "post"),
        ("/v1/chat", "post"),
        ("/v1/sources", "get"),
        ("/v1/sources/{source_id}", "get"),
        ("/v1/catalog/jobs", "get"),
        ("/v1/ingest/jobs", "post"),
    ):
        assert path in schema["paths"], path
        assert method in schema["paths"][path], f"{method.upper()} {path}"
