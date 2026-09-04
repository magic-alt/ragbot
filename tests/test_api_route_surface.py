from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.app.api import app


def test_final_app_openapi_exposes_server_managed_pdf_upload() -> None:
    schema = app.openapi()
    paths = schema.get("paths", {})
    assert "/ingest/upload/pdf" in paths
    assert "post" in paths["/ingest/upload/pdf"]


def test_final_app_dispatches_server_managed_pdf_upload_route() -> None:
    # text/plain is deliberately invalid for the upload endpoint. Reaching the
    # endpoint must therefore return 415. A 404 means the final application did
    # not actually mount the managed-upload router.
    response = TestClient(app).post(
        "/ingest/upload/pdf?tenant_id=engineering&filename=probe.pdf",
        content=b"not-a-pdf",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 415, response.text


def test_runtime_identity_is_stable_for_one_api_process() -> None:
    with TestClient(app) as client:
        first = client.get("/admin/runtime")
        second = client.get("/admin/runtime")

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["service"] == "ragbot-api"
    assert first_payload["api_version"] == "0.5.0"
    assert first_payload["boot_id"]
    assert first_payload["boot_id"] == second_payload["boot_id"]
    assert first_payload["pid"] == second_payload["pid"]
    assert "server-managed-pdf-upload" in first_payload["capabilities"]
