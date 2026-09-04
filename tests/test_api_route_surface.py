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
