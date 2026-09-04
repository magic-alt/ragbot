from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.api.app.routes import uploads as upload_routes
from services.api.app.routes.quick_import import QuickSourceSpec, _run_quick_import
from services.api.app.storage.repo import InMemoryRepo
from services.api.app.storage.upload_support import ensure_upload_repository
from services.worker.connectors.security import validate_local_source_path
from services.worker.uploads import FilesystemUploadStore, upload_object_id, upload_uri
from services.worker.uploads.lifecycle import gc_uploaded_objects, retire_uploaded_object_for_source


def test_upload_uri_is_canonical_and_rejects_path_traversal() -> None:
    object_id = "0123456789abcdef0123456789abcdef"
    assert upload_uri(object_id) == f"ragbot-upload:///{object_id}"
    assert upload_object_id(upload_uri(object_id)) == object_id
    with pytest.raises(ValueError):
        upload_object_id("ragbot-upload:///../../etc/passwd")


def test_filesystem_store_separates_logical_objects_from_content_blob(tmp_path: Path) -> None:
    store = FilesystemUploadStore(tmp_path / "uploads")
    payload = b"%PDF-same-content"
    digest = hashlib.sha256(payload).hexdigest()
    object_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    object_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    for object_id in (object_a, object_b):
        temporary = store.temporary_path(object_id)
        temporary.write_bytes(payload)
        store.commit_pdf(temporary, object_id=object_id, sha256=digest, size_bytes=len(payload))

    path_a = store.local_path(upload_uri(object_a))
    path_b = store.local_path(upload_uri(object_b))
    assert path_a != path_b
    assert path_a.read_bytes() == path_b.read_bytes() == payload
    assert len(list(store.blob_root.glob("*.pdf"))) == 1


def test_managed_upload_bypasses_generic_local_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "uploads"
    store = FilesystemUploadStore(root)
    object_id = "0123456789abcdef0123456789abcdef"
    payload = b"%PDF-test"
    digest = hashlib.sha256(payload).hexdigest()
    temporary = store.temporary_path(object_id)
    temporary.write_bytes(payload)
    store.commit_pdf(temporary, object_id=object_id, sha256=digest, size_bytes=len(payload))

    monkeypatch.setenv("RAGBOT_UPLOAD_DIR", str(root))
    monkeypatch.setenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", str(tmp_path / "different-root"))
    resolved = validate_local_source_path(upload_uri(object_id))
    assert Path(resolved).read_bytes() == payload


def test_retire_and_gc_remove_unreferenced_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = ensure_upload_repository(InMemoryRepo())
    root = tmp_path / "uploads"
    monkeypatch.setenv("RAGBOT_UPLOAD_DIR", str(root))
    store = FilesystemUploadStore(root)
    object_id = "0123456789abcdef0123456789abcdef"
    payload = b"%PDF-test"
    digest = hashlib.sha256(payload).hexdigest()
    temporary = store.temporary_path(object_id)
    temporary.write_bytes(payload)
    stored = store.commit_pdf(temporary, object_id=object_id, sha256=digest, size_bytes=len(payload))

    from services.api.app.storage.models import Source, UploadedObject

    repo.add_uploaded_object(
        UploadedObject(
            object_id=object_id,
            tenant_id="tenant-a",
            sha256=digest,
            storage_backend=stored.storage_backend,
            storage_key=stored.storage_key,
            size_bytes=len(payload),
            media_type="application/pdf",
            original_filename="guide.pdf",
            state="active",
            ref_count=1,
        )
    )
    source = Source(
        source_id="source-a",
        tenant_id="tenant-a",
        source_type="pdf",
        name="guide.pdf",
        config={"path": upload_uri(object_id), "upload_object_id": object_id},
    )

    assert retire_uploaded_object_for_source(repo, source) is True
    assert repo.get_uploaded_object(object_id).state == "retired"
    stats = gc_uploaded_objects(repo, tenant_id="tenant-a", retention_seconds=0)
    assert stats["deleted"] == 1
    assert repo.get_uploaded_object(object_id).state == "deleted"
    assert not (store.object_root / f"{object_id}.pdf").exists()


def test_generic_quick_import_cannot_forge_managed_upload_uri() -> None:
    repo = ensure_upload_repository(InMemoryRepo())
    services = SimpleNamespace(repo=repo)
    object_id = "0123456789abcdef0123456789abcdef"

    with pytest.raises(HTTPException, match="server-managed") as exc_info:
        _run_quick_import(
            tenant_id="tenant-a",
            spec=QuickSourceSpec(
                location=upload_uri(object_id),
                source_type="pdf",
            ),
            services=services,
        )

    assert exc_info.value.status_code == 422


def test_upload_endpoint_persists_object_uri_not_client_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = ensure_upload_repository(InMemoryRepo())
    services = SimpleNamespace(repo=repo)
    monkeypatch.setenv("RAGBOT_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(
        upload_routes,
        "_run_quick_import",
        lambda **kwargs: {
            "status": "accepted",
            "source_id": "source-1",
            "source_type": "pdf",
            "job_id": "job-1",
        },
    )
    app = FastAPI()
    app.include_router(upload_routes.create_upload_router(lambda: services, lambda: None))

    with patch.dict("os.environ", {}, clear=False):
        response = TestClient(app).post(
            "/ingest/upload/pdf?tenant_id=engineering&filename=%2FUsers%2Fkaermax%2Fsecret%2Fguide.pdf",
            content=b"%PDF-test-body",
            headers={"Content-Type": "application/pdf"},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["location"].startswith("ragbot-upload:///")
    assert "/Users/kaermax" not in payload["location"]
    objects = repo.list_uploaded_objects(tenant_id="engineering")
    assert len(objects) == 1
    assert objects[0].original_filename == "guide.pdf"
    assert objects[0].state == "active"
    assert objects[0].ref_count == 1


def test_upload_endpoint_rejects_non_pdf_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = ensure_upload_repository(InMemoryRepo())
    services = SimpleNamespace(repo=repo)
    monkeypatch.setenv("RAGBOT_UPLOAD_DIR", str(tmp_path / "uploads"))
    app = FastAPI()
    app.include_router(upload_routes.create_upload_router(lambda: services, lambda: None))

    response = TestClient(app).post(
        "/ingest/upload/pdf?tenant_id=engineering&filename=guide.pdf",
        content=b"not-a-pdf",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 415
    assert repo.list_uploaded_objects(tenant_id="engineering") == []
