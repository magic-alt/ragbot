"""Server-managed upload surface for client-local documents."""
from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.api.app.storage.models import UploadedObject
from services.worker.uploads import build_upload_store_from_env, upload_uri
from services.worker.uploads.lifecycle import gc_uploaded_objects

from ..auth.principal import authorize_tenant, require_operator
from .quick_import import QuickSourceSpec, _run_quick_import


def create_upload_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/ingest", tags=["ingest"])

    @router.post("/upload/pdf", status_code=202)
    async def upload_pdf(
        request: Request,
        tenant_id: str = Query(min_length=1),
        filename: str = Query(min_length=1),
        name: Optional[str] = Query(default=None, min_length=1),
        tag: list[str] = Query(default=[]),
        chunk_size: Optional[int] = Query(default=None, ge=1),
        chunk_overlap: Optional[int] = Query(default=None, ge=0),
        _key: Optional[str] = Depends(auth_dep),
    ):
        authorize_tenant(_key, tenant_id)
        require_operator(_key)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="PDF upload requires application/pdf")

        services = get_services()
        # Opportunistic GC prevents single-node deployments from requiring a
        # dedicated maintenance process. The explicit endpoint below remains
        # available for deterministic operational cleanup.
        try:
            gc_uploaded_objects(services.repo)
        except Exception:
            pass
        try:
            store = build_upload_store_from_env()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        object_id = uuid.uuid4().hex
        temporary = store.temporary_path(object_id)
        digest, size = await _stream_pdf(
            request,
            temporary,
            max_bytes=_positive_int_env("RAGBOT_PDF_MAX_BYTES", 25 * 1024 * 1024),
        )
        stored = store.commit_pdf(
            temporary,
            object_id=object_id,
            sha256=digest,
            size_bytes=size,
        )
        now = datetime.now(timezone.utc).isoformat()
        uploaded = UploadedObject(
            object_id=object_id,
            tenant_id=tenant_id,
            sha256=digest,
            storage_backend=stored.storage_backend,
            storage_key=stored.storage_key,
            size_bytes=size,
            media_type="application/pdf",
            original_filename=Path(filename).name,
            state="staged",
            ref_count=0,
            created_at=now,
        )
        services.repo.add_uploaded_object(uploaded)

        location = upload_uri(object_id)
        config: dict[str, Any] = {
            "upload_object_id": object_id,
            "upload_sha256": digest,
            "upload_size_bytes": size,
            "original_filename": Path(filename).name,
        }
        if chunk_size is not None:
            config["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            config["chunk_overlap"] = chunk_overlap

        try:
            result = _run_quick_import(
                tenant_id=tenant_id,
                spec=QuickSourceSpec(
                    location=location,
                    source_type="pdf",
                    name=name or Path(filename).name,
                    tags=list(tag),
                    config=config,
                    reuse_source=False,
                ),
                services=services,
            )
        except Exception:
            services.repo.update_uploaded_object(
                object_id,
                state="orphaned",
                ref_count=0,
                retired_at=datetime.now(timezone.utc).isoformat(),
            )
            raise

        services.repo.update_uploaded_object(
            object_id,
            state="active",
            ref_count=1,
            last_referenced_at=datetime.now(timezone.utc).isoformat(),
            retired_at=None,
        )
        return {
            **result,
            "location": location,
            "object_id": object_id,
            "filename": Path(filename).name,
            "sha256": digest,
            "size_bytes": size,
        }

    @router.get("/uploads")
    async def list_uploads(
        tenant_id: str = Query(min_length=1),
        _key: Optional[str] = Depends(auth_dep),
    ):
        authorize_tenant(_key, tenant_id)
        require_operator(_key)
        objects = get_services().repo.list_uploaded_objects(tenant_id=tenant_id)
        return {"objects": [asdict(obj) for obj in objects]}

    @router.post("/uploads/gc")
    async def collect_uploads(
        retention_seconds: Optional[int] = Query(default=None, ge=0),
        _key: Optional[str] = Depends(auth_dep),
    ):
        require_operator(_key)
        return gc_uploaded_objects(
            get_services().repo,
            retention_seconds=retention_seconds,
        )

    return router


async def _stream_pdf(request: Request, temporary: Path, *, max_bytes: int) -> tuple[str, int]:
    temporary.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    written = 0
    signature = bytearray()
    try:
        with temporary.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF upload exceeds {max_bytes} byte limit",
                    )
                if len(signature) < 5:
                    signature.extend(chunk[: 5 - len(signature)])
                hasher.update(chunk)
                handle.write(chunk)
        if written == 0:
            raise HTTPException(status_code=422, detail="PDF upload body is empty")
        if not bytes(signature).startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="Uploaded body is not a PDF file")
        return hasher.hexdigest(), written
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"{name} must be an integer") from exc
    if value <= 0:
        raise HTTPException(status_code=503, detail=f"{name} must be > 0")
    return value
