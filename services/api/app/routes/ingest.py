"""Ingestion jobs API routes and reusable queue helpers.

With PostgreSQL, jobs are persisted first and consumed by the independent
``services.worker.main`` process. In-memory development keeps the historical
executor path so a one-process local setup remains convenient.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.principal import allowed_tenants, authorize_tenant
from ..storage.models import IngestionJob


class TriggerJobRequest(BaseModel):
    source_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)


class TriggerJobResponse(BaseModel):
    status: str
    job_id: str
    source_id: str


def assert_source_ingestible(source) -> None:
    """Reject ingestion for sources that are not currently active."""
    if source.status != "active":
        raise HTTPException(
            status_code=409,
            detail=f"Source is not active: {source.source_id} ({source.status})",
        )


def latest_active_ingestion_job(repo, *, tenant_id: str, source_id: str) -> Optional[IngestionJob]:
    """Return the newest pending/running job for a source, if one exists."""
    active = [
        job
        for job in repo.list_jobs(tenant_id=tenant_id, source_id=source_id)
        if job.status in {"pending", "running"}
    ]
    if not active:
        return None
    return max(active, key=lambda job: (job.created_at or "", job.job_id))


def enqueue_ingestion_job(source, services, job_id: Optional[str] = None) -> IngestionJob:
    """Persist an ingestion job and schedule inline execution when configured.

    Connector type/config are deeply snapshotted into the Job so queued/retried
    work is not silently redirected by a later mutable Source config edit. This
    helper is shared by the low-level Job API and higher-level Quick Import.
    """
    assert_source_ingestible(source)
    job_id = job_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    source_config = deepcopy(source.config or {})
    job = IngestionJob(
        job_id=job_id,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        source_type=source.source_type,
        source_config=source_config,
        status="pending",
        created_at=now,
        available_at=now,
    )
    services.repo.add_job(job)

    if not _use_durable_worker():
        from services.worker.pipeline import run_ingest_pipeline

        execution_source = replace(
            source,
            source_type=job.source_type,
            config=deepcopy(job.source_config),
        )
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            run_ingest_pipeline,
            execution_source,
            services.repo,
            services.qdrant,
            job_id,
            services.embedder,
            True,
        )
    return job


def create_ingest_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/ingest", tags=["ingest"])

    @router.post("/jobs", status_code=202, response_model=TriggerJobResponse)
    async def trigger_job(
        payload: TriggerJobRequest,
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        source = services.repo.get_source(payload.source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, f"Source not found: {payload.source_id}")
        if source.tenant_id != payload.tenant_id:
            raise HTTPException(403, "Tenant mismatch")
        authorize_tenant(_key, source.tenant_id)

        job = enqueue_ingestion_job(source, services)
        return TriggerJobResponse(status="accepted", job_id=job.job_id, source_id=source.source_id)

    @router.get("/jobs")
    async def list_jobs(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        _key: Optional[str] = Depends(auth_dep),
    ):
        services = get_services()
        if tenant_id:
            authorize_tenant(_key, tenant_id)
        jobs = services.repo.list_jobs(tenant_id=tenant_id, source_id=source_id)
        tenant_scope = allowed_tenants(_key)
        if tenant_scope is not None:
            jobs = [job for job in jobs if job.tenant_id in tenant_scope]
        return {"jobs": [asdict(job) for job in jobs]}

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str, _key: Optional[str] = Depends(auth_dep)):
        services = get_services()
        job = services.repo.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        authorize_tenant(_key, job.tenant_id)
        return asdict(job)

    @router.post("/jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str, _key: Optional[str] = Depends(auth_dep)):
        services = get_services()
        old_job = services.repo.get_job(job_id)
        if not old_job:
            raise HTTPException(404, f"Job not found: {job_id}")
        authorize_tenant(_key, old_job.tenant_id)
        if old_job.status != "failed":
            raise HTTPException(400, "Only failed jobs can be retried")

        source = services.repo.get_source(old_job.source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, f"Source not found: {old_job.source_id}")
        authorize_tenant(_key, source.tenant_id)

        job = enqueue_ingestion_job(source, services)
        return {"status": "accepted", "job_id": job.job_id, "retried_from": job_id}

    return router


def _use_durable_worker() -> bool:
    mode = os.getenv("RAGBOT_INGESTION_MODE", "auto").strip().lower()
    if mode not in {"auto", "inline", "worker"}:
        raise RuntimeError("RAGBOT_INGESTION_MODE must be one of: auto, inline, worker")
    if mode == "worker":
        return True
    if mode == "inline":
        return False
    return bool(os.getenv("POSTGRES_DSN"))
