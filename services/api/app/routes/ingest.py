"""Ingestion jobs API routes.

Provides endpoints for triggering ingestion jobs, listing jobs, and
checking job status.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel


class TriggerJobRequest(BaseModel):
    source_id: str
    tenant_id: str


class TriggerJobResponse(BaseModel):
    status: str
    job_id: str
    source_id: str


def create_ingest_router(get_services: Callable, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/ingest", tags=["ingest"])

    @router.post("/jobs", status_code=202, response_model=TriggerJobResponse)
    async def trigger_job(payload: TriggerJobRequest, _key=Depends(auth_dep)):
        services = get_services()
        source = services.repo.get_source(payload.source_id)
        if not source or source.status == "deleted":
            raise HTTPException(404, f"Source not found: {payload.source_id}")
        if source.tenant_id != payload.tenant_id:
            raise HTTPException(403, "Tenant mismatch")

        job_id = uuid.uuid4().hex

        # Run pipeline in background thread to avoid blocking
        from services.worker.pipeline import run_ingest_pipeline
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            run_ingest_pipeline,
            source,
            services.repo,
            services.qdrant,
            job_id,
        )

        return TriggerJobResponse(status="accepted", job_id=job_id, source_id=source.source_id)

    @router.get("/jobs")
    async def list_jobs(
        tenant_id: Optional[str] = None,
        source_id: Optional[str] = None,
        _key=Depends(auth_dep),
    ):
        services = get_services()
        jobs = services.repo.list_jobs(tenant_id=tenant_id, source_id=source_id)
        return {"jobs": [asdict(j) for j in jobs]}

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str, _key=Depends(auth_dep)):
        services = get_services()
        job = services.repo.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        return asdict(job)

    @router.post("/jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str, _key=Depends(auth_dep)):
        services = get_services()
        old_job = services.repo.get_job(job_id)
        if not old_job:
            raise HTTPException(404, f"Job not found: {job_id}")
        if old_job.status != "failed":
            raise HTTPException(400, "Only failed jobs can be retried")

        source = services.repo.get_source(old_job.source_id)
        if not source:
            raise HTTPException(404, f"Source not found: {old_job.source_id}")

        new_job_id = uuid.uuid4().hex
        from services.worker.pipeline import run_ingest_pipeline
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            run_ingest_pipeline,
            source,
            services.repo,
            services.qdrant,
            new_job_id,
        )

        return {"status": "accepted", "job_id": new_job_id, "retried_from": job_id}

    return router
