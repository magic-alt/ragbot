"""Non-secret process identity used to verify runtime routing.

The bootstrap controller compares this endpoint through the Docker-internal
loopback and the published host address. Matching boot IDs prove that the host
port reaches the exact API process that Compose just started instead of a stale
local uvicorn process or another Ragbot stack.
"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter

_BOOT_ID = uuid.uuid4().hex


def create_runtime_identity_router() -> APIRouter:
    router = APIRouter(tags=["runtime"])

    @router.get("/admin/runtime")
    async def runtime_identity() -> dict:
        return {
            "service": "ragbot-api",
            "api_version": "0.5.0",
            "boot_id": _BOOT_ID,
            "pid": os.getpid(),
            "capabilities": ["server-managed-pdf-upload"],
        }

    return router
