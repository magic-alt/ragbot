from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware

from .observability.prometheus import observe_http

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request_id/method/path/status/latency and records Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            latency_seconds = max(0.0, time.perf_counter() - start)
            observe_http(request.method, request.url.path, status, latency_seconds)

        latency_ms = int(latency_seconds * 1000)
        client_ip = request.client.host if request.client else "-"
        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%d client=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            client_ip,
        )

        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app) -> None:
    """Configure CORS and request logging middleware on the FastAPI app."""
    cors_origins = os.getenv("RAGBOT_CORS_ORIGINS", "")
    if cors_origins.strip():
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestLoggingMiddleware)
