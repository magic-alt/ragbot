from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .agent.callbacks import AsyncQueueCallback
from .agent.context import process_client_context
from .agent.graph import run_agent
from .agent.state import Constraints, SourceType
from .auth.acl import compute_security_scope
from .auth.principal import (
    CAP_FEEDBACK_WRITE,
    CAP_KNOWLEDGE_QUERY,
    authorize_identity,
    require_admin,
    require_capability,
)
from .factory import build_services_from_env
from .main import chat
from .middleware import setup_middleware
from .observability.metrics import get_metrics_collector
from .observability.otel_metrics import setup_otel_metrics
from .observability.prometheus import render_prometheus
from .observability.tracing import setup_tracing
from .routes.admin_ui import create_admin_ui_router
from .routes.control_plane import create_control_plane_router
from .routes.ingest import create_ingest_router
from .routes.openai_compat import create_openai_compat_endpoint
from .routes.quick_import import create_quick_import_router
from .routes.search import create_search_endpoint
from .routes.sources import create_sources_router
from .routes.uploads import create_upload_router
from .runtime import validate_production_environment

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_API_KEYS: Optional[set] = None


def _load_api_keys() -> Optional[set]:
    raw = os.getenv("RAGBOT_API_KEYS", "")
    if not raw.strip():
        return None
    return {key.strip() for key in raw.split(",") if key.strip()}


async def verify_api_key(api_key: Optional[str] = Depends(_API_KEY_HEADER)) -> Optional[str]:
    if _VALID_API_KEYS is None:
        return api_key
    if not api_key or api_key not in _VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


_services = None


def _get_services():
    global _services
    if _services is None:
        _services = build_services_from_env()
    return _services


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _VALID_API_KEYS
    validate_production_environment()
    _VALID_API_KEYS = _load_api_keys()
    setup_tracing()
    setup_otel_metrics()
    yield

    if _services is not None:
        closed: set[int] = set()
        for resource in (_services.sql_engine, _services.repo, _services.qdrant):
            if id(resource) in closed:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
            closed.add(id(resource))


app = FastAPI(title="ragbot API", version="0.5.0", lifespan=lifespan)
setup_middleware(app)

app.include_router(create_search_endpoint(_get_services, verify_api_key))
app.include_router(create_openai_compat_endpoint(_get_services, verify_api_key))
app.include_router(create_sources_router(_get_services, verify_api_key))
app.include_router(create_ingest_router(_get_services, verify_api_key))
app.include_router(create_quick_import_router(_get_services, verify_api_key))
app.include_router(create_upload_router(_get_services, verify_api_key))
app.include_router(create_control_plane_router(_get_services, verify_api_key))
app.include_router(create_admin_ui_router())


class ConstraintsModel(BaseModel):
    source_types: Optional[List[SourceType]] = None
    doc_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    path_prefix: Optional[str] = None
    url_prefix: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    model_config = {"extra": "forbid"}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: Optional[str] = None
    stream: bool = False
    constraints: Optional[ConstraintsModel] = None
    client_context: Optional[dict] = None


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest, _key: Optional[str] = Depends(verify_api_key)):
    require_capability(_key, CAP_KNOWLEDGE_QUERY)
    services = _get_services()
    constraints = _constraints_from_model(payload.constraints)
    constraints, initial_evidence = process_client_context(payload.client_context, constraints)
    trusted_user_id, constraints = _apply_trusted_identity(
        services, _key, payload.tenant_id, payload.user_id, constraints
    )

    if payload.stream:
        return StreamingResponse(
            _chat_stream_realtime(
                payload,
                trusted_user_id=trusted_user_id,
                constraints=constraints,
                initial_evidence=initial_evidence,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await chat(
        payload.query,
        payload.tenant_id,
        trusted_user_id,
        services,
        constraints,
        payload.session_id,
        initial_evidence=initial_evidence,
    )


@app.get("/admin/health")
async def health_endpoint() -> dict:
    return {"status": "ok"}


@app.get("/admin/ready")
async def readiness_endpoint() -> dict:
    try:
        services = _get_services()
        checks = {}
        for name, resource in (("repository", services.repo), ("vector_store", services.qdrant)):
            checker = getattr(resource, "healthcheck", None)
            checks[name] = bool(checker()) if callable(checker) else True
        if not all(checks.values()):
            raise HTTPException(status_code=503, detail="Dependencies are not ready")
        return {"status": "ready", "checks": checks}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="Dependencies are not ready")


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics_endpoint(_key: Optional[str] = Depends(verify_api_key)) -> Response:
    require_admin(_key)
    payload, content_type = render_prometheus(_get_services().repo)
    return Response(content=payload, media_type=content_type)


@app.get("/admin/metrics")
async def metrics_endpoint(_key: Optional[str] = Depends(verify_api_key)) -> dict:
    """Return process-local diagnostic aggregation, not production monitoring state."""
    require_admin(_key)
    aggregate = get_metrics_collector().aggregate()
    return {
        "scope": "process-local-diagnostics",
        "production_metrics": "/metrics",
        "total_requests": aggregate.total_requests,
        "citation_coverage": round(aggregate.citation_coverage, 4),
        "retrieval_hit_rate": round(aggregate.retrieval_hit_rate, 4),
        "tool_failure_rate": round(aggregate.tool_failure_rate, 4),
        "avg_duration_ms": round(aggregate.avg_duration_ms, 1),
        "p95_duration_ms": round(aggregate.p95_duration_ms, 1),
        "avg_iterations": round(aggregate.avg_iterations, 2),
        "confidence_distribution": aggregate.confidence_distribution,
        "tool_stats": aggregate.tool_stats,
        "feedback_score": round(aggregate.feedback_score, 4),
        "positive_feedback": aggregate.positive_feedback,
        "negative_feedback": aggregate.negative_feedback,
    }


@app.get("/admin/metrics/history")
async def metrics_history_endpoint(last_n: int = 100, _key: Optional[str] = Depends(verify_api_key)) -> dict:
    require_admin(_key)
    return {
        "scope": "process-local-diagnostics",
        "requests": get_metrics_collector().get_history(last_n),
    }


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    feedback: str = Field(pattern="^(positive|negative)$")


@app.post("/admin/feedback")
async def feedback_endpoint(payload: FeedbackRequest, _key: Optional[str] = Depends(verify_api_key)):
    require_capability(_key, CAP_FEEDBACK_WRITE)
    collector = get_metrics_collector()
    found = collector.record_feedback(payload.request_id, payload.feedback)
    if not found:
        raise HTTPException(status_code=404, detail="Request not found in this API replica's diagnostic history")
    return {"status": "ok"}


@app.get("/admin/cost")
async def cost_endpoint(_key: Optional[str] = Depends(verify_api_key)) -> dict:
    require_admin(_key)
    from .llm.router import CostTracker
    return CostTracker().summary()


def _constraints_from_model(model: Optional[ConstraintsModel]) -> Optional[Constraints]:
    if not model:
        return None
    return Constraints(**model.model_dump())


def _apply_trusted_identity(
    services,
    api_key: Optional[str],
    tenant_id: str,
    requested_user_id: str,
    constraints: Optional[Constraints],
) -> tuple[str, Constraints]:
    trusted_user_id, groups, roles = authorize_identity(api_key, tenant_id, requested_user_id)
    resolved = constraints or Constraints()
    policies = services.repo.list_policies(tenant_id)
    resolved.security_scope = {
        "tenant_id": tenant_id,
        "acl_hashes": compute_security_scope(
            trusted_user_id,
            policies,
            groups=list(groups),
            roles=list(roles),
        ),
    }
    return trusted_user_id, resolved


async def _chat_stream_realtime(
    payload: ChatRequest,
    *,
    trusted_user_id: str,
    constraints: Constraints,
    initial_evidence: Optional[list],
) -> AsyncIterator[str]:
    services = _get_services()
    cb = AsyncQueueCallback()

    async def _run() -> None:
        await run_agent(
            query=payload.query,
            tenant_id=payload.tenant_id,
            user_id=trusted_user_id,
            services=services,
            constraints=constraints,
            session_id=payload.session_id,
            callback=cb,
            initial_evidence=initial_evidence,
        )

    task = asyncio.create_task(_run())
    request_id = None
    try:
        async for event in cb:
            data = dict(event.data)
            if "request_id" in data:
                request_id = data["request_id"]
            if event.event_type == "tool_call":
                yield _sse("tool_call", data)
            elif event.event_type == "tool_result":
                yield _sse("tool_result", data)
            elif event.event_type == "error":
                yield _sse("error", data)
            elif event.event_type == "final":
                answer = data.get("answer", "")
                for delta in _iter_tokens(answer):
                    yield _sse("token", {"request_id": request_id, "delta": delta})
                yield _sse("final", data)
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Streaming agent task failed")


def _iter_tokens(text: str, size: int = 8) -> Iterator[str]:
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
