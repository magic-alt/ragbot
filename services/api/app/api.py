from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .agent.callbacks import AsyncQueueCallback
from .agent.context import compress_evidence, dedup_evidence, process_client_context
from .agent.graph import run_agent
from .agent.state import Constraints, SourceType
from .factory import build_services_from_env
from .main import chat
from .middleware import setup_middleware
from .observability.metrics import get_metrics_collector
from .observability.tracing import setup_tracing
from .routes.ingest import create_ingest_router
from .routes.openai_compat import create_openai_compat_endpoint
from .routes.search import create_search_endpoint
from .routes.sources import create_sources_router

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_API_KEYS: Optional[set] = None


def _load_api_keys() -> Optional[set]:
    raw = os.getenv("RAGBOT_API_KEYS", "")
    if not raw.strip():
        return None
    return {k.strip() for k in raw.split(",") if k.strip()}


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
    _VALID_API_KEYS = _load_api_keys()
    setup_tracing()
    yield

    # Do not instantiate external services just to shut them down. If services
    # were used, close every distinct closeable resource (SQL, repo, Qdrant).
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
async def chat_endpoint(payload: ChatRequest, _key: str = Depends(verify_api_key)):
    services = _get_services()
    if payload.stream:
        return StreamingResponse(
            _chat_stream_realtime(payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    constraints = _constraints_from_model(payload.constraints)
    constraints, initial_evidence = process_client_context(payload.client_context, constraints)
    result = await chat(
        payload.query,
        payload.tenant_id,
        payload.user_id,
        services,
        constraints,
        payload.session_id,
        initial_evidence=initial_evidence,
    )
    return result


@app.get("/admin/health")
async def health_endpoint() -> dict:
    """Liveness endpoint: the API process is running."""
    return {"status": "ok"}


@app.get("/admin/ready")
async def readiness_endpoint() -> dict:
    """Readiness endpoint: configured persistence/vector dependencies respond."""
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


@app.get("/admin/metrics")
async def metrics_endpoint(_key: str = Depends(verify_api_key)) -> dict:
    collector = get_metrics_collector()
    agg = collector.aggregate()
    return {
        "total_requests": agg.total_requests,
        "citation_coverage": round(agg.citation_coverage, 4),
        "retrieval_hit_rate": round(agg.retrieval_hit_rate, 4),
        "tool_failure_rate": round(agg.tool_failure_rate, 4),
        "avg_duration_ms": round(agg.avg_duration_ms, 1),
        "p95_duration_ms": round(agg.p95_duration_ms, 1),
        "avg_iterations": round(agg.avg_iterations, 2),
        "confidence_distribution": agg.confidence_distribution,
        "tool_stats": agg.tool_stats,
        "feedback_score": round(agg.feedback_score, 4),
        "positive_feedback": agg.positive_feedback,
        "negative_feedback": agg.negative_feedback,
    }


@app.get("/admin/metrics/history")
async def metrics_history_endpoint(
    last_n: int = 100,
    _key: str = Depends(verify_api_key),
) -> dict:
    collector = get_metrics_collector()
    return {"requests": collector.get_history(last_n)}


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    feedback: str = Field(pattern="^(positive|negative)$")


@app.post("/admin/feedback")
async def feedback_endpoint(payload: FeedbackRequest, _key: str = Depends(verify_api_key)):
    collector = get_metrics_collector()
    found = collector.record_feedback(payload.request_id, payload.feedback)
    if not found:
        raise HTTPException(status_code=404, detail="Request not found")
    return {"status": "ok"}


@app.get("/admin/cost")
async def cost_endpoint(_key: str = Depends(verify_api_key)) -> dict:
    from .llm.router import CostTracker

    return CostTracker().summary()


@app.get("/admin/cache")
async def cache_endpoint(_key: str = Depends(verify_api_key)) -> dict:
    from .cache.cache import get_embedding_cache, get_retrieval_cache, is_cache_enabled

    return {
        "enabled": is_cache_enabled(),
        "retrieval": get_retrieval_cache().stats(),
        "embedding": get_embedding_cache().stats(),
    }


def _constraints_from_model(model: Optional[ConstraintsModel]) -> Optional[Constraints]:
    if not model:
        return None
    return Constraints(**model.model_dump())


async def _chat_stream_realtime(payload: ChatRequest) -> AsyncIterator[str]:
    """Stream agent events while preserving non-streaming request semantics."""
    services = _get_services()
    constraints = _constraints_from_model(payload.constraints)
    constraints, initial_evidence = process_client_context(payload.client_context, constraints)
    cb = AsyncQueueCallback()

    async def _run() -> None:
        await run_agent(
            query=payload.query,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
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
            # run_agent already emitted a sanitized SSE error event. Keep the
            # transport valid and retain the original stack trace server-side.
            logger.exception("Streaming agent task failed")


def _iter_tokens(text: str, size: int = 8) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
