from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .agent.callbacks import QueueCallback
from .agent.context import process_client_context, dedup_evidence, compress_evidence
from .agent.graph import run_agent
from .agent.state import Constraints, SourceType
from .factory import build_services_from_env
from .main import chat
from .middleware import setup_middleware
from .routes.openai_compat import create_openai_compat_endpoint
from .routes.ingest import create_ingest_router
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
    yield
    services = _get_services()
    engine = services.sql_engine
    if hasattr(engine, "close"):
        engine.close()


app = FastAPI(title="ragbot API", version="0.4.0", lifespan=lifespan)

# Register middleware (CORS + request logging)
setup_middleware(app)

# Register routers
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
    tenant_id: str
    user_id: str
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
        )
    constraints = _constraints_from_model(payload.constraints)
    # Process client_context (IDE context injection)
    constraints, initial_evidence = process_client_context(
        payload.client_context, constraints,
    )
    result = await asyncio.to_thread(
        chat,
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
    return {"status": "ok"}


def _constraints_from_model(model: Optional[ConstraintsModel]) -> Optional[Constraints]:
    if not model:
        return None
    return Constraints(**model.model_dump())


async def _chat_stream_realtime(payload: ChatRequest) -> AsyncIterator[str]:
    """Real-time SSE streaming using QueueCallback.

    The agent runs in a background thread and emits events to a thread-safe
    queue. This async generator reads from the queue and yields SSE events.
    """
    services = _get_services()
    constraints = _constraints_from_model(payload.constraints)
    cb = QueueCallback()

    async def _run_agent():
        await asyncio.to_thread(
            run_agent,
            query=payload.query,
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            services=services,
            constraints=constraints,
            session_id=payload.session_id,
            callback=cb,
        )

    task = asyncio.create_task(_run_agent())
    request_id = None

    try:
        while True:
            try:
                event = await asyncio.to_thread(cb.get, 0.5)
            except Exception:
                if task.done():
                    break
                continue

            if event is None:
                break

            data = dict(event.data)
            if "request_id" in data:
                request_id = data["request_id"]

            if event.event_type == "tool_call":
                yield _sse("tool_call", data)
            elif event.event_type == "tool_result":
                yield _sse("tool_result", data)
            elif event.event_type == "final":
                answer = data.get("answer", "")
                # Stream the final answer as token chunks
                for delta in _iter_tokens(answer):
                    yield _sse("token", {"request_id": request_id, "delta": delta})
                yield _sse("final", data)
    finally:
        await task


def _iter_tokens(text: str, size: int = 8) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
