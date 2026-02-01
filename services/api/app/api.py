from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent.graph import run_agent
from .agent.state import Constraints, SourceType
from .factory import build_services_from_env
from .main import chat

app = FastAPI(title="ragbot API", version="0.1.0")
_services = None


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

    class Config:
        extra = "forbid"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str
    user_id: str
    session_id: Optional[str] = None
    stream: bool = False
    constraints: Optional[ConstraintsModel] = None
    client_context: Optional[dict] = None


class IngestRequest(BaseModel):
    source_type: str
    source_config: dict


@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    services = _get_services()
    if payload.stream:
        return StreamingResponse(
            _chat_stream(payload),
            media_type="text/event-stream",
        )
    constraints = _constraints_from_model(payload.constraints)
    return chat(
        payload.query,
        payload.tenant_id,
        payload.user_id,
        services,
        constraints=constraints,
        session_id=payload.session_id,
    )


@app.post("/ingest", status_code=202)
def ingest_endpoint(payload: IngestRequest) -> dict:
    job_id = payload.source_config.get("job_id", "demo")
    return {"status": "accepted", "job_id": job_id}


@app.get("/admin/health")
def health_endpoint() -> dict:
    return {"status": "ok"}


@app.on_event("shutdown")
def shutdown_event() -> None:
    services = _get_services()
    engine = services.sql_engine
    if hasattr(engine, "close"):
        engine.close()


def _constraints_from_model(model: Optional[ConstraintsModel]) -> Optional[Constraints]:
    if not model:
        return None
    return Constraints(**model.dict())


def _chat_stream(payload: ChatRequest) -> Iterable[str]:
    services = _get_services()
    constraints = _constraints_from_model(payload.constraints)
    state = run_agent(
        query=payload.query,
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        services=services,
        constraints=constraints,
        session_id=payload.session_id,
    )
    request_id = state.request_id

    for call in state.tool_calls:
        yield _sse(
            "tool_call",
            {"request_id": request_id, "name": call.name, "args": call.args},
        )
        yield _sse(
            "tool_result",
            {
                "request_id": request_id,
                "name": call.name,
                "ok": call.ok,
                "meta": call.result_preview,
                "error": call.error,
            },
        )

    if state.final:
        for chunk in _chunk_text(state.final.answer, size=20):
            yield _sse("token", {"request_id": request_id, "delta": chunk})
        if state.final.citations:
            yield _sse(
                "citation",
                {"request_id": request_id, "citations": [asdict(cite) for cite in state.final.citations]},
            )
        yield _sse(
            "final",
            {
                "request_id": request_id,
                "answer": state.final.answer,
                "citations": [asdict(cite) for cite in state.final.citations],
                "confidence": state.final.confidence,
                "followups": state.final.followups,
            },
        )


def _chunk_text(text: str, size: int = 20) -> Iterable[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _get_services():
    global _services
    if _services is None:
        _services = build_services_from_env()
    return _services

