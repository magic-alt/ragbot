from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .agent.graph import run_agent
from .agent.state import Constraints, SourceType
from .factory import build_services_from_env
from .main import chat

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


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _VALID_API_KEYS
    _VALID_API_KEYS = _load_api_keys()
    yield
    services = _get_services()
    engine = services.sql_engine
    if hasattr(engine, "close"):
        engine.close()


app = FastAPI(title="ragbot API", version="0.1.0", lifespan=lifespan)


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
async def chat_endpoint(payload: ChatRequest, _key: str = Depends(verify_api_key)):
    services = _get_services()
    if payload.stream:
        return StreamingResponse(
            _chat_stream(payload),
            media_type="text/event-stream",
        )
    constraints = _constraints_from_model(payload.constraints)
    result = await asyncio.to_thread(
        chat,
        payload.query,
        payload.tenant_id,
        payload.user_id,
        services,
        constraints,
        payload.session_id,
    )
    return result


@app.post("/ingest", status_code=202)
async def ingest_endpoint(payload: IngestRequest, _key: str = Depends(verify_api_key)) -> dict:
    job_id = payload.source_config.get("job_id", "demo")
    return {"status": "accepted", "job_id": job_id}


@app.get("/admin/health")
async def health_endpoint() -> dict:
    return {"status": "ok"}


def _constraints_from_model(model: Optional[ConstraintsModel]) -> Optional[Constraints]:
    if not model:
        return None
    return Constraints(**model.dict())


def _chat_stream(payload: ChatRequest) -> Iterator[str]:
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

    final_answer = state.final.answer if state.final else ""
    streamed_answer = final_answer

    if state.final and state.final.confidence != "low":
        llm = getattr(services, "llm", None)
        if llm and getattr(llm, "enabled", False) and state.draft and state.draft.answer_outline:
            streamed_answer = ""
            system, user = _build_stream_prompt(payload.query, state.draft.answer_outline)
            try:
                for delta in llm.stream_text(system=system, user=user, temperature=0.2):
                    streamed_answer += delta
                    yield _sse("token", {"request_id": request_id, "delta": delta})
            except Exception:
                streamed_answer = final_answer
                for delta in _iter_tokens(final_answer):
                    yield _sse("token", {"request_id": request_id, "delta": delta})
        else:
            for delta in _iter_tokens(final_answer):
                yield _sse("token", {"request_id": request_id, "delta": delta})
    else:
        for delta in _iter_tokens(final_answer):
            yield _sse("token", {"request_id": request_id, "delta": delta})

    if state.final:
        if state.final.citations:
            yield _sse(
                "citation",
                {"request_id": request_id, "citations": [asdict(cite) for cite in state.final.citations]},
            )
        yield _sse(
            "final",
            {
                "request_id": request_id,
                "answer": streamed_answer,
                "citations": [asdict(cite) for cite in state.final.citations],
                "confidence": state.final.confidence,
                "followups": state.final.followups,
            },
        )


def _iter_tokens(text: str, size: int = 8) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _build_stream_prompt(question: str, claim_lines: List[str]) -> tuple[str, str]:
    system = (
        "You are a helpful assistant. Rewrite the provided claims into a concise answer. "
        "Do not add new facts. Preserve citations exactly as given."
    )
    claims = "\n".join(f"- {line}" for line in claim_lines)
    user = f"Question:\n{question}\n\nClaims:\n{claims}\n\nAnswer:"
    return system, user


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _get_services():
    global _services
    if _services is None:
        _services = build_services_from_env()
    return _services
