"""OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional

import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.callbacks import AgentEvent, QueueCallback, NullCallback
from ..agent.graph import run_agent
from ..agent.state import Constraints

router = APIRouter(tags=["openai-compat"])


class Message(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "ragbot"
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.2
    max_tokens: Optional[int] = None


def create_openai_compat_endpoint(get_services, verify_api_key):
    """Register the /v1/chat/completions endpoint on the router."""

    @router.post("/v1/chat/completions")
    async def openai_chat_completions(
        payload: OpenAIChatRequest,
        request: Request,
        _key: str = Depends(verify_api_key),
    ):
        services = get_services()
        tenant_id = request.headers.get("X-Tenant-ID", "default")
        user_id = request.headers.get("X-User-ID", "anonymous")

        # Extract query from last user message
        query = ""
        for msg in reversed(payload.messages):
            if msg.role == "user":
                query = msg.content
                break
        if not query:
            query = payload.messages[-1].content if payload.messages else ""

        request_id = uuid.uuid4().hex

        if payload.stream:
            return StreamingResponse(
                _stream_response(query, tenant_id, user_id, services, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming: run agent and format as OpenAI response
        state = await asyncio.to_thread(
            run_agent,
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            services=services,
            request_id=request_id,
        )
        answer = state.final.answer if state.final else ""
        citations = [asdict(c) for c in state.final.citations] if state.final else []

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(query),
                "completion_tokens": len(answer),
                "total_tokens": len(query) + len(answer),
            },
            "citations": citations,
        }

    async def _stream_response(
        query: str,
        tenant_id: str,
        user_id: str,
        services: Any,
        request_id: str,
    ) -> AsyncIterator[str]:
        cb = QueueCallback()

        async def _run():
            await asyncio.to_thread(
                run_agent,
                query=query,
                tenant_id=tenant_id,
                user_id=user_id,
                services=services,
                request_id=request_id,
                callback=cb,
            )

        task = asyncio.create_task(_run())

        created = int(time.time())
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

                if event.event_type == "final":
                    answer = event.data.get("answer", "")
                    # Stream the answer as token chunks
                    for i in range(0, len(answer), 8):
                        chunk = answer[i:i + 8]
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "ragbot",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # Send final stop chunk
            stop_data = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "ragbot",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await task

    return router
