"""OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.callbacks import AsyncQueueCallback
from ..agent.graph import run_agent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compat"])


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "ragbot"
    messages: List[Message] = Field(min_length=1)
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
        tenant_id = request.headers.get("X-Tenant-ID", "default").strip() or "default"
        user_id = request.headers.get("X-User-ID", "anonymous").strip() or "anonymous"

        query = ""
        for msg in reversed(payload.messages):
            if msg.role == "user" and msg.content.strip():
                query = msg.content
                break
        if not query:
            query = payload.messages[-1].content
        if not query.strip():
            raise HTTPException(status_code=400, detail="At least one non-empty message is required")

        request_id = uuid.uuid4().hex

        if payload.stream:
            return StreamingResponse(
                _stream_response(
                    query,
                    tenant_id,
                    user_id,
                    services,
                    request_id,
                    payload.model,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        state = await run_agent(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            services=services,
            request_id=request_id,
        )
        answer = state.final.answer if state.final else ""
        citations = [asdict(c) for c in state.final.citations] if state.final else []

        # Ragbot providers do not currently expose provider token accounting at
        # this adapter boundary. Do not report character counts as token counts.
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
            "citations": citations,
        }

    async def _stream_response(
        query: str,
        tenant_id: str,
        user_id: str,
        services: Any,
        request_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        cb = AsyncQueueCallback()

        async def _run() -> None:
            await run_agent(
                query=query,
                tenant_id=tenant_id,
                user_id=user_id,
                services=services,
                request_id=request_id,
                callback=cb,
            )

        task = asyncio.create_task(_run())
        created = int(time.time())
        failed = False
        try:
            async for event in cb:
                if event.event_type == "error":
                    failed = True
                    error_data = {
                        "error": {
                            "message": event.data.get("error", "Agent execution failed"),
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                elif event.event_type == "final":
                    answer = event.data.get("answer", "")
                    for i in range(0, len(answer), 8):
                        chunk = answer[i:i + 8]
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if not failed:
                stop_data = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(stop_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("OpenAI-compatible streaming agent task failed")

    return router
