"""OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.callbacks import AsyncQueueCallback
from ..agent.graph import run_agent
from ..agent.state import Constraints
from ..auth.acl import compute_security_scope
from ..auth.principal import CAP_KNOWLEDGE_QUERY, authorize_identity, require_capability

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compat"])


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "ragbot"
    messages: List[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _prepare_messages(messages: List[Message]) -> Tuple[str, List[Dict[str, str]], Optional[str]]:
    """Keep the last user turn as retrieval query and preserve prior context."""
    query_index: Optional[int] = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role.strip().lower() == "user" and message.content.strip():
            query_index = index
            break
    if query_index is None:
        query_index = len(messages) - 1

    query = messages[query_index].content.strip()
    if not query:
        raise HTTPException(status_code=400, detail="At least one non-empty message is required")

    system_parts: List[str] = []
    conversation: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        role = message.role.strip().lower()
        content = message.content.strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if index == query_index:
            continue
        if role in {"user", "assistant"}:
            conversation.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return query, conversation, system_prompt


def create_openai_compat_endpoint(get_services, verify_api_key):
    @router.post("/v1/chat/completions")
    async def openai_chat_completions(
        payload: OpenAIChatRequest,
        request: Request,
        _key: Optional[str] = Depends(verify_api_key),
    ):
        require_capability(_key, CAP_KNOWLEDGE_QUERY)
        services = get_services()
        tenant_id = request.headers.get("X-Tenant-ID", "default").strip() or "default"
        requested_user_id = request.headers.get("X-User-ID", "anonymous").strip() or "anonymous"
        user_id, groups, roles = authorize_identity(_key, tenant_id, requested_user_id)
        policies = services.repo.list_policies(tenant_id)
        constraints = Constraints(
            security_scope={
                "tenant_id": tenant_id,
                "acl_hashes": compute_security_scope(
                    user_id,
                    policies,
                    groups=list(groups),
                    roles=list(roles),
                ),
            }
        )

        query, conversation, system_prompt = _prepare_messages(payload.messages)
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
                    constraints,
                    conversation,
                    system_prompt,
                    payload.temperature,
                    payload.max_tokens,
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
            constraints=constraints,
            conversation_messages=conversation,
            system_prompt=system_prompt,
            generation_temperature=payload.temperature,
            generation_max_tokens=payload.max_tokens,
        )
        answer = state.final.answer if state.final else ""
        citations = [asdict(citation) for citation in state.final.citations] if state.final else []
        prompt_text = "\n".join(message.content for message in payload.messages)
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(answer)
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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
            },
            "citations": citations,
        }

    async def _stream_response(
        query: str,
        tenant_id: str,
        user_id: str,
        services: Any,
        request_id: str,
        model: str,
        constraints: Constraints,
        conversation: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> AsyncIterator[str]:
        callback = AsyncQueueCallback()

        async def _run() -> None:
            await run_agent(
                query=query,
                tenant_id=tenant_id,
                user_id=user_id,
                services=services,
                request_id=request_id,
                callback=callback,
                constraints=constraints,
                conversation_messages=conversation,
                system_prompt=system_prompt,
                generation_temperature=temperature,
                generation_max_tokens=max_tokens,
            )

        task = asyncio.create_task(_run())
        created = int(time.time())
        failed = False
        try:
            async for event in callback:
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
                    for index in range(0, len(answer), 8):
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": answer[index:index + 8]},
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
