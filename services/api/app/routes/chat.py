from __future__ import annotations

from typing import Dict

from ..main import chat


async def handle_chat(payload: Dict[str, str]) -> Dict[str, object]:
    return await chat(payload["query"], payload["tenant_id"], payload["user_id"])

