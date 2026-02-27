from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ...auth.acl import compute_security_scope
from ..state import ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL, ROUTE_WEB, AgentState

logger = logging.getLogger(__name__)

SQL_HINTS = ("select", "from", "join", "where", "group by", "报表", "统计", "sum", "count")
CODE_HINTS = ("stacktrace", "error", "exception", "报错", "函数", "方法", "class", "代码", "repo")
DOC_HINTS = ("规范", "文档", "说明", "教程", "指南")

_LLM_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["doc_rag", "sql", "code", "mixed", "web_fallback"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["route", "reasoning"],
    "additionalProperties": False,
}


def route_node(state: AgentState, services: Any) -> AgentState:
    query = state.query.lower()
    policies = services.repo.list_policies(state.tenant_id)
    acl_hashes = compute_security_scope(state.user_id, policies)
    state.constraints.security_scope = {
        "tenant_id": state.tenant_id,
        "acl_hashes": acl_hashes,
    }

    llm = getattr(services, "llm", None)
    if llm and getattr(llm, "enabled", False):
        try:
            state.route = _llm_route(llm, state.query)
        except Exception as exc:
            logger.warning("LLM routing failed, falling back to keyword: %s", exc)
            state.route = _keyword_route(query)
    else:
        state.route = _keyword_route(query)

    _infer_constraints(state)
    state.plan = _build_plan(state.route)
    return state


def _llm_route(llm: Any, query: str) -> str:
    system = (
        "You are a router for a RAG system. Classify the user question into one of: "
        "doc_rag (document retrieval), sql (structured data query), code (code search), "
        "mixed (multiple sources needed), web_fallback (web search needed). "
        "Return JSON with fields: route, reasoning."
    )
    user = f"Question: {query}"
    result = llm.chat_json(system=system, user=user, schema=_LLM_ROUTE_SCHEMA)
    route = result.get("route", "mixed")
    valid = {ROUTE_DOC_RAG, ROUTE_SQL, ROUTE_CODE, ROUTE_MIXED, ROUTE_WEB}
    if route not in valid:
        return ROUTE_MIXED
    return route


def _keyword_route(query: str) -> str:
    if _matches(query, SQL_HINTS):
        return ROUTE_SQL
    if _matches(query, CODE_HINTS):
        return ROUTE_CODE
    if _matches(query, DOC_HINTS):
        return ROUTE_DOC_RAG
    return ROUTE_MIXED


def _matches(query: str, hints: tuple) -> bool:
    return any(hint in query for hint in hints)


def _infer_constraints(state: AgentState) -> None:
    query = state.query
    match = re.search(r"repo:([\w\-_/]+)", query)
    if match:
        state.constraints.repo = state.constraints.repo or match.group(1)
    if "pdf" in query:
        state.constraints.source_types = state.constraints.source_types or ["pdf"]
    if "网页" in query or "web" in query:
        state.constraints.source_types = state.constraints.source_types or ["web"]


def _build_plan(route: Optional[str]) -> list[str]:
    if route == ROUTE_SQL:
        return ["sql_query", "retrieve(doc) if needed", "finalize"]
    if route == ROUTE_CODE:
        return ["code_search", "retrieve(doc) if needed", "finalize"]
    if route == ROUTE_DOC_RAG:
        return ["retrieve(doc)", "synthesize", "verify", "finalize"]
    return ["retrieve(doc)", "verify", "retrieve/sql/code if needed", "finalize"]

