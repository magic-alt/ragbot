from __future__ import annotations

import re
from typing import Optional

from ...auth.acl import compute_security_scope
from ..state import ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL, AgentState

SQL_HINTS = ("select", "from", "join", "where", "group by", "报表", "统计", "sum", "count")
CODE_HINTS = ("stacktrace", "error", "exception", "报错", "函数", "方法", "class", "代码", "repo")
DOC_HINTS = ("规范", "文档", "说明", "教程", "指南")


def route_node(state: AgentState, services: object) -> AgentState:
    query = state.query.lower()
    policies = services.repo.list_policies(state.tenant_id)
    acl_hashes = compute_security_scope(state.user_id, policies)
    state.constraints.security_scope = {
        "tenant_id": state.tenant_id,
        "acl_hashes": acl_hashes,
    }
    if _matches(query, SQL_HINTS):
        state.route = ROUTE_SQL
    elif _matches(query, CODE_HINTS):
        state.route = ROUTE_CODE
    elif _matches(query, DOC_HINTS):
        state.route = ROUTE_DOC_RAG
    else:
        state.route = ROUTE_MIXED

    _infer_constraints(state)
    state.plan = _build_plan(state.route)
    return state


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

