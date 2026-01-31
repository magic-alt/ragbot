from __future__ import annotations

import re
from typing import Dict

from ..state import ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL
from contracts.types import AgentState

SQL_HINTS = ("select", "from", "join", "where", "group by", "报表", "统计", "sum", "count")
CODE_HINTS = ("stacktrace", "error", "exception", "报错", "函数", "方法", "class", "代码", "repo")
DOC_HINTS = ("规范", "文档", "说明", "教程", "指南")


def route_node(state: AgentState) -> AgentState:
    query = state.query.lower()
    if _matches(query, SQL_HINTS):
        state.route = ROUTE_SQL
    elif _matches(query, CODE_HINTS):
        state.route = ROUTE_CODE
    elif _matches(query, DOC_HINTS):
        state.route = ROUTE_DOC_RAG
    else:
        state.route = ROUTE_MIXED

    state.constraints = _infer_constraints(state.query)
    return state


def _matches(query: str, hints: tuple) -> bool:
    return any(hint in query for hint in hints)


def _infer_constraints(query: str) -> Dict[str, str]:
    constraints: Dict[str, str] = {}
    match = re.search(r"repo:([\w\-_/]+)", query)
    if match:
        constraints["repo"] = match.group(1)
    if "pdf" in query:
        constraints["source_type"] = "pdf"
    if "网页" in query or "web" in query:
        constraints["source_type"] = "web"
    return constraints

