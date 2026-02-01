from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..retrieval.qdrant import InMemoryQdrant
from ..retrieval.service import Retriever
from ..storage.repo import InMemoryRepo
from .nodes.code import CodeSearch, code_node
from .nodes.finalize import finalize_node
from .nodes.retrieve import retrieve_node
from .nodes.route import route_node
from .nodes.sql import SqlEngine, sql_node
from .nodes.synthesize import synthesize_node
from .nodes.verify import verify_node
from .nodes.web import web_node
from .state import (
    AgentState,
    Constraints,
    ROUTE_CODE,
    ROUTE_DOC_RAG,
    ROUTE_MIXED,
    ROUTE_SQL,
    ROUTE_WEB,
    build_initial_state,
)


@dataclass
class AgentServices:
    repo: InMemoryRepo
    qdrant: Any
    retriever: Retriever
    sql_engine: Any
    code_search: CodeSearch


def build_default_services(repo: Optional[InMemoryRepo] = None) -> AgentServices:
    repo = repo or InMemoryRepo()
    qdrant = InMemoryQdrant()
    retriever = Retriever(repo, qdrant)
    sql_engine = SqlEngine(repo)
    code_search = CodeSearch(repo_roots={"default": "."})
    return AgentServices(
        repo=repo,
        qdrant=qdrant,
        retriever=retriever,
        sql_engine=sql_engine,
        code_search=code_search,
    )


def run_agent(
    query: str,
    tenant_id: str,
    user_id: str,
    services: AgentServices,
    constraints: Optional[Constraints] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> AgentState:
    state = build_initial_state(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        constraints=constraints,
        request_id=request_id,
    )
    state = route_node(state, services)

    action = _initial_action(state)
    while True:
        state.iteration += 1
        if action == "sql_query":
            state = sql_node(state, services)
        elif action == "code_search":
            state = code_node(state, services)
        elif action == "retrieve":
            state = retrieve_node(state, services)
        elif action == "web_search":
            state = web_node(state, services)

        state = synthesize_node(state, services)
        state = verify_node(state, services)
        if not _should_continue(state):
            break
        action = _next_step(state)

    state = finalize_node(state, services)
    return state


def _initial_action(state: AgentState) -> str:
    if state.route == ROUTE_SQL:
        return "sql_query"
    if state.route == ROUTE_CODE:
        return "code_search"
    if state.route == ROUTE_DOC_RAG:
        return "retrieve"
    if state.route == ROUTE_MIXED:
        return "retrieve"
    if state.route == ROUTE_WEB:
        return "web_search"
    return "retrieve"


def _should_continue(state: AgentState) -> bool:
    if state.hard_fail:
        return False
    if state.iteration >= state.max_iterations:
        return False
    if state.verification and state.verification.enough_evidence:
        return False
    return True


def _next_step(state: AgentState) -> str:
    if state.verification and state.verification.next_action:
        return state.verification.next_action
    return "retrieve"

