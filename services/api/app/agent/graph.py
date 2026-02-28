from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, runtime_checkable

from typing import Protocol

from ..llm.provider import ModelProvider, build_model_provider
from ..retrieval.qdrant import InMemoryQdrant
from ..retrieval.service import Retriever
from ..storage.repo import InMemoryRepo
from .callbacks import AgentEvent, EventCallback, NullCallback
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
from contracts.types import SqlResult


@runtime_checkable
class QdrantInterface(Protocol):
    @property
    def dim(self) -> int: ...

    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None: ...

    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]: ...


@runtime_checkable
class SqlEngineInterface(Protocol):
    def query(self, query: str, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> SqlResult: ...


@dataclass
class AgentServices:
    repo: InMemoryRepo
    qdrant: QdrantInterface
    retriever: Retriever
    sql_engine: SqlEngineInterface
    code_search: CodeSearch
    llm: ModelProvider


def build_default_services(repo: Optional[InMemoryRepo] = None) -> AgentServices:
    repo = repo or InMemoryRepo()
    qdrant = InMemoryQdrant()
    retriever = Retriever(repo, qdrant)
    sql_engine = SqlEngine(repo)
    code_search = CodeSearch(repo_roots={"default": "."})
    llm = build_model_provider()
    return AgentServices(
        repo=repo,
        qdrant=qdrant,
        retriever=retriever,
        sql_engine=sql_engine,
        code_search=code_search,
        llm=llm,
    )


def run_agent(
    query: str,
    tenant_id: str,
    user_id: str,
    services: AgentServices,
    constraints: Optional[Constraints] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    callback: Optional[EventCallback] = None,
) -> AgentState:
    cb = callback or NullCallback()
    state = build_initial_state(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        constraints=constraints,
        request_id=request_id,
    )
    state = route_node(state, services)
    cb.emit(AgentEvent("route", {"route": state.route, "request_id": state.request_id}))

    action = _initial_action(state)
    while True:
        state.iteration += 1
        prev_calls = len(state.tool_calls)

        if action == "sql_query":
            state = sql_node(state, services)
        elif action == "code_search":
            state = code_node(state, services)
        elif action == "retrieve":
            state = retrieve_node(state, services)
        elif action == "web_search":
            state = web_node(state, services)

        # Emit events for any new tool calls
        for call in state.tool_calls[prev_calls:]:
            cb.emit(AgentEvent("tool_call", {"name": call.name, "args": call.args, "request_id": state.request_id}))
            cb.emit(AgentEvent("tool_result", {
                "name": call.name, "ok": call.ok,
                "meta": call.result_preview, "error": call.error,
                "request_id": state.request_id,
            }))

        state = synthesize_node(state, services)
        state = verify_node(state, services)
        if not _should_continue(state):
            break
        action = _next_step(state)

    state = finalize_node(state, services)
    if state.final:
        cb.emit(AgentEvent("final", {
            "request_id": state.request_id,
            "answer": state.final.answer,
            "confidence": state.final.confidence,
        }))
    cb.close()
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
    if state.verification and state.verification.next_query:
        state.query = state.verification.next_query
    if state.verification and state.verification.next_action:
        return state.verification.next_action
    return "retrieve"

