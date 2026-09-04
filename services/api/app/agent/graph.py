from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from contracts.types import SqlResult

from ..llm.provider import ModelProvider, build_model_provider
from ..observability.metrics import build_request_metrics, get_metrics_collector
from ..observability.tracing import RequestTracer
from ..retrieval.cross_encoder import NoOpReranker, Reranker
from ..retrieval.embedder import Embedder, HashEmbedder
from ..retrieval.qdrant import InMemoryQdrant
from ..retrieval.service import Retriever
from ..storage.protocol import Repo
from ..storage.repo import InMemoryRepo
from .callbacks import AgentEvent, EventCallback, NullCallback
from .nodes.code import CodeSearch, apply_patch_node, code_node, explain_error_node, open_file_node
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


@runtime_checkable
class QdrantInterface(Protocol):
    @property
    def dim(self) -> int: ...
    def upsert(self, points: Iterable[Tuple[str, List[float], Dict[str, Any]]]) -> None: ...
    def delete_points(self, point_ids: Iterable[str]) -> int: ...
    def delete_by_doc_ids(self, doc_ids: Iterable[str]) -> int: ...
    def healthcheck(self) -> bool: ...
    def search(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]: ...


@runtime_checkable
class SqlEngineInterface(Protocol):
    def query(self, query: str, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> SqlResult: ...


@dataclass
class AgentServices:
    repo: Repo
    qdrant: QdrantInterface
    retriever: Retriever
    sql_engine: SqlEngineInterface
    code_search: CodeSearch
    llm: ModelProvider
    embedder: Embedder = None  # type: ignore[assignment]
    reranker: Reranker = None  # type: ignore[assignment]


def build_default_services(repo: Optional[InMemoryRepo] = None) -> AgentServices:
    repo = repo or InMemoryRepo()
    qdrant = InMemoryQdrant()
    embedder = HashEmbedder()
    reranker = NoOpReranker()
    retriever = Retriever(repo, qdrant, embedder=embedder, reranker=reranker)
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
        embedder=embedder,
        reranker=reranker,
    )


async def run_agent(
    query: str,
    tenant_id: str,
    user_id: str,
    services: AgentServices,
    constraints: Optional[Constraints] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    callback: Optional[EventCallback] = None,
    initial_evidence: Optional[list] = None,
    conversation_messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    generation_temperature: float = 0.2,
    generation_max_tokens: Optional[int] = None,
) -> AgentState:
    """Run the agent graph and always terminate the event stream."""
    cb = callback or NullCallback()
    state = build_initial_state(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        constraints=constraints,
        request_id=request_id,
        conversation_messages=conversation_messages,
        system_prompt=system_prompt,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
    )
    tracer = RequestTracer(request_id=state.request_id)

    try:
        if initial_evidence:
            state.evidence.extend(initial_evidence)

        with tracer.span("route") as span:
            state = await route_node(state, services)
            span.attributes["route"] = state.route or ""
        cb.emit(AgentEvent("route", {"route": state.route, "request_id": state.request_id}))

        action = _initial_action(state)
        while True:
            state.iteration += 1
            prev_calls = len(state.tool_calls)

            with tracer.span(action, iteration=state.iteration) as span:
                if action == "sql_query":
                    state = await sql_node(state, services)
                elif action == "code_search":
                    state = await code_node(state, services)
                elif action == "open_file":
                    state = await open_file_node(state, services)
                elif action == "apply_patch":
                    state = await apply_patch_node(state, services)
                elif action == "explain_error":
                    state = await explain_error_node(state, services)
                elif action == "retrieve":
                    state = await retrieve_node(state, services)
                elif action == "web_search":
                    state = await web_node(state, services)

                new_calls = state.tool_calls[prev_calls:]
                span.attributes["tool_calls"] = len(new_calls)
                span.attributes["evidence_total"] = len(state.evidence)
                for call in new_calls:
                    if not call.ok:
                        span.attributes["has_failure"] = True

            for call in state.tool_calls[prev_calls:]:
                cb.emit(AgentEvent("tool_call", {
                    "name": call.name,
                    "args": call.args,
                    "request_id": state.request_id,
                }))
                cb.emit(AgentEvent("tool_result", {
                    "name": call.name,
                    "ok": call.ok,
                    "meta": call.result_preview,
                    "error": call.error,
                    "request_id": state.request_id,
                }))

            with tracer.span("synthesize", iteration=state.iteration):
                state = await synthesize_node(state, services)
            with tracer.span("verify", iteration=state.iteration):
                state = await verify_node(state, services)
            if not _should_continue(state):
                break
            action = _next_step(state)

        with tracer.span("finalize"):
            state = await finalize_node(state, services)

        trace_record = tracer.finish()
        metrics = build_request_metrics(state, trace_record)
        get_metrics_collector().record(metrics)

        if state.final:
            cb.emit(AgentEvent("final", {
                "request_id": state.request_id,
                "answer": state.final.answer,
                "citations": [asdict(citation) for citation in state.final.citations],
                "confidence": state.final.confidence,
                "followups": list(state.final.followups),
            }))
        return state
    except Exception:
        cb.emit(AgentEvent("error", {
            "request_id": state.request_id,
            "error": "Agent execution failed",
        }))
        raise
    finally:
        cb.close()


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
