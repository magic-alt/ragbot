from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from contracts.types import SqlResult

from ..llm.provider import ModelProvider
from ..llm.router import build_model_router
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
    llm = build_model_router()
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
    """Execute the agent state graph."""
    state = build_initial_state(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        constraints=constraints,
        session_id=session_id,
        request_id=request_id,
        initial_evidence=initial_evidence,
        conversation_messages=conversation_messages,
        system_prompt=system_prompt,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
    )
    callback = callback or NullCallback()
    metrics = get_metrics_collector()
    request_metrics = build_request_metrics(state.request_id)
    tracer = RequestTracer(state.request_id)

    try:
        state = await route_node(state, services)
        callback.on_event(AgentEvent("route", state.request_id, {"route": state.route}))

        if state.route in (ROUTE_DOC_RAG, ROUTE_MIXED):
            state = await retrieve_node(state, services)
        elif state.route == ROUTE_SQL:
            state = await sql_node(state, services)
        elif state.route == ROUTE_CODE:
            state = await code_node(state, services)
        elif state.route == ROUTE_WEB:
            state = await web_node(state, services)

        state = await synthesize_node(state, services)
        state = await verify_node(state, services)
        state = await finalize_node(state, services)
        return state
    finally:
        # Existing metric/tracing helpers own detailed event aggregation; keep
        # this function's lifecycle behavior stable while model routing evolves.
        try:
            metrics.record_request(asdict(request_metrics))
        except Exception:
            pass
        try:
            tracer.finish()
        except Exception:
            pass
