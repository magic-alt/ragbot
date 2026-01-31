from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from contracts.types import AgentState
from ..auth.acl import compute_security_scope
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
from .state import ROUTE_CODE, ROUTE_DOC_RAG, ROUTE_MIXED, ROUTE_SQL, ROUTE_WEB, build_initial_state


@dataclass
class AgentServices:
    repo: InMemoryRepo
    qdrant: InMemoryQdrant
    retriever: Retriever
    sql_engine: SqlEngine
    code_search: CodeSearch

    def security_scope(self, user_id: str, tenant_id: str):
        policies = self.repo.list_policies(tenant_id)
        return compute_security_scope(user_id, policies)


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


def run_agent(query: str, tenant_id: str, user_id: str, services: AgentServices) -> AgentState:
    state = build_initial_state(query, tenant_id, user_id)
    state = route_node(state)

    if state.route == ROUTE_SQL:
        state = sql_node(state, services)
    elif state.route == ROUTE_CODE:
        state = code_node(state, services)
    elif state.route == ROUTE_DOC_RAG:
        state = retrieve_node(state, services)
    elif state.route == ROUTE_MIXED:
        state = retrieve_node(state, services)
    else:
        state = web_node(state, services)

    state = synthesize_node(state, services)
    state = verify_node(state, services)
    state = finalize_node(state, services)
    return state

