from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .agent.graph import AgentServices
from .agent.nodes.code import CodeSearch
from .agent.nodes.sql import PostgresSqlEngine, SqlEngine
from .llm.provider import build_model_provider
from .retrieval.cross_encoder import build_reranker
from .retrieval.embedder import build_embedder
from .retrieval.qdrant import InMemoryQdrant, QdrantClientAdapter
from .retrieval.service import Retriever
from .storage.repo import InMemoryRepo

logger = logging.getLogger(__name__)


def build_services_from_env(repo: Optional[Any] = None) -> AgentServices:
    # ── Repo ──────────────────────────────────────────────────────────
    postgres_dsn = os.getenv("POSTGRES_DSN")
    if repo is None:
        if postgres_dsn:
            from .storage.pg_repo import PostgresRepo
            repo = PostgresRepo(dsn=postgres_dsn)
            logger.info("Using PostgresRepo")
        else:
            repo = InMemoryRepo()
            logger.info("Using InMemoryRepo")

    # ── Qdrant + Embedder ─────────────────────────────────────────────
    # The vector-store dimension is a single source of truth for both
    # ingestion and query embeddings. Keep the in-memory default lightweight,
    # while retaining the historical 1536 production default for Qdrant.
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    qdrant_dim_raw = os.getenv("QDRANT_DIM")
    qdrant_dim = int(qdrant_dim_raw) if qdrant_dim_raw else (1536 if qdrant_url else 64)

    if qdrant_url:
        qdrant = QdrantClientAdapter(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection,
            dim=qdrant_dim,
        )
    else:
        qdrant = InMemoryQdrant(dim=qdrant_dim)

    embedder = build_embedder(dimension=qdrant_dim)
    if embedder.dimension != qdrant.dim:
        raise RuntimeError(
            "Embedding dimension does not match vector store: "
            f"embedder={embedder.dimension}, qdrant={qdrant.dim}. "
            "Set QDRANT_DIM consistently and reindex after changing embedding models."
        )

    # ── Reranker ──────────────────────────────────────────────────────
    reranker = build_reranker()
    retriever = Retriever(repo, qdrant, embedder=embedder, reranker=reranker)

    # ── SQL Engine ────────────────────────────────────────────────────
    if postgres_dsn:
        allowed_schemas_raw = os.getenv("POSTGRES_ALLOWED_SCHEMAS")
        allowed_schemas = (
            [s.strip() for s in allowed_schemas_raw.split(",") if s.strip()]
            if allowed_schemas_raw
            else None
        )
        sql_limit = int(os.getenv("POSTGRES_SQL_LIMIT", "200"))
        sql_timeout = int(os.getenv("POSTGRES_SQL_TIMEOUT_MS", "3000"))
        sql_engine = PostgresSqlEngine(
            dsn=postgres_dsn,
            allowed_schemas=allowed_schemas,
            limit=sql_limit,
            timeout_ms=sql_timeout,
        )
    else:
        sql_engine = SqlEngine(repo)

    # ── Code + LLM ────────────────────────────────────────────────────
    repo_root = os.getenv("CODE_REPO_ROOT", ".")
    code_search = CodeSearch(repo_roots={"default": repo_root})
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
