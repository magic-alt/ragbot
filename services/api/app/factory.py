from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .agent.graph import AgentServices
from .agent.nodes.code import CodeSearch
from .agent.nodes.sql import PostgresSqlEngine, SqlEngine
from .agent.sql_disabled import DisabledSqlEngine
from .llm.provider import build_model_provider
from .retrieval.cross_encoder import build_reranker
from .retrieval.embedder import HashEmbedder, build_embedder, model_dimension
from .retrieval.qdrant import InMemoryQdrant, QdrantClientAdapter
from .retrieval.service import Retriever
from .runtime import is_production, validate_production_environment
from .storage.generation_support import ensure_generation_repository
from .storage.repo import InMemoryRepo
from .storage.upload_support import ensure_upload_repository

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_services_from_env(repo: Optional[Any] = None) -> AgentServices:
    validate_production_environment()

    postgres_dsn = os.getenv("POSTGRES_DSN")
    if repo is None:
        if postgres_dsn:
            from .storage.managed_pg_repo import ManagedPostgresRepo
            repo = ManagedPostgresRepo(dsn=postgres_dsn)
            logger.info("Using ManagedPostgresRepo")
        else:
            repo = InMemoryRepo()
            logger.info("Using InMemoryRepo")

    ensure_generation_repository(repo)
    ensure_upload_repository(repo)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    qdrant_dim_raw = os.getenv("QDRANT_DIM")
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
    inferred_dim = model_dimension(embedding_model)
    if qdrant_dim_raw:
        qdrant_dim = int(qdrant_dim_raw)
    elif inferred_dim:
        qdrant_dim = inferred_dim
    else:
        qdrant_dim = 1536 if qdrant_url else 64

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

    if is_production():
        unsafe = []
        if isinstance(repo, InMemoryRepo):
            unsafe.append("InMemoryRepo")
        if isinstance(qdrant, InMemoryQdrant):
            unsafe.append("InMemoryQdrant")
        if isinstance(embedder, HashEmbedder):
            unsafe.append("HashEmbedder")
        if unsafe:
            raise RuntimeError(
                "Production services cannot use development fallbacks: " + ", ".join(unsafe)
            )

    reranker = build_reranker()
    retriever = Retriever(repo, qdrant, embedder=embedder, reranker=reranker)

    sql_enabled = _env_flag("RAGBOT_SQL_TOOL_ENABLED", False)
    if sql_enabled:
        sql_dsn = (os.getenv("RAGBOT_SQL_DSN") or "").strip()
        if postgres_dsn and not sql_dsn and not is_production():
            sql_dsn = postgres_dsn
        if sql_dsn:
            allowed_schemas_raw = os.getenv("RAGBOT_SQL_ALLOWED_SCHEMAS", "")
            allowed_schemas = [
                s.strip() for s in allowed_schemas_raw.split(",") if s.strip()
            ] or None
            sql_engine = PostgresSqlEngine(
                dsn=sql_dsn,
                allowed_schemas=allowed_schemas,
                limit=int(os.getenv("RAGBOT_SQL_LIMIT", "200")),
                timeout_ms=int(os.getenv("RAGBOT_SQL_TIMEOUT_MS", "3000")),
            )
        elif isinstance(repo, InMemoryRepo):
            sql_engine = SqlEngine(repo)
        else:
            raise RuntimeError(
                "RAGBOT_SQL_TOOL_ENABLED=true requires RAGBOT_SQL_DSN "
                "or an in-memory development repository"
            )
    else:
        sql_engine = DisabledSqlEngine()

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
