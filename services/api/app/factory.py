from __future__ import annotations

import logging
import os
from typing import Any, Optional

from services.platform import RuntimeProfile
from services.worker.connectors.registry import connector_registry

from .agent.graph import AgentServices
from .agent.nodes.code import CodeSearch
from .agent.nodes.sql import PostgresSqlEngine, SqlEngine
from .agent.sql_disabled import DisabledSqlEngine
from .llm.router import build_model_router
from .retrieval.embedder import HashEmbedder, model_dimension
from .retrieval.embedding_router import ActiveIndexEmbedder, build_embedding_router
from .retrieval.index_lifecycle import IndexLifecycleService
from .retrieval.qdrant import InMemoryQdrant
from .retrieval.service import Retriever
from .runtime import is_production, validate_production_environment
from .runtime_registry import runtime_component_registry
from .storage.generation_support import ensure_generation_repository
from .storage.index_support import ensure_index_repository, supports_index_lifecycle
from .storage.publication_barrier import ensure_index_cutover_gate, ensure_worker_claim_gate
from .storage.repo import InMemoryRepo
from .storage.upload_support import ensure_upload_repository

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_profile() -> RuntimeProfile:
    return RuntimeProfile.from_environment(
        connector_ids=(spec.component_id for spec in connector_registry().specs())
    )


def build_services_from_env(repo: Optional[Any] = None) -> AgentServices:
    validate_production_environment()
    profile = resolve_runtime_profile()
    components = runtime_component_registry()

    postgres_dsn = os.getenv("POSTGRES_DSN")
    if repo is None:
        repo = components.build("repository", profile.repository_provider, {"dsn": postgres_dsn})
        logger.info("Using repository provider: %s (%s)", profile.repository_provider, type(repo).__name__)

    ensure_generation_repository(repo)
    ensure_upload_repository(repo)
    ensure_index_repository(repo)
    ensure_worker_claim_gate(repo)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    explicit_alias = os.getenv("QDRANT_INDEX_ALIAS", "").strip()
    qdrant_alias = (explicit_alias or f"{qdrant_collection}_active") if qdrant_url else ""
    qdrant_dim_raw = os.getenv("QDRANT_DIM")
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
    inferred_dim = model_dimension(embedding_model)

    active_index = None
    if qdrant_alias and supports_index_lifecycle(repo):
        active_index = repo.get_active_index_version(qdrant_alias)
    active_dense = (active_index.vector_schema.get("dense") or {}) if active_index else {}
    active_dim = int(active_dense.get("dimension") or 0)
    if active_dim:
        qdrant_dim = active_dim
    elif qdrant_dim_raw:
        qdrant_dim = int(qdrant_dim_raw)
    elif inferred_dim:
        qdrant_dim = inferred_dim
    else:
        qdrant_dim = 1536 if qdrant_url else 64

    qdrant = components.build(
        "vector",
        profile.vector_provider,
        {
            "url": qdrant_url,
            "api_key": qdrant_api_key,
            "collection_name": qdrant_collection,
            "alias_name": qdrant_alias or None,
            "dim": qdrant_dim,
        },
    )
    default_embedder = components.build(
        "embedding", profile.embedding_provider, {"dimension": qdrant_dim}
    )
    embedding_router = build_embedding_router(default_embedder)
    index_lifecycle = None
    embedder = default_embedder

    if qdrant_alias and supports_index_lifecycle(repo) and not isinstance(qdrant, InMemoryQdrant):
        index_lifecycle = IndexLifecycleService(
            repo, qdrant, embedding_router, alias_name=qdrant_alias
        )
        try:
            index_lifecycle.bootstrap_current(default_embedder)
        except Exception:
            physical = qdrant.active_collection_name()
            concurrent = repo.get_index_version_by_collection(qdrant_alias, physical)
            if concurrent is None:
                raise
            logger.info(
                "IndexVersion bootstrap completed concurrently: alias=%s collection=%s version=%s",
                qdrant_alias,
                physical,
                concurrent.index_version_id,
            )
            index_lifecycle.bootstrap_current(default_embedder)
        ensure_index_cutover_gate(index_lifecycle)
        embedder = ActiveIndexEmbedder(
            repo,
            embedding_router,
            qdrant_alias,
            default_embedder,
            vector_store=qdrant,
        )

    if embedder.dimension != qdrant.dim:
        raise RuntimeError(
            "Active embedding contract does not match vector index: "
            f"embedder={embedder.dimension}, vector={qdrant.dim}. "
            "Register the active embedding contract or reconcile the index alias before serving traffic."
        )

    if is_production():
        unsafe = []
        if isinstance(repo, InMemoryRepo):
            unsafe.append("InMemoryRepo")
        if isinstance(qdrant, InMemoryQdrant):
            unsafe.append("InMemoryQdrant")
        try:
            active_raw_embedder = embedding_router.get(embedder.contract_id)
        except (KeyError, AttributeError):
            active_raw_embedder = default_embedder
        if isinstance(active_raw_embedder, HashEmbedder):
            unsafe.append("HashEmbedder")
        if unsafe:
            raise RuntimeError("Production services cannot use development fallbacks: " + ", ".join(unsafe))

    reranker = components.build("reranker", profile.reranker_provider)
    retriever = Retriever(repo, qdrant, embedder=embedder, reranker=reranker)

    sql_enabled = _env_flag("RAGBOT_SQL_TOOL_ENABLED", False)
    if sql_enabled:
        sql_dsn = (os.getenv("RAGBOT_SQL_DSN") or "").strip()
        if postgres_dsn and not sql_dsn and not is_production():
            sql_dsn = postgres_dsn
        if sql_dsn:
            allowed_schemas_raw = os.getenv("RAGBOT_SQL_ALLOWED_SCHEMAS", "")
            allowed_schemas = [s.strip() for s in allowed_schemas_raw.split(",") if s.strip()] or None
            sql_engine = PostgresSqlEngine(
                dsn=sql_dsn,
                allowed_schemas=allowed_schemas,
                limit=int(os.getenv("RAGBOT_SQL_LIMIT", "200")),
                timeout_ms=int(os.getenv("RAGBOT_SQL_TIMEOUT_MS", "3000")),
            )
        elif isinstance(repo, InMemoryRepo):
            sql_engine = SqlEngine(repo)
        else:
            raise RuntimeError("RAGBOT_SQL_TOOL_ENABLED=true requires RAGBOT_SQL_DSN or an in-memory development repository")
    else:
        sql_engine = DisabledSqlEngine()

    repo_root = os.getenv("CODE_REPO_ROOT", ".")
    code_search = CodeSearch(repo_roots={"default": repo_root})
    llm = build_model_router()

    logger.info("Resolved Ragbot runtime profile: %s", profile.as_public_dict())
    logger.info("Resolved model router: %s", llm.diagnostics())
    logger.info("Resolved embedding contracts: %s", embedding_router.public_metadata())
    services = AgentServices(
        repo=repo,
        qdrant=qdrant,
        retriever=retriever,
        sql_engine=sql_engine,
        code_search=code_search,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
    )
    setattr(services, "embedding_router", embedding_router)
    setattr(services, "index_lifecycle", index_lifecycle)
    return services
