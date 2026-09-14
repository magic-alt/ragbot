from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.api.app.storage.quality_support import ensure_quality_repository

from .contracts import (
    RagRun,
    model_contract_id,
    query_hash,
    reranker_contract_id,
    retrieval_contract_id,
)


class QualityRecorder:
    """Persist non-secret runtime lineage and evidence summaries.

    A minimal run record is always written when the repository supports the
    quality capability. `RAGBOT_TRACE_DETAIL_SAMPLE_RATE` controls only detailed
    trace payloads; request/contract lineage is never sampled away.
    """

    def __init__(self, repo: Any) -> None:
        self.repo = ensure_quality_repository(repo)
        self.store_content = _env_flag("RAGBOT_TRACE_STORE_CONTENT", False)
        self.detail_sample_rate = _float_env(
            "RAGBOT_TRACE_DETAIL_SAMPLE_RATE", 1.0, minimum=0.0, maximum=1.0
        )
        self.retention_days = _int_env("RAGBOT_TRACE_RETENTION_DAYS", 30, minimum=0)

    def record_search(
        self,
        *,
        services: Any,
        request_id: str,
        tenant_id: str,
        user_id: str,
        query: str,
        request: Any,
        response: Any = None,
        status: str = "completed",
        started_monotonic: Optional[float] = None,
        started_at: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> RagRun:
        completed = datetime.now(timezone.utc)
        sampled = _sample_request(request_id, self.detail_sample_rate)
        contracts = _runtime_contracts(services, request)
        trace_dict = response.trace.as_dict() if response is not None else {}
        chunks = list(getattr(response, "chunks", None) or [])
        retrieved = [_retrieved_summary(chunk, sampled=sampled) for chunk in chunks]
        citations = _citation_summaries(chunks)
        duration_ms = _duration_ms(started_monotonic)
        if trace_dict.get("stage_ms"):
            duration_ms = max(
                duration_ms,
                int(sum(float(item) for item in trace_dict.get("stage_ms", {}).values())),
            )
        run = RagRun(
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            run_kind="search",
            status=status,
            query_hash=query_hash(query),
            query_text=query if self.store_content else None,
            retrieval_plan=getattr(getattr(request, "plan", None), "value", None)
            or str(getattr(request, "plan", "") or "")
            or None,
            retrieval_contract_id=contracts.get("retrieval_contract_id"),
            embedding_contract_id=contracts.get("embedding_contract_id"),
            index_version_id=contracts.get("index_version_id"),
            reranker_contract_id=contracts.get("reranker_contract_id"),
            stage_latency_ms=dict(trace_dict.get("stage_ms") or {}),
            retrieved=retrieved,
            citations=citations,
            trace=trace_dict if sampled else {},
            trace_sampled=sampled,
            total_duration_ms=duration_ms,
            error_code=type(error).__name__ if error is not None else None,
            error_message=_safe_error(error),
            started_at=started_at or completed.isoformat(),
            completed_at=completed.isoformat(),
            expires_at=_expires_at(completed, self.retention_days),
        )
        self.repo.add_rag_run(run)
        return run

    def record_agent(
        self,
        *,
        services: Any,
        state: Any,
        original_query: str,
        trace_record: Any,
        status: str = "completed",
        started_at: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> RagRun:
        completed = datetime.now(timezone.utc)
        sampled = _sample_request(state.request_id, self.detail_sample_rate)
        retrieval = _agent_retrieval_summary(state)
        model_records = _model_usage_for_request(services, state.request_id)
        model_contracts = _model_contracts(model_records)
        usage = _aggregate_model_usage(model_records)
        trace_dict = trace_record.to_dict() if trace_record is not None else {}
        stage_latency = {
            str(span.get("name")): int(span.get("duration_ms") or 0)
            for span in trace_dict.get("spans", [])
        }
        retrieval_context = dict(retrieval.get("context") or {})
        stage_latency.update(
            {
                f"retrieval.{key}": value
                for key, value in dict(retrieval_context.get("stage_ms") or {}).items()
            }
        )
        final = getattr(state, "final", None)
        citations = [asdict(item) for item in (getattr(final, "citations", None) or [])]
        run = RagRun(
            request_id=state.request_id,
            trace_id=getattr(trace_record, "trace_id", None),
            tenant_id=state.tenant_id,
            user_id=state.user_id,
            run_kind="agent",
            status=status,
            query_hash=query_hash(original_query),
            query_text=original_query if self.store_content else None,
            route=getattr(state, "route", None),
            retrieval_plan=retrieval.get("retrieval_plan"),
            retrieval_contract_id=retrieval.get("retrieval_contract_id"),
            embedding_contract_id=retrieval.get("embedding_contract_id"),
            index_version_id=retrieval.get("index_version_id"),
            reranker_contract_id=retrieval.get("reranker_contract_id"),
            model_contracts=model_contracts,
            stage_latency_ms=stage_latency,
            retrieved=list(retrieval.get("retrieved") or []),
            citations=citations,
            usage=usage,
            trace=(
                {
                    "agent": trace_dict,
                    "retrieval": retrieval_context,
                }
                if sampled
                else {}
            ),
            trace_sampled=sampled,
            total_duration_ms=int(getattr(trace_record, "total_duration_ms", 0) or 0),
            error_code=type(error).__name__ if error is not None else None,
            error_message=_safe_error(error),
            started_at=started_at or completed.isoformat(),
            completed_at=completed.isoformat(),
            expires_at=_expires_at(completed, self.retention_days),
        )
        self.repo.add_rag_run(run)
        return run


def _runtime_contracts(services: Any, request: Any) -> dict[str, Optional[str]]:
    embedder = getattr(services, "embedder", None)
    embedding_id = str(getattr(embedder, "contract_id", "") or "") or None
    reranker = getattr(services, "reranker", None)
    plan = getattr(getattr(request, "plan", None), "value", None) or str(
        getattr(request, "plan", "") or ""
    )
    index_id = None
    representations: dict[str, Any] = {}
    repo = getattr(services, "repo", None)
    qdrant = getattr(services, "qdrant", None)
    alias = getattr(qdrant, "alias_name", None)
    selected_index = str(getattr(request, "index_version_id", "") or "").strip() or None
    version = None
    if selected_index:
        get_version = getattr(repo, "get_index_version", None)
        if callable(get_version):
            try:
                version = get_version(selected_index)
            except Exception:
                version = None
    elif alias:
        getter = getattr(repo, "get_active_index_version", None)
        if callable(getter):
            try:
                version = getter(alias)
            except Exception:
                version = None
    if version is not None:
        index_id = str(getattr(version, "index_version_id", "") or "") or None
        embedding_id = (
            str(getattr(version, "embedding_contract_id", "") or "") or embedding_id
        )
        if embedding_id:
            representations["dense"] = embedding_id
        vector_schema = dict(getattr(version, "vector_schema", None) or {})
        sparse = vector_schema.get("sparse")
        if isinstance(sparse, dict) and sparse.get("contract_id"):
            representations["sparse"] = str(sparse["contract_id"])
    elif embedding_id:
        representations["dense"] = embedding_id

    retrieval_id = retrieval_contract_id(
        plan=plan,
        top_k=int(getattr(request, "top_k", 20)),
        candidate_pool=getattr(request, "candidate_pool", None),
        rerank=bool(getattr(request, "rerank", True)),
        diversity=bool(getattr(request, "diversity", False)),
        representation_contracts=representations,
        index_version_id=selected_index,
    )
    return {
        "retrieval_contract_id": retrieval_id,
        "embedding_contract_id": embedding_id,
        "index_version_id": index_id,
        "reranker_contract_id": reranker_contract_id(reranker),
    }


def _retrieved_summary(chunk: Any, *, sampled: bool) -> dict[str, Any]:
    metadata = dict(getattr(chunk, "metadata", None) or {})
    trace = metadata.get("_retrieval") if isinstance(metadata, dict) else None
    summary: dict[str, Any] = {
        "chunk_id": getattr(chunk, "chunk_id", None),
        "doc_id": getattr(chunk, "doc_id", None),
        "score": float(getattr(chunk, "score", 0.0) or 0.0),
        "citations": list(getattr(chunk, "citations", None) or []),
    }
    if isinstance(trace, dict):
        summary["final_rank"] = trace.get("final_rank")
        summary["rerank_score"] = trace.get("rerank_score")
        summary["fusion_score"] = trace.get("fusion_score")
        if sampled:
            summary["candidate_trace"] = {
                key: trace.get(key)
                for key in ("dense", "vector", "lexical", "qdrant_dense_sparse")
                if trace.get(key) is not None
            }
    return summary


def _citation_summaries(chunks: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chunk in chunks:
        for citation in list(getattr(chunk, "citations", None) or []):
            output.append(
                {
                    "citation_id": str(citation),
                    "chunk_id": getattr(chunk, "chunk_id", None),
                    "doc_id": getattr(chunk, "doc_id", None),
                }
            )
    return output


def _agent_retrieval_summary(state: Any) -> dict[str, Any]:
    for call in reversed(list(getattr(state, "tool_calls", None) or [])):
        if getattr(call, "name", None) != "retrieve" or not getattr(call, "ok", False):
            continue
        preview = dict(getattr(call, "result_preview", None) or {})
        quality = preview.get("quality")
        if isinstance(quality, dict):
            return dict(quality)
    return {}


def _model_usage_for_request(services: Any, request_id: str) -> list[Any]:
    llm = getattr(services, "llm", None)
    tracker = getattr(llm, "cost_tracker", None)
    getter = getattr(tracker, "records_for_request", None)
    if callable(getter):
        return list(getter(request_id))
    return []


def _model_contracts(records: list[Any]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        provider = str(getattr(record, "provider", "unknown"))
        model = str(getattr(record, "model", "unknown"))
        contract_id = model_contract_id(provider, model)
        item = output.setdefault(
            contract_id,
            {
                "contract_id": contract_id,
                "provider": provider,
                "model": model,
                "tasks": [],
            },
        )
        task = str(getattr(record, "task", "default"))
        if task not in item["tasks"]:
            item["tasks"].append(task)
    return list(output.values())


def _aggregate_model_usage(records: list[Any]) -> dict[str, Any]:
    return {
        "calls": len(records),
        "input_tokens": sum(int(getattr(item, "input_tokens", 0) or 0) for item in records),
        "output_tokens": sum(int(getattr(item, "output_tokens", 0) or 0) for item in records),
        "cached_input_tokens": sum(
            int(getattr(item, "cached_input_tokens", 0) or 0) for item in records
        ),
        "reasoning_tokens": sum(
            int(getattr(item, "reasoning_tokens", 0) or 0) for item in records
        ),
        "total_tokens": sum(int(getattr(item, "total_tokens", 0) or 0) for item in records),
        "estimated_cost_usd": round(
            sum(float(getattr(item, "estimated_cost_usd", 0.0) or 0.0) for item in records),
            8,
        ),
    }


def _duration_ms(started_monotonic: Optional[float]) -> int:
    if started_monotonic is None:
        return 0
    return max(0, int((time.monotonic() - started_monotonic) * 1000))


def _sample_request(request_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    import hashlib

    bucket = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def _expires_at(completed: datetime, retention_days: int) -> Optional[str]:
    if retention_days <= 0:
        return None
    return (completed + timedelta(days=retention_days)).isoformat()


def _safe_error(error: Optional[BaseException]) -> Optional[str]:
    if error is None:
        return None
    text = str(error).replace("\n", " ").strip()
    return text[:500]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value
