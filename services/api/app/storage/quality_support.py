from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime
from types import MethodType
from typing import Any, Mapping, Optional

from services.api.app.quality.contracts import EvaluationRun, PromotionDecision, RagRun


_RUN_JSON_FIELDS = {
    "model_contracts",
    "stage_latency_ms",
    "retrieved",
    "citations",
    "usage",
    "trace",
}
_EVAL_JSON_FIELDS = {
    "runtime_contracts",
    "config",
    "metrics",
    "latency",
    "cost",
    "artifacts",
}


def ensure_quality_repository(repo: Any) -> Any:
    """Attach the optional durable quality control-plane capability."""

    required = (
        "add_rag_run",
        "get_rag_run",
        "add_evaluation_run",
        "get_evaluation_run",
        "add_quality_feedback",
        "add_promotion_decision",
    )
    if all(callable(getattr(repo, name, None)) for name in required):
        return repo

    if hasattr(repo, "_pool"):
        repo.add_rag_run = MethodType(_pg_add_rag_run, repo)
        repo.get_rag_run = MethodType(_pg_get_rag_run, repo)
        repo.list_rag_runs = MethodType(_pg_list_rag_runs, repo)
        repo.purge_expired_rag_runs = MethodType(_pg_purge_expired_rag_runs, repo)
        repo.add_evaluation_run = MethodType(_pg_add_evaluation_run, repo)
        repo.get_evaluation_run = MethodType(_pg_get_evaluation_run, repo)
        repo.list_evaluation_runs = MethodType(_pg_list_evaluation_runs, repo)
        repo.add_quality_feedback = MethodType(_pg_add_quality_feedback, repo)
        repo.add_promotion_decision = MethodType(_pg_add_promotion_decision, repo)
        repo.get_promotion_decision = MethodType(_pg_get_promotion_decision, repo)
        return repo

    if not hasattr(repo, "_quality_lock"):
        repo._quality_lock = threading.Lock()
    if not hasattr(repo, "_rag_runs"):
        repo._rag_runs = {}
    if not hasattr(repo, "_evaluation_runs"):
        repo._evaluation_runs = {}
    if not hasattr(repo, "_quality_feedback"):
        repo._quality_feedback = {}
    if not hasattr(repo, "_promotion_decisions"):
        repo._promotion_decisions = {}
    repo.add_rag_run = MethodType(_mem_add_rag_run, repo)
    repo.get_rag_run = MethodType(_mem_get_rag_run, repo)
    repo.list_rag_runs = MethodType(_mem_list_rag_runs, repo)
    repo.purge_expired_rag_runs = MethodType(_mem_purge_expired_rag_runs, repo)
    repo.add_evaluation_run = MethodType(_mem_add_evaluation_run, repo)
    repo.get_evaluation_run = MethodType(_mem_get_evaluation_run, repo)
    repo.list_evaluation_runs = MethodType(_mem_list_evaluation_runs, repo)
    repo.add_quality_feedback = MethodType(_mem_add_quality_feedback, repo)
    repo.add_promotion_decision = MethodType(_mem_add_promotion_decision, repo)
    repo.get_promotion_decision = MethodType(_mem_get_promotion_decision, repo)
    return repo


def _mem_add_rag_run(self: Any, run: RagRun) -> None:
    with self._quality_lock:
        self._rag_runs[run.request_id] = run


def _mem_get_rag_run(self: Any, request_id: str) -> Optional[RagRun]:
    with self._quality_lock:
        return self._rag_runs.get(request_id)


def _mem_list_rag_runs(self: Any, tenant_id: Optional[str] = None, limit: int = 100) -> list[RagRun]:
    with self._quality_lock:
        values = list(self._rag_runs.values())
    if tenant_id is not None:
        values = [item for item in values if item.tenant_id == tenant_id]
    values.sort(key=lambda item: item.completed_at or "", reverse=True)
    return values[: max(1, int(limit))]


def _mem_purge_expired_rag_runs(self: Any, now: Optional[str] = None) -> int:
    cutoff = _parse_datetime(now) if now else datetime.utcnow()
    with self._quality_lock:
        stale = [
            request_id
            for request_id, item in self._rag_runs.items()
            if item.expires_at and _parse_datetime(item.expires_at) <= cutoff
        ]
        for request_id in stale:
            del self._rag_runs[request_id]
    return len(stale)


def _mem_add_evaluation_run(self: Any, run: EvaluationRun) -> None:
    with self._quality_lock:
        if run.evaluation_run_id in self._evaluation_runs:
            raise ValueError(f"EvaluationRun is immutable and already exists: {run.evaluation_run_id}")
        self._evaluation_runs[run.evaluation_run_id] = run


def _mem_get_evaluation_run(self: Any, evaluation_run_id: str) -> Optional[EvaluationRun]:
    with self._quality_lock:
        return self._evaluation_runs.get(evaluation_run_id)


def _mem_list_evaluation_runs(self: Any, limit: int = 100) -> list[EvaluationRun]:
    with self._quality_lock:
        values = list(self._evaluation_runs.values())
    values.sort(key=lambda item: item.created_at or "", reverse=True)
    return values[: max(1, int(limit))]


def _mem_add_quality_feedback(
    self: Any,
    *,
    feedback_id: str,
    request_id: str,
    tenant_id: str,
    user_id: str,
    feedback_type: str,
    comment: Optional[str] = None,
    citation_id: Optional[str] = None,
    rating: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    with self._quality_lock:
        self._quality_feedback[feedback_id] = {
            "feedback_id": feedback_id,
            "request_id": request_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "feedback_type": feedback_type,
            "comment": comment,
            "citation_id": citation_id,
            "rating": rating,
            "metadata": dict(metadata or {}),
        }


def _mem_add_promotion_decision(self: Any, decision: PromotionDecision) -> None:
    with self._quality_lock:
        if decision.promotion_decision_id in self._promotion_decisions:
            raise ValueError(f"Promotion decision already exists: {decision.promotion_decision_id}")
        self._promotion_decisions[decision.promotion_decision_id] = decision


def _mem_get_promotion_decision(self: Any, decision_id: str) -> Optional[PromotionDecision]:
    with self._quality_lock:
        return self._promotion_decisions.get(decision_id)


def _pg_add_rag_run(self: Any, run: RagRun) -> None:
    params = asdict(run)
    for key in _RUN_JSON_FIELDS:
        params[key] = self._jsonb(params[key])
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_runs (
                request_id, trace_id, tenant_id, user_id, run_kind, status,
                query_hash, query_text, route, retrieval_plan,
                retrieval_contract_id, embedding_contract_id, index_version_id,
                reranker_contract_id, model_contracts, stage_latency_ms,
                retrieved, citations, usage, trace, trace_sampled,
                total_duration_ms, error_code, error_message,
                started_at, completed_at, expires_at
            ) VALUES (
                %(request_id)s, %(trace_id)s, %(tenant_id)s, %(user_id)s,
                %(run_kind)s, %(status)s, %(query_hash)s, %(query_text)s,
                %(route)s, %(retrieval_plan)s, %(retrieval_contract_id)s,
                %(embedding_contract_id)s, %(index_version_id)s,
                %(reranker_contract_id)s, %(model_contracts)s,
                %(stage_latency_ms)s, %(retrieved)s, %(citations)s,
                %(usage)s, %(trace)s, %(trace_sampled)s,
                %(total_duration_ms)s, %(error_code)s, %(error_message)s,
                COALESCE(%(started_at)s, NOW()), COALESCE(%(completed_at)s, NOW()),
                %(expires_at)s
            )
            ON CONFLICT (request_id) DO UPDATE SET
                trace_id = EXCLUDED.trace_id,
                status = EXCLUDED.status,
                route = EXCLUDED.route,
                retrieval_plan = EXCLUDED.retrieval_plan,
                retrieval_contract_id = EXCLUDED.retrieval_contract_id,
                embedding_contract_id = EXCLUDED.embedding_contract_id,
                index_version_id = EXCLUDED.index_version_id,
                reranker_contract_id = EXCLUDED.reranker_contract_id,
                model_contracts = EXCLUDED.model_contracts,
                stage_latency_ms = EXCLUDED.stage_latency_ms,
                retrieved = EXCLUDED.retrieved,
                citations = EXCLUDED.citations,
                usage = EXCLUDED.usage,
                trace = EXCLUDED.trace,
                trace_sampled = EXCLUDED.trace_sampled,
                total_duration_ms = EXCLUDED.total_duration_ms,
                error_code = EXCLUDED.error_code,
                error_message = EXCLUDED.error_message,
                completed_at = EXCLUDED.completed_at,
                expires_at = EXCLUDED.expires_at
            """,
            params,
        )


def _pg_get_rag_run(self: Any, request_id: str) -> Optional[RagRun]:
    with self._pool.connection() as conn:
        row = conn.execute("SELECT * FROM rag_runs WHERE request_id = %s", (request_id,)).fetchone()
    return _row_to_rag_run(row) if row else None


def _pg_list_rag_runs(self: Any, tenant_id: Optional[str] = None, limit: int = 100) -> list[RagRun]:
    if tenant_id is None:
        sql = "SELECT * FROM rag_runs ORDER BY completed_at DESC LIMIT %s"
        params = (max(1, int(limit)),)
    else:
        sql = "SELECT * FROM rag_runs WHERE tenant_id = %s ORDER BY completed_at DESC LIMIT %s"
        params = (tenant_id, max(1, int(limit)))
    with self._pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_rag_run(row) for row in rows]


def _pg_purge_expired_rag_runs(self: Any, now: Optional[str] = None) -> int:
    with self._pool.connection() as conn:
        row = conn.execute(
            "DELETE FROM rag_runs WHERE expires_at IS NOT NULL AND expires_at <= COALESCE(%s::timestamptz, NOW()) RETURNING request_id",
            (now,),
        ).fetchall()
    return len(row)


def _pg_add_evaluation_run(self: Any, run: EvaluationRun) -> None:
    params = asdict(run)
    for key in _EVAL_JSON_FIELDS:
        params[key] = self._jsonb(params[key])
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_runs (
                evaluation_run_id, evaluation_contract_id, tenant_id,
                dataset_name, dataset_version, code_revision, candidate_ref,
                baseline_ref, runtime_contracts, config, metrics, latency,
                cost, artifacts, status, created_at, completed_at
            ) VALUES (
                %(evaluation_run_id)s, %(evaluation_contract_id)s, %(tenant_id)s,
                %(dataset_name)s, %(dataset_version)s, %(code_revision)s,
                %(candidate_ref)s, %(baseline_ref)s, %(runtime_contracts)s,
                %(config)s, %(metrics)s, %(latency)s, %(cost)s, %(artifacts)s,
                %(status)s, COALESCE(%(created_at)s, NOW()), %(completed_at)s
            )
            """,
            params,
        )


def _pg_get_evaluation_run(self: Any, evaluation_run_id: str) -> Optional[EvaluationRun]:
    with self._pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM evaluation_runs WHERE evaluation_run_id = %s",
            (evaluation_run_id,),
        ).fetchone()
    return _row_to_evaluation_run(row) if row else None


def _pg_list_evaluation_runs(self: Any, limit: int = 100) -> list[EvaluationRun]:
    with self._pool.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT %s",
            (max(1, int(limit)),),
        ).fetchall()
    return [_row_to_evaluation_run(row) for row in rows]


def _pg_add_quality_feedback(
    self: Any,
    *,
    feedback_id: str,
    request_id: str,
    tenant_id: str,
    user_id: str,
    feedback_type: str,
    comment: Optional[str] = None,
    citation_id: Optional[str] = None,
    rating: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                feedback_id, request_id, tenant_id, user_id, feedback_type,
                comment, citation_id, rating, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                feedback_id,
                request_id,
                tenant_id,
                user_id,
                feedback_type,
                comment,
                citation_id,
                rating,
                self._jsonb(dict(metadata or {})),
            ),
        )


def _pg_add_promotion_decision(self: Any, decision: PromotionDecision) -> None:
    params = asdict(decision)
    params["policy"] = self._jsonb(params["policy"])
    params["deltas"] = self._jsonb(params["deltas"])
    params["reasons"] = self._jsonb(params["reasons"])
    with self._pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO promotion_decisions (
                promotion_decision_id, baseline_evaluation_id,
                candidate_evaluation_id, decision, policy, deltas, reasons,
                created_at
            ) VALUES (
                %(promotion_decision_id)s, %(baseline_evaluation_id)s,
                %(candidate_evaluation_id)s, %(decision)s, %(policy)s,
                %(deltas)s, %(reasons)s, COALESCE(%(created_at)s, NOW())
            )
            """,
            params,
        )


def _pg_get_promotion_decision(self: Any, decision_id: str) -> Optional[PromotionDecision]:
    with self._pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM promotion_decisions WHERE promotion_decision_id = %s",
            (decision_id,),
        ).fetchone()
    return _row_to_promotion_decision(row) if row else None


def _row_to_rag_run(row: Any) -> RagRun:
    data = _row_dict(row, [
        "request_id", "trace_id", "tenant_id", "user_id", "run_kind", "status",
        "query_hash", "query_text", "route", "retrieval_plan", "retrieval_contract_id",
        "embedding_contract_id", "index_version_id", "reranker_contract_id",
        "model_contracts", "stage_latency_ms", "retrieved", "citations", "usage",
        "trace", "trace_sampled", "total_duration_ms", "error_code", "error_message",
        "started_at", "completed_at", "expires_at",
    ])
    for key in ("started_at", "completed_at", "expires_at"):
        data[key] = _iso(data.get(key))
    return RagRun(**data)


def _row_to_evaluation_run(row: Any) -> EvaluationRun:
    data = _row_dict(row, [
        "evaluation_run_id", "evaluation_contract_id", "tenant_id", "dataset_name",
        "dataset_version", "code_revision", "candidate_ref", "baseline_ref",
        "runtime_contracts", "config", "metrics", "latency", "cost", "artifacts",
        "status", "created_at", "completed_at",
    ])
    data["created_at"] = _iso(data.get("created_at"))
    data["completed_at"] = _iso(data.get("completed_at"))
    return EvaluationRun(**data)


def _row_to_promotion_decision(row: Any) -> PromotionDecision:
    data = _row_dict(row, [
        "promotion_decision_id", "baseline_evaluation_id", "candidate_evaluation_id",
        "decision", "policy", "deltas", "reasons", "created_at",
    ])
    data["created_at"] = _iso(data.get("created_at"))
    return PromotionDecision(**data)


def _row_dict(row: Any, columns: list[str]) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    return dict(zip(columns, row))


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
