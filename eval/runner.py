"""Evaluation runner for ragbot.

Executes evaluation datasets against the agent and produces
structured results with pass/fail verdicts and failure analysis.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .datasets import EvalCase, EvalResult

logger = logging.getLogger(__name__)


def compute_mrr_at_k(expected: List[str], retrieved: List[str], k: int = 10) -> float:
    """Compute MRR@k: 1/rank of first relevant item in top-k, 0 if not found."""
    if not expected:
        return 0.0
    expected_set = set(expected)
    for i, rid in enumerate(retrieved[:k]):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(expected: List[str], retrieved: List[str], k: int = 10) -> float:
    """Compute Recall@k: fraction of expected items found in top-k."""
    if not expected:
        return 0.0
    expected_set = set(expected)
    found = sum(1 for rid in retrieved[:k] if rid in expected_set)
    return found / len(expected_set)


async def run_eval_case(
    case: EvalCase,
    services=None,
) -> EvalResult:
    """Run a single evaluation case and check expectations."""
    from services.api.app.agent.graph import AgentServices, build_default_services, run_agent
    from services.api.app.agent.nodes.code import CodeSearch
    from services.api.app.agent.nodes.sql import SqlEngine
    from services.api.app.agent.state import Constraints
    from services.api.app.storage.models import TableData

    services = services or build_default_services()

    # Setup tables if provided
    if case.setup_tables:
        for table_def in case.setup_tables:
            table = TableData(
                name=table_def["name"],
                columns=table_def["columns"],
                rows=table_def.get("rows", []),
            )
            services.repo.register_table(table)

    # Setup files if provided
    if case.setup_files:
        services.code_search = CodeSearch(
            repo_roots={},
            in_memory_files=case.setup_files,
        )

    constraints = None
    if case.constraints:
        constraints = Constraints(**case.constraints)

    result = EvalResult(case_id=case.case_id, category=case.category, passed=True)

    start = time.monotonic()
    try:
        state = await run_agent(
            query=case.query,
            tenant_id=case.tenant_id,
            user_id=case.user_id,
            services=services,
            constraints=constraints,
        )
        result.duration_ms = int((time.monotonic() - start) * 1000)
        result.actual_route = state.route or ""
        result.actual_evidence_count = len(state.evidence)

        if state.final:
            result.actual_answer = state.final.answer
            result.actual_confidence = state.final.confidence
            result.actual_citation_count = len(state.final.citations)

        # Collect retrieved chunk IDs from evidence
        result.retrieved_chunk_ids = [
            e.chunk_id for e in state.evidence if hasattr(e, "chunk_id") and e.chunk_id
        ]

        # Compute MRR@10 and Recall@10 if expected_chunk_ids are provided
        if case.expected_chunk_ids:
            result.mrr_at_10 = compute_mrr_at_k(case.expected_chunk_ids, result.retrieved_chunk_ids, k=10)
            result.recall_at_10 = compute_recall_at_k(case.expected_chunk_ids, result.retrieved_chunk_ids, k=10)

        # Run checks
        result.checks = _run_checks(case, result)
        result.passed = all(result.checks.values()) if result.checks else True

        # Failure analysis
        if not result.passed:
            result.failure_category = _analyze_failure(case, result, state)

    except Exception as exc:
        result.duration_ms = int((time.monotonic() - start) * 1000)
        result.passed = False
        result.error = str(exc)
        result.failure_category = "error"

    return result


def _run_checks(case: EvalCase, result: EvalResult) -> Dict[str, bool]:
    """Run all applicable checks on the result."""
    checks: Dict[str, bool] = {}

    if case.expected_answer_contains:
        for keyword in case.expected_answer_contains:
            key = f"answer_contains_{keyword}"
            checks[key] = keyword.lower() in result.actual_answer.lower()

    if case.expected_answer_not_contains:
        for keyword in case.expected_answer_not_contains:
            key = f"answer_not_contains_{keyword}"
            checks[key] = keyword.lower() not in result.actual_answer.lower()

    if case.expected_route:
        checks["route"] = result.actual_route == case.expected_route

    if case.expected_confidence:
        checks["confidence"] = result.actual_confidence == case.expected_confidence

    if case.expected_min_citations > 0:
        checks["min_citations"] = result.actual_citation_count >= case.expected_min_citations

    if case.expected_min_evidence > 0:
        checks["min_evidence"] = result.actual_evidence_count >= case.expected_min_evidence

    return checks


def _analyze_failure(case: EvalCase, result: EvalResult, state) -> str:
    """Determine the failure category: bad_retrieval, bad_synthesis, or bad_tool."""
    # Check if route was wrong
    if case.expected_route and result.actual_route != case.expected_route:
        return "bad_routing"

    # Check evidence count
    if case.expected_min_evidence > 0 and result.actual_evidence_count < case.expected_min_evidence:
        # Had tool calls but not enough evidence
        tool_failures = sum(1 for tc in state.tool_calls if not tc.ok)
        if tool_failures > 0:
            return "bad_tool"
        return "bad_retrieval"

    # Had evidence but answer was wrong
    if result.actual_evidence_count > 0:
        return "bad_synthesis"

    return "bad_retrieval"


async def run_eval_suite(
    cases: List[EvalCase],
    services=None,
    category_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
) -> List[EvalResult]:
    """Run a full evaluation suite."""
    from services.api.app.agent.graph import build_default_services

    filtered = cases
    if category_filter:
        filtered = [c for c in filtered if c.category == category_filter]
    if tag_filter:
        filtered = [c for c in filtered if tag_filter in c.tags]

    results: List[EvalResult] = []
    for i, case in enumerate(filtered, 1):
        logger.info("Running eval %d/%d: %s (%s)", i, len(filtered), case.case_id, case.category)
        # Fresh services per case to avoid cross-contamination
        svc = services or build_default_services()
        result = await run_eval_case(case, svc)
        results.append(result)
        status = "PASS" if result.passed else f"FAIL ({result.failure_category})"
        logger.info("  %s: %s (%.0fms)", case.case_id, status, result.duration_ms)

    return results


def summarize_results(results: List[EvalResult]) -> Dict[str, Any]:
    """Produce a summary report from evaluation results."""
    if not results:
        return {"total": 0, "passed": 0, "failed": 0}

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # By category
    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "failed": 0}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1

    # Failure categories
    failure_categories: Dict[str, int] = {}
    for r in results:
        if not r.passed and r.failure_category:
            failure_categories[r.failure_category] = (
                failure_categories.get(r.failure_category, 0) + 1
            )

    # Average duration
    avg_ms = sum(r.duration_ms for r in results) / total

    # MRR@10 and Recall@10 averages (only for cases that have retrieval metrics)
    mrr_results = [r for r in results if r.mrr_at_10 > 0 or r.recall_at_10 > 0]
    avg_mrr_at_10 = round(sum(r.mrr_at_10 for r in mrr_results) / len(mrr_results), 4) if mrr_results else 0.0
    avg_recall_at_10 = round(sum(r.recall_at_10 for r in mrr_results) / len(mrr_results), 4) if mrr_results else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4),
        "avg_duration_ms": round(avg_ms, 1),
        "avg_mrr_at_10": avg_mrr_at_10,
        "avg_recall_at_10": avg_recall_at_10,
        "retrieval_eval_count": len(mrr_results),
        "by_category": by_category,
        "failure_categories": failure_categories,
        "failed_cases": [
            {"case_id": r.case_id, "failure": r.failure_category, "checks": r.checks}
            for r in results if not r.passed
        ],
    }
