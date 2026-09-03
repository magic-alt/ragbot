#!/usr/bin/env python3
"""Evaluate a running Ragbot knowledge base and write JSON/Markdown/HTML reports.

The script intentionally tests the live HTTP API so it measures the same index,
embedding model, vector store and retrieval stack that users exercise from the
CLI/Admin UI.

Examples:
    python3 scripts/rag_eval.py eval/datasets/pdf_retrieval_smoke.json \
      --tenant engineering --open

    python3 scripts/rag_eval.py eval/datasets/pdf_retrieval_smoke.json \
      --tenant engineering --with-answers --fail-on-threshold
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "tmp" / "ragbot-runtime.json"
DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_REPORT_DIR = ROOT / "reports" / "rag-eval"


@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    labeled: bool
    retrieval_pass: Optional[bool]
    first_relevant_rank: Optional[int]
    reciprocal_rank_at_10: Optional[float]
    recall_at_10: Optional[float]
    search_ms: float
    answer_ms: Optional[float]
    answer_pass: Optional[bool]
    answer: Optional[str]
    answer_confidence: Optional[str]
    citation_count: Optional[int]
    diagnostics: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    warnings: List[str]
    error: Optional[str] = None


def _runtime_server() -> str:
    if not STATE_FILE.exists():
        return DEFAULT_SERVER
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SERVER
    return str(data.get("server") or DEFAULT_SERVER).rstrip("/")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * percentile
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _slug(value: str) -> str:
    chars = []
    for char in value.lower().strip():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "rag-eval"


def _http_json(
    server: str,
    path: str,
    payload: Optional[Dict[str, Any]],
    *,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
        method = "POST"
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ragbot at {server}: {exc}") from exc


def _load_dataset(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Evaluation dataset must be a JSON object")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset requires a non-empty 'cases' array")
    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = str(case.get("id") or "").strip()
        query = str(case.get("query") or "").strip()
        if not case_id or not query:
            raise ValueError(f"cases[{index}] requires id and query")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
    return raw


def _case_filters(dataset: Dict[str, Any], case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    defaults = dataset.get("defaults") or {}
    filters = dict(defaults.get("filters") or {})
    filters.update(case.get("filters") or {})
    return filters or None


def _case_top_k(dataset: Dict[str, Any], case: Dict[str, Any], cli_top_k: Optional[int]) -> int:
    if cli_top_k:
        return cli_top_k
    if case.get("top_k"):
        return int(case["top_k"])
    return int((dataset.get("defaults") or {}).get("top_k") or 10)


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _relevance_config(case: Dict[str, Any]) -> Dict[str, Any]:
    raw = case.get("relevance") or {}
    return raw if isinstance(raw, dict) else {}


def _is_labeled(case: Dict[str, Any]) -> bool:
    rel = _relevance_config(case)
    selectors = (
        rel.get("expected_chunk_ids"),
        rel.get("doc_ids"),
        rel.get("pages"),
        rel.get("path_contains"),
        rel.get("all_terms"),
        rel.get("any_terms"),
    )
    return any(bool(item) for item in selectors)


def _chunk_relevant(chunk: Dict[str, Any], case: Dict[str, Any]) -> bool:
    rel = _relevance_config(case)
    expected_chunk_ids = {str(v) for v in rel.get("expected_chunk_ids") or []}
    if expected_chunk_ids:
        return str(chunk.get("chunk_id") or "") in expected_chunk_ids

    metadata = chunk.get("metadata") or {}
    text = _norm_text(chunk.get("text"))

    doc_ids = {str(v) for v in rel.get("doc_ids") or []}
    if doc_ids and str(chunk.get("doc_id") or "") not in doc_ids:
        return False

    pages = {str(v) for v in rel.get("pages") or []}
    if pages and str(metadata.get("page")) not in pages:
        return False

    path_needles = [_norm_text(v) for v in rel.get("path_contains") or []]
    if path_needles:
        path = _norm_text(metadata.get("path") or metadata.get("url"))
        if not any(needle in path for needle in path_needles):
            return False

    all_terms = [_norm_text(v) for v in rel.get("all_terms") or []]
    if all_terms and not all(term in text for term in all_terms):
        return False

    any_terms = [_norm_text(v) for v in rel.get("any_terms") or []]
    if any_terms and not any(term in text for term in any_terms):
        return False

    return bool(doc_ids or pages or path_needles or all_terms or any_terms)


def _first_relevant_rank(chunks: Sequence[Dict[str, Any]], case: Dict[str, Any]) -> Optional[int]:
    for rank, chunk in enumerate(chunks, 1):
        if _chunk_relevant(chunk, case):
            return rank
    return None


def _exact_recall_at_10(chunks: Sequence[Dict[str, Any]], case: Dict[str, Any]) -> Optional[float]:
    rel = _relevance_config(case)
    expected_ids = {str(v) for v in rel.get("expected_chunk_ids") or []}
    if expected_ids:
        found = {str(chunk.get("chunk_id") or "") for chunk in chunks[:10]}
        return len(expected_ids & found) / len(expected_ids)

    expected_pages = {str(v) for v in rel.get("pages") or []}
    if expected_pages:
        found_pages = {
            str((chunk.get("metadata") or {}).get("page"))
            for chunk in chunks[:10]
        }
        return len(expected_pages & found_pages) / len(expected_pages)
    return None


def _answer_expectation(case: Dict[str, Any]) -> Dict[str, Any]:
    raw = case.get("answer") or {}
    return raw if isinstance(raw, dict) else {}


def _evaluate_answer(answer_payload: Dict[str, Any], case: Dict[str, Any]) -> Optional[bool]:
    expected = _answer_expectation(case)
    if not expected:
        return None
    answer = _norm_text(answer_payload.get("answer"))
    checks: List[bool] = []
    all_terms = [_norm_text(v) for v in expected.get("contains_all") or []]
    if all_terms:
        checks.append(all(term in answer for term in all_terms))
    any_terms = [_norm_text(v) for v in expected.get("contains_any") or []]
    if any_terms:
        checks.append(any(term in answer for term in any_terms))
    min_citations = expected.get("min_citations")
    if min_citations is not None:
        checks.append(len(answer_payload.get("citations") or []) >= int(min_citations))
    return all(checks) if checks else None


def _search_payload(
    query: str,
    *,
    tenant: str,
    user: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query,
        "tenant_id": tenant,
        "user_id": user,
        "top_k": top_k,
        "filters": filters,
    }


def _chat_payload(query: str, *, tenant: str, user: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    constraints = None
    if filters:
        constraints = {
            key: value
            for key, value in {
                "source_types": filters.get("source_types"),
                "doc_ids": filters.get("doc_ids"),
                "tags": filters.get("tags"),
                "path_prefix": filters.get("path_prefix"),
                "url_prefix": filters.get("url_prefix"),
                "time_from": filters.get("time_from"),
                "time_to": filters.get("time_to"),
            }.items()
            if value is not None
        } or None
    return {
        "query": query,
        "tenant_id": tenant,
        "user_id": user,
        "constraints": constraints,
    }


def run_case(
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    *,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    timeout: float,
    cli_top_k: Optional[int],
    with_answers: bool,
) -> CaseResult:
    query = str(case["query"])
    case_id = str(case["id"])
    category = str(case.get("category") or "default")
    filters = _case_filters(dataset, case)
    top_k = _case_top_k(dataset, case, cli_top_k)
    labeled = _is_labeled(case)

    start = time.perf_counter()
    search = _http_json(
        server,
        "/search",
        _search_payload(query, tenant=tenant, user=user, top_k=top_k, filters=filters),
        api_key=api_key,
        timeout=timeout,
    )
    search_ms = (time.perf_counter() - start) * 1000.0
    chunks = list(search.get("chunks") or [])
    diagnostics = dict(search.get("diagnostics") or {})
    warnings = [str(v) for v in diagnostics.get("warnings") or []]

    first_rank = _first_relevant_rank(chunks, case) if labeled else None
    max_rank = int(_relevance_config(case).get("max_rank") or 5)
    retrieval_pass = (first_rank is not None and first_rank <= max_rank) if labeled else None
    reciprocal = (
        1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
    ) if labeled else None
    recall = _exact_recall_at_10(chunks, case) if labeled else None

    answer_payload: Optional[Dict[str, Any]] = None
    answer_ms: Optional[float] = None
    answer_pass: Optional[bool] = None
    if with_answers or bool(_answer_expectation(case)):
        answer_start = time.perf_counter()
        answer_payload = _http_json(
            server,
            "/chat",
            _chat_payload(query, tenant=tenant, user=user, filters=filters),
            api_key=api_key,
            timeout=max(timeout, 120.0),
        )
        answer_ms = (time.perf_counter() - answer_start) * 1000.0
        answer_pass = _evaluate_answer(answer_payload, case)

    return CaseResult(
        case_id=case_id,
        category=category,
        query=query,
        labeled=labeled,
        retrieval_pass=retrieval_pass,
        first_relevant_rank=first_rank,
        reciprocal_rank_at_10=reciprocal,
        recall_at_10=recall,
        search_ms=round(search_ms, 2),
        answer_ms=round(answer_ms, 2) if answer_ms is not None else None,
        answer_pass=answer_pass,
        answer=str(answer_payload.get("answer")) if answer_payload else None,
        answer_confidence=str(answer_payload.get("confidence")) if answer_payload else None,
        citation_count=len(answer_payload.get("citations") or []) if answer_payload else None,
        diagnostics=diagnostics,
        chunks=chunks,
        warnings=warnings,
    )


def _case_pass(result: CaseResult) -> Optional[bool]:
    checks = [value for value in (result.retrieval_pass, result.answer_pass) if value is not None]
    return all(checks) if checks else None


def _summary(results: Sequence[CaseResult]) -> Dict[str, Any]:
    labeled = [r for r in results if r.labeled and r.error is None]
    search_times = [r.search_ms for r in results if r.error is None]
    answer_times = [r.answer_ms for r in results if r.answer_ms is not None]
    reciprocal = [r.reciprocal_rank_at_10 for r in labeled if r.reciprocal_rank_at_10 is not None]
    exact_recall = [r.recall_at_10 for r in labeled if r.recall_at_10 is not None]
    passed_cases = [r for r in results if _case_pass(r) is True]
    failed_cases = [r for r in results if _case_pass(r) is False or r.error]
    evaluated_cases = [r for r in results if _case_pass(r) is not None or r.error]

    def hit_at(k: int) -> float:
        if not labeled:
            return 0.0
        return sum(1 for r in labeled if r.first_relevant_rank is not None and r.first_relevant_rank <= k) / len(labeled)

    categories: Dict[str, Dict[str, Any]] = {}
    for category in sorted({r.category for r in results}):
        items = [r for r in results if r.category == category]
        labeled_items = [r for r in items if r.labeled and not r.error]
        rr = [r.reciprocal_rank_at_10 for r in labeled_items if r.reciprocal_rank_at_10 is not None]
        categories[category] = {
            "cases": len(items),
            "labeled": len(labeled_items),
            "hit_at_5": round(
                sum(1 for r in labeled_items if r.first_relevant_rank is not None and r.first_relevant_rank <= 5) / len(labeled_items), 4
            ) if labeled_items else None,
            "mrr_at_10": round(statistics.fmean(rr), 4) if rr else None,
            "avg_search_ms": round(statistics.fmean([r.search_ms for r in items if not r.error]), 2)
            if any(not r.error for r in items) else None,
        }

    diagnostics = next((r.diagnostics for r in results if r.diagnostics), {})
    warnings = list(dict.fromkeys(w for r in results for w in r.warnings))
    return {
        "total_cases": len(results),
        "evaluated_cases": len(evaluated_cases),
        "labeled_retrieval_cases": len(labeled),
        "passed_cases": len(passed_cases),
        "failed_cases": len(failed_cases),
        "pass_rate": round(len(passed_cases) / len(evaluated_cases), 4) if evaluated_cases else None,
        "hit_at_1": round(hit_at(1), 4),
        "hit_at_3": round(hit_at(3), 4),
        "hit_at_5": round(hit_at(5), 4),
        "hit_at_10": round(hit_at(10), 4),
        "mrr_at_10": round(statistics.fmean(reciprocal), 4) if reciprocal else None,
        "recall_at_10": round(statistics.fmean(exact_recall), 4) if exact_recall else None,
        "exact_recall_cases": len(exact_recall),
        "avg_search_ms": round(statistics.fmean(search_times), 2) if search_times else None,
        "p50_search_ms": round(_percentile(search_times, 0.50), 2) if search_times else None,
        "p95_search_ms": round(_percentile(search_times, 0.95), 2) if search_times else None,
        "avg_answer_ms": round(statistics.fmean(answer_times), 2) if answer_times else None,
        "p95_answer_ms": round(_percentile(answer_times, 0.95), 2) if answer_times else None,
        "runtime": diagnostics,
        "warnings": warnings,
        "categories": categories,
    }


def _evaluate_thresholds(summary: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    def minimum(metric: str, key: str) -> None:
        if key not in thresholds:
            return
        actual = summary.get(metric)
        expected = float(thresholds[key])
        checks[key] = {
            "pass": actual is not None and float(actual) >= expected,
            "actual": actual,
            "expected": expected,
        }

    def maximum(metric: str, key: str) -> None:
        if key not in thresholds:
            return
        actual = summary.get(metric)
        expected = float(thresholds[key])
        checks[key] = {
            "pass": actual is not None and float(actual) <= expected,
            "actual": actual,
            "expected": expected,
        }

    minimum("pass_rate", "pass_rate_min")
    minimum("hit_at_1", "hit_at_1_min")
    minimum("hit_at_3", "hit_at_3_min")
    minimum("hit_at_5", "hit_at_5_min")
    minimum("hit_at_10", "hit_at_10_min")
    minimum("mrr_at_10", "mrr_at_10_min")
    minimum("recall_at_10", "recall_at_10_min")
    maximum("p95_search_ms", "p95_search_ms_max")
    maximum("p95_answer_ms", "p95_answer_ms_max")

    if thresholds.get("semantic_embedding_required"):
        actual = bool((summary.get("runtime") or {}).get("semantic_embedding"))
        checks["semantic_embedding_required"] = {
            "pass": actual,
            "actual": actual,
            "expected": True,
        }

    passed = all(item["pass"] for item in checks.values()) if checks else True
    return {"passed": passed, "checks": checks}


def _result_dict(result: CaseResult) -> Dict[str, Any]:
    chunks = []
    for rank, chunk in enumerate(result.chunks, 1):
        metadata = chunk.get("metadata") or {}
        trace = metadata.get("_retrieval") or {}
        chunks.append(
            {
                "rank": rank,
                "relevant": None if not result.labeled else None,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "score": chunk.get("score"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "path": metadata.get("path") or metadata.get("url"),
                "text": chunk.get("text"),
                "retrieval": trace,
            }
        )
    return {
        "case_id": result.case_id,
        "category": result.category,
        "query": result.query,
        "labeled": result.labeled,
        "case_pass": _case_pass(result),
        "retrieval_pass": result.retrieval_pass,
        "first_relevant_rank": result.first_relevant_rank,
        "reciprocal_rank_at_10": result.reciprocal_rank_at_10,
        "recall_at_10": result.recall_at_10,
        "search_ms": result.search_ms,
        "answer_ms": result.answer_ms,
        "answer_pass": result.answer_pass,
        "answer": result.answer,
        "answer_confidence": result.answer_confidence,
        "citation_count": result.citation_count,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "error": result.error,
        "chunks": chunks,
    }


def _fmt_percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def _markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    gate = report["gate"]
    runtime = summary.get("runtime") or {}
    lines = [
        f"# Ragbot RAG Evaluation — {report['name']}",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Server: `{report['server']}`  ",
        f"Tenant: `{report['tenant']}`",
        "",
        "## Executive summary",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Gate | {'PASS' if gate['passed'] else 'FAIL'} |",
        f"| Cases | {summary['total_cases']} |",
        f"| Pass rate | {_fmt_percent(summary['pass_rate'])} |",
        f"| Hit@1 | {_fmt_percent(summary['hit_at_1'])} |",
        f"| Hit@3 | {_fmt_percent(summary['hit_at_3'])} |",
        f"| Hit@5 | {_fmt_percent(summary['hit_at_5'])} |",
        f"| Hit@10 | {_fmt_percent(summary['hit_at_10'])} |",
        f"| MRR@10 | {summary['mrr_at_10'] if summary['mrr_at_10'] is not None else '-'} |",
        f"| Recall@10 | {summary['recall_at_10'] if summary['recall_at_10'] is not None else '-'} |",
        f"| Search p50 | {summary['p50_search_ms']} ms |",
        f"| Search p95 | {summary['p95_search_ms']} ms |",
        "",
        "## Runtime",
        "",
        f"- Embedding: `{runtime.get('embedding_model', '?')}` ({runtime.get('embedding_backend', '?')}, {runtime.get('embedding_dimension', '?')}D)",
        f"- Semantic embedding: `{runtime.get('semantic_embedding', '?')}`",
        f"- Vector store: `{runtime.get('vector_store', '?')}`",
        f"- Repository: `{runtime.get('repository', '?')}`",
        f"- Reranker: `{runtime.get('reranker', '?')}` / enabled=`{runtime.get('reranker_enabled', False)}`",
    ]
    for warning in summary.get("warnings") or []:
        lines.append(f"- ⚠️ {warning}")

    lines.extend(["", "## Thresholds", "", "| Check | Actual | Expected | Result |", "| --- | ---: | ---: | --- |"])
    if gate["checks"]:
        for key, item in gate["checks"].items():
            lines.append(f"| `{key}` | {item['actual']} | {item['expected']} | {'PASS' if item['pass'] else 'FAIL'} |")
    else:
        lines.append("| _none configured_ | - | - | PASS |")

    lines.extend(["", "## Cases", "", "| Case | Category | Rank | MRR@10 | Recall@10 | Search | Status |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for case in report["cases"]:
        status = "ERROR" if case["error"] else ("PASS" if case["case_pass"] is True else "FAIL" if case["case_pass"] is False else "OBSERVE")
        lines.append(
            f"| `{case['case_id']}` | {case['category']} | {case['first_relevant_rank'] or '-'} | "
            f"{case['reciprocal_rank_at_10'] if case['reciprocal_rank_at_10'] is not None else '-'} | "
            f"{case['recall_at_10'] if case['recall_at_10'] is not None else '-'} | {case['search_ms']} ms | {status} |"
        )

    for case in report["cases"]:
        lines.extend(["", f"### {case['case_id']}", "", f"> {case['query']}", ""])
        if case["error"]:
            lines.append(f"**Error:** `{case['error']}`")
            continue
        lines.append(f"First relevant rank: `{case['first_relevant_rank']}`; search latency: `{case['search_ms']} ms`.")
        if case["answer"] is not None:
            lines.extend(["", "**Answer**", "", case["answer"], ""])
        lines.extend(["", "Top results:", ""])
        for chunk in case["chunks"][:5]:
            text = " ".join(str(chunk.get("text") or "").split())[:240]
            trace = chunk.get("retrieval") or {}
            vector = trace.get("vector") or {}
            lexical = trace.get("lexical") or {}
            lines.append(
                f"- #{chunk['rank']} score=`{chunk.get('score')}` page=`{chunk.get('page')}` "
                f"vector=`{vector.get('rank')}/{vector.get('score')}` lexical=`{lexical.get('rank')}/{lexical.get('score')}` — {text}"
            )
    return "\n".join(lines) + "\n"


def _html_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    gate = report["gate"]
    runtime = summary.get("runtime") or {}
    rows = []
    details = []
    for case in report["cases"]:
        status = "error" if case["error"] else ("pass" if case["case_pass"] is True else "fail" if case["case_pass"] is False else "observe")
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(case['case_id'])}</code></td>"
            f"<td>{html.escape(case['category'])}</td>"
            f"<td>{case['first_relevant_rank'] or '-'}</td>"
            f"<td>{case['reciprocal_rank_at_10'] if case['reciprocal_rank_at_10'] is not None else '-'}</td>"
            f"<td>{case['search_ms']} ms</td>"
            f"<td><span class='badge {status}'>{status.upper()}</span></td>"
            "</tr>"
        )
        result_cards = []
        for chunk in case["chunks"][:5]:
            trace = chunk.get("retrieval") or {}
            vector = trace.get("vector") or {}
            lexical = trace.get("lexical") or {}
            text = html.escape(" ".join(str(chunk.get("text") or "").split())[:500])
            result_cards.append(
                "<div class='result'>"
                f"<b>#{chunk['rank']}</b> score={chunk.get('score')} page={chunk.get('page')} "
                f"<span class='muted'>vector={vector.get('rank')}/{vector.get('score')} · lexical={lexical.get('rank')}/{lexical.get('score')}</span>"
                f"<p>{text}</p></div>"
            )
        answer = ""
        if case["answer"] is not None:
            answer = f"<h4>Answer</h4><p>{html.escape(case['answer'])}</p><p class='muted'>confidence={html.escape(str(case['answer_confidence']))} · citations={case['citation_count']}</p>"
        details.append(
            "<details>"
            f"<summary><b>{html.escape(case['case_id'])}</b> — {html.escape(case['query'])}</summary>"
            f"<p>First relevant rank: {case['first_relevant_rank']} · Search: {case['search_ms']} ms</p>"
            f"{answer}{''.join(result_cards)}</details>"
        )

    warning_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in summary.get("warnings") or [])
    threshold_rows = "".join(
        f"<tr><td><code>{html.escape(key)}</code></td><td>{item['actual']}</td><td>{item['expected']}</td><td><span class='badge {'pass' if item['pass'] else 'fail'}'>{'PASS' if item['pass'] else 'FAIL'}</span></td></tr>"
        for key, item in gate["checks"].items()
    ) or "<tr><td colspan='4'>No thresholds configured</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Ragbot RAG Evaluation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f4f6f8;color:#17202a}}
main{{max-width:1200px;margin:0 auto;padding:32px}} h1,h2{{margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.card{{background:white;border:1px solid #dfe3e8;border-radius:10px;padding:16px}} .value{{font-size:28px;font-weight:700}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:10px;border-bottom:1px solid #e6e9ed;text-align:left}}
.badge{{padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700}} .pass{{background:#dff5e5;color:#176b34}} .fail,.error{{background:#fde2e1;color:#9f2923}} .observe{{background:#eef1f4;color:#5d6772}}
details{{background:white;border:1px solid #dfe3e8;border-radius:10px;padding:12px 16px;margin:12px 0}} summary{{cursor:pointer}}
.result{{border-top:1px solid #edf0f2;padding:10px 0}} .muted{{color:#66717d;font-size:13px}} code{{background:#eef1f4;padding:2px 5px;border-radius:4px}}
ul.warning{{background:#fff4d6;border:1px solid #f1cf70;border-radius:8px;padding:12px 30px}}
</style></head><body><main>
<h1>Ragbot RAG Evaluation</h1><p class='muted'>{html.escape(report['name'])} · {html.escape(report['generated_at'])} · tenant={html.escape(report['tenant'])}</p>
<div class='grid'>
<div class='card'><div class='muted'>Gate</div><div class='value'>{'PASS' if gate['passed'] else 'FAIL'}</div></div>
<div class='card'><div class='muted'>Hit@5</div><div class='value'>{_fmt_percent(summary['hit_at_5'])}</div></div>
<div class='card'><div class='muted'>MRR@10</div><div class='value'>{summary['mrr_at_10'] if summary['mrr_at_10'] is not None else '-'}</div></div>
<div class='card'><div class='muted'>P95 search</div><div class='value'>{summary['p95_search_ms']} ms</div></div>
<div class='card'><div class='muted'>Embedding</div><div class='value' style='font-size:18px'>{html.escape(str(runtime.get('embedding_model','?')))}</div></div>
</div>
<h2>Runtime</h2><div class='card'><p>backend=<code>{html.escape(str(runtime.get('embedding_backend','?')))}</code> · dimension=<code>{runtime.get('embedding_dimension','?')}</code> · semantic=<code>{runtime.get('semantic_embedding','?')}</code> · vector=<code>{html.escape(str(runtime.get('vector_store','?')))}</code> · reranker=<code>{html.escape(str(runtime.get('reranker','?')))}</code></p></div>
{f"<ul class='warning'>{warning_html}</ul>" if warning_html else ''}
<h2>Thresholds</h2><table><thead><tr><th>Check</th><th>Actual</th><th>Expected</th><th>Result</th></tr></thead><tbody>{threshold_rows}</tbody></table>
<h2>Cases</h2><table><thead><tr><th>Case</th><th>Category</th><th>Rank</th><th>MRR@10</th><th>Latency</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Case details</h2>{''.join(details)}
</main></body></html>"""


def _write_reports(report: Dict[str, Any], report_dir: Path) -> Dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = report_dir / f"{_slug(report['name'])}-{stamp}"
    paths = {
        "json": base.with_suffix(".json"),
        "markdown": base.with_suffix(".md"),
        "html": base.with_suffix(".html"),
    }
    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    paths["html"].write_text(_html_report(report), encoding="utf-8")
    latest = report_dir / "latest.html"
    latest.write_text(paths["html"].read_text(encoding="utf-8"), encoding="utf-8")
    paths["latest"] = latest
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a live Ragbot knowledge base and generate reports")
    parser.add_argument("dataset", help="JSON evaluation dataset")
    parser.add_argument("--server", help="Ragbot API URL; defaults to tmp/ragbot-runtime.json")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="rag-eval")
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--top-k", type=int, help="Override dataset top_k")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--with-answers", action="store_true", help="Also call /chat for every case")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--open", action="store_true", help="Open the generated HTML report")
    parser.add_argument("--fail-on-threshold", action="store_true", help="Return exit code 1 when configured thresholds fail")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    try:
        dataset = _load_dataset(dataset_path)
        server = (args.server or _runtime_server()).rstrip("/")
        health = _http_json(server, "/admin/ready", None, api_key=args.api_key, timeout=min(args.timeout, 10.0))
        if health.get("status") != "ready":
            raise RuntimeError(f"Ragbot is not ready: {health}")

        print(f"Dataset: {dataset.get('name') or dataset_path.name}")
        print(f"Server: {server}")
        print(f"Tenant: {args.tenant}")
        print(f"Cases: {len(dataset['cases'])}")

        results: List[CaseResult] = []
        for index, case in enumerate(dataset["cases"], 1):
            print(f"[{index}/{len(dataset['cases'])}] {case['id']}: {case['query']}")
            try:
                result = run_case(
                    dataset,
                    case,
                    server=server,
                    tenant=args.tenant,
                    user=args.user,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    cli_top_k=args.top_k,
                    with_answers=args.with_answers,
                )
                status = _case_pass(result)
                print(
                    f"  rank={result.first_relevant_rank or '-'} search={result.search_ms:.1f}ms "
                    f"status={'PASS' if status is True else 'FAIL' if status is False else 'OBSERVE'}"
                )
            except Exception as exc:
                result = CaseResult(
                    case_id=str(case["id"]), category=str(case.get("category") or "default"), query=str(case["query"]),
                    labeled=_is_labeled(case), retrieval_pass=False if _is_labeled(case) else None,
                    first_relevant_rank=None, reciprocal_rank_at_10=0.0 if _is_labeled(case) else None,
                    recall_at_10=None, search_ms=0.0, answer_ms=None, answer_pass=None, answer=None,
                    answer_confidence=None, citation_count=None, diagnostics={}, chunks=[], warnings=[], error=str(exc),
                )
                print(f"  ERROR: {exc}")
            results.append(result)

        summary = _summary(results)
        gate = _evaluate_thresholds(summary, dataset.get("thresholds") or {})
        report = {
            "schema_version": 1,
            "name": str(dataset.get("name") or dataset_path.stem),
            "description": str(dataset.get("description") or ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "server": server,
            "tenant": args.tenant,
            "dataset": str(dataset_path),
            "summary": summary,
            "gate": gate,
            "cases": [_result_dict(result) for result in results],
        }
        paths = _write_reports(report, Path(args.report_dir).resolve())

        print()
        print("Evaluation summary")
        print(f"  Hit@1:  {_fmt_percent(summary['hit_at_1'])}")
        print(f"  Hit@5:  {_fmt_percent(summary['hit_at_5'])}")
        print(f"  Hit@10: {_fmt_percent(summary['hit_at_10'])}")
        print(f"  MRR@10: {summary['mrr_at_10']}")
        print(f"  Recall@10: {summary['recall_at_10']}")
        print(f"  P95 search: {summary['p95_search_ms']} ms")
        print(f"  Gate: {'PASS' if gate['passed'] else 'FAIL'}")
        print()
        print(f"JSON: {paths['json']}")
        print(f"Markdown: {paths['markdown']}")
        print(f"HTML: {paths['html']}")
        print(f"Latest: {paths['latest']}")

        if args.open:
            webbrowser.open(paths["html"].as_uri())
        if args.fail_on_threshold and not gate["passed"]:
            return 1
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
