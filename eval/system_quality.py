"""Self-contained live quality gate for a running Ragbot deployment.

This module deliberately exercises the public HTTP surface and production
pipeline. It uploads a deterministic PDF, waits for ingestion, probes all
retrieval modes, validates agent answers/citations, and emits machine-readable
quality metrics.
"""
from __future__ import annotations

import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


DEFAULT_SERVER = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    query: str
    sentinel: str
    category: str
    semantic_only: bool = False


@dataclass
class ProbeResult:
    case_id: str
    mode: str
    rank: Optional[int]
    reciprocal_rank: float
    latency_ms: float
    diagnostics: dict[str, Any]
    rerank_requested: bool
    error: Optional[str] = None


CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "exact-sentinel",
        "quartz-harbor-417",
        "quartz-harbor-417",
        "lexical",
    ),
    QualityCase(
        "semantic-service-interval",
        "What is the maintenance interval for the auxiliary power cell?",
        "amber-cell-918",
        "semantic",
        semantic_only=True,
    ),
    QualityCase(
        "semantic-thermal-calibration",
        "For how long can the actuator calibration be trusted after heat cycling?",
        "thermal-orbit-362",
        "semantic",
        semantic_only=True,
    ),
    QualityCase(
        "semantic-network-recovery",
        "What must happen before motion is enabled again after a fieldbus timing fault?",
        "fieldbus-lantern-744",
        "semantic",
        semantic_only=True,
    ),
    QualityCase(
        "cross-lingual-emergency-stop",
        "急停解除后，伺服系统重新允许运动之前需要完成什么检查？",
        "safety-comet-531",
        "cross_lingual",
        semantic_only=True,
    ),
)


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "semantic_embedding_required": False,
        "hybrid_hit_at_5_min": 0.70,
        "hybrid_mrr_at_10_min": 0.45,
        "p95_search_ms_max": 3000.0,
        "answer_pass_rate_min": 1.0,
    },
    "standard": {
        "semantic_embedding_required": True,
        "vector_semantic_hit_at_5_min": 0.75,
        "hybrid_hit_at_5_min": 0.85,
        "hybrid_mrr_at_10_min": 0.65,
        "p95_search_ms_max": 2000.0,
        "answer_pass_rate_min": 1.0,
    },
    "strict": {
        "semantic_embedding_required": True,
        "vector_semantic_hit_at_5_min": 0.90,
        "hybrid_hit_at_5_min": 0.92,
        "hybrid_mrr_at_10_min": 0.75,
        "p95_search_ms_max": 1200.0,
        "answer_pass_rate_min": 1.0,
    },
}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * percentile
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _headers(api_key: Optional[str], *, content_type: Optional[str] = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _request_json(
    server: str,
    method: str,
    path: str,
    *,
    api_key: Optional[str],
    payload: Optional[dict[str, Any]] = None,
    raw_body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    if payload is not None and raw_body is not None:
        raise ValueError("payload and raw_body are mutually exclusive")
    body = raw_body
    resolved_type = content_type
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        resolved_type = "application/json"
    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=body,
        headers=_headers(api_key, content_type=resolved_type),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ragbot at {server}: {exc}") from exc


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_fixture_pdf() -> bytes:
    """Build a small ASCII-only PDF without external libraries."""
    pages = [
        [
            "Ragbot automated quality corpus.",
            "Marker quartz-harbor-417 identifies the exact lexical control page.",
            "This page exists to prove deterministic PDF parsing and exact-term retrieval.",
        ],
        [
            "Marker amber-cell-918 identifies the auxiliary energy maintenance page.",
            "The backup battery must be replaced every eighteen months during scheduled service.",
            "The replacement interval is independent of normal servo operating hours.",
        ],
        [
            "Marker thermal-orbit-362 identifies the thermal calibration page.",
            "After a thermal cycle, actuator calibration remains valid for thirty-six hours.",
            "A fresh calibration is required when that validity window expires.",
        ],
        [
            "Marker fieldbus-lantern-744 identifies the network recovery page.",
            "Following a fieldbus timing fault, the controller must re-establish distributed clock synchronization.",
            "Motion may be enabled only after synchronization is stable and drive state is verified.",
        ],
        [
            "Marker safety-comet-531 identifies the emergency stop recovery page.",
            "After emergency-stop release, the servo system must verify the safety chain and drive state before motion is re-enabled.",
            "The check prevents unexpected movement during recovery.",
        ],
    ]

    objects: list[bytes] = []
    page_ids: list[int] = []
    content_ids: list[int] = []
    next_id = 4
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_id, content_id, lines in zip(page_ids, content_ids, pages):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        commands = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
        for index, line in enumerate(lines):
            if index:
                commands.append("T*")
            commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        content_obj = (
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        objects.extend([page_obj, content_obj])

    output = bytearray(b"%PDF-1.4\n%ragbot-quality-gate\n")
    offsets = [0]
    for object_id, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def _upload_fixture(
    server: str,
    *,
    api_key: Optional[str],
    tenant: str,
    timeout: float,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "tenant_id": tenant,
            "filename": "ragbot-system-quality.pdf",
            "name": f"ragbot-system-quality-{int(time.time())}",
            "tag": "ragbot-quality-gate",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    )
    return _request_json(
        server,
        "POST",
        f"/ingest/upload/pdf?{query}",
        api_key=api_key,
        raw_body=build_fixture_pdf(),
        content_type="application/pdf",
        timeout=timeout,
    )


def _wait_job(
    server: str,
    job_id: str,
    *,
    api_key: Optional[str],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _request_json(
            server,
            "GET",
            f"/ingest/jobs/{job_id}",
            api_key=api_key,
            timeout=10.0,
        )
        status = str(last.get("status") or "")
        if status == "completed":
            return last
        if status in {"failed", "dead_lettered", "cancelled"}:
            raise RuntimeError(
                f"Ingestion job {job_id} ended as {status}: {last.get('error')}"
            )
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for ingestion job {job_id}; last={last}")


def _search(
    server: str,
    *,
    api_key: Optional[str],
    tenant: str,
    user: str,
    query: str,
    mode: str,
    doc_id: Optional[str],
    top_k: int,
    rerank: bool,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    filters = {"doc_ids": [doc_id]} if doc_id else {"tags": ["ragbot-quality-gate"]}
    payload = {
        "query": query,
        "tenant_id": tenant,
        "user_id": user,
        "top_k": top_k,
        "mode": mode,
        "candidate_pool": max(20, top_k * 4),
        "rerank": rerank,
        "explain": True,
        "filters": filters,
    }
    start = time.perf_counter()
    result = _request_json(
        server,
        "POST",
        "/search",
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )
    return result, (time.perf_counter() - start) * 1000.0


def _first_rank(chunks: Sequence[dict[str, Any]], sentinel: str) -> Optional[int]:
    needle = sentinel.casefold()
    for rank, chunk in enumerate(chunks, 1):
        if needle in str(chunk.get("text") or "").casefold():
            return rank
    return None


def _probe_mode(
    server: str,
    *,
    api_key: Optional[str],
    tenant: str,
    user: str,
    mode: str,
    doc_id: Optional[str],
    top_k: int,
    rerank: bool,
    timeout: float,
    repetitions: int,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for case in CASES:
        for _ in range(max(1, repetitions)):
            try:
                response, elapsed = _search(
                    server,
                    api_key=api_key,
                    tenant=tenant,
                    user=user,
                    query=case.query,
                    mode=mode,
                    doc_id=doc_id,
                    top_k=top_k,
                    rerank=rerank,
                    timeout=timeout,
                )
                rank = _first_rank(list(response.get("chunks") or []), case.sentinel)
                results.append(
                    ProbeResult(
                        case.case_id,
                        mode,
                        rank,
                        (1.0 / rank) if rank and rank <= 10 else 0.0,
                        round(elapsed, 2),
                        dict(response.get("diagnostics") or {}),
                        rerank,
                    )
                )
            except Exception as exc:
                results.append(
                    ProbeResult(
                        case.case_id,
                        mode,
                        None,
                        0.0,
                        0.0,
                        {},
                        rerank,
                        str(exc),
                    )
                )
    return results


def _mode_metrics(results: Sequence[ProbeResult], mode: str) -> dict[str, Any]:
    mode_results = [item for item in results if item.mode == mode]
    valid = [item for item in mode_results if not item.error]
    by_case: dict[str, list[ProbeResult]] = {}
    for item in valid:
        by_case.setdefault(item.case_id, []).append(item)

    def hit_at(k: int, *, semantic_only: bool = False) -> float:
        wanted = [case for case in CASES if not semantic_only or case.semantic_only]
        if not wanted:
            return 0.0
        hits = 0
        for case in wanted:
            runs = by_case.get(case.case_id, [])
            if runs and all(run.rank is not None and run.rank <= k for run in runs):
                hits += 1
        return hits / len(wanted)

    case_mrr: list[float] = []
    for case in CASES:
        runs = by_case.get(case.case_id, [])
        if runs:
            case_mrr.append(statistics.fmean(run.reciprocal_rank for run in runs))
        else:
            case_mrr.append(0.0)
    latencies = [item.latency_ms for item in valid]
    return {
        "requests": len(mode_results),
        "errors": len(mode_results) - len(valid),
        "hit_at_1": round(hit_at(1), 4),
        "hit_at_5": round(hit_at(5), 4),
        "semantic_hit_at_5": round(hit_at(5, semantic_only=True), 4),
        "mrr_at_10": round(statistics.fmean(case_mrr), 4) if case_mrr else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50), 2) if latencies else None,
        "p95_ms": round(_percentile(latencies, 0.95), 2) if latencies else None,
    }


def _answer_probe(
    server: str,
    *,
    api_key: Optional[str],
    tenant: str,
    user: str,
    doc_id: Optional[str],
    timeout: float,
) -> dict[str, Any]:
    constraints = {"doc_ids": [doc_id]} if doc_id else {"tags": ["ragbot-quality-gate"]}
    cases = (
        (
            "thermal-answer",
            "How long is calibration valid after a thermal cycle?",
            ("thirty-six", "36"),
        ),
        (
            "battery-answer",
            "How often should the backup battery be replaced?",
            ("eighteen", "18"),
        ),
    )
    items: list[dict[str, Any]] = []
    for case_id, query, alternatives in cases:
        start = time.perf_counter()
        try:
            response = _request_json(
                server,
                "POST",
                "/chat",
                api_key=api_key,
                payload={
                    "query": query,
                    "tenant_id": tenant,
                    "user_id": user,
                    "constraints": constraints,
                },
                timeout=max(timeout, 120.0),
            )
            latency = (time.perf_counter() - start) * 1000.0
            answer = str(response.get("answer") or "")
            citations = list(response.get("citations") or [])
            correct = any(token.casefold() in answer.casefold() for token in alternatives)
            passed = bool(answer.strip()) and correct and len(citations) >= 1
            items.append(
                {
                    "case_id": case_id,
                    "passed": passed,
                    "latency_ms": round(latency, 2),
                    "citation_count": len(citations),
                    "confidence": response.get("confidence"),
                    "answer": answer,
                }
            )
        except Exception as exc:
            items.append({"case_id": case_id, "passed": False, "error": str(exc)})
    pass_rate = sum(1 for item in items if item.get("passed")) / len(items)
    return {"pass_rate": round(pass_rate, 4), "cases": items}


def _runtime_diagnostics(results: Iterable[ProbeResult]) -> dict[str, Any]:
    for item in results:
        if item.diagnostics:
            return dict(item.diagnostics)
    return {}


def _check(name: str, actual: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "passed": bool(passed),
    }


def evaluate_gate(summary: dict[str, Any], profile: str) -> dict[str, Any]:
    thresholds = dict(PROFILES[profile])
    checks: list[dict[str, Any]] = []
    runtime = summary.get("runtime") or {}
    hybrid = (summary.get("retrieval") or {}).get("hybrid") or {}
    vector = (summary.get("retrieval") or {}).get("vector") or {}
    answer = summary.get("answer") or {}

    if thresholds.get("semantic_embedding_required"):
        actual = bool(runtime.get("semantic_embedding"))
        checks.append(_check("semantic_embedding", actual, True, actual))
    if "vector_semantic_hit_at_5_min" in thresholds:
        expected = float(thresholds["vector_semantic_hit_at_5_min"])
        actual = float(vector.get("semantic_hit_at_5") or 0.0)
        checks.append(
            _check(
                "vector_semantic_hit_at_5",
                actual,
                f">={expected}",
                actual >= expected,
            )
        )

    expected = float(thresholds["hybrid_hit_at_5_min"])
    actual = float(hybrid.get("hit_at_5") or 0.0)
    checks.append(_check("hybrid_hit_at_5", actual, f">={expected}", actual >= expected))

    expected = float(thresholds["hybrid_mrr_at_10_min"])
    actual = float(hybrid.get("mrr_at_10") or 0.0)
    checks.append(
        _check("hybrid_mrr_at_10", actual, f">={expected}", actual >= expected)
    )

    p95_values = [
        float(metrics["p95_ms"])
        for metrics in (summary.get("retrieval") or {}).values()
        if metrics.get("p95_ms") is not None
    ]
    actual_p95 = max(p95_values) if p95_values else float("inf")
    expected_p95 = float(thresholds["p95_search_ms_max"])
    checks.append(
        _check(
            "search_p95_ms",
            round(actual_p95, 2),
            f"<={expected_p95}",
            actual_p95 <= expected_p95,
        )
    )

    expected_answer = float(thresholds["answer_pass_rate_min"])
    actual_answer = float(answer.get("pass_rate") or 0.0)
    checks.append(
        _check(
            "answer_pass_rate",
            actual_answer,
            f">={expected_answer}",
            actual_answer >= expected_answer,
        )
    )

    ingestion = summary.get("ingestion") or {}
    counts_ok = (
        int(ingestion.get("doc_count") or 0) >= 1
        and int(ingestion.get("chunk_count") or 0) >= 1
    )
    checks.append(
        _check(
            "ingestion_counts",
            {
                "documents": ingestion.get("doc_count"),
                "chunks": ingestion.get("chunk_count"),
            },
            "documents>=1, chunks>=1",
            counts_ok,
        )
    )

    all_errors = sum(
        int(metrics.get("errors") or 0)
        for metrics in (summary.get("retrieval") or {}).values()
    )
    checks.append(_check("retrieval_errors", all_errors, 0, all_errors == 0))
    return {
        "profile": profile,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "thresholds": thresholds,
    }


def run_live_gate(
    *,
    server: str = DEFAULT_SERVER,
    api_key: Optional[str] = None,
    tenant: str = "default",
    user: str = "ragbot-quality-gate",
    profile: str = "standard",
    timeout: float = 180.0,
    top_k: int = 10,
    repetitions: int = 1,
    rerank: bool = True,
    chunk_size: int = 420,
    chunk_overlap: int = 40,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}")
    ready = _request_json(
        server,
        "GET",
        "/admin/ready",
        api_key=api_key,
        timeout=10.0,
    )
    if ready.get("status") != "ready":
        raise RuntimeError(f"Ragbot is not ready: {ready}")
    runtime_identity = _request_json(
        server,
        "GET",
        "/admin/runtime",
        api_key=api_key,
        timeout=10.0,
    )
    upload = _upload_fixture(
        server,
        api_key=api_key,
        tenant=tenant,
        timeout=timeout,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    job_id = str(upload.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"PDF upload did not return job_id: {upload}")
    job = _wait_job(server, job_id, api_key=api_key, timeout=timeout)

    discovery, _ = _search(
        server,
        api_key=api_key,
        tenant=tenant,
        user=user,
        query="quartz-harbor-417",
        mode="hybrid",
        doc_id=None,
        top_k=20,
        rerank=False,
        timeout=timeout,
    )
    doc_id = next(
        (
            str(chunk.get("doc_id"))
            for chunk in discovery.get("chunks") or []
            if "quartz-harbor-417" in str(chunk.get("text") or "").casefold()
        ),
        None,
    )
    if not doc_id:
        raise RuntimeError(
            "Ingestion completed but the fixture sentinel could not be retrieved"
        )

    probes: list[ProbeResult] = []
    for mode in ("lexical", "vector", "hybrid"):
        probes.extend(
            _probe_mode(
                server,
                api_key=api_key,
                tenant=tenant,
                user=user,
                mode=mode,
                doc_id=doc_id,
                top_k=top_k,
                rerank=rerank if mode == "hybrid" else False,
                timeout=timeout,
                repetitions=repetitions,
            )
        )

    runtime = _runtime_diagnostics(probes)
    retrieval = {
        mode: _mode_metrics(probes, mode)
        for mode in ("lexical", "vector", "hybrid")
    }
    answer = _answer_probe(
        server,
        api_key=api_key,
        tenant=tenant,
        user=user,
        doc_id=doc_id,
        timeout=timeout,
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "server": server,
        "tenant": tenant,
        "runtime_identity": runtime_identity,
        "ingestion": {
            "source_id": upload.get("source_id"),
            "job_id": job_id,
            "doc_id": doc_id,
            "status": job.get("status"),
            "doc_count": job.get("doc_count"),
            "chunk_count": job.get("chunk_count"),
            "stats": job.get("stats") or {},
            "upload_sha256": upload.get("sha256"),
            "upload_size_bytes": upload.get("size_bytes"),
        },
        "runtime": runtime,
        "retrieval": retrieval,
        "answer": answer,
        "probes": [asdict(item) for item in probes],
    }
    summary["gate"] = evaluate_gate(summary, profile)
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    gate = report["gate"]
    runtime = report.get("runtime") or {}
    ingestion = report.get("ingestion") or {}
    lines = [
        "# Ragbot automated system quality gate",
        "",
        f"- Result: **{'PASS' if gate['passed'] else 'FAIL'}**",
        f"- Profile: `{gate['profile']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Server: `{report['server']}`",
        f"- Tenant: `{report['tenant']}`",
        f"- Embedding: `{runtime.get('embedding_model', '?')}` / "
        f"`{runtime.get('embedding_backend', '?')}` / "
        f"semantic=`{runtime.get('semantic_embedding', '?')}`",
        f"- Ingestion: status=`{ingestion.get('status')}`, "
        f"documents=`{ingestion.get('doc_count')}`, chunks=`{ingestion.get('chunk_count')}`",
        "",
        "## Retrieval",
        "",
        "| Mode | Hit@1 | Hit@5 | Semantic Hit@5 | MRR@10 | p95 | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("lexical", "vector", "hybrid"):
        metrics = report["retrieval"][mode]
        lines.append(
            f"| {mode} | {metrics['hit_at_1']:.2f} | {metrics['hit_at_5']:.2f} | "
            f"{metrics['semantic_hit_at_5']:.2f} | {metrics['mrr_at_10']:.2f} | "
            f"{metrics['p95_ms']} ms | {metrics['errors']} |"
        )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            "| Check | Actual | Expected | Result |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in gate["checks"]:
        lines.append(
            f"| `{item['name']}` | `{item['actual']}` | `{item['expected']}` | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This self-contained corpus validates the production RAG path and provides a "
            "repeatable regression signal. It does not replace a domain Golden Dataset "
            "built from your own documents and real user questions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"system-quality-{stamp}.json"
    md_path = report_dir / f"system-quality-{stamp}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(markdown_report(report), encoding="utf-8")
    latest_json = report_dir / "latest-system-quality.json"
    latest_md = report_dir / "latest-system-quality.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": md_path,
        "latest_json": latest_json,
        "latest_markdown": latest_md,
    }
