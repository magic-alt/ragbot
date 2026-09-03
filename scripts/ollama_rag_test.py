#!/usr/bin/env python3
"""End-to-end local Ollama + Ragbot Qdrant smoke/evaluation helper.

This script intentionally uses only the Python standard library. It verifies:
1. the requested Ollama model is installed and callable;
2. Ragbot is ready and its vector store is healthy;
3. the query retrieves evidence through Ragbot's /search endpoint;
4. Ragbot's /chat endpoint produces a grounded answer from the configured LLM.

Typical use:
    python scripts/ollama_rag_test.py \
        --model qwen3.8:27b \
        "What does the indexed document say about the system architecture?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_RAGBOT_URL = "http://127.0.0.1:8000"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class ProbeError(RuntimeError):
    """Actionable local validation failure."""


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    body = None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ProbeError(f"HTTP {exc.code} from {url}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ProbeError(f"Cannot reach {url}: {exc.reason}") from None
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ProbeError(f"Non-JSON response from {url}: {raw[:500]}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"Unexpected JSON response from {url}: expected object")
    return value


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def _model_ids(payload: Dict[str, Any]) -> List[str]:
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            result.append(str(row["id"]))
    return result


def check_ollama_model(
    base_url: str,
    model: str,
    *,
    timeout: float,
    run_generation: bool = True,
    reasoning_effort: Optional[str] = "none",
) -> Dict[str, Any]:
    base_url = _normalize_base_url(base_url)
    started = time.perf_counter()
    models_payload = _request_json("GET", f"{base_url}/v1/models", timeout=timeout)
    models = _model_ids(models_payload)
    list_ms = round((time.perf_counter() - started) * 1000, 1)
    if model not in models:
        preview = ", ".join(models[:12]) or "<none>"
        raise ProbeError(
            f"Ollama model {model!r} is not installed. Available models: {preview}. "
            f"Run: ollama pull {model}"
        )

    generation_ms: Optional[float] = None
    content = ""
    if run_generation:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: RAGBOT_OLLAMA_OK",
                }
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": 32,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()
        result = _request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout=timeout,
        )
        generation_ms = round((time.perf_counter() - started) * 1000, 1)
        try:
            content = str(result["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProbeError("Ollama chat response did not contain choices[0].message.content") from exc
        if not content:
            raise ProbeError("Ollama model returned an empty completion")

    return {
        "url": base_url,
        "model": model,
        "available": True,
        "installed_model_count": len(models),
        "model_list_latency_ms": list_ms,
        "generation_latency_ms": generation_ms,
        "generation_preview": content[:160],
    }


def check_ragbot_ready(
    base_url: str,
    *,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    base_url = _normalize_base_url(base_url)
    started = time.perf_counter()
    ready = _request_json(
        "GET",
        f"{base_url}/admin/ready",
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    if ready.get("status") != "ready":
        raise ProbeError(f"Ragbot is not ready: {ready}")
    checks = ready.get("checks") or {}
    if isinstance(checks, dict) and checks.get("vector_store") is False:
        raise ProbeError("Ragbot vector store readiness check failed")
    return {"url": base_url, "latency_ms": elapsed_ms, **ready}


def ragbot_search(
    base_url: str,
    query: str,
    *,
    tenant: str,
    user: str,
    top_k: int,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "tenant_id": tenant,
        "user_id": user,
        "top_k": top_k,
    }
    started = time.perf_counter()
    result = _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/search",
        payload=payload,
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def ragbot_chat(
    base_url: str,
    query: str,
    *,
    tenant: str,
    user: str,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "tenant_id": tenant,
        "user_id": user,
        "stream": False,
    }
    started = time.perf_counter()
    result = _request_json(
        "POST",
        f"{_normalize_base_url(base_url)}/chat",
        payload=payload,
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def _chunk_ids(chunks: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("chunk_id"):
            result.add(str(chunk["chunk_id"]))
    return result


def _citation_chunk_ids(citations: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for citation in citations:
        if isinstance(citation, dict) and citation.get("chunk_id"):
            result.add(str(citation["chunk_id"]))
    return result


def evaluate(
    *,
    ragbot_url: str,
    ollama_url: str,
    model: str,
    query: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    top_k: int,
    min_retrieved: int,
    min_top_score: Optional[float],
    require_citations: bool,
    timeout: float,
    run_direct_generation: bool,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    ollama = check_ollama_model(
        ollama_url,
        model,
        timeout=timeout,
        run_generation=run_direct_generation,
        reasoning_effort=reasoning_effort,
    )
    ready = check_ragbot_ready(ragbot_url, api_key=api_key, timeout=timeout)
    search = ragbot_search(
        ragbot_url,
        query,
        tenant=tenant,
        user=user,
        top_k=top_k,
        api_key=api_key,
        timeout=timeout,
    )
    chat = ragbot_chat(
        ragbot_url,
        query,
        tenant=tenant,
        user=user,
        api_key=api_key,
        timeout=timeout,
    )

    chunks = search.get("chunks") if isinstance(search.get("chunks"), list) else []
    answer = str(chat.get("answer") or "").strip()
    citations = chat.get("citations") if isinstance(chat.get("citations"), list) else []
    retrieved_ids = _chunk_ids(chunks)
    cited_ids = _citation_chunk_ids(citations)
    overlap = retrieved_ids & cited_ids
    top_score = None
    if chunks and isinstance(chunks[0], dict):
        raw_score = chunks[0].get("score")
        if isinstance(raw_score, (int, float)):
            top_score = float(raw_score)

    gates = {
        "ollama_model_available": bool(ollama.get("available")),
        "ragbot_ready": ready.get("status") == "ready",
        "retrieval_count": len(chunks) >= min_retrieved,
        "answer_nonempty": bool(answer),
        "citations_present": bool(citations) if require_citations else True,
        "top_score": (
            True
            if min_top_score is None
            else top_score is not None and top_score >= min_top_score
        ),
    }
    passed = all(gates.values())

    return {
        "passed": passed,
        "query": query,
        "tenant": tenant,
        "user": user,
        "expected_llm": {"provider": "ollama", "model": model},
        "ollama": ollama,
        "ragbot": ready,
        "retrieval": {
            "count": len(chunks),
            "top_score": top_score,
            "latency_ms": search.get("latency_ms"),
            "diagnostics": search.get("diagnostics") or {},
            "chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "score": c.get("score"),
                    "text_preview": str(c.get("text") or "")[:240],
                }
                for c in chunks[:top_k]
                if isinstance(c, dict)
            ],
        },
        "generation": {
            "latency_ms": chat.get("latency_ms"),
            "answer_chars": len(answer),
            "answer_preview": answer[:1000],
            "citation_count": len(citations),
            "confidence": chat.get("confidence"),
        },
        "grounding": {
            "retrieved_chunk_ids": sorted(retrieved_ids),
            "cited_chunk_ids": sorted(cited_ids),
            "retrieved_citation_overlap": sorted(overlap),
            "overlap_ratio": (
                round(len(overlap) / len(cited_ids), 4) if cited_ids else None
            ),
        },
        "gates": gates,
    }


def _print_human(report: Dict[str, Any]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"Ollama + Ragbot RAG validation: {status}")
    print(
        f"  Ollama: {report['ollama']['model']} @ {report['ollama']['url']} "
        f"(generation={report['ollama'].get('generation_latency_ms')} ms)"
    )
    print(
        f"  Ragbot: {report['ragbot']['url']} "
        f"(vector_store={report['ragbot'].get('checks', {}).get('vector_store')})"
    )
    print(
        f"  Retrieval: {report['retrieval']['count']} chunks, "
        f"top_score={report['retrieval']['top_score']}, "
        f"latency={report['retrieval']['latency_ms']} ms"
    )
    print(
        f"  Generation: {report['generation']['answer_chars']} chars, "
        f"citations={report['generation']['citation_count']}, "
        f"latency={report['generation']['latency_ms']} ms"
    )
    print("  Gates:")
    for name, ok in report["gates"].items():
        print(f"    {'OK' if ok else 'FAIL'}  {name}")
    preview = report["generation"].get("answer_preview")
    if preview:
        print("\n--- Answer preview ---")
        print(preview)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local Ollama/Qwen against Ragbot's indexed vector knowledge"
    )
    parser.add_argument("query", nargs="+", help="A question answerable from indexed Ragbot documents")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3.8:27b"),
        help="Installed Ollama model (default: OLLAMA_MODEL or qwen3.8:27b)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        help=f"Host Ollama URL (default: OLLAMA_BASE_URL or {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--server",
        default=os.getenv("RAGBOT_SERVER", DEFAULT_RAGBOT_URL),
        help=f"Ragbot API URL (default: RAGBOT_SERVER or {DEFAULT_RAGBOT_URL})",
    )
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="ollama-rag-test")
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-retrieved", type=int, default=1)
    parser.add_argument("--min-top-score", type=float)
    parser.add_argument(
        "--allow-no-citations",
        action="store_true",
        help="Do not fail when the final answer has no citations",
    )
    parser.add_argument(
        "--skip-direct-generation",
        action="store_true",
        help="Only verify model presence; skip the direct Ollama generation probe",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max"],
        default=os.getenv("OLLAMA_REASONING_EFFORT", "none") or None,
        help="Reasoning effort for the direct Ollama probe (default: none)",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Write the full JSON report to this path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the human summary")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    query = " ".join(args.query).strip()
    try:
        report = evaluate(
            ragbot_url=args.server,
            ollama_url=args.ollama_url,
            model=args.model,
            query=query,
            tenant=args.tenant,
            user=args.user,
            api_key=args.api_key,
            top_k=max(1, args.top_k),
            min_retrieved=max(1, args.min_retrieved),
            min_top_score=args.min_top_score,
            require_citations=not args.allow_no_citations,
            timeout=args.timeout,
            run_direct_generation=not args.skip_direct_generation,
            reasoning_effort=args.reasoning_effort,
        )
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
