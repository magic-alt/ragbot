#!/usr/bin/env python3
"""Explainable Ragbot retrieval smoke test.

This helper is intentionally separate from answer generation: it shows whether
vector and lexical retrieval found the expected chunks before an LLM can hide a
retrieval problem behind a plausible answer.

Examples:
    python scripts/search_test.py "What is the difference between LoRA and QLoRA?" --tenant engineering
    python scripts/search_test.py "大模型部署时如何减少显存占用？" --tenant engineering --top-k 10
    python scripts/search_test.py --query-file eval/my_queries.txt --tenant engineering
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "tmp" / "ragbot-runtime.json"
DEFAULT_SERVER = "http://127.0.0.1:8000"


def _runtime_server() -> str:
    if not STATE_FILE.exists():
        return DEFAULT_SERVER
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SERVER
    return str(data.get("server") or DEFAULT_SERVER)


def _queries(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    if args.query:
        values.append(" ".join(args.query).strip())
    if args.query_file:
        for line in Path(args.query_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line)
    return [value for value in values if value]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def _request(server: str, args: argparse.Namespace, query: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if args.source_type:
        filters["source_types"] = args.source_type
    if args.doc_id:
        filters["doc_ids"] = args.doc_id
    if args.tag:
        filters["tags"] = args.tag
    payload = {
        "query": query,
        "tenant_id": args.tenant,
        "user_id": args.user,
        "top_k": args.top_k,
        "filters": filters or None,
    }
    headers = {"Content-Type": "application/json"}
    api_key = args.api_key or os.getenv("RAGBOT_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    response = requests.post(
        f"{server.rstrip('/')}/search",
        json=payload,
        headers=headers,
        timeout=args.timeout,
    )
    response.raise_for_status()
    return response.json()


def _print_result(query: str, result: Dict[str, Any]) -> None:
    diagnostics = result.get("diagnostics") or {}
    print("=" * 88)
    print(f"QUERY: {query}")
    print(
        "Runtime: "
        f"embedding={diagnostics.get('embedding_model', '?')} "
        f"({diagnostics.get('embedding_backend', '?')}, {diagnostics.get('embedding_dimension', '?')}D), "
        f"semantic={diagnostics.get('semantic_embedding', '?')}, "
        f"vector={diagnostics.get('vector_store', '?')}, "
        f"repo={diagnostics.get('repository', '?')}, "
        f"reranker={'on' if diagnostics.get('reranker_enabled') else 'off'}"
    )
    for warning in diagnostics.get("warnings") or []:
        print(f"WARNING: {warning}")

    chunks = result.get("chunks") or []
    vector_supported = 0
    lexical_supported = 0
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        trace = metadata.get("_retrieval") or {}
        vector = trace.get("vector")
        lexical = trace.get("lexical")
        if vector:
            vector_supported += 1
        if lexical:
            lexical_supported += 1
        page = metadata.get("page")
        section = metadata.get("section")
        text = " ".join(str(chunk.get("text") or "").split())
        print()
        print(
            f"#{index} final={_fmt(chunk.get('score'))} "
            f"page={page if page is not None else '-'} section={section or '-'} "
            f"chunk={str(chunk.get('chunk_id') or '')[:18]}"
        )
        print(
            "   "
            f"vector={('#' + str(vector.get('rank')) + '/' + _fmt(vector.get('score'))) if vector else '-'}  "
            f"lexical={('#' + str(lexical.get('rank')) + '/' + _fmt(lexical.get('score'))) if lexical else '-'}  "
            f"rrf={_fmt(trace.get('rrf_score'))}  rerank={_fmt(trace.get('rerank_score'))}"
        )
        print(f"   {text[:320]}")

    print()
    print(
        f"Summary: results={len(chunks)}, vector-supported={vector_supported}, "
        f"lexical-supported={lexical_supported}, request={str(result.get('request_id') or '')[:12]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explainable retrieval tests against a running Ragbot API"
    )
    parser.add_argument("query", nargs="*", help="Search query; quote it when convenient")
    parser.add_argument("--query-file", help="UTF-8 text file with one query per line")
    parser.add_argument("--server", default=None, help="API URL; defaults to tmp/ragbot-runtime.json")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="search-test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-type", action="append", choices=["pdf", "local_fs", "web", "repo", "s3", "gdrive", "notion", "confluence"])
    parser.add_argument("--doc-id", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true", help="Print raw JSON results")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queries = _queries(args)
    if not queries:
        print("ERROR: provide a query or --query-file", file=sys.stderr)
        return 2
    server = args.server or _runtime_server()
    try:
        for query in queries:
            result = _request(server, args, query)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                _print_result(query, result)
    except (OSError, requests.RequestException, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
