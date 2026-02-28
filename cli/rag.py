"""ragbot CLI client.

Provides command-line access to ragbot functionality:
  rag ask "query"      - Ask a question via the agent
  rag search "query"   - Search for code/docs
  rag patch "query"    - Generate a code patch
  rag ingest <path>    - Ingest a local directory or repo

Can target either a local ragbot instance or a remote API server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional


def _api_request(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Make an HTTP request to the ragbot API."""
    import requests

    url = f"{base_url.rstrip('/')}{path}"
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if method.upper() == "GET":
        resp = requests.get(url, headers=hdrs, timeout=60)
    else:
        resp = requests.post(url, json=payload, headers=hdrs, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _local_chat(
    query: str,
    tenant_id: str,
    user_id: str,
    constraints: Optional[Dict[str, Any]] = None,
    client_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a chat query using local services (no HTTP)."""
    from services.api.app.main import chat
    from services.api.app.agent.graph import build_default_services
    from services.api.app.agent.state import Constraints

    services = build_default_services()
    c = None
    if constraints:
        c = Constraints(**constraints)
    return asyncio.run(chat(query, tenant_id, user_id, services, c))


def cmd_ask(args: argparse.Namespace) -> None:
    """Handle 'rag ask' command."""
    query = " ".join(args.query)
    constraints = {}
    if args.repo:
        constraints["repo"] = args.repo
    if args.ref:
        constraints["ref"] = args.ref
    if args.tags:
        constraints["tags"] = args.tags

    if args.server:
        payload = {
            "query": query,
            "tenant_id": args.tenant,
            "user_id": args.user,
            "constraints": constraints or None,
        }
        headers = {}
        if args.api_key:
            headers["X-API-Key"] = args.api_key
        result = _api_request(args.server, "POST", "/chat", payload, headers)
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)

    _print_result(result, args.json)


def cmd_search(args: argparse.Namespace) -> None:
    """Handle 'rag search' command."""
    query = " ".join(args.query)

    if args.server:
        payload = {
            "query": query,
            "tenant_id": args.tenant,
            "user_id": args.user,
            "top_k": args.top_k,
        }
        headers = {}
        if args.api_key:
            headers["X-API-Key"] = args.api_key
        result = _api_request(args.server, "POST", "/search", payload, headers)
    else:
        from services.api.app.agent.graph import build_default_services
        services = build_default_services()
        from services.api.app.auth.acl import compute_security_scope
        scope = compute_security_scope(args.user, services.repo.list_policies())
        filters = {"tenant_id": args.tenant, "acl_hashes": scope}
        chunks = services.retriever.retrieve(query, filters, top_k=args.top_k)
        result = {
            "chunks": [{"chunk_id": c.chunk_id, "text": c.text, "score": c.score} for c in chunks],
            "total": len(chunks),
        }

    _print_result(result, args.json)


def cmd_patch(args: argparse.Namespace) -> None:
    """Handle 'rag patch' command."""
    query = " ".join(args.query)
    constraints = {"repo": args.repo} if args.repo else {}

    if args.server:
        payload = {
            "query": query,
            "tenant_id": args.tenant,
            "user_id": args.user,
            "constraints": constraints or None,
        }
        headers = {}
        if args.api_key:
            headers["X-API-Key"] = args.api_key
        result = _api_request(args.server, "POST", "/chat", payload, headers)
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)

    _print_result(result, args.json)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Handle 'rag ingest' command."""
    path = args.path

    if args.server:
        # Create source + trigger job via API
        headers = {}
        if args.api_key:
            headers["X-API-Key"] = args.api_key

        source_type = args.type or ("repo" if path.endswith(".git") or "github.com" in path else "local_fs")
        source_payload = {
            "tenant_id": args.tenant,
            "source_type": source_type,
            "name": args.name or path,
            "config": {"path": path},
        }
        source = _api_request(args.server, "POST", "/sources", source_payload, headers)
        print(f"Source created: {source.get('source_id')}")

        job_payload = {
            "source_id": source["source_id"],
            "tenant_id": args.tenant,
        }
        job = _api_request(args.server, "POST", "/ingest/jobs", job_payload, headers)
        print(f"Ingestion job started: {job.get('job_id')}")
        if args.json:
            print(json.dumps({"source": source, "job": job}, indent=2, ensure_ascii=False))
    else:
        from services.api.app.agent.graph import build_default_services
        from services.api.app.storage.models import Source
        from services.worker.pipeline import run_ingest_pipeline
        import uuid

        services = build_default_services()
        source_type = args.type or "local_fs"
        source = Source(
            source_id=uuid.uuid4().hex,
            tenant_id=args.tenant,
            source_type=source_type,
            name=args.name or path,
            config={"path": path},
        )
        services.repo.add_source(source)
        job = run_ingest_pipeline(source, services.repo, services.qdrant)
        print(f"Ingestion complete: status={job.status}, chunks={job.chunk_count}")
        if args.json:
            from dataclasses import asdict
            print(json.dumps(asdict(job), indent=2, ensure_ascii=False))


def _print_result(result: Dict[str, Any], as_json: bool) -> None:
    """Print a result dict, either as JSON or human-readable."""
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if "answer" in result:
        print(result["answer"])
        if result.get("citations"):
            print("\n--- Citations ---")
            for cite in result["citations"]:
                parts = []
                if cite.get("path"):
                    loc = cite["path"]
                    if cite.get("line_start"):
                        loc += f":{cite['line_start']}"
                        if cite.get("line_end"):
                            loc += f"-{cite['line_end']}"
                    parts.append(loc)
                if cite.get("url"):
                    parts.append(cite["url"])
                if cite.get("chunk_id"):
                    parts.append(f"chunk:{cite['chunk_id']}")
                print(f"  [{cite.get('kind', '?')}] {' | '.join(parts)}")
        if result.get("confidence"):
            print(f"\n[confidence: {result['confidence']}]")
    elif "chunks" in result:
        chunks = result["chunks"]
        print(f"Found {result.get('total', len(chunks))} results:\n")
        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("score", "?")
            text_preview = chunk.get("text", "")[:120].replace("\n", " ")
            print(f"  {i}. [{score}] {text_preview}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="ragbot CLI - RAG-powered code & document assistant",
    )
    parser.add_argument("--server", "-s", help="API server URL (e.g., http://localhost:8000). If omitted, runs locally.")
    parser.add_argument("--api-key", "-k", help="API key for authentication")
    parser.add_argument("--tenant", "-t", default="default", help="Tenant ID (default: 'default')")
    parser.add_argument("--user", "-u", default="cli-user", help="User ID (default: 'cli-user')")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # rag ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", nargs="+", help="The question to ask")
    ask_parser.add_argument("--repo", help="Restrict to a specific repo")
    ask_parser.add_argument("--ref", help="Git ref (branch/tag)")
    ask_parser.add_argument("--tags", nargs="*", help="Filter by tags")
    ask_parser.set_defaults(func=cmd_ask)

    # rag search
    search_parser = subparsers.add_parser("search", help="Search for code or documents")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    search_parser.set_defaults(func=cmd_search)

    # rag patch
    patch_parser = subparsers.add_parser("patch", help="Generate a code patch")
    patch_parser.add_argument("query", nargs="+", help="Patch instruction")
    patch_parser.add_argument("--repo", help="Target repo")
    patch_parser.set_defaults(func=cmd_patch)

    # rag ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a local directory or repo")
    ingest_parser.add_argument("path", help="Path to directory or repo URL")
    ingest_parser.add_argument("--type", choices=["local_fs", "repo", "pdf", "web"], help="Source type (auto-detected if omitted)")
    ingest_parser.add_argument("--name", help="Source name")
    ingest_parser.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
