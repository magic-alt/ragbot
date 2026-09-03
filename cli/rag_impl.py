"""ragbot CLI client.

Product-oriented commands:
  rag ingest <location>          - Create/reuse a Source and ingest it
  rag import <manifest.json>     - Batch-create a knowledge base from a manifest
  rag doctor                     - Check API/dependency readiness
  rag ask "query"                - Ask a question via the agent
  rag search "query"             - Search indexed knowledge
  rag patch "query"              - Generate a code patch

The CLI can target either a local ragbot instance or a remote API server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _api_request(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120,
) -> Dict[str, Any]:
    import requests

    url = f"{base_url.rstrip('/')}{path}"
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if method.upper() == "GET":
        resp = requests.get(url, headers=hdrs, timeout=timeout)
    else:
        resp = requests.post(url, json=payload, headers=hdrs, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _local_chat(
    query: str,
    tenant_id: str,
    user_id: str,
    constraints: Optional[Dict[str, Any]] = None,
    client_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.api.app.main import chat
    from services.api.app.agent.graph import build_default_services
    from services.api.app.agent.state import Constraints

    services = build_default_services()
    c = Constraints(**constraints) if constraints else None
    return asyncio.run(chat(query, tenant_id, user_id, services, c))


def _wait_for_job(
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
    poll_interval: float,
    quiet: bool = False,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    previous_status: Optional[str] = None
    while True:
        job = _api_request(server, "GET", f"/ingest/jobs/{job_id}", headers=headers, timeout=60)
        status = str(job.get("status", "unknown"))
        if not quiet and status != previous_status:
            details = []
            if job.get("doc_count") is not None:
                details.append(f"docs={job.get('doc_count', 0)}")
            if job.get("chunk_count") is not None:
                details.append(f"chunks={job.get('chunk_count', 0)}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"Ingestion {job_id}: {status}{suffix}")
            previous_status = status
        if status == "completed":
            return job
        if status == "failed":
            raise RuntimeError(job.get("error") or f"Ingestion job failed: {job_id}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for ingestion job {job_id} after {timeout:.0f}s")
        time.sleep(max(0.1, poll_interval))


def _ingest_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build non-secret connector config; credential values are never CLI args."""
    config: Dict[str, Any] = {}
    for attr, key in (
        ("ref", "ref"),
        ("credential_ref", "credential_ref"),
        ("credential_type", "credential_type"),
        ("base_url", "base_url"),
        ("email", "email"),
        ("auth_type", "auth_type"),
        ("root_page_id", "root_page_id"),
        ("notion_version", "notion_version"),
    ):
        value = getattr(args, attr, None)
        if value:
            config[key] = value
    if getattr(args, "chunk_size", None):
        config["chunk_size"] = args.chunk_size
    if getattr(args, "chunk_overlap", None) is not None:
        config["chunk_overlap"] = args.chunk_overlap
    if getattr(args, "max_file_bytes", None):
        config["max_file_bytes"] = args.max_file_bytes
    if getattr(args, "no_recursive", False):
        config["recursive"] = False
    raw_json = getattr(args, "config_json", None)
    if raw_json:
        extra = json.loads(raw_json)
        if not isinstance(extra, dict):
            raise ValueError("--config-json must decode to a JSON object")
        config.update(extra)
    return config


def _normalize_manifest(raw: Any, default_tenant: str) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(raw, list):
        return default_tenant, raw
    if not isinstance(raw, dict):
        raise ValueError("Manifest must be a JSON object or array")
    tenant_id = str(raw.get("tenant_id") or default_tenant)
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest object requires a non-empty 'sources' array")
    return tenant_id, sources


def _manifest_source_spec(item: Dict[str, Any]) -> Dict[str, Any]:
    from services.api.app.routes.quick_import import infer_source_type

    if not isinstance(item, dict):
        raise ValueError("Each manifest source must be an object")
    location = item.get("location") or item.get("path") or item.get("url")
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Each manifest source requires location/path/url")
    source_type = item.get("source_type") or item.get("type") or infer_source_type(location)
    spec: Dict[str, Any] = {"location": location, "source_type": source_type}
    for key in (
        "name", "tags", "acl_policy_id", "config", "reuse_source",
        "sync_source_metadata", "dedupe_active_job", "idempotency_key",
    ):
        if key in item:
            spec[key] = item[key]
    return spec


def _ingest_local_spec(tenant_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    from services.api.app.agent.graph import build_default_services
    from services.api.app.routes.quick_import import build_source_config, deterministic_source_id, infer_source_type
    from services.api.app.routes.sources import _validate_source_config
    from services.api.app.storage.models import Source
    from services.worker.pipeline import run_ingest_pipeline

    services = build_default_services()
    location = str(spec["location"])
    source_type = spec.get("source_type") or infer_source_type(location)
    config = build_source_config(source_type, location, spec.get("config"))
    _validate_source_config(source_type, config)
    source_id = deterministic_source_id(tenant_id, source_type, location)
    existing = services.repo.get_source(source_id)
    source = Source(
        source_id=source_id,
        tenant_id=tenant_id,
        source_type=source_type,
        name=spec.get("name") or (existing.name if existing else location),
        config=config,
        status="active",
        acl_policy_id=spec.get("acl_policy_id") if spec.get("acl_policy_id") is not None else (existing.acl_policy_id if existing else None),
        tags=spec.get("tags") if spec.get("tags") is not None else (existing.tags if existing else []),
        created_at=existing.created_at if existing else None,
    )
    services.repo.add_source(source)
    job = run_ingest_pipeline(source, services.repo, services.qdrant, embedder=services.embedder)
    return {
        "status": job.status,
        "source_id": source.source_id,
        "source_type": source.source_type,
        "job_id": job.job_id,
        "job": asdict(job),
    }


def cmd_ask(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    constraints = {}
    if args.repo:
        constraints["repo"] = args.repo
    if args.ref:
        constraints["ref"] = args.ref
    if args.tags:
        constraints["tags"] = args.tags
    if args.server:
        payload = {"query": query, "tenant_id": args.tenant, "user_id": args.user, "constraints": constraints or None}
        result = _api_request(args.server, "POST", "/chat", payload, _auth_headers(args.api_key))
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)
    _print_result(result, args.json)


def cmd_search(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    if args.server:
        payload = {"query": query, "tenant_id": args.tenant, "user_id": args.user, "top_k": args.top_k}
        result = _api_request(args.server, "POST", "/search", payload, _auth_headers(args.api_key))
    else:
        from services.api.app.agent.graph import build_default_services
        from services.api.app.auth.acl import compute_security_scope
        services = build_default_services()
        scope = compute_security_scope(args.user, services.repo.list_policies())
        filters = {"tenant_id": args.tenant, "acl_hashes": scope}
        chunks = services.retriever.retrieve(query, filters, top_k=args.top_k)
        result = {"chunks": [{"chunk_id": c.chunk_id, "text": c.text, "score": c.score} for c in chunks], "total": len(chunks)}
    _print_result(result, args.json)


def cmd_patch(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    constraints = {"repo": args.repo} if args.repo else {}
    if args.server:
        payload = {"query": query, "tenant_id": args.tenant, "user_id": args.user, "constraints": constraints or None}
        result = _api_request(args.server, "POST", "/chat", payload, _auth_headers(args.api_key))
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)
    _print_result(result, args.json)


def cmd_ingest(args: argparse.Namespace) -> None:
    from services.api.app.routes.quick_import import infer_source_type

    location = args.path
    source_type = args.type or infer_source_type(location)
    spec: Dict[str, Any] = {
        "location": location,
        "source_type": source_type,
        "name": args.name,
        "tags": args.tags,
        "config": _ingest_config(args),
        "reuse_source": not args.no_reuse_source,
        "dedupe_active_job": not args.force_new_job,
        "idempotency_key": args.idempotency_key,
    }
    if args.server:
        payload = {"tenant_id": args.tenant, **spec}
        result = _api_request(args.server, "POST", "/ingest/quick", payload, _auth_headers(args.api_key))
        final_job = None
        if args.wait and result.get("job_id"):
            final_job = _wait_for_job(
                args.server, result["job_id"], headers=_auth_headers(args.api_key),
                timeout=args.timeout, poll_interval=args.poll_interval, quiet=args.json,
            )
        if args.json:
            output = {"submission": result}
            if final_job is not None:
                output["job"] = final_job
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"Source {result.get('source_id')} [{result.get('source_type')}] -> job {result.get('job_id')} ({result.get('status')})")
            if final_job is not None:
                print(f"Knowledge ready: docs={final_job.get('doc_count', 0)}, chunks={final_job.get('chunk_count', 0)}")
        return

    result = _ingest_local_spec(args.tenant, spec)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        job = result["job"]
        print(f"Ingestion complete: source={result['source_id']}, status={job['status']}, docs={job['doc_count']}, chunks={job['chunk_count']}")


def cmd_import(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    tenant_id, items = _normalize_manifest(raw, args.tenant)
    specs = [_manifest_source_spec(item) for item in items]
    if args.server:
        payload = {"tenant_id": tenant_id, "sources": specs}
        result = _api_request(args.server, "POST", "/ingest/batch", payload, _auth_headers(args.api_key), timeout=180)
        jobs: Dict[str, Dict[str, Any]] = {}
        if args.wait:
            for item in result.get("items", []):
                job_id = item.get("job_id")
                if not job_id or item.get("status") == "error" or job_id in jobs:
                    continue
                jobs[job_id] = _wait_for_job(
                    args.server, job_id, headers=_auth_headers(args.api_key),
                    timeout=args.timeout, poll_interval=args.poll_interval, quiet=args.json,
                )
        if args.json:
            print(json.dumps({"submission": result, "jobs": jobs}, indent=2, ensure_ascii=False))
        else:
            print(f"Batch submitted: total={result.get('total', 0)}, accepted={result.get('accepted', 0)}, failed={result.get('failed', 0)}")
            if args.wait:
                completed = sum(1 for job in jobs.values() if job.get("status") == "completed")
                chunks = sum(int(job.get("chunk_count", 0)) for job in jobs.values())
                print(f"Knowledge ready: completed_jobs={completed}, chunks={chunks}")
        if result.get("failed"):
            raise RuntimeError(f"{result['failed']} manifest source(s) failed to submit")
        return

    results = [_ingest_local_spec(tenant_id, spec) for spec in specs]
    if args.json:
        print(json.dumps({"tenant_id": tenant_id, "items": results}, indent=2, ensure_ascii=False))
    else:
        chunks = sum(int(item["job"].get("chunk_count", 0)) for item in results)
        print(f"Batch ingestion complete: sources={len(results)}, chunks={chunks}")


def cmd_doctor(args: argparse.Namespace) -> None:
    checks: Dict[str, Any] = {}
    if args.server:
        headers = _auth_headers(args.api_key)
        try:
            checks["liveness"] = _api_request(args.server, "GET", "/admin/health", headers=headers, timeout=30)
        except Exception as exc:
            checks["liveness"] = {"status": "failed", "error": str(exc)}
        try:
            checks["readiness"] = _api_request(args.server, "GET", "/admin/ready", headers=headers, timeout=30)
        except Exception as exc:
            checks["readiness"] = {"status": "failed", "error": str(exc)}
        ok = checks.get("liveness", {}).get("status") == "ok" and checks.get("readiness", {}).get("status") == "ready"
    else:
        from services.api.app.agent.graph import build_default_services
        try:
            services = build_default_services()
            repo_check = getattr(services.repo, "healthcheck", None)
            qdrant_check = getattr(services.qdrant, "healthcheck", None)
            checks = {
                "repository": bool(repo_check()) if callable(repo_check) else True,
                "vector_store": bool(qdrant_check()) if callable(qdrant_check) else True,
                "embedder": type(services.embedder).__name__,
            }
            ok = bool(checks["repository"] and checks["vector_store"])
        except Exception as exc:
            checks = {"startup": False, "error": str(exc)}
            ok = False
    if args.json:
        print(json.dumps({"ok": ok, "checks": checks}, indent=2, ensure_ascii=False))
    else:
        print("ragbot doctor: " + ("READY" if ok else "NOT READY"))
        for name, value in checks.items():
            print(f"  {name}: {value}")
    if not ok:
        raise RuntimeError("ragbot is not ready")


def _print_result(result: Dict[str, Any], as_json: bool) -> None:
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


def _add_wait_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="Wait until ingestion completes")
    parser.add_argument("--timeout", type=float, default=900, help="Maximum wait time in seconds (default: 900)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Job polling interval in seconds (default: 1)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="rag", description="ragbot CLI - build and query a RAG knowledge base")
    parser.add_argument("--server", "-s", help="API server URL (e.g., http://localhost:8000). If omitted, runs locally.")
    parser.add_argument("--api-key", "-k", help="API key for authentication")
    parser.add_argument("--tenant", "-t", default="default", help="Tenant ID (default: 'default')")
    parser.add_argument("--user", "-u", default="cli-user", help="User ID (default: 'cli-user')")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", nargs="+", help="The question to ask")
    ask_parser.add_argument("--repo", help="Restrict to a specific repo")
    ask_parser.add_argument("--ref", help="Git ref (branch/tag)")
    ask_parser.add_argument("--tags", nargs="*", help="Filter by tags")
    ask_parser.set_defaults(func=cmd_ask)

    search_parser = subparsers.add_parser("search", help="Search indexed knowledge")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    search_parser.set_defaults(func=cmd_search)

    patch_parser = subparsers.add_parser("patch", help="Generate a code patch")
    patch_parser.add_argument("query", nargs="+", help="Patch instruction")
    patch_parser.add_argument("--repo", help="Target repo")
    patch_parser.set_defaults(func=cmd_patch)

    ingest_parser = subparsers.add_parser("ingest", help="Create/reuse a source and ingest it")
    ingest_parser.add_argument(
        "path",
        help="Local/HTTP/S3/Google Drive/Notion/Confluence location",
    )
    ingest_parser.add_argument(
        "--type",
        choices=["local_fs", "repo", "pdf", "web", "s3", "gdrive", "notion", "confluence"],
        help="Source type (auto-detected if omitted)",
    )
    ingest_parser.add_argument("--name", help="Source name")
    ingest_parser.add_argument("--tag", dest="tags", action="append", help="Source tag; repeat for multiple tags")
    ingest_parser.add_argument("--ref", help="Git ref for repository sources")
    ingest_parser.add_argument("--credential-ref", help="Secret reference such as env:RAGBOT_NOTION_TOKEN; never pass token values")
    ingest_parser.add_argument("--credential-type", choices=["access_token", "google_json"], help="Google Drive credential type")
    ingest_parser.add_argument("--base-url", help="Confluence base URL when location is only a space key")
    ingest_parser.add_argument("--email", help="Confluence account email for basic auth")
    ingest_parser.add_argument("--auth-type", choices=["basic", "bearer"], help="Confluence authentication mode")
    ingest_parser.add_argument("--root-page-id", help="Optional Confluence root page restriction")
    ingest_parser.add_argument("--notion-version", help="Notion API version override")
    ingest_parser.add_argument("--no-recursive", action="store_true", help="Disable recursive Drive/Notion traversal")
    ingest_parser.add_argument("--max-file-bytes", type=int, help="Per-file hard download limit for cloud connectors")
    ingest_parser.add_argument("--config-json", help="Additional non-secret connector config as a JSON object")
    ingest_parser.add_argument("--chunk-size", type=int, help="Override connector chunk size")
    ingest_parser.add_argument("--chunk-overlap", type=int, help="Override connector chunk overlap")
    ingest_parser.add_argument("--idempotency-key", help="Return the same job for repeated submissions with this key")
    ingest_parser.add_argument("--no-reuse-source", action="store_true", help="Do not reuse a source with the same tenant/type/location")
    ingest_parser.add_argument("--force-new-job", action="store_true", help="Queue a new job even if one is pending/running")
    _add_wait_options(ingest_parser)
    ingest_parser.set_defaults(func=cmd_ingest)

    import_parser = subparsers.add_parser("import", help="Build a knowledge base from a JSON manifest")
    import_parser.add_argument("manifest", help="Path to manifest JSON")
    _add_wait_options(import_parser)
    import_parser.set_defaults(func=cmd_import)

    doctor_parser = subparsers.add_parser("doctor", help="Check ragbot deployment readiness")
    doctor_parser.set_defaults(func=cmd_doctor)

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
