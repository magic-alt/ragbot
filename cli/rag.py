"""Canonical Ragbot product CLI.

This module is the single implementation behind both ``rag`` and
``python -m cli.rag``. The repository bootstrap controller in
``scripts/ragbot.py`` owns install/start/stop/log operations and delegates
knowledge operations here; there is no second product CLI implementation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .job_wait import format_job_knowledge, job_chunk_stats, wait_for_job


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
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if method.upper() == "GET":
        response = requests.get(url, headers=request_headers, timeout=timeout)
    else:
        response = requests.post(url, json=payload, headers=request_headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _wait_for_job(
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
    poll_interval: float,
    quiet: bool = False,
) -> Dict[str, Any]:
    return wait_for_job(
        _api_request,
        server,
        job_id,
        headers=headers,
        timeout=timeout,
        poll_interval=poll_interval,
        quiet=quiet,
    )


def _local_chat(
    query: str,
    tenant_id: str,
    user_id: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.api.app.agent.graph import build_default_services
    from services.api.app.agent.state import Constraints
    from services.api.app.main import chat

    services = build_default_services()
    parsed = Constraints(**constraints) if constraints else None
    return asyncio.run(chat(query, tenant_id, user_id, services, parsed))


def _ingest_config(args: argparse.Namespace) -> Dict[str, Any]:
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
    if getattr(args, "config_json", None):
        extra = json.loads(args.config_json)
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
        "name",
        "tags",
        "acl_policy_id",
        "config",
        "reuse_source",
        "sync_source_metadata",
        "dedupe_active_job",
        "idempotency_key",
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
        acl_policy_id=(
            spec.get("acl_policy_id")
            if spec.get("acl_policy_id") is not None
            else (existing.acl_policy_id if existing else None)
        ),
        tags=spec.get("tags") if spec.get("tags") is not None else (existing.tags if existing else []),
        created_at=existing.created_at if existing else None,
        updated_at=existing.updated_at if existing else None,
    )
    services.repo.add_source(source)
    persisted = services.repo.get_source(source.source_id) or source
    job = run_ingest_pipeline(persisted, services.repo, services.qdrant, embedder=services.embedder)
    return {
        "status": job.status,
        "source_id": source.source_id,
        "source_type": source.source_type,
        "job_id": job.job_id,
        "job": asdict(job),
    }


def cmd_ask(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    constraints: Dict[str, Any] = {}
    if args.repo:
        constraints["repo"] = args.repo
    if args.ref:
        constraints["ref"] = args.ref
    if args.tags:
        constraints["tags"] = args.tags
    if args.server:
        result = _api_request(
            args.server,
            "POST",
            "/chat",
            {"query": query, "tenant_id": args.tenant, "user_id": args.user, "constraints": constraints or None},
            _auth_headers(args.api_key),
        )
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)
    _print_result(result, args.json)


def cmd_search(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    if args.server:
        result = _api_request(
            args.server,
            "POST",
            "/search",
            {"query": query, "tenant_id": args.tenant, "user_id": args.user, "top_k": args.top_k},
            _auth_headers(args.api_key),
        )
    else:
        from services.api.app.agent.graph import build_default_services
        from services.api.app.auth.acl import compute_security_scope

        services = build_default_services()
        scope = compute_security_scope(args.user, services.repo.list_policies())
        chunks = services.retriever.retrieve(
            query,
            {"tenant_id": args.tenant, "acl_hashes": scope},
            top_k=args.top_k,
        )
        result = {
            "chunks": [
                {"chunk_id": chunk.chunk_id, "text": chunk.text, "score": chunk.score}
                for chunk in chunks
            ],
            "total": len(chunks),
        }
    _print_result(result, args.json)


def cmd_patch(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    constraints = {"repo": args.repo} if args.repo else {}
    if args.server:
        result = _api_request(
            args.server,
            "POST",
            "/chat",
            {"query": query, "tenant_id": args.tenant, "user_id": args.user, "constraints": constraints or None},
            _auth_headers(args.api_key),
        )
    else:
        result = _local_chat(query, args.tenant, args.user, constraints or None)
    _print_result(result, args.json)


def cmd_ingest(args: argparse.Namespace) -> None:
    from services.api.app.routes.quick_import import infer_source_type

    location = args.path
    spec: Dict[str, Any] = {
        "location": location,
        "source_type": args.type or infer_source_type(location),
        "name": args.name,
        "tags": args.tags,
        "config": _ingest_config(args),
        "reuse_source": not args.no_reuse_source,
        "dedupe_active_job": not args.force_new_job,
        "idempotency_key": args.idempotency_key,
    }
    if not args.server:
        result = _ingest_local_spec(args.tenant, spec)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            job = result["job"]
            print(
                f"Ingestion complete: source={result['source_id']}, status={job['status']}, "
                f"{format_job_knowledge(job)}"
            )
        return

    submission = _api_request(
        args.server,
        "POST",
        "/ingest/quick",
        {"tenant_id": args.tenant, **spec},
        _auth_headers(args.api_key),
    )
    final_job = None
    if args.wait and submission.get("job_id"):
        final_job = _wait_for_job(
            args.server,
            submission["job_id"],
            headers=_auth_headers(args.api_key),
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            quiet=args.json,
        )
    if args.json:
        output: Dict[str, Any] = {"submission": submission}
        if final_job is not None:
            output["job"] = final_job
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(
            f"Source {submission.get('source_id')} [{submission.get('source_type')}] -> "
            f"job {submission.get('job_id')} ({submission.get('status')})"
        )
        if final_job is not None:
            print(f"Knowledge ready: {format_job_knowledge(final_job)}")


def cmd_import(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tenant_id, items = _normalize_manifest(raw, args.tenant)
    specs = [_manifest_source_spec(item) for item in items]
    if not args.server:
        results = [_ingest_local_spec(tenant_id, spec) for spec in specs]
        if args.json:
            print(json.dumps({"tenant_id": tenant_id, "items": results}, indent=2, ensure_ascii=False))
        else:
            counts = [job_chunk_stats(item["job"]) for item in results]
            print(
                f"Batch ingestion complete: sources={len(results)}, "
                f"chunks={sum(item['total'] for item in counts)}, "
                f"written={sum(item['written'] for item in counts)}, "
                f"reused={sum(item['reused'] for item in counts)}"
            )
        return

    submission = _api_request(
        args.server,
        "POST",
        "/ingest/batch",
        {"tenant_id": tenant_id, "sources": specs},
        _auth_headers(args.api_key),
        timeout=180,
    )
    jobs: Dict[str, Dict[str, Any]] = {}
    if args.wait:
        for item in submission.get("items", []):
            job_id = item.get("job_id")
            if not job_id or item.get("status") == "error" or job_id in jobs:
                continue
            jobs[job_id] = _wait_for_job(
                args.server,
                job_id,
                headers=_auth_headers(args.api_key),
                timeout=args.timeout,
                poll_interval=args.poll_interval,
                quiet=args.json,
            )
    if args.json:
        print(json.dumps({"submission": submission, "jobs": jobs}, indent=2, ensure_ascii=False))
    else:
        print(
            f"Batch submitted: total={submission.get('total', 0)}, "
            f"accepted={submission.get('accepted', 0)}, failed={submission.get('failed', 0)}"
        )
        if args.wait:
            counts = [job_chunk_stats(job) for job in jobs.values()]
            print(
                "Knowledge ready: completed_jobs="
                f"{sum(1 for job in jobs.values() if job.get('status') == 'completed')}, "
                f"chunks={sum(item['total'] for item in counts)}, "
                f"written={sum(item['written'] for item in counts)}, "
                f"reused={sum(item['reused'] for item in counts)}"
            )
    if submission.get("failed"):
        raise RuntimeError(f"{submission['failed']} manifest source(s) failed to submit")


def cmd_doctor(args: argparse.Namespace) -> None:
    checks: Dict[str, Any] = {}
    if args.server:
        headers = _auth_headers(args.api_key)
        for name, path in (("liveness", "/admin/health"), ("readiness", "/admin/ready")):
            try:
                checks[name] = _api_request(args.server, "GET", path, headers=headers, timeout=30)
            except Exception as exc:
                checks[name] = {"status": "failed", "error": str(exc)}
        ok = (
            checks.get("liveness", {}).get("status") == "ok"
            and checks.get("readiness", {}).get("status") == "ready"
        )
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
    elif "answer" in result:
        print(result["answer"])
        if result.get("citations"):
            print("\n--- Citations ---")
            for citation in result["citations"]:
                parts = []
                if citation.get("path"):
                    location = citation["path"]
                    if citation.get("line_start"):
                        location += f":{citation['line_start']}"
                    parts.append(location)
                if citation.get("url"):
                    parts.append(citation["url"])
                if citation.get("chunk_id"):
                    parts.append(f"chunk:{citation['chunk_id']}")
                print(f"  [{citation.get('kind', '?')}] {' | '.join(parts)}")
        if result.get("confidence"):
            print(f"\n[confidence: {result['confidence']}]")
    elif "chunks" in result:
        chunks = result["chunks"]
        print(f"Found {result.get('total', len(chunks))} results:\n")
        for index, chunk in enumerate(chunks, 1):
            preview = chunk.get("text", "")[:120].replace("\n", " ")
            print(f"  {index}. [{chunk.get('score', '?')}] {preview}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def _add_wait_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="Wait until ingestion reaches a terminal state")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--poll-interval", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Ragbot product CLI")
    parser.add_argument("--server", "-s", help="API server URL; omit for in-process development")
    parser.add_argument("--api-key", "-k")
    parser.add_argument("--tenant", "-t", default="default")
    parser.add_argument("--user", "-u", default="cli-user")
    parser.add_argument("--json", "-j", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    ask = subparsers.add_parser("ask")
    ask.add_argument("query", nargs="+")
    ask.add_argument("--repo")
    ask.add_argument("--ref")
    ask.add_argument("--tags", nargs="*")
    ask.set_defaults(func=cmd_ask)

    search = subparsers.add_parser("search")
    search.add_argument("query", nargs="+")
    search.add_argument("--top-k", type=int, default=10)
    search.set_defaults(func=cmd_search)

    patch = subparsers.add_parser("patch")
    patch.add_argument("query", nargs="+")
    patch.add_argument("--repo")
    patch.set_defaults(func=cmd_patch)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("path")
    ingest.add_argument("--type", choices=["local_fs", "repo", "pdf", "web", "s3", "gdrive", "notion", "confluence"])
    ingest.add_argument("--name")
    ingest.add_argument("--tag", dest="tags", action="append")
    ingest.add_argument("--ref")
    ingest.add_argument("--credential-ref")
    ingest.add_argument("--credential-type", choices=["access_token", "google_json"])
    ingest.add_argument("--base-url")
    ingest.add_argument("--email")
    ingest.add_argument("--auth-type", choices=["basic", "bearer"])
    ingest.add_argument("--root-page-id")
    ingest.add_argument("--notion-version")
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--max-file-bytes", type=int)
    ingest.add_argument("--config-json")
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)
    ingest.add_argument("--idempotency-key")
    ingest.add_argument("--no-reuse-source", action="store_true")
    ingest.add_argument("--force-new-job", action="store_true")
    _add_wait_options(ingest)
    ingest.set_defaults(func=cmd_ingest)

    manifest = subparsers.add_parser("import")
    manifest.add_argument("manifest")
    _add_wait_options(manifest)
    manifest.set_defaults(func=cmd_import)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def _normalize_ingest_argv(argv: Sequence[str]) -> List[str]:
    """Join split ingest locations while preserving subsequent --options."""
    tokens = list(argv)
    try:
        ingest_index = tokens.index("ingest")
    except ValueError:
        return tokens
    start = ingest_index + 1
    if start >= len(tokens):
        return tokens
    end = start
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    if end - start > 1:
        tokens[start:end] = [" ".join(tokens[start:end])]
    return tokens


def main(argv: Optional[List[str]] = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(_normalize_ingest_argv(raw))
    if not args.command:
        parser.print_help()
        raise SystemExit(1)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
