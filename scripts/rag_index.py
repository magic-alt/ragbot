#!/usr/bin/env python3
"""Operational CLI for Ragbot IndexVersion build/promotion/rollback.

Run this as a separate process from the API. Large reindex builds therefore do
not occupy an API request worker; build progress and ETA are persisted in
PostgreSQL after each batch and printed to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from services.api.app.factory import build_services_from_env


def _json_file(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _service(services):
    service = getattr(services, "index_lifecycle", None)
    if service is None:
        raise SystemExit("Configured runtime does not provide index lifecycle")
    return service


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ragbot vector index lifecycle")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    sub.add_parser("contracts")

    create = sub.add_parser("create")
    create.add_argument("--contract", required=True)
    create.add_argument("--id")
    create.add_argument("--collection")

    build = sub.add_parser("build")
    build.add_argument("index_version_id")
    build.add_argument("--batch-size", type=int, default=100)

    shadow = sub.add_parser("shadow")
    shadow.add_argument("index_version_id")
    shadow.add_argument("--cases", required=True, help="JSON list of {query, expected_chunk_ids?}")
    shadow.add_argument("--top-k", type=int, default=10)
    shadow.add_argument("--output")

    ready = sub.add_parser("ready")
    ready.add_argument("index_version_id")
    ready.add_argument("--evidence", required=True)
    ready.add_argument("--approve", action="store_true")

    activate = sub.add_parser("activate")
    activate.add_argument("index_version_id")
    activate.add_argument("--retention-seconds", type=int, default=7 * 24 * 3600)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("index_version_id")
    rollback.add_argument("--retention-seconds", type=int, default=7 * 24 * 3600)

    sub.add_parser("reconcile")
    sub.add_parser("prune")

    args = parser.parse_args(argv)
    services = build_services_from_env()
    try:
        service = _service(services)
        if args.command == "status":
            _print(
                {
                    "alias_name": service.alias_name,
                    "active_physical_collection": service.vector_store.active_collection_name(),
                    "items": [
                        asdict(item)
                        for item in service.repo.list_index_versions(alias_name=service.alias_name)
                    ],
                }
            )
        elif args.command == "contracts":
            _print({"items": service.embedding_router.public_metadata()})
        elif args.command == "create":
            _print(
                asdict(
                    service.create_candidate(
                        args.contract,
                        index_version_id=args.id,
                        physical_collection=args.collection,
                    )
                )
            )
        elif args.command == "build":
            def progress(stats: dict[str, Any]) -> None:
                print(json.dumps(stats, ensure_ascii=False), file=sys.stderr, flush=True)

            _print(
                asdict(
                    service.build(
                        args.index_version_id,
                        batch_size=max(1, args.batch_size),
                        progress=progress,
                    )
                )
            )
        elif args.command == "shadow":
            cases = _json_file(args.cases)
            if not isinstance(cases, list):
                raise SystemExit("--cases must contain a JSON list")
            evidence = service.shadow_compare(
                args.index_version_id, cases, top_k=max(1, args.top_k)
            )
            if args.output:
                Path(args.output).write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            _print(evidence)
        elif args.command == "ready":
            evidence = _json_file(args.evidence)
            if not isinstance(evidence, dict):
                raise SystemExit("--evidence must contain a JSON object")
            _print(
                asdict(
                    service.mark_ready(
                        args.index_version_id,
                        evidence,
                        approved=bool(args.approve),
                    )
                )
            )
        elif args.command == "activate":
            _print(
                asdict(
                    service.activate(
                        args.index_version_id,
                        retention_seconds=max(0, args.retention_seconds),
                    )
                )
            )
        elif args.command == "rollback":
            _print(
                asdict(
                    service.rollback(
                        args.index_version_id,
                        retention_seconds=max(0, args.retention_seconds),
                    )
                )
            )
        elif args.command == "reconcile":
            _print(service.reconcile())
        elif args.command == "prune":
            _print(service.prune_retired())
        return 0
    finally:
        closed: set[int] = set()
        for resource in (
            getattr(services, "embedding_router", None),
            services.sql_engine,
            services.repo,
            services.qdrant,
        ):
            if resource is None or id(resource) in closed:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
            closed.add(id(resource))


if __name__ == "__main__":
    raise SystemExit(main())
