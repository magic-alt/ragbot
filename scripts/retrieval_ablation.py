#!/usr/bin/env python3
"""Compare vector, lexical and hybrid retrieval on one live Ragbot dataset.

This intentionally reuses ``scripts/rag_eval.py`` dataset semantics so relevance
labels, filters and top-k behavior have one contract. It calls the live /search
endpoint in each retrieval mode and reports Hit@K / MRR@10 side by side.

Example:
    python scripts/retrieval_ablation.py \
      eval/datasets/deepseek_in_action_retrieval.json \
      --tenant engineering --candidate-pool 50 --output reports/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import rag_eval

MODES = ("vector", "lexical", "hybrid")


def _mode_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [case for case in cases if case["labeled"] and not case.get("error")]

    def hit_at(k: int) -> float:
        if not labeled:
            return 0.0
        return sum(
            1
            for case in labeled
            if case.get("first_relevant_rank") is not None
            and int(case["first_relevant_rank"]) <= k
        ) / len(labeled)

    reciprocal = []
    for case in labeled:
        rank = case.get("first_relevant_rank")
        reciprocal.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)

    return {
        "labeled_cases": len(labeled),
        "hit_at_1": round(hit_at(1), 4),
        "hit_at_3": round(hit_at(3), 4),
        "hit_at_5": round(hit_at(5), 4),
        "hit_at_10": round(hit_at(10), 4),
        "mrr_at_10": round(statistics.fmean(reciprocal), 4) if reciprocal else None,
    }


def _run_mode(
    dataset: Dict[str, Any],
    *,
    mode: str,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    timeout: float,
    top_k_override: Optional[int],
    candidate_pool: Optional[int],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    for case in dataset["cases"]:
        top_k = rag_eval._case_top_k(dataset, case, top_k_override)
        payload = {
            "query": str(case["query"]),
            "tenant_id": tenant,
            "user_id": user,
            "top_k": top_k,
            "filters": rag_eval._case_filters(dataset, case),
            "mode": mode,
            "candidate_pool": candidate_pool,
            "explain": True,
        }
        try:
            response = rag_eval._http_json(
                server,
                "/search",
                payload,
                api_key=api_key,
                timeout=timeout,
            )
            chunks = list(response.get("chunks") or [])
            labeled = rag_eval._is_labeled(case)
            first_rank = rag_eval._first_relevant_rank(chunks, case) if labeled else None
            diagnostics = dict(response.get("diagnostics") or {})
            if diagnostics and not runtime:
                runtime = diagnostics
            cases.append(
                {
                    "id": str(case["id"]),
                    "category": str(case.get("category") or "default"),
                    "query": str(case["query"]),
                    "labeled": labeled,
                    "first_relevant_rank": first_rank,
                    "top_chunk_id": chunks[0].get("chunk_id") if chunks else None,
                    "top_score": chunks[0].get("score") if chunks else None,
                    "top_trace": ((chunks[0].get("metadata") or {}).get("_retrieval") or {}) if chunks else {},
                    "error": None,
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "id": str(case["id"]),
                    "category": str(case.get("category") or "default"),
                    "query": str(case["query"]),
                    "labeled": rag_eval._is_labeled(case),
                    "first_relevant_rank": None,
                    "top_chunk_id": None,
                    "top_score": None,
                    "top_trace": {},
                    "error": str(exc),
                }
            )
    return {"mode": mode, "summary": _mode_summary(cases), "runtime": runtime, "cases": cases}


def _print_table(results: list[dict[str, Any]]) -> None:
    print()
    print("Retrieval ablation")
    print("mode      Hit@1   Hit@3   Hit@5   Hit@10  MRR@10")
    print("--------  ------  ------  ------  ------  ------")
    for item in results:
        summary = item["summary"]
        print(
            f"{item['mode']:<8}  "
            f"{summary['hit_at_1']:<6.3f}  "
            f"{summary['hit_at_3']:<6.3f}  "
            f"{summary['hit_at_5']:<6.3f}  "
            f"{summary['hit_at_10']:<6.3f}  "
            f"{(summary['mrr_at_10'] or 0.0):<6.3f}"
        )

    hybrid = next((item for item in results if item["mode"] == "hybrid"), None)
    if hybrid:
        runtime = hybrid.get("runtime") or {}
        policy = runtime.get("fusion_policy") or {}
        print()
        print(
            "hybrid runtime: embedding={model} semantic={semantic} "
            "candidate_pool={pool} reranker={reranker}".format(
                model=runtime.get("embedding_model", "?"),
                semantic=runtime.get("semantic_embedding", "?"),
                pool=runtime.get("candidate_pool", "?"),
                reranker=runtime.get("reranker_enabled", False),
            )
        )
        if policy:
            print(
                "sample fusion policy: vector_weight={vw} lexical_weight={lw} reason={reason}".format(
                    vw=policy.get("vector_weight"),
                    lw=policy.get("lexical_weight"),
                    reason=policy.get("reason"),
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vector/lexical/hybrid retrieval ablation against a live Ragbot server"
    )
    parser.add_argument("dataset")
    parser.add_argument("--server", help="Ragbot API URL; defaults to rag_eval runtime state")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="retrieval-ablation")
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help="Modes to evaluate; default runs all three",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.dataset).resolve()
    try:
        dataset = rag_eval._load_dataset(path)
        server = (args.server or rag_eval._runtime_server()).rstrip("/")
        health = rag_eval._http_json(
            server,
            "/admin/ready",
            None,
            api_key=args.api_key,
            timeout=min(args.timeout, 10.0),
        )
        if health.get("status") != "ready":
            raise RuntimeError(f"Ragbot is not ready: {health}")

        results = [
            _run_mode(
                dataset,
                mode=mode,
                server=server,
                tenant=args.tenant,
                user=args.user,
                api_key=args.api_key,
                timeout=args.timeout,
                top_k_override=args.top_k,
                candidate_pool=args.candidate_pool,
            )
            for mode in args.modes
        ]
        report = {
            "schema_version": 1,
            "dataset": str(path),
            "name": dataset.get("name") or path.stem,
            "server": server,
            "tenant": args.tenant,
            "candidate_pool": args.candidate_pool,
            "results": results,
        }
        _print_table(results)
        if args.output:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nJSON: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
