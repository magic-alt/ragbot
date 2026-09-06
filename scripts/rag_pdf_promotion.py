#!/usr/bin/env python3
"""Run the PDF ingestion control/candidate A/B promotion gate."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPORT_DIR = ROOT / "reports" / "pdf-promotion"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--profile", choices=["development", "production"], default="production")
    parser.add_argument("--embedding", choices=["env", "hash"], default="env")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--coalesce-target-multiplier", type=float, default=4.0)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--no-fail-on-gate", action="store_true", help="return zero even when promotion gate fails")
    return parser


def _source(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"corpus not found: {path}")
    return path


def _pipeline_row(item: dict[str, Any]) -> str:
    q = item.get("quality") or {}
    t = item.get("timing") or {}
    s = item.get("chunk_shape") or {}
    f = item.get("fidelity") or {}
    peak = int((item.get("memory") or {}).get("tracemalloc_peak_bytes") or 0) / 1024 / 1024
    return (
        f"| `{item['name']}` | {q.get('hit_at_5', 0):.1%} | {q.get('mrr_at_10', 0):.3f} | "
        f"{q.get('ndcg_at_10', 0):.3f} | {item.get('raw_blocks', 0)}→{item.get('bridge_blocks', 0)} | "
        f"{item.get('chunks', 0)} | {s.get('non_boundary_end_rate', 0):.1%} | "
        f"{f.get('label_text_coverage', 0):.1%} | {t.get('parser_seconds', 0):.2f}s | "
        f"{t.get('embedding_seconds', 0):.1f}s | {peak:.0f} MB |"
    )


def markdown_report(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    audit = payload["suite_audit"]
    lines = [
        f"# PDF ingestion promotion gate — {payload['dataset_name']}",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Corpus: `{payload['corpus']}`  ",
        f"Embedding: `{payload['embedding_model']}` / `{payload['embedding_dimension']}`  ",
        f"Profile: `{payload['profile']}`  ",
        f"Decision: **{'PROMOTE' if gate['passed'] else 'HOLD'}**",
        "",
        "| Pipeline | Hit@5 | MRR@10 | nDCG@10 | Blocks | Chunks | Non-boundary | Label text | Parse | Embed | Peak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _pipeline_row(payload["control"]),
        _pipeline_row(payload["candidate"]),
        "",
        "## Gate checks",
        "",
        "| Check | Actual | Expected | Result |",
        "| --- | --- | --- | --- |",
    ]
    for check in gate.get("checks") or []:
        lines.append(
            f"| `{check['name']}` | `{check.get('actual')}` | `{check.get('expected')}` | "
            f"{'PASS' if check.get('passed') else 'FAIL'} |"
        )

    lines.extend(["", "## Suite audit", ""])
    golden = audit.get("golden_dataset") or {}
    stats = golden.get("stats") or {}
    suite_stats = audit.get("stats") or {}
    lines.extend(
        [
            f"- Audit: **{'PASS' if audit.get('passed') else 'FAIL'}**",
            f"- Cases: {stats.get('cases', 0)}",
            f"- Stable label rate: {stats.get('stable_label_rate', 0):.1%}",
            f"- Documents: {suite_stats.get('documents', 0)}",
            f"- Approved review rate: {suite_stats.get('approved_review_rate', 0):.1%}",
            f"- Document types: {', '.join(suite_stats.get('document_types') or []) or 'n/a'}",
            f"- Case tags: {', '.join(suite_stats.get('case_tags') or []) or 'n/a'}",
        ]
    )
    for warning in audit.get("warnings") or []:
        lines.append(f"- Warning: {warning}")

    diagnostics = gate.get("diagnostics") or {}
    lines.extend(["", "## Regression diagnostics", ""])
    new_failures = diagnostics.get("new_case_failures") or []
    rank_regressions = diagnostics.get("rank_regressions") or []
    category_regressions = diagnostics.get("category_regressions") or []
    lines.append(f"- New retrieval failures: {len(new_failures)}")
    if new_failures:
        lines.append(f"  - {', '.join(new_failures)}")
    lines.append(f"- Rank regressions: {len(rank_regressions)}")
    for item in rank_regressions[:20]:
        lines.append(
            f"  - `{item['id']}` ({item.get('category')}): rank {item.get('control_rank')} → {item.get('candidate_rank')}"
        )
    lines.append(f"- Category MRR regressions beyond threshold: {len(category_regressions)}")
    for item in category_regressions:
        lines.append(f"  - `{item['category']}`: ΔMRR={item['mrr_delta']:+.4f}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PROMOTE` means the candidate passed the configured offline promotion contract; it does not mutate production defaults.",
            "- Peak memory is steady-state Python allocation measured after warming the exact splitter runtime; one-time optional-framework imports are excluded from the ratio gate.",
            "- Production profile requires a reviewed, stable-label, source-pinned PDF Structure Golden Suite.",
            "- After an offline PROMOTE, validate the candidate in isolated Qdrant/PostgreSQL ingestion and live hybrid/reranker A/B before changing the default Source contract.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(argv: Sequence[str] | None = None) -> tuple[dict[str, Any], int]:
    args = build_parser().parse_args(argv)
    if args.chunk_size < 2 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-size/overlap must satisfy size>=2 and 0<=overlap<size")
    if args.top_k < 1 or args.repetitions < 1:
        raise ValueError("top-k and repetitions must be >=1")
    if args.coalesce_target_multiplier < 1.0:
        raise ValueError("coalesce-target-multiplier must be >=1")

    from benchmarks.component_compare import load_raw_pdfs
    from benchmarks.factorial_compare import splitter_config
    from benchmarks.pdf_promotion import (
        CANDIDATE_PIPELINE,
        CONTROL_PIPELINE,
        PipelineSpec,
        audit_pdf_structure_suite,
        evaluate_promotion_gate,
        prepare_pipelines,
        run_pipeline,
    )
    from benchmarks.rag_native_compare import load_golden_dataset
    from benchmarks.runtime_warmup import warm_chunker_runtime
    from services.api.app.retrieval.embedder import HashEmbedder, build_embedder

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = load_golden_dataset(dataset_path)
    cases = list(dataset.get("cases") or [])
    source = _source(args.corpus)
    documents = load_raw_pdfs(source)
    suite_audit = audit_pdf_structure_suite(dataset, documents, profile=args.profile)
    embedder = build_embedder() if args.embedding == "env" else HashEmbedder(args.hash_dimension)

    control_parser, candidate_parser = prepare_pipelines(documents)
    candidate_spec = PipelineSpec(
        name=CANDIDATE_PIPELINE.name,
        parser=CANDIDATE_PIPELINE.parser,
        splitter=CANDIDATE_PIPELINE.splitter,
        coalescing=CANDIDATE_PIPELINE.coalescing,
        coalesce_target_multiplier=args.coalesce_target_multiplier,
    )

    def warm_spec(spec: PipelineSpec) -> dict[str, object]:
        config = splitter_config(
            spec.splitter,
            coalesced=spec.coalescing == "coalesced",
            chunk_size=args.chunk_size,
            coalesce_target_multiplier=spec.coalesce_target_multiplier,
        )
        return warm_chunker_runtime(
            config,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    runtime_warmup = {
        "control": warm_spec(CONTROL_PIPELINE),
        "candidate": warm_spec(candidate_spec),
    }

    control = run_pipeline(
        control_parser,
        CONTROL_PIPELINE,
        cases,
        embedder=embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        repetitions=args.repetitions,
    )
    candidate = run_pipeline(
        candidate_parser,
        candidate_spec,
        cases,
        embedder=embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        repetitions=args.repetitions,
    )
    gate = evaluate_promotion_gate(control, candidate, suite_audit)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_name": str(dataset.get("name") or dataset_path.stem),
        "corpus": str(source),
        "profile": args.profile,
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "configuration": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "coalesce_target_multiplier": args.coalesce_target_multiplier,
            "memory_measurement": "steady_state_tracemalloc_after_exact_splitter_warmup",
        },
        "runtime_warmup": runtime_warmup,
        "suite_audit": suite_audit,
        "control": control,
        "candidate": candidate,
        "gate": gate,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"pdf-promotion-{stamp}.json"
    md_path = report_dir / f"pdf-promotion-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = markdown_report(payload)
    md_path.write_text(md, encoding="utf-8")
    (report_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (report_dir / "latest.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    status = 0 if gate["passed"] or args.no_fail_on_gate else 1
    return payload, status


def main() -> int:
    try:
        _payload, status = run()
        return status
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
