#!/usr/bin/env python3
"""Run parser × block-coalescing × splitter factorial retrieval benchmarks."""
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

DEFAULT_REPORT_DIR = ROOT / "reports" / "rag-factorial"


def _csv(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _resolve_corpus(raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("--corpus is required")
    path = Path(value).expanduser()
    path = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists():
        raise ValueError(f"corpus not found: {path}")
    if path.is_file() and path.suffix.lower() != ".pdf":
        raise ValueError("factorial parser benchmark requires PDF input")
    return path


def _row(item: dict[str, Any]) -> str:
    quality = item.get("quality") or {}
    shape = item.get("chunk_shape") or {}
    timing = item.get("timing") or {}
    memory_mb = float((item.get("memory") or {}).get("tracemalloc_peak_bytes") or 0) / (1024 * 1024)
    return (
        f"| `{item['cell']}` | {quality.get('hit_at_1', 0):.1%} | "
        f"{quality.get('mrr_at_10', 0):.3f} | {quality.get('ndcg_at_10', 0):.3f} | "
        f"{quality.get('query_latency_ms_p50', 0):.1f} ms | "
        f"{item.get('raw_blocks', 0)}→{item.get('bridge_blocks', 0)} | {item.get('chunks', 0)} | "
        f"{shape.get('non_boundary_end_rate', 0):.1%} | {shape.get('character_inflation_ratio', 0):.3f} | "
        f"{timing.get('parser_seconds', 0):.2f} s | {timing.get('embedding_seconds', 0):.1f} s | {memory_mb:.0f} MB |"
    )


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [item for item in payload.get("cells") or [] if item.get("status") != "skipped"]
    skipped = [item for item in payload.get("cells") or [] if item.get("status") == "skipped"]
    lines = [
        f"# Parser × coalescing × splitter factorial benchmark — {payload['dataset_name']}",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Corpus: `{payload['corpus']}`  ",
        f"Embedding: `{payload['embedding_model']}` / `{payload['embedding_dimension']}`  ",
        f"Design: `{payload['configuration']['design']}`  ",
        "",
        "Each cell uses the production parser bridge. Parser output, optional block coalescing and splitter are the experimental factors; embedding, cosine retrieval, Golden Dataset and top-k are fixed.",
        "",
        "| Pipeline cell | Hit@1 | MRR@10 | nDCG@10 | p50 | Blocks | Chunks | Non-boundary | Char inflation | Parse | Embed | Peak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_row(item) for item in rows],
        "",
    ]
    for item in skipped:
        lines.append(f"- `{item['cell']}` skipped: {item['error']}")
    if skipped:
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- `raw_blocks→bridge_blocks` exposes whether coalescing actually removes structured-parser fragmentation.",
            "- Parser time is cached per parser backend and repeated in each cell for comparison; it is not re-paid for every splitter during the run.",
            "- Embedding time is paid per cell because each pipeline may produce different chunk text and cardinality.",
            "- Do not promote production defaults from a development Golden Dataset alone; confirm on reviewed stable labels and a structurally diverse PDF corpus.",
            "- The default compact design omits PyPDF2+coalesced because PyPDF2 already emits page-scale blocks and that row is largely redundant.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Golden Dataset JSON")
    parser.add_argument("--corpus", required=True, help="PDF file or directory")
    parser.add_argument("--parsers", default="pypdf2,pymupdf")
    parser.add_argument("--splitters", default="ragbot,llamaindex")
    parser.add_argument("--design", choices=["compact", "full"], default="compact")
    parser.add_argument("--coalesce-target-multiplier", type=float, default=4.0)
    parser.add_argument("--embedding", choices=["env", "hash"], default="env")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.chunk_size < 2:
        raise ValueError("chunk-size must be >= 2")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must satisfy 0 <= overlap < chunk-size")
    if args.top_k < 1 or args.repetitions < 1:
        raise ValueError("top-k and repetitions must be >= 1")
    if args.coalesce_target_multiplier < 1.0:
        raise ValueError("coalesce-target-multiplier must be >= 1.0")

    from benchmarks.component_compare import load_raw_pdfs
    from benchmarks.factorial_compare import (
        compact_factorial_cells,
        prepare_parser_backend,
        run_factorial_cell,
    )
    from benchmarks.rag_native_compare import load_golden_dataset
    from services.api.app.retrieval.embedder import HashEmbedder, build_embedder

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = load_golden_dataset(dataset_path)
    cases = list(dataset.get("cases") or [])
    corpus = _resolve_corpus(args.corpus)
    documents = load_raw_pdfs(corpus)
    embedder = build_embedder() if args.embedding == "env" else HashEmbedder(args.hash_dimension)
    if args.embedding == "env" and isinstance(embedder, HashEmbedder):
        raise ValueError(
            "--embedding env resolved to HashEmbedder; configure the same semantic embedding model used by Ragbot"
        )

    parsers = _csv(args.parsers)
    splitters = _csv(args.splitters)
    if not parsers or not splitters:
        raise ValueError("at least one parser and splitter are required")

    if args.design == "compact":
        matrix = compact_factorial_cells(parsers, splitters)
    else:
        matrix = [
            (parser_name, splitter_name, mode)
            for parser_name in parsers
            for mode in ("raw", "coalesced")
            for splitter_name in splitters
        ]

    prepared: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for parser_name in parsers:
        try:
            prepared[parser_name] = prepare_parser_backend(parser_name, documents)
        except (ImportError, RuntimeError) as exc:
            if args.strict_backends:
                raise
            unavailable[parser_name] = str(exc)

    cells: list[dict[str, Any]] = []
    for parser_name, splitter_name, mode in matrix:
        cell_name = f"{parser_name}+{mode}+{splitter_name}"
        if parser_name in unavailable:
            cells.append({"component": "factorial", "cell": cell_name, "status": "skipped", "error": unavailable[parser_name]})
            continue
        try:
            cells.append(
                run_factorial_cell(
                    prepared[parser_name],
                    splitter_name,
                    mode,
                    cases,
                    embedder=embedder,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    top_k=args.top_k,
                    repetitions=args.repetitions,
                    coalesce_target_multiplier=args.coalesce_target_multiplier,
                )
            )
        except (ImportError, RuntimeError) as exc:
            if args.strict_backends:
                raise
            cells.append({"component": "factorial", "cell": cell_name, "status": "skipped", "error": str(exc)})

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_name": str(dataset.get("name") or dataset_path.stem),
        "corpus": str(corpus),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "configuration": {
            "design": args.design,
            "parsers": parsers,
            "splitters": splitters,
            "coalesce_target_multiplier": args.coalesce_target_multiplier,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
        },
        "cells": cells,
    }

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"factorial-benchmark-{stamp}.json"
    md_path = report_dir / f"factorial-benchmark-{stamp}.md"
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    md = markdown_report(payload)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    (report_dir / "latest.json").write_text(json_text, encoding="utf-8")
    (report_dir / "latest.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return payload


def main() -> int:
    try:
        run()
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
