#!/usr/bin/env python3
"""Run controlled splitter/parser benchmarks on one real corpus.

The command is deliberately local: it does not compare live Ragbot with native
framework stores. Instead it isolates one component at a time under the same
Golden Dataset, embedding model and cosine retrieval harness.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPORT_DIR = ROOT / "reports" / "rag-components"
SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".pdf"}


def _csv(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _resolve_source(raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("--corpus is required")
    path = Path(value).expanduser()
    path = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists():
        raise ValueError(f"corpus not found: {path}")
    if path.is_file() and path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported corpus file: {path.name}")
    return path


def _load_units(source: Path):
    from benchmarks.rag_native_compare import load_corpus_units

    if source.is_dir():
        return load_corpus_units(source)
    with tempfile.TemporaryDirectory(prefix="ragbot-component-corpus-") as tmp:
        staging = Path(tmp)
        target = staging / source.name
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        return load_corpus_units(staging)


def _result_row(item: dict[str, Any]) -> str:
    quality = item.get("quality") or {}
    latency = quality
    if item.get("component") == "splitter":
        timing = item.get("timing") or {}
        detail = item.get("chunk_shape") or {}
        return (
            f"| `{item['backend']}` | {quality.get('hit_at_1', 0):.1%} | "
            f"{quality.get('hit_at_5', 0):.1%} | {quality.get('mrr_at_10', 0):.3f} | "
            f"{quality.get('ndcg_at_10', 0):.3f} | {quality.get('query_latency_ms_p50', 0):.2f} ms | "
            f"{item.get('chunks', 0)} | {timing.get('split_seconds', 0):.3f} s | "
            f"{detail.get('non_boundary_end_rate', 0):.1%} | {detail.get('character_inflation_ratio', 0):.3f} |"
        )
    timing = item.get("timing") or {}
    fidelity = item.get("fidelity") or {}
    return (
        f"| `{item['backend']}` | {quality.get('hit_at_1', 0):.1%} | "
        f"{quality.get('hit_at_5', 0):.1%} | {quality.get('mrr_at_10', 0):.3f} | "
        f"{quality.get('ndcg_at_10', 0):.3f} | {quality.get('query_latency_ms_p50', 0):.2f} ms | "
        f"{timing.get('parser_seconds', 0):.3f} s | {item.get('chunks', 0)} | "
        f"{fidelity.get('page_metadata_coverage', 0):.1%} | "
        f"{fidelity.get('label_text_coverage', 0):.1%} | "
        f"{fidelity.get('table_block_rate', 0):.1%} |"
    )


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# Local RAG component benchmark — {payload['dataset_name']}",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Corpus: `{payload['corpus']}`  ",
        f"Embedding: `{payload['embedding_model']}` / `{payload['embedding_dimension']}`  ",
        "",
        "The benchmark isolates one component at a time; whole-framework conclusions must not be drawn from it.",
        "",
    ]
    splitter = payload.get("splitter") or []
    if splitter:
        lines.extend(
            [
                "## Splitter-only",
                "",
                "Fixed: parsed page text + embedding + cosine search + Golden Dataset + top-k. Changed: splitter.",
                "",
                "| Splitter | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | p50 | Chunks | Split | Non-boundary end | Char inflation |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[_result_row(item) for item in splitter if item.get("status") != "skipped"],
                "",
            ]
        )
        skipped = [item for item in splitter if item.get("status") == "skipped"]
        for item in skipped:
            lines.append(f"- `{item['backend']}` skipped: {item['error']}")
        if skipped:
            lines.append("")

    parser = payload.get("parser") or []
    if parser:
        lines.extend(
            [
                "## Parser-only",
                "",
                "Fixed: Ragbot chunking + embedding + cosine search + Golden Dataset + top-k. Changed: parser.",
                "",
                "| Parser | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | p50 | Parse | Chunks | Page metadata | Label text | Table blocks |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[_result_row(item) for item in parser if item.get("status") != "skipped"],
                "",
            ]
        )
        skipped = [item for item in parser if item.get("status") == "skipped"]
        for item in skipped:
            lines.append(f"- `{item['backend']}` skipped: {item['error']}")
        if skipped:
            lines.append("")

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- Splitter results attribute differences only to segmentation because the retrieval implementation is shared.",
            "- Parser retrieval metrics should be read together with page/label/structure fidelity; fast parsing alone is not sufficient.",
            "- `label_text_coverage` measures whether Golden Dataset answer-bearing terms survive parsing before retrieval.",
            "- `character_inflation_ratio` measures chunk overlap/segmentation expansion, not semantic quality.",
            "- Use a production Golden Dataset with reviewed stable labels before changing production defaults.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Golden Dataset JSON")
    parser.add_argument("--corpus", required=True, help="PDF/text file or corpus directory")
    parser.add_argument("--components", default="splitter,parser", help="splitter,parser")
    parser.add_argument("--splitters", default="ragbot,langchain,llamaindex")
    parser.add_argument("--parsers", default="pypdf2,pymupdf")
    parser.add_argument("--embedding", choices=["env", "hash"], default="env")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--strict-backends", action="store_true", help="fail instead of skipping unavailable optional parser/splitter backends")
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

    from benchmarks.component_compare import (
        load_raw_pdfs,
        run_parser_backend,
        run_splitter_backend,
    )
    from benchmarks.rag_native_compare import load_golden_dataset
    from services.api.app.retrieval.embedder import HashEmbedder, build_embedder

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = load_golden_dataset(dataset_path)
    cases = list(dataset.get("cases") or [])
    source = _resolve_source(args.corpus)
    embedder = build_embedder() if args.embedding == "env" else HashEmbedder(args.hash_dimension)
    components = set(_csv(args.components))
    unknown = components - {"splitter", "parser"}
    if unknown:
        raise ValueError(f"unsupported components: {sorted(unknown)}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_name": str(dataset.get("name") or dataset_path.stem),
        "corpus": str(source),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "configuration": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
        },
        "splitter": [],
        "parser": [],
    }

    if "splitter" in components:
        units = _load_units(source)
        for backend in _csv(args.splitters):
            try:
                payload["splitter"].append(
                    run_splitter_backend(
                        backend,
                        units,
                        cases,
                        embedder=embedder,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                        top_k=args.top_k,
                        repetitions=args.repetitions,
                    )
                )
            except (ImportError, RuntimeError) as exc:
                if args.strict_backends:
                    raise
                payload["splitter"].append({"component": "splitter", "backend": backend, "status": "skipped", "error": str(exc)})

    if "parser" in components:
        if source.is_file() and source.suffix.lower() != ".pdf":
            raise ValueError("parser component benchmark requires PDF input")
        documents = load_raw_pdfs(source)
        for backend in _csv(args.parsers):
            try:
                payload["parser"].append(
                    run_parser_backend(
                        backend,
                        documents,
                        cases,
                        embedder=embedder,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                        top_k=args.top_k,
                        repetitions=args.repetitions,
                    )
                )
            except (ImportError, RuntimeError) as exc:
                if args.strict_backends:
                    raise
                payload["parser"].append({"component": "parser", "backend": backend, "status": "skipped", "error": str(exc)})

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"component-benchmark-{stamp}.json"
    md_path = report_dir / f"component-benchmark-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = markdown_report(payload)
    md_path.write_text(md, encoding="utf-8")
    (report_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
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
