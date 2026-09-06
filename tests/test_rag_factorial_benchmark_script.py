from __future__ import annotations

from scripts.rag_factorial_benchmark import build_parser, markdown_report


def test_factorial_cli_defaults_to_compact_two_parser_two_splitter_design():
    args = build_parser().parse_args(
        [
            "--dataset",
            "golden.json",
            "--corpus",
            "manual.pdf",
        ]
    )
    assert args.parsers == "pypdf2,pymupdf"
    assert args.splitters == "ragbot,llamaindex"
    assert args.design == "compact"
    assert args.coalesce_target_multiplier == 4.0


def test_factorial_markdown_surfaces_fragmentation_and_cost_metrics():
    payload = {
        "dataset_name": "demo",
        "generated_at": "2026-09-06T00:00:00+00:00",
        "corpus": "/tmp/demo.pdf",
        "embedding_model": "semantic-demo",
        "embedding_dimension": 4,
        "configuration": {"design": "compact"},
        "cells": [
            {
                "component": "factorial",
                "cell": "pymupdf+coalesced+llamaindex",
                "raw_blocks": 100,
                "bridge_blocks": 20,
                "chunks": 30,
                "quality": {
                    "hit_at_1": 1.0,
                    "mrr_at_10": 1.0,
                    "ndcg_at_10": 1.0,
                    "query_latency_ms_p50": 12.5,
                },
                "chunk_shape": {
                    "non_boundary_end_rate": 0.1,
                    "character_inflation_ratio": 1.02,
                },
                "timing": {"parser_seconds": 2.0, "embedding_seconds": 10.0},
                "memory": {"tracemalloc_peak_bytes": 10 * 1024 * 1024},
            }
        ],
    }
    report = markdown_report(payload)
    assert "pymupdf+coalesced+llamaindex" in report
    assert "100→20" in report
    assert "30" in report
    assert "10.0%" in report
    assert "10 MB" in report
