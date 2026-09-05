#!/usr/bin/env python3
"""Run Ragbot Level 2 Golden Dataset evaluation and Level 3 framework comparison."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.rag_native_compare import (
    audit_golden_dataset,
    load_corpus_units,
    load_golden_dataset,
    markdown_report as native_markdown_report,
    run_comparison,
    write_reports as write_native_reports,
)

DEFAULT_REPORT_DIR = ROOT / "reports" / "rag-benchmark"
DEFAULT_SERVER = "http://127.0.0.1:8000"


def _runtime_server() -> str:
    state = ROOT / "tmp" / "ragbot-runtime.json"
    if not state.exists():
        return DEFAULT_SERVER
    try:
        payload = json.loads(state.read_text(encoding="utf-8"))
        return str(payload.get("server") or DEFAULT_SERVER).rstrip("/")
    except Exception:
        return DEFAULT_SERVER


def _print_audit(audit: dict[str, Any]) -> None:
    stats = audit["stats"]
    print("[dataset] Golden Dataset audit")
    print(
        f"  profile={audit['profile']} cases={stats['cases']} labeled={stats['labeled_cases']} "
        f"stable={stats['stable_label_rate']:.0%} categories={stats['category_count']}"
    )
    for check in audit["checks"]:
        print(
            f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}: "
            f"actual={check['actual']} expected={check['expected']}"
        )


def _run_level2(
    *,
    dataset_path: Path,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    top_k: int,
    timeout: float,
    with_answers: bool,
    report_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "rag_eval.py"),
        str(dataset_path),
        "--server",
        server,
        "--tenant",
        tenant,
        "--user",
        user,
        "--top-k",
        str(top_k),
        "--timeout",
        str(timeout),
        "--report-dir",
        str(report_dir),
        "--fail-on-threshold",
    ]
    if with_answers:
        command.append("--with-answers")
    if api_key:
        command.extend(["--api-key", api_key])

    print("\n[level2] live Ragbot Golden Dataset benchmark")
    result = subprocess.run(command, cwd=ROOT, check=False)
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "report_dir": str(report_dir),
    }


def _run_level3(
    *,
    dataset: dict[str, Any],
    corpus_dir: Path,
    backends: str,
    embedding: str,
    hash_dimension: int,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    repetitions: int,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    timeout: float,
    ragbot_mode: str,
    rerank: bool,
    enforce_embedding_match: bool,
    report_dir: Path,
) -> dict[str, Any]:
    print("\n[level3] native Ragbot / LangChain / LlamaIndex comparison")
    units = load_corpus_units(corpus_dir)
    report = run_comparison(
        dataset=dataset,
        units=units,
        backends=backends,
        embedding=embedding,
        hash_dimension=hash_dimension,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        repetitions=repetitions,
        server=server,
        tenant=tenant,
        user=user,
        api_key=api_key,
        timeout=timeout,
        ragbot_mode=ragbot_mode,
        rerank=rerank,
        enforce_embedding_match=enforce_embedding_match,
    )
    paths = write_native_reports(report, report_dir)
    print(native_markdown_report(report))
    return {
        "status": "passed",
        "report": report,
        "json": str(paths["json"]),
        "markdown": str(paths["markdown"]),
    }


def _write_summary(
    *,
    dataset_path: Path,
    audit: dict[str, Any],
    level2: Optional[dict[str, Any]],
    level3: Optional[dict[str, Any]],
    report_dir: Path,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_audit": audit,
        "level2": level2,
        "level3": {
            key: value
            for key, value in (level3 or {}).items()
            if key != "report"
        } if level3 else None,
        "overall_passed": bool(
            audit["passed"]
            and (level2 is None or level2.get("status") == "passed")
            and (level3 is None or level3.get("status") == "passed")
        ),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"benchmark-summary-{stamp}.json"
    md_path = report_dir / f"benchmark-summary-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Ragbot Level 2 + Level 3 benchmark summary",
        "",
        f"- Result: **{'PASS' if payload['overall_passed'] else 'FAIL'}**",
        f"- Dataset: `{dataset_path}`",
        f"- Dataset profile: `{audit['profile']}`",
        f"- Cases: `{audit['stats']['cases']}`",
        f"- Stable-label rate: `{audit['stats']['stable_label_rate']:.0%}`",
        "",
        "## Stages",
        "",
        f"- Level 2 live Ragbot Golden Dataset: `{(level2 or {}).get('status', 'not-run')}`",
        f"- Level 3 native framework comparison: `{(level3 or {}).get('status', 'not-run')}`",
    ]
    if level3 and level3.get("markdown"):
        lines.append(f"- Level 3 report: `{level3['markdown']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "latest-summary.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (report_dir / "latest-summary.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Level 2 live Golden Dataset evaluation and Level 3 native "
            "Ragbot/LangChain/LlamaIndex comparison."
        )
    )
    parser.add_argument("--dataset", required=True, help="Golden Dataset JSON")
    parser.add_argument(
        "--level",
        choices=["2", "3", "all"],
        default="all",
        help="Benchmark stage to run (default: all)",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=["off", "development", "production"],
        default="development",
        help="Dataset maturity gate; production requires >=50 cases and >=80% stable labels",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Local corpus for Level 3; must match the corpus indexed in live Ragbot",
    )
    parser.add_argument("--backends", default="ragbot,langchain,llamaindex")
    parser.add_argument("--embedding", choices=["env", "hash"], default="env")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--server", help="Ragbot API URL; defaults to tmp/ragbot-runtime.json")
    parser.add_argument("--tenant", default=os.getenv("RAGBOT_EVAL_TENANT", "default"))
    parser.add_argument("--user", default=os.getenv("RAGBOT_EVAL_USER", "rag-benchmark"))
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--ragbot-mode", choices=["vector", "lexical", "hybrid"], default="vector")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-enforce-embedding-match", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    if args.top_k < 10:
        print("ERROR: --top-k must be >=10 because the benchmark reports @10 metrics", file=sys.stderr)
        return 2
    if args.repetitions < 1:
        print("ERROR: --repetitions must be >=1", file=sys.stderr)
        return 2
    if args.level in {"3", "all"} and not args.corpus_dir:
        print("ERROR: --corpus-dir is required for Level 3", file=sys.stderr)
        return 2

    try:
        dataset = load_golden_dataset(dataset_path)
        audit = audit_golden_dataset(dataset, args.dataset_profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Ragbot Level 2 + Level 3 benchmark")
    _print_audit(audit)
    if not audit["passed"]:
        print("Overall: FAIL (Golden Dataset maturity gate)")
        return 1

    server = (args.server or _runtime_server()).rstrip("/")
    report_dir = Path(args.report_dir).resolve()
    level2: Optional[dict[str, Any]] = None
    level3: Optional[dict[str, Any]] = None

    if args.level in {"2", "all"}:
        level2 = _run_level2(
            dataset_path=dataset_path,
            server=server,
            tenant=args.tenant,
            user=args.user,
            api_key=args.api_key,
            top_k=args.top_k,
            timeout=args.timeout,
            with_answers=args.with_answers,
            report_dir=report_dir / "level2",
        )

    if args.level in {"3", "all"}:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.is_absolute():
            corpus_dir = (Path.cwd() / corpus_dir).resolve()
        if not corpus_dir.is_dir():
            print(f"ERROR: corpus directory not found: {corpus_dir}", file=sys.stderr)
            return 2
        try:
            level3 = _run_level3(
                dataset=dataset,
                corpus_dir=corpus_dir,
                backends=args.backends,
                embedding=args.embedding,
                hash_dimension=args.hash_dimension,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=args.top_k,
                repetitions=args.repetitions,
                server=server,
                tenant=args.tenant,
                user=args.user,
                api_key=args.api_key,
                timeout=args.timeout,
                ragbot_mode=args.ragbot_mode,
                rerank=args.rerank,
                enforce_embedding_match=not args.no_enforce_embedding_match,
                report_dir=report_dir / "level3",
            )
        except Exception as exc:
            print(f"[level3] ERROR: {exc}", file=sys.stderr)
            level3 = {"status": "failed", "error": str(exc)}

    paths = _write_summary(
        dataset_path=dataset_path,
        audit=audit,
        level2=level2,
        level3=level3,
        report_dir=report_dir,
    )
    failed = (
        (level2 is not None and level2.get("status") != "passed")
        or (level3 is not None and level3.get("status") != "passed")
    )
    print(f"\nSummary: {paths['markdown']}")
    print("Overall: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
