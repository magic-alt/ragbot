#!/usr/bin/env python3
"""Run Ragbot's automated functional + live RAG quality verification."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.system_quality import PROFILES, run_live_gate, write_report

DEFAULT_REPORT_DIR = ROOT / "reports" / "quality-gate"
DEFAULT_SERVER = "http://127.0.0.1:8000"
VENV = ROOT / ".venv"
FUNCTIONAL_EXTRAS = "dev,postgres,qdrant,worker,s3,saas,observability"


def _runtime_server() -> str:
    state = ROOT / "tmp" / "ragbot-runtime.json"
    if not state.exists():
        return DEFAULT_SERVER
    try:
        import json

        payload = json.loads(state.read_text(encoding="utf-8"))
        return str(payload.get("server") or DEFAULT_SERVER).rstrip("/")
    except Exception:
        return DEFAULT_SERVER


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _python_has_pytest(executable: Path | str) -> bool:
    try:
        result = subprocess.run(
            [str(executable), "-c", "import pytest"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _existing_pytest_python() -> Optional[Path]:
    """Prefer the repository virtualenv, then the invoking interpreter."""
    venv_python = _venv_python()
    candidates = []
    if venv_python.exists():
        candidates.append(venv_python)
    current = Path(sys.executable)
    if current not in candidates:
        candidates.append(current)
    for executable in candidates:
        if _python_has_pytest(executable):
            return executable
    return None


def _bootstrap_functional_test_env() -> tuple[Optional[Path], Optional[str]]:
    """Create/repair .venv with the same functional extras used by GitHub CI."""
    venv_python = _venv_python()
    if not venv_python.exists():
        print(f"[functional] creating test virtualenv: {VENV}")
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV)],
            cwd=ROOT,
            check=False,
        )
        if created.returncode != 0:
            return None, f"could not create {VENV} (exit {created.returncode})"

    print(
        "[functional] installing/repairing test dependencies in .venv "
        f"(extras: {FUNCTIONAL_EXTRAS})"
    )
    install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-e",
            f".[{FUNCTIONAL_EXTRAS}]",
        ],
        cwd=ROOT,
        check=False,
    )
    if install.returncode != 0:
        return None, f"test dependency installation failed (exit {install.returncode})"
    if not _python_has_pytest(venv_python):
        return None, "pytest is still unavailable after bootstrapping .venv"
    return venv_python, None


def _resolve_pytest_python(mode: str) -> tuple[Optional[Path], dict[str, object]]:
    if mode == "off":
        return None, {"status": "skipped", "reason": "disabled by --pytest=off"}

    executable = _existing_pytest_python()
    if executable is not None:
        return executable, {"status": "ready", "python": str(executable), "bootstrapped": False}

    if mode == "auto":
        return None, {
            "status": "skipped",
            "reason": "pytest not installed; use --pytest on to bootstrap .venv automatically",
        }

    executable, error = _bootstrap_functional_test_env()
    if executable is None:
        return None, {"status": "failed", "reason": error or "test environment bootstrap failed"}
    return executable, {"status": "ready", "python": str(executable), "bootstrapped": True}


def _run_pytest(mode: str) -> dict[str, object]:
    executable, resolution = _resolve_pytest_python(mode)
    if executable is None:
        return resolution

    print(f"\n[functional] running repository pytest suite with {executable}")
    start = time.perf_counter()
    result = subprocess.run(
        [str(executable), "-m", "pytest", "-q"],
        cwd=ROOT,
        check=False,
    )
    elapsed = round((time.perf_counter() - start) * 1000.0, 2)
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_ms": elapsed,
        "python": str(executable),
        "bootstrapped": bool(resolution.get("bootstrapped")),
    }


def _run_domain_dataset(
    dataset: Path,
    *,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    report_dir: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "rag_eval.py"),
        str(dataset),
        "--server",
        server,
        "--tenant",
        tenant,
        "--user",
        user,
        "--report-dir",
        str(report_dir),
        "--fail-on-threshold",
    ]
    if api_key:
        command.extend(["--api-key", api_key])

    print(f"\n[domain] {dataset}")
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    return {
        "dataset": str(dataset),
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a running Ragbot with a deterministic production-path RAG corpus, "
            "optional full pytest suite, and optional domain Golden Datasets."
        )
    )
    parser.add_argument(
        "--server",
        help="Ragbot API URL; defaults to tmp/ragbot-runtime.json",
    )
    parser.add_argument(
        "--tenant",
        default=os.getenv("RAGBOT_EVAL_TENANT", "default"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("RAGBOT_EVAL_USER", "ragbot-quality-gate"),
    )
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat each retrieval case to expose instability",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for the hybrid probe",
    )
    parser.add_argument(
        "--pytest",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Functional suite policy: auto runs when pytest already exists; "
            "on bootstraps repository .venv/test extras when needed; off disables it"
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Optional domain Golden Dataset; repeat to evaluate multiple corpora "
            "after the system gate"
        ),
    )
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 5 <= args.top_k <= 50:
        print(
            "ERROR: --top-k must be between 5 and 50 for the built-in Hit@5/candidate-pool gates",
            file=sys.stderr,
        )
        return 2
    if args.repetitions < 1:
        print("ERROR: --repetitions must be >= 1", file=sys.stderr)
        return 2

    server = (args.server or _runtime_server()).rstrip("/")
    report_dir = Path(args.report_dir).resolve()
    print("Ragbot automated quality gate")
    print(f"  server: {server}")
    print(f"  tenant: {args.tenant}")
    print(f"  profile: {args.profile}")

    functional = _run_pytest(args.pytest)
    detail = functional.get("reason") or functional.get("python") or ""
    print(f"[functional] {functional['status']}: {detail}")

    try:
        print(
            "\n[live-rag] uploading deterministic PDF and exercising "
            "the production RAG path"
        )
        live = run_live_gate(
            server=server,
            api_key=args.api_key,
            tenant=args.tenant,
            user=args.user,
            profile=args.profile,
            timeout=args.timeout,
            top_k=args.top_k,
            repetitions=args.repetitions,
            rerank=not args.no_rerank,
        )
    except Exception as exc:
        print(f"[live-rag] ERROR: {exc}", file=sys.stderr)
        return 2

    live["functional_suite"] = functional
    paths = write_report(live, report_dir)
    gate = live["gate"]
    print(f"[live-rag] {'PASS' if gate['passed'] else 'FAIL'}")
    for mode, metrics in live["retrieval"].items():
        print(
            f"  {mode:7s} hit@5={metrics['hit_at_5']:.0%} "
            f"semantic-hit@5={metrics['semantic_hit_at_5']:.0%} "
            f"mrr@10={metrics['mrr_at_10']:.3f} "
            f"p95={metrics['p95_ms']}ms"
        )
    runtime = live.get("runtime") or {}
    print(
        "  embedding={model} backend={backend} semantic={semantic}".format(
            model=runtime.get("embedding_model", "?"),
            backend=runtime.get("embedding_backend", "?"),
            semantic=runtime.get("semantic_embedding", "?"),
        )
    )
    print(f"  answer-pass={live['answer']['pass_rate']:.0%}")
    print(f"  report={paths['markdown']}")

    domain_results: list[dict[str, object]] = []
    for raw_dataset in args.dataset:
        dataset = Path(raw_dataset)
        if not dataset.is_absolute():
            dataset = (Path.cwd() / dataset).resolve()
        if not dataset.exists():
            domain_results.append(
                {
                    "dataset": str(dataset),
                    "status": "failed",
                    "reason": "file not found",
                }
            )
            print(f"[domain] missing dataset: {dataset}", file=sys.stderr)
            continue
        domain_results.append(
            _run_domain_dataset(
                dataset,
                server=server,
                tenant=args.tenant,
                user=args.user,
                api_key=args.api_key,
                report_dir=report_dir / "domain",
            )
        )

    failed = not bool(gate["passed"])
    if functional.get("status") == "failed":
        failed = True
    if any(item.get("status") == "failed" for item in domain_results):
        failed = True

    print("\nOverall: " + ("FAIL" if failed else "PASS"))
    if functional.get("status") == "skipped":
        print(
            "Note: functional pytest suite was skipped; use --pytest on for a "
            "release-grade full-function gate."
        )
    if not args.dataset:
        print(
            "Note: add --dataset <your-golden.json> to score retrieval quality "
            "on your real document corpus."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
