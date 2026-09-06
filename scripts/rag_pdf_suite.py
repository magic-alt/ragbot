#!/usr/bin/env python3
"""Initialize or validate a source-backed PDF Structure Golden Suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="scan a PDF corpus and write an authoring skeleton")
    init.add_argument("--corpus", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--name", default="PDF Structure Golden Suite")
    init.add_argument("--force", action="store_true")

    validate = sub.add_parser("validate", help="validate corpus identity and Golden Suite coverage")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--corpus", required=True)
    validate.add_argument("--profile", choices=["development", "production"], default="production")
    validate.add_argument("--json", action="store_true", help="print machine-readable audit JSON")
    return parser


def _source(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"corpus not found: {path}")
    return path


def _init(args: argparse.Namespace) -> int:
    from benchmarks.component_compare import load_raw_pdfs
    from benchmarks.pdf_promotion import build_suite_skeleton

    source = _source(args.corpus)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise ValueError(f"output already exists: {output}; pass --force to replace")
    documents = load_raw_pdfs(source)
    payload = build_suite_skeleton(documents, name=args.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote authoring skeleton: {output}")
    print(f"Documents: {len(documents)}")
    print("Next: replace every document_type=TODO, add source-backed cases, stable relevance labels, tags and review.status=approved.")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from benchmarks.component_compare import load_raw_pdfs
    from benchmarks.pdf_promotion import audit_pdf_structure_suite

    source = _source(args.corpus)
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise ValueError(f"dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object")
    documents = load_raw_pdfs(source)
    audit = audit_pdf_structure_suite(dataset, documents, profile=args.profile)
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(f"PDF Structure Golden Suite: {'PASS' if audit['passed'] else 'FAIL'} ({args.profile})")
        golden = audit.get("golden_dataset") or {}
        stats = golden.get("stats") or {}
        print(
            f"Cases={stats.get('cases', 0)} stable={stats.get('stable_label_rate', 0):.1%} "
            f"documents={audit.get('stats', {}).get('documents', 0)} reviewed={audit.get('stats', {}).get('approved_review_rate', 0):.1%}"
        )
        for check in list(golden.get("checks") or []) + list(audit.get("checks") or []):
            mark = "PASS" if check.get("passed") else "FAIL"
            print(f"- {mark}: {check.get('name')} actual={check.get('actual')} expected={check.get('expected')}")
        for warning in audit.get("warnings") or []:
            print(f"- WARN: {warning}")
    return 0 if audit["passed"] else 1


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return _init(args)
    return _validate(args)


def main() -> int:
    try:
        return run()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
