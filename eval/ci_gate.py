"""CI gate for evaluation quality.

Usage:
    python -m eval.ci_gate --threshold 0.70 --dataset full
    python -m eval.ci_gate --threshold 0.50 --dataset sample
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .datasets import build_full_dataset, build_sample_dataset
from .runner import run_eval_suite, summarize_results

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ragbot evaluation CI gate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum pass rate to pass the gate (0.0-1.0, default: 0.70)",
    )
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        default="sample",
        help="Which dataset to use (default: sample)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (doc_qa, db_qa, code_task)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filter by tag",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write detailed results JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load dataset
    if args.dataset == "full":
        cases = build_full_dataset()
    else:
        cases = build_sample_dataset()

    logger.info("Loaded %d evaluation cases (dataset=%s)", len(cases), args.dataset)

    # Run evaluation
    results = run_eval_suite(
        cases,
        category_filter=args.category,
        tag_filter=args.tag,
    )

    # Summarize
    summary = summarize_results(results)

    # Output
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total:       {summary['total']}")
    print(f"Passed:      {summary['passed']}")
    print(f"Failed:      {summary['failed']}")
    print(f"Pass rate:   {summary['pass_rate']:.1%}")
    print(f"Avg latency: {summary['avg_duration_ms']:.0f}ms")
    if summary.get("retrieval_eval_count", 0) > 0:
        print(f"MRR@10:      {summary['avg_mrr_at_10']:.4f}")
        print(f"Recall@10:   {summary['avg_recall_at_10']:.4f}")
    print()

    if summary.get("by_category"):
        print("By category:")
        for cat, stats in summary["by_category"].items():
            rate = stats["passed"] / stats["total"] if stats["total"] else 0
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0%})")
        print()

    if summary.get("failure_categories"):
        print("Failure breakdown:")
        for fcat, count in summary["failure_categories"].items():
            print(f"  {fcat}: {count}")
        print()

    # Write detailed results
    if args.output:
        from dataclasses import asdict
        data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Detailed results written to %s", args.output)

    # Gate check
    threshold = args.threshold
    pass_rate = summary["pass_rate"]

    print("=" * 60)
    if pass_rate >= threshold:
        print(f"GATE PASSED: {pass_rate:.1%} >= {threshold:.1%}")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"GATE FAILED: {pass_rate:.1%} < {threshold:.1%}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
