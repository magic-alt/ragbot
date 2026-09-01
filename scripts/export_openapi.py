"""Export the canonical FastAPI OpenAPI document.

Usage:
    python scripts/export_openapi.py > openapi.json
    python scripts/export_openapi.py --output build/openapi.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.api.app.api import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Ragbot FastAPI OpenAPI JSON")
    parser.add_argument("--output", type=Path, help="Optional output path; stdout by default")
    args = parser.parse_args()

    payload = json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
