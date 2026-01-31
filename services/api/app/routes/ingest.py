from __future__ import annotations

from typing import Dict


def handle_ingest(payload: Dict[str, object]) -> Dict[str, object]:
    return {"status": "accepted", "job_id": payload.get("job_id", "demo")}

