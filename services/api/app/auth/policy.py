from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def compute_policy_hash(rules: Dict[str, Any]) -> str:
    payload = json.dumps(rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

