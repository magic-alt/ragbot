#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_DSN:?POSTGRES_DSN is required}"
: "${QDRANT_URL:?QDRANT_URL is required}"
: "${QDRANT_COLLECTION:?QDRANT_COLLECTION is required}"

BACKUP_DIR="${1:-./backups/ragbot-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$BACKUP_DIR"

command -v pg_dump >/dev/null || { echo "pg_dump is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }

printf 'Backing up PostgreSQL...\n'
pg_dump --dbname="$POSTGRES_DSN" --format=custom --no-owner --file="$BACKUP_DIR/postgres.dump"

headers=()
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  headers=(-H "api-key: ${QDRANT_API_KEY}")
fi

printf 'Creating Qdrant collection snapshot for %s...\n' "$QDRANT_COLLECTION"
snapshot_json="$(curl -fsS -X POST "${QDRANT_URL%/}/collections/${QDRANT_COLLECTION}/snapshots?wait=true" "${headers[@]}")"
snapshot_name="$(SNAPSHOT_JSON="$snapshot_json" python - <<'PY'
import json, os
payload = json.loads(os.environ['SNAPSHOT_JSON'])
name = (payload.get('result') or {}).get('name')
if not name:
    raise SystemExit('Qdrant snapshot response did not contain result.name')
print(name)
PY
)"

curl -fsS \
  "${QDRANT_URL%/}/collections/${QDRANT_COLLECTION}/snapshots/${snapshot_name}" \
  "${headers[@]}" \
  --output "$BACKUP_DIR/qdrant.snapshot"

POSTGRES_DUMP="$BACKUP_DIR/postgres.dump" \
QDRANT_SNAPSHOT="$BACKUP_DIR/qdrant.snapshot" \
QDRANT_COLLECTION="$QDRANT_COLLECTION" \
QDRANT_SNAPSHOT_NAME="$snapshot_name" \
python - <<'PY' > "$BACKUP_DIR/manifest.json"
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

pg = Path(os.environ['POSTGRES_DUMP'])
qd = Path(os.environ['QDRANT_SNAPSHOT'])
print(json.dumps({
    'created_at': datetime.now(timezone.utc).isoformat(),
    'postgres': {'file': pg.name, 'sha256': sha256(pg), 'bytes': pg.stat().st_size},
    'qdrant': {
        'collection': os.environ['QDRANT_COLLECTION'],
        'snapshot_name': os.environ['QDRANT_SNAPSHOT_NAME'],
        'file': qd.name,
        'sha256': sha256(qd),
        'bytes': qd.stat().st_size,
    },
}, indent=2))
PY

printf 'Backup complete: %s\n' "$BACKUP_DIR"
printf 'Manifest: %s/manifest.json\n' "$BACKUP_DIR"
