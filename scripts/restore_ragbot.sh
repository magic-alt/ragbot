#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_DSN:?POSTGRES_DSN is required}"
: "${QDRANT_URL:?QDRANT_URL is required}"
: "${QDRANT_COLLECTION:?QDRANT_COLLECTION is required}"

BACKUP_DIR="${1:?usage: bash scripts/restore_ragbot.sh BACKUP_DIR}"
POSTGRES_DUMP="$BACKUP_DIR/postgres.dump"
QDRANT_SNAPSHOT="$BACKUP_DIR/qdrant.snapshot"
MANIFEST="$BACKUP_DIR/manifest.json"

for path in "$POSTGRES_DUMP" "$QDRANT_SNAPSHOT" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing backup artifact: $path" >&2; exit 2; }
done

command -v pg_restore >/dev/null || { echo "pg_restore is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }

POSTGRES_DUMP="$POSTGRES_DUMP" QDRANT_SNAPSHOT="$QDRANT_SNAPSHOT" MANIFEST="$MANIFEST" python - <<'PY'
import hashlib, json, os

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

with open(os.environ['MANIFEST'], encoding='utf-8') as f:
    manifest = json.load(f)
checks = [
    (os.environ['POSTGRES_DUMP'], manifest['postgres']['sha256']),
    (os.environ['QDRANT_SNAPSHOT'], manifest['qdrant']['sha256']),
]
for path, expected in checks:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f'Checksum mismatch for {path}: {actual} != {expected}')
print('Backup checksums verified')
PY

printf 'Restoring PostgreSQL...\n'
pg_restore \
  --dbname="$POSTGRES_DSN" \
  --clean --if-exists --no-owner --no-privileges \
  "$POSTGRES_DUMP"

headers=()
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  headers=(-H "api-key: ${QDRANT_API_KEY}")
fi

printf 'Restoring Qdrant collection %s...\n' "$QDRANT_COLLECTION"
curl -fsS -X POST \
  "${QDRANT_URL%/}/collections/${QDRANT_COLLECTION}/snapshots/upload?wait=true&priority=snapshot" \
  "${headers[@]}" \
  -F "snapshot=@${QDRANT_SNAPSHOT}" >/dev/null

printf 'Restore complete. Run application readiness and retrieval smoke before reopening traffic.\n'
