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

restore_collection="$(MANIFEST="$MANIFEST" QDRANT_COLLECTION="$QDRANT_COLLECTION" python - <<'PY'
import json, os
with open(os.environ['MANIFEST'], encoding='utf-8') as f:
    manifest = json.load(f)
print((manifest.get('qdrant') or {}).get('collection') or os.environ['QDRANT_COLLECTION'])
PY
)"
restore_alias="$(MANIFEST="$MANIFEST" QDRANT_COLLECTION="$QDRANT_COLLECTION" python - <<'PY'
import json, os
with open(os.environ['MANIFEST'], encoding='utf-8') as f:
    manifest = json.load(f)
qdrant = manifest.get('qdrant') or {}
print(qdrant.get('alias') or os.environ.get('QDRANT_INDEX_ALIAS') or f"{os.environ['QDRANT_COLLECTION']}_active")
PY
)"

printf 'Restoring PostgreSQL...\n'
pg_restore \
  --dbname="$POSTGRES_DSN" \
  --clean --if-exists --no-owner --no-privileges \
  "$POSTGRES_DUMP"

headers=()
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  headers=(-H "api-key: ${QDRANT_API_KEY}")
fi

printf 'Restoring Qdrant physical collection %s...\n' "$restore_collection"
curl -fsS -X POST \
  "${QDRANT_URL%/}/collections/${restore_collection}/snapshots/upload?wait=true&priority=snapshot" \
  "${headers[@]}" \
  -F "snapshot=@${QDRANT_SNAPSHOT}" >/dev/null

# Collection-level Qdrant snapshots intentionally do not contain aliases. The
# alias is restored as a separate atomic control-plane operation and verified
# before this script returns success.
printf 'Restoring Qdrant alias %s -> %s...\n' "$restore_alias" "$restore_collection"

alias_restored=false
last_aliases='{}'
for attempt in $(seq 1 10); do
  aliases_json="$(curl -fsS "${QDRANT_URL%/}/aliases" "${headers[@]}")"
  last_aliases="$aliases_json"
  current_target="$(
    ALIASES_JSON="$aliases_json" RESTORE_ALIAS="$restore_alias" python - <<'PY'
import json, os
payload = json.loads(os.environ['ALIASES_JSON'])
alias = os.environ['RESTORE_ALIAS']
for item in (payload.get('result') or {}).get('aliases', []) or []:
    if item.get('alias_name') == alias:
        print(item.get('collection_name') or '')
        break
else:
    print('')
PY
  )"

  if [[ "$current_target" == "$restore_collection" ]]; then
    alias_restored=true
    break
  fi

  alias_actions="$(
    CURRENT_TARGET="$current_target" \
    RESTORE_ALIAS="$restore_alias" \
    RESTORE_COLLECTION="$restore_collection" \
    python - <<'PY'
import json, os
alias = os.environ['RESTORE_ALIAS']
collection = os.environ['RESTORE_COLLECTION']
current = os.environ.get('CURRENT_TARGET') or ''
actions = []
if current and current != collection:
    actions.append({'delete_alias': {'alias_name': alias}})
actions.append({'create_alias': {'collection_name': collection, 'alias_name': alias}})
print(json.dumps({'actions': actions}, separators=(',', ':')))
PY
  )"

  curl -fsS -X POST \
    "${QDRANT_URL%/}/collections/aliases?timeout=10" \
    "${headers[@]}" \
    -H 'Content-Type: application/json' \
    --data-raw "$alias_actions" >/dev/null

  if [[ "$attempt" -lt 10 ]]; then
    sleep 0.2
  fi
done

if [[ "$alias_restored" != true ]]; then
  last_aliases="$(curl -fsS "${QDRANT_URL%/}/aliases" "${headers[@]}")"
  final_target="$(
    ALIASES_JSON="$last_aliases" RESTORE_ALIAS="$restore_alias" python - <<'PY'
import json, os
payload = json.loads(os.environ['ALIASES_JSON'])
alias = os.environ['RESTORE_ALIAS']
for item in (payload.get('result') or {}).get('aliases', []) or []:
    if item.get('alias_name') == alias:
        print(item.get('collection_name') or '')
        break
else:
    print('')
PY
  )"
  if [[ "$final_target" == "$restore_collection" ]]; then
    alias_restored=true
  fi
fi

if [[ "$alias_restored" != true ]]; then
  echo "Failed to restore Qdrant alias '$restore_alias' -> '$restore_collection'." >&2
  echo "Observed alias state: $last_aliases" >&2
  exit 1
fi

printf 'Qdrant alias verified: %s -> %s\n' "$restore_alias" "$restore_collection"
printf 'Restore complete. Run `python scripts/rag_index.py reconcile` and application readiness before reopening traffic.\n'
