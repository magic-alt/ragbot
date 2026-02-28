#!/usr/bin/env bash
# Initialize Qdrant collection for ragbot.
# Usage: ./init_collection.sh [QDRANT_URL] [COLLECTION_NAME] [VECTOR_DIM]
#
# Environment variables (overridden by positional args):
#   QDRANT_URL       - default: http://localhost:6333
#   QDRANT_COLLECTION - default: rag_chunks
#   QDRANT_DIM        - default: 1536

set -euo pipefail

QDRANT_URL="${1:-${QDRANT_URL:-http://localhost:6333}}"
COLLECTION="${2:-${QDRANT_COLLECTION:-rag_chunks}}"
DIM="${3:-${QDRANT_DIM:-1536}}"

echo "Initializing Qdrant collection '${COLLECTION}' at ${QDRANT_URL} with dim=${DIM}"

# Check if collection already exists
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/collections/${COLLECTION}")

if [ "$STATUS" = "200" ]; then
    echo "Collection '${COLLECTION}' already exists. Skipping creation."
    exit 0
fi

# Create collection with cosine distance
curl -s -X PUT "${QDRANT_URL}/collections/${COLLECTION}" \
    -H "Content-Type: application/json" \
    -d "{
        \"vectors\": {
            \"size\": ${DIM},
            \"distance\": \"Cosine\"
        },
        \"optimizers_config\": {
            \"indexing_threshold\": 10000
        }
    }"

echo ""
echo "Collection '${COLLECTION}' created successfully."

# Create payload indexes for common filter fields
for FIELD in tenant_id doc_id acl_hash; do
    curl -s -X PUT "${QDRANT_URL}/collections/${COLLECTION}/index" \
        -H "Content-Type: application/json" \
        -d "{
            \"field_name\": \"${FIELD}\",
            \"field_schema\": \"keyword\"
        }"
    echo "  Index on '${FIELD}' created."
done

# Create index for tags (array of keywords)
curl -s -X PUT "${QDRANT_URL}/collections/${COLLECTION}/index" \
    -H "Content-Type: application/json" \
    -d "{
        \"field_name\": \"tags\",
        \"field_schema\": \"keyword\"
    }"
echo "  Index on 'tags' created."

echo "Qdrant initialization complete."
