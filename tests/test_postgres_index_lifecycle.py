from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.api.app.storage.index_support import ensure_index_repository
from services.api.app.storage.models import IndexVersion
from services.api.app.storage.pg_repo import PostgresRepo

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_DSN"),
    reason="POSTGRES_TEST_DSN not configured",
)


def _version(suffix: str, name: str, *, status: str, contract: str, dim: int) -> IndexVersion:
    now = datetime.now(timezone.utc).isoformat()
    return IndexVersion(
        index_version_id=f"idx-{name}-{suffix}",
        alias_name=f"rag-index-{suffix}-active",
        physical_collection=f"rag-index-{suffix}-{name}",
        embedding_contract_id=contract,
        embedding_spec={"provider_id": "test", "model": name, "dimension": dim, "distance": "cosine"},
        vector_schema={"dense": {"dimension": dim, "distance": "cosine"}},
        status=status,
        created_at=now,
        ready_at=now if status in {"ready", "active"} else None,
        activated_at=now if status == "active" else None,
        validation_evidence={"smoke": True} if status in {"ready", "active"} else {},
    )


def test_postgres_index_version_activation_rollback_and_retention() -> None:
    repo = PostgresRepo(os.environ["POSTGRES_TEST_DSN"], pool_min=1, pool_max=2)
    ensure_index_repository(repo)
    suffix = uuid.uuid4().hex[:12]
    try:
        old = _version(suffix, "old", status="active", contract="emb-old", dim=4)
        new = _version(suffix, "new", status="ready", contract="emb-new", dim=6)
        repo.add_index_version(old)
        repo.add_index_version(new)

        assert repo.get_active_index_version(old.alias_name).index_version_id == old.index_version_id
        assert repo.get_index_version_by_collection(old.alias_name, new.physical_collection).index_version_id == new.index_version_id

        delete_after = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        activated = repo.activate_index_version(
            new.index_version_id,
            previous_index_version_id=old.index_version_id,
            delete_after=delete_after,
        )
        assert activated.status == "active"
        assert repo.get_active_index_version(old.alias_name).index_version_id == new.index_version_id
        retired_old = repo.get_index_version(old.index_version_id)
        assert retired_old.status == "retired"
        assert retired_old.delete_after is not None

        rolled_back = repo.activate_index_version(
            old.index_version_id,
            previous_index_version_id=new.index_version_id,
            delete_after=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        )
        assert rolled_back.status == "active"
        assert repo.get_active_index_version(old.alias_name).index_version_id == old.index_version_id
        retired_new = repo.get_index_version(new.index_version_id)
        assert retired_new.status == "retired"

        repo.update_index_version(
            new.index_version_id,
            delete_after=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        )
        prunable = repo.list_prunable_index_versions(datetime.now(timezone.utc).isoformat())
        assert new.index_version_id in {item.index_version_id for item in prunable}
    finally:
        with repo._pool.connection() as conn:
            conn.execute(
                "DELETE FROM vector_index_versions WHERE alias_name = %s",
                (f"rag-index-{suffix}-active",),
            )
        repo.close()
