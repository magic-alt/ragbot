from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from cli.job_wait import wait_for_job
from services.api.app.agent.nodes.route import route_node
from services.api.app.agent.sql_disabled import DisabledSqlEngine
from services.api.app.agent.state import build_initial_state
from services.api.app.middleware import _metric_path
from services.api.app.observability.prometheus import render_prometheus
from services.api.app.routes.openai_compat import Message, _prepare_messages
from services.api.app.runtime import validate_production_environment
from services.api.app.storage.models import IngestionJob, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker.main import _execute_claimed_job
from services.worker.pipeline import run_ingest_pipeline
from services.worker.source_fence import job_stats_for_source, source_generation


class SqlSecurityBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sql_route_falls_back_when_tool_disabled(self):
        repo = InMemoryRepo()
        services = SimpleNamespace(
            repo=repo,
            sql_engine=DisabledSqlEngine(),
            llm=SimpleNamespace(enabled=False),
        )
        state = build_initial_state("select * from chunks", "tenant-a", "user-a")
        state = await route_node(state, services)
        self.assertEqual(state.route, "doc_rag")

    def test_disabled_sql_engine_is_fail_closed(self):
        with self.assertRaises(PermissionError):
            DisabledSqlEngine().query("select 1")

    def test_production_rejects_control_plane_dsn_reuse_for_agent_sql(self):
        env = {
            "RAGBOT_ENV": "production",
            "POSTGRES_DSN": "postgresql://internal/ragbot",
            "QDRANT_URL": "http://qdrant:6333",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_API_KEY": "test",
            "RAGBOT_API_KEYS": "key-a",
            "RAGBOT_API_KEY_PRINCIPALS": "configured",
            "RAGBOT_INGESTION_MODE": "worker",
            "RAGBOT_SQL_TOOL_ENABLED": "true",
            "RAGBOT_SQL_DSN": "postgresql://internal/ragbot",
            "RAGBOT_SQL_ALLOWED_SCHEMAS": "analytics",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "must not equal POSTGRES_DSN"):
                validate_production_environment()


class SourceGenerationFenceTests(unittest.TestCase):
    def _source(self, updated_at) -> Source:
        return Source(
            source_id="source-a",
            tenant_id="tenant-a",
            source_type="local_fs",
            name="source-a",
            config={"path": "/definitely/not/read"},
            created_at="2026-09-04T00:00:00+00:00",
            updated_at=updated_at,
        )

    def test_api_string_and_postgres_datetime_generation_are_equal(self):
        api_source = self._source("2026-09-04T08:15:00+08:00")
        pg_source = self._source(datetime(2026, 9, 4, 0, 15, tzinfo=timezone.utc))
        self.assertEqual(source_generation(api_source), source_generation(pg_source))

    def test_pipeline_fails_before_connector_when_generation_changed(self):
        repo = InMemoryRepo()
        submitted = self._source("2026-09-04T00:00:00+00:00")
        repo.add_source(self._source("2026-09-04T00:01:00+00:00"))
        job = IngestionJob(
            job_id="job-a",
            tenant_id="tenant-a",
            source_id="source-a",
            source_type="local_fs",
            source_config=dict(submitted.config),
            status="running",
            stats=job_stats_for_source(submitted),
        )
        repo.add_job(job)

        result = run_ingest_pipeline(
            submitted,
            repo,
            SimpleNamespace(),
            job_id="job-a",
            existing_job=True,
            expected_source_generation=source_generation(submitted),
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("generation changed", result.error or "")

    def test_worker_dead_letters_stale_generation_before_execution(self):
        repo = InMemoryRepo()
        submitted = self._source("2026-09-04T00:00:00+00:00")
        repo.add_source(self._source("2026-09-04T00:02:00+00:00"))
        job = IngestionJob(
            job_id="job-worker",
            tenant_id="tenant-a",
            source_id="source-a",
            source_type="local_fs",
            source_config=dict(submitted.config),
            status="running",
            attempts=1,
            lease_owner="worker-a",
            stats=job_stats_for_source(submitted),
        )
        repo.add_job(job)
        services = SimpleNamespace(repo=repo, qdrant=SimpleNamespace(), embedder=None)

        _execute_claimed_job(
            job,
            services,
            worker_id="worker-a",
            lease_seconds=30,
        )
        latest = repo.get_job("job-worker")
        self.assertEqual(latest.status, "dead_lettered")
        self.assertEqual(latest.failure_class, "source_generation_mismatch")


class CliTerminalStateTests(unittest.TestCase):
    def test_dead_lettered_job_raises_without_waiting_for_timeout(self):
        calls = []

        def request_fn(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "status": "dead_lettered",
                "failure_class": "configuration_error",
                "error": "bad connector configuration",
            }

        with self.assertRaisesRegex(RuntimeError, "configuration_error"):
            wait_for_job(
                request_fn,
                "http://localhost:8000",
                "job-dlq",
                headers={},
                timeout=60,
                poll_interval=5,
                quiet=True,
            )
        self.assertEqual(len(calls), 1)


class OpenAIConversationTests(unittest.TestCase):
    def test_prepare_messages_preserves_history_and_last_user_query(self):
        query, conversation, system_prompt = _prepare_messages([
            Message(role="system", content="Answer concisely."),
            Message(role="user", content="What is lease recovery?"),
            Message(role="assistant", content="It repairs expired work."),
            Message(role="user", content="How does it interact with DLQ?"),
        ])
        self.assertEqual(query, "How does it interact with DLQ?")
        self.assertEqual(system_prompt, "Answer concisely.")
        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": "What is lease recovery?"},
                {"role": "assistant", "content": "It repairs expired work."},
            ],
        )


class ProductionMetricsTests(unittest.TestCase):
    def test_metric_path_uses_route_template(self):
        request = SimpleNamespace(
            scope={"route": SimpleNamespace(path="/sources/{source_id}")},
            url=SimpleNamespace(path="/sources/secret-source-id"),
        )
        self.assertEqual(_metric_path(request), "/sources/{source_id}")

    def test_prometheus_export_contains_queue_metrics(self):
        try:
            payload, content_type = render_prometheus(InMemoryRepo())
        except RuntimeError as exc:
            self.skipTest(str(exc))
        text = payload.decode("utf-8")
        self.assertIn("ragbot_ingestion_jobs", text)
        self.assertIn("ragbot_http_requests_total", text)
        self.assertIn("text/plain", content_type)


if __name__ == "__main__":
    unittest.main()
