from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from scripts import ollama_rag_test
from services.api.app.llm.ollama import OllamaAdapter


class OllamaAdapterCompatibilityTests(unittest.TestCase):
    def test_rejects_unknown_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            OllamaAdapter(reasoning_effort="extreme")

    def test_reads_qwen_runtime_options_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OLLAMA_MODEL": "qwen3.8:27b",
                "OLLAMA_TIMEOUT_SECONDS": "420",
                "OLLAMA_REASONING_EFFORT": "none",
            },
            clear=False,
        ):
            adapter = OllamaAdapter()
        self.assertEqual(adapter.model, "qwen3.8:27b")
        self.assertEqual(adapter.timeout, 420.0)
        self.assertEqual(adapter.reasoning_effort, "none")

    def test_chat_json_uses_openai_compatible_fields(self) -> None:
        adapter = OllamaAdapter(
            base_url="http://localhost:11434",
            model="qwen3.8:27b",
            timeout=10,
            reasoning_effort="none",
        )
        adapter._post_json = AsyncMock(  # type: ignore[method-assign]
            return_value={"choices": [{"message": {"content": '{"route":"retrieve"}'}}]}
        )
        schema = {
            "type": "object",
            "properties": {"route": {"type": "string"}},
            "required": ["route"],
        }

        result = asyncio.run(
            adapter.chat_json(
                system="Route the request",
                user="Find the architecture section",
                schema=schema,
                max_output_tokens=128,
            )
        )

        self.assertEqual(result, {"route": "retrieve"})
        adapter._post_json.assert_awaited_once()
        path, payload = adapter._post_json.await_args.args
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen3.8:27b")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertNotIn("format", payload)
        self.assertNotIn("options", payload)


class OllamaRagEvaluationTests(unittest.TestCase):
    @patch.object(ollama_rag_test, "ragbot_chat")
    @patch.object(ollama_rag_test, "ragbot_search")
    @patch.object(ollama_rag_test, "check_ragbot_ready")
    @patch.object(ollama_rag_test, "check_ollama_model")
    def test_evaluate_passes_with_retrieval_and_grounded_answer(
        self,
        check_model,
        check_ready,
        search,
        chat,
    ) -> None:
        check_model.return_value = {
            "available": True,
            "model": "qwen3.8:27b",
            "url": "http://127.0.0.1:11434",
        }
        check_ready.return_value = {
            "status": "ready",
            "url": "http://127.0.0.1:8000",
            "checks": {"vector_store": True},
        }
        search.return_value = {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "score": 0.88,
                    "text": "Ragbot stores semantic vectors in Qdrant.",
                }
            ],
            "latency_ms": 12.0,
            "diagnostics": {"backend": "qdrant"},
        }
        chat.return_value = {
            "answer": "Ragbot uses Qdrant for semantic vector retrieval.",
            "citations": [{"kind": "chunk", "chunk_id": "chunk-1"}],
            "confidence": "high",
            "latency_ms": 101.0,
        }

        report = ollama_rag_test.evaluate(
            ragbot_url="http://127.0.0.1:8000",
            ollama_url="http://127.0.0.1:11434",
            model="qwen3.8:27b",
            query="Where are semantic vectors stored?",
            tenant="default",
            user="tester",
            api_key=None,
            top_k=5,
            min_retrieved=1,
            min_top_score=0.5,
            require_citations=True,
            timeout=300,
            run_direct_generation=True,
            reasoning_effort="none",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["retrieval"]["count"], 1)
        self.assertEqual(report["grounding"]["retrieved_citation_overlap"], ["chunk-1"])
        self.assertTrue(all(report["gates"].values()))

    @patch.object(ollama_rag_test, "ragbot_chat")
    @patch.object(ollama_rag_test, "ragbot_search")
    @patch.object(ollama_rag_test, "check_ragbot_ready")
    @patch.object(ollama_rag_test, "check_ollama_model")
    def test_evaluate_fails_when_answer_has_no_citations(
        self,
        check_model,
        check_ready,
        search,
        chat,
    ) -> None:
        check_model.return_value = {"available": True, "model": "qwen3.8:27b", "url": "ollama"}
        check_ready.return_value = {"status": "ready", "url": "ragbot", "checks": {"vector_store": True}}
        search.return_value = {
            "chunks": [{"chunk_id": "chunk-1", "doc_id": "doc-1", "score": 0.8, "text": "evidence"}],
            "latency_ms": 1.0,
        }
        chat.return_value = {"answer": "An answer without grounding.", "citations": [], "latency_ms": 2.0}

        report = ollama_rag_test.evaluate(
            ragbot_url="ragbot",
            ollama_url="ollama",
            model="qwen3.8:27b",
            query="question",
            tenant="default",
            user="tester",
            api_key=None,
            top_k=5,
            min_retrieved=1,
            min_top_score=None,
            require_citations=True,
            timeout=300,
            run_direct_generation=False,
            reasoning_effort="none",
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["citations_present"])


if __name__ == "__main__":
    unittest.main()
