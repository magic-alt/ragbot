from __future__ import annotations

from benchmarks.runtime_warmup import warm_chunker_runtime


def test_warm_chunker_runtime_uses_exact_chunking_contract():
    config = {
        "provider": "ragbot",
        "strategy": "fixed",
        "block_coalescing": {
            "enabled": True,
            "target_chars": 320,
            "respect_page": True,
            "respect_section": True,
        },
    }
    result = warm_chunker_runtime(
        config,
        chunk_size=80,
        chunk_overlap=10,
    )
    assert result["provider"] == "ragbot"
    assert result["strategy"] == "fixed"
    assert isinstance(result["config_hash"], str)
    assert result["config_hash"]
    assert result["chunks"] >= 1
