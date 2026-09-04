from __future__ import annotations

from .protocol import ChunkingSpec


class LangChainChunker:
    """Optional LangChain text/code splitter behind Ragbot's Chunker port."""

    def __init__(self, spec: ChunkingSpec) -> None:
        try:
            from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "LangChain chunking requires ragbot[document-transformers] "
                "or langchain-text-splitters"
            ) from exc

        self._spec = spec
        kwargs = {
            "chunk_size": spec.chunk_size,
            "chunk_overlap": spec.chunk_overlap,
            "length_function": len,
        }
        if spec.strategy == "recursive":
            self._splitter = RecursiveCharacterTextSplitter(**kwargs)
        elif spec.strategy == "code":
            language = _langchain_language(Language, spec.language)
            if language is None:
                # Unknown languages still benefit from recursive structural
                # boundaries without Ragbot maintaining its own separator table.
                self._splitter = RecursiveCharacterTextSplitter(**kwargs)
            else:
                self._splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    **kwargs,
                )
        else:  # pragma: no cover - registry validates strategies
            raise ValueError(f"Unsupported LangChain chunking strategy: {spec.strategy}")

    @property
    def spec(self) -> ChunkingSpec:
        return self._spec

    def split(self, text: str) -> list[str]:
        return [part.strip() for part in self._splitter.split_text(text) if part.strip()]


class LlamaIndexChunker:
    """Optional LlamaIndex sentence splitter behind Ragbot's Chunker port."""

    def __init__(self, spec: ChunkingSpec) -> None:
        try:
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "LlamaIndex chunking requires ragbot[document-transformers] or llama-index-core"
            ) from exc

        if spec.strategy != "sentence":  # pragma: no cover - registry validates strategies
            raise ValueError(f"Unsupported LlamaIndex chunking strategy: {spec.strategy}")
        self._spec = spec
        # Existing Ragbot Source configs define chunk sizes in characters. Using
        # list as tokenizer preserves that contract while gaining sentence-aware
        # boundaries. A future explicit token-budget strategy can version this.
        self._splitter = SentenceSplitter(
            chunk_size=spec.chunk_size,
            chunk_overlap=spec.chunk_overlap,
            tokenizer=list,
        )

    @property
    def spec(self) -> ChunkingSpec:
        return self._spec

    def split(self, text: str) -> list[str]:
        try:
            from llama_index.core import Document
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LlamaIndex chunking requires llama-index-core") from exc
        nodes = self._splitter.get_nodes_from_documents([Document(text=text)])
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]


def _langchain_language(language_enum, language: str | None):
    normalized = str(language or "").strip().lower()
    mapping = {
        "python": "PYTHON",
        "javascript": "JS",
        "typescript": "TS",
        "java": "JAVA",
        "go": "GO",
        "rust": "RUST",
        "c": "C",
        "cpp": "CPP",
        "csharp": "CSHARP",
        "ruby": "RUBY",
        "php": "PHP",
        "kotlin": "KOTLIN",
        "html": "HTML",
        "markdown": "MARKDOWN",
        "rst": "RST",
        "proto": "PROTO",
    }
    member = mapping.get(normalized)
    return getattr(language_enum, member, None) if member else None
