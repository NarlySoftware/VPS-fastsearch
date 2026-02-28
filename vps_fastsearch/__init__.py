"""VPS-FastSearch - Fast memory/vector search for CPU-only VPS."""

from .chunker import chunk_markdown, chunk_text
from .client import DaemonNotRunningError, FastSearchClient, FastSearchError, embed, search
from .config import FastSearchConfig, create_default_config, load_config
from .core import (
    BM25Result,
    Embedder,
    HybridResult,
    Reranker,
    RerankResult,
    SearchDB,
    VectorResult,
    get_embedder,
    get_reranker,
)

__version__ = "0.3.0"
__all__ = [
    # Core classes
    "Embedder",
    "Reranker",
    "SearchDB",
    "get_embedder",
    "get_reranker",
    # Search result types
    "BM25Result",
    "VectorResult",
    "HybridResult",
    "RerankResult",
    # Chunking
    "chunk_text",
    "chunk_markdown",
    # Client
    "FastSearchClient",
    "FastSearchError",
    "DaemonNotRunningError",
    "search",
    "embed",
    # Config
    "FastSearchConfig",
    "load_config",
    "create_default_config",
]
