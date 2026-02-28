"""VPS-FastSearch - Fast memory/vector search for CPU-only VPS."""

from .chunker import chunk_markdown, chunk_text
from .client import DaemonNotRunningError, FastSearchClient, embed, search
from .config import FastSearchConfig, create_default_config, load_config
from .core import Embedder, Reranker, SearchDB, get_embedder, get_reranker

__version__ = "0.3.0"
__all__ = [
    # Core classes
    "Embedder",
    "Reranker",
    "SearchDB",
    "get_embedder",
    "get_reranker",
    # Chunking
    "chunk_text",
    "chunk_markdown",
    # Client
    "FastSearchClient",
    "DaemonNotRunningError",
    "search",
    "embed",
    # Config
    "FastSearchConfig",
    "load_config",
    "create_default_config",
]
