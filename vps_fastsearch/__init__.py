"""FastSearch - Fast memory/vector search for CPU-only VPS."""

from .core import Embedder, Reranker, SearchDB, get_embedder, get_reranker
from .chunker import chunk_text, chunk_markdown
from .client import FastSearchClient, DaemonNotRunningError, search, embed
from .config import FastSearchConfig, load_config, create_default_config

__version__ = "0.2.0"
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
