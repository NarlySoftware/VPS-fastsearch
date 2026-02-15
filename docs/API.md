# Python API Reference

FastSearch provides a Python API for programmatic access to all features.

## Installation

```bash
pip install vps-fastsearch

# With reranking support
pip install "vps-fastsearch[rerank]"
```

## Quick Start

```python
from vps_fastsearch import FastSearchClient, search, embed

# Use the client (recommended)
with FastSearchClient() as client:
    results = client.search("query")

# Or use convenience functions
results = search("query")
vectors = embed(["text1", "text2"])
```

---

## FastSearchClient

The main client class for connecting to the FastSearch daemon.

### Constructor

```python
FastSearchClient(
    socket_path: str | None = None,
    config_path: str | None = None,
    timeout: float = 30.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `socket_path` | `str \| None` | `None` | Unix socket path (from config if None) |
| `config_path` | `str \| None` | `None` | Config file path |
| `timeout` | `float` | `30.0` | Socket timeout in seconds |

### Example

```python
from vps_fastsearch import FastSearchClient

# Default connection
client = FastSearchClient()

# Custom socket
client = FastSearchClient(socket_path="/tmp/custom.sock")

# Custom config
client = FastSearchClient(config_path="/etc/vps_fastsearch/config.yaml")

# Custom timeout
client = FastSearchClient(timeout=60.0)
```

---

### search()

Search indexed documents.

```python
def search(
    query: str,
    db_path: str = "vps_fastsearch.db",
    limit: int = 10,
    mode: str = "hybrid",
    rerank: bool = False,
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Search query text |
| `db_path` | `str` | `"vps_fastsearch.db"` | Path to database file |
| `limit` | `int` | `10` | Maximum results |
| `mode` | `str` | `"hybrid"` | Search mode: `hybrid`, `bm25`, `vector` |
| `rerank` | `bool` | `False` | Apply cross-encoder reranking |

**Returns:** `dict` with:
- `query`: Original query
- `mode`: Search mode used
- `reranked`: Whether reranking was applied
- `search_time_ms`: Search latency in milliseconds
- `results`: List of result dictionaries

**Result dictionary:**
```python
{
    "id": 15,
    "source": "/path/to/file.md",
    "chunk_index": 2,
    "content": "The chunk text...",
    "metadata": {"section": "Installation"},
    "rrf_score": 0.0323,      # hybrid mode
    "bm25_rank": 1,           # hybrid mode
    "vec_rank": 3,            # hybrid mode
    "rerank_score": 0.892,    # if reranked
    "rank": 1
}
```

### Examples

```python
# Basic search
result = client.search("how to configure")
for r in result["results"]:
    print(f"{r['rank']}. {r['source']} - {r['content'][:100]}")

# Search with options
result = client.search(
    query="authentication setup",
    limit=20,
    mode="hybrid",
    rerank=True
)

# BM25 only (keyword search)
result = client.search("socket_path", mode="bm25")

# Vector only (semantic search)
result = client.search("how to set up the system", mode="vector")

# Search different database
result = client.search("query", db_path="/var/lib/vps_fastsearch/main.db")
```

---

### embed()

Generate embeddings for texts.

```python
def embed(texts: list[str]) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `list[str]` | required | List of texts to embed |

**Returns:** `dict` with:
- `embeddings`: List of 768-dimensional vectors
- `count`: Number of embeddings generated
- `embed_time_ms`: Embedding latency in milliseconds

### Examples

```python
# Single text
result = client.embed(["Hello, world!"])
vector = result["embeddings"][0]  # 768-dim list

# Multiple texts (more efficient)
result = client.embed([
    "First document",
    "Second document",
    "Third document"
])
vectors = result["embeddings"]  # List of 3 vectors

# Use for custom similarity
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

result = client.embed(["query", "document"])
similarity = cosine_similarity(
    result["embeddings"][0],
    result["embeddings"][1]
)
```

---

### rerank()

Rerank documents against a query using cross-encoder.

```python
def rerank(query: str, documents: list[str]) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Query text |
| `documents` | `list[str]` | required | List of document texts |

**Returns:** `dict` with:
- `scores`: Raw relevance scores (same order as input)
- `ranked`: Sorted list of `{"index": int, "score": float}`
- `rerank_time_ms`: Reranking latency in milliseconds

### Examples

```python
# Rerank documents
result = client.rerank(
    query="How to install Python?",
    documents=[
        "Python can be installed using apt-get on Ubuntu.",
        "JavaScript is a programming language.",
        "To install Python, download from python.org.",
        "Python is a snake species."
    ]
)

# Get best match
best_idx = result["ranked"][0]["index"]
best_score = result["ranked"][0]["score"]
print(f"Best: Document {best_idx} (score: {best_score:.4f})")

# Iterate ranked results
for item in result["ranked"]:
    print(f"#{item['index']}: {item['score']:.4f}")
```

---

### status()

Get daemon status.

```python
def status() -> dict[str, Any]
```

**Returns:** `dict` with:
- `uptime_seconds`: Daemon uptime
- `request_count`: Total requests handled
- `socket_path`: Unix socket path
- `loaded_models`: Dict of loaded model info
- `total_memory_mb`: Current memory usage
- `max_memory_mb`: Memory budget

### Example

```python
status = client.status()
print(f"Uptime: {status['uptime_seconds']:.0f}s")
print(f"Requests: {status['request_count']}")
print(f"Memory: {status['total_memory_mb']:.0f}MB / {status['max_memory_mb']}MB")

for slot, info in status["loaded_models"].items():
    print(f"  {slot}: {info['memory_mb']:.0f}MB (idle: {info['idle_seconds']:.0f}s)")
```

---

### load_model() / unload_model()

Manually control model loading.

```python
def load_model(slot: str) -> dict[str, Any]
def unload_model(slot: str) -> dict[str, Any]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `slot` | `str` | Model slot: `"embedder"` or `"reranker"` |

### Example

```python
# Pre-load reranker
result = client.load_model("reranker")
print(f"Loaded: {result['slot']}, Memory: {result['memory_mb']}MB")

# Unload reranker (free memory)
result = client.unload_model("reranker")
print(f"Unloaded: {result['slot']}")
```

---

### reload_config()

Reload daemon configuration without restart.

```python
def reload_config(config_path: str | None = None) -> dict[str, Any]
```

### Example

```python
result = client.reload_config("/etc/vps_fastsearch/config.yaml")
print(f"Reloaded: {result['reloaded']}")
```

---

### ping() / shutdown() / close()

Connection management.

```python
def ping() -> bool           # Check if daemon responds
def shutdown() -> dict        # Stop the daemon
def close()                   # Close client connection
```

### Example

```python
# Check daemon
if client.ping():
    print("Daemon is running")

# Clean up
client.close()

# Or use context manager (auto-closes)
with FastSearchClient() as client:
    results = client.search("query")
```

---

### Static Methods

```python
@staticmethod
def is_daemon_running(socket_path: str | None = None) -> bool
```

Check if daemon is running without creating a client.

```python
from vps_fastsearch import FastSearchClient

if FastSearchClient.is_daemon_running():
    print("Daemon is available")
else:
    print("Start the daemon with: vps-fastsearch daemon start")
```

---

## Convenience Functions

Top-level functions that automatically use daemon or fall back to direct mode.

### search()

```python
from vps_fastsearch import search

results = search("query")
results = search("query", limit=20, rerank=True)
```

### embed()

```python
from vps_fastsearch import embed

vectors = embed(["text1", "text2"])
```

---

## Core Classes

Lower-level classes for direct usage without daemon.

### Embedder

Generate embeddings directly.

```python
from vps_fastsearch import Embedder

class Embedder:
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DIMENSIONS = 768
    
    def __init__(self, model_name: str | None = None)
    def embed(self, texts: list[str]) -> list[list[float]]
    def embed_single(self, text: str) -> list[float]
```

### Example

```python
from vps_fastsearch import Embedder

# Create embedder (loads model)
embedder = Embedder()

# Generate embeddings
vectors = embedder.embed(["Hello world", "Goodbye world"])

# Single text
vector = embedder.embed_single("Hello world")
print(f"Dimensions: {len(vector)}")  # 768
```

### Singleton Access

```python
from vps_fastsearch import get_embedder

# Get singleton instance (reuses loaded model)
embedder = get_embedder()
```

---

### Reranker

Cross-encoder reranking.

```python
from vps_fastsearch import Reranker

class Reranker:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str | None = None)
    def rerank(self, query: str, documents: list[str]) -> list[float]
    def rerank_with_indices(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[tuple[int, float]]
```

### Example

```python
from vps_fastsearch import Reranker

reranker = Reranker()

# Get scores
scores = reranker.rerank("query", ["doc1", "doc2", "doc3"])

# Get sorted indices
ranked = reranker.rerank_with_indices("query", docs, top_k=5)
for idx, score in ranked:
    print(f"Doc {idx}: {score:.4f}")
```

### Singleton Access

```python
from vps_fastsearch import get_reranker

reranker = get_reranker()
```

---

### SearchDB

Database operations.

```python
from vps_fastsearch import SearchDB

class SearchDB:
    def __init__(self, db_path: str | Path = "vps_fastsearch.db")
    
    # Indexing
    def index_document(
        self,
        source: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> int
    
    def index_batch(
        self,
        items: list[tuple[str, int, str, list[float], dict | None]],
    ) -> list[int]
    
    # Searching
    def search_bm25(self, query: str, limit: int = 10) -> list[dict]
    def search_vector(self, embedding: list[float], limit: int = 10) -> list[dict]
    def search_hybrid(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        k: int = 60,
        bm25_weight: float = 1.0,
        vec_weight: float = 1.0,
    ) -> list[dict]
    def search_hybrid_reranked(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        rerank_top_k: int = 20,
        reranker: Reranker | None = None,
    ) -> list[dict]
    
    # Management
    def delete_source(self, source: str) -> int
    def get_stats(self) -> dict
    def close(self)
```

### Example

```python
from vps_fastsearch import SearchDB, Embedder

# Open database
db = SearchDB("myproject.db")
embedder = Embedder()

# Index a document
embedding = embedder.embed_single("This is the content")
doc_id = db.index_document(
    source="/path/to/file.md",
    chunk_index=0,
    content="This is the content",
    embedding=embedding,
    metadata={"section": "Introduction"}
)

# Search
query_embedding = embedder.embed_single("search query")
results = db.search_hybrid("search query", query_embedding, limit=10)

# Get stats
stats = db.get_stats()
print(f"Total chunks: {stats['total_chunks']}")

# Clean up
db.close()
```

---

### Chunking Functions

```python
from vps_fastsearch import chunk_text, chunk_markdown

def chunk_text(
    text: str,
    target_chars: int = 2000,
    overlap_chars: int = 200,
) -> Iterator[str]

def chunk_markdown(
    text: str,
    target_chars: int = 2000,
    overlap_chars: int = 200,
) -> Iterator[tuple[str, dict]]
```

### Example

```python
from vps_fastsearch import chunk_text, chunk_markdown

# Plain text
text = open("document.txt").read()
for chunk in chunk_text(text):
    print(f"Chunk: {len(chunk)} chars")

# Markdown (with section metadata)
markdown = open("README.md").read()
for chunk, metadata in chunk_markdown(markdown):
    print(f"Section: {metadata['section']}, {len(chunk)} chars")
```

---

## Configuration API

```python
from vps_fastsearch import (
    FastSearchConfig,
    load_config,
    create_default_config
)

# Load config
config = load_config()  # Uses default path
config = load_config("/path/to/config.yaml")

# Access settings
print(config.daemon.socket_path)
print(config.memory.max_ram_mb)
print(config.models["embedder"].name)

# Create default config file
path = create_default_config()
path = create_default_config("/custom/path/config.yaml")
```

---

## Exceptions

```python
from vps_fastsearch import DaemonNotRunningError

class FastSearchError(Exception):
    """Base exception"""

class DaemonNotRunningError(FastSearchError):
    """Daemon is not running or unreachable"""
```

### Example

```python
from vps_fastsearch import FastSearchClient, DaemonNotRunningError

try:
    client = FastSearchClient()
    results = client.search("query")
except DaemonNotRunningError:
    print("Daemon not running. Start with: vps-fastsearch daemon start")
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Example: Index and search with FastSearch."""

from pathlib import Path
from vps_fastsearch import (
    FastSearchClient,
    SearchDB,
    Embedder,
    chunk_markdown,
    DaemonNotRunningError
)


def index_directory(db_path: str, directory: str):
    """Index all markdown files in a directory."""
    db = SearchDB(db_path)
    embedder = Embedder()
    
    total_chunks = 0
    for path in Path(directory).glob("**/*.md"):
        content = path.read_text()
        chunks = list(chunk_markdown(content))
        
        if not chunks:
            continue
        
        # Generate embeddings
        texts = [c[0] for c in chunks]
        embeddings = embedder.embed(texts)
        
        # Index
        for i, ((text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            db.index_document(
                source=str(path),
                chunk_index=i,
                content=text,
                embedding=embedding,
                metadata=metadata,
            )
        
        total_chunks += len(chunks)
        print(f"Indexed {path.name}: {len(chunks)} chunks")
    
    db.close()
    print(f"\nTotal: {total_chunks} chunks")


def search_with_fallback(query: str, db_path: str = "vps_fastsearch.db"):
    """Search using daemon if available, otherwise direct."""
    try:
        with FastSearchClient() as client:
            result = client.search(query, db_path=db_path)
            print(f"[daemon] {result['search_time_ms']:.1f}ms")
            return result["results"]
    except DaemonNotRunningError:
        # Fall back to direct
        db = SearchDB(db_path)
        embedder = Embedder()
        embedding = embedder.embed_single(query)
        results = db.search_hybrid(query, embedding)
        db.close()
        print("[direct]")
        return results


if __name__ == "__main__":
    # Index
    index_directory("example.db", "./docs")
    
    # Search
    results = search_with_fallback("how to configure", "example.db")
    for r in results[:3]:
        print(f"\n{r['rank']}. {r['source']}")
        print(f"   {r['content'][:150]}...")
```
