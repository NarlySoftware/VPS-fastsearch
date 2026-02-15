# FastSearch

**Fast hybrid search for CPU-only VPS using ONNX Runtime.**

FastSearch combines BM25 full-text search with vector similarity search using Reciprocal Rank Fusion (RRF). Designed for resource-constrained environments, it features a daemon mode that keeps models loaded for instant search latency.

[![Tests](https://github.com/NarlySoftware/VPS-fastsearch/actions/workflows/test.yml/badge.svg)](https://github.com/NarlySoftware/VPS-fastsearch/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **🔍 Hybrid Search** — Combines BM25 (full-text) + vector similarity using Reciprocal Rank Fusion
- **🎯 Cross-Encoder Reranking** — Optional ms-marco-MiniLM reranker for improved accuracy
- **⚡ Daemon Mode** — Unix socket server keeps models loaded for instant search (~5ms latency)
- **💾 Memory Management** — LRU eviction with configurable memory budget
- **🖥️ CPU Optimized** — Uses ONNX Runtime and FastEmbed for fast inference without GPU
- **📦 Single File Storage** — SQLite database with sqlite-vec for vector operations
- **📝 Markdown Aware** — Smart chunking with section awareness for documentation

## 🤖 Perfect for AI Assistants on VPS

FastSearch was built to solve a specific problem: **giving AI assistants like [OpenClaw](https://github.com/clawdbot/clawdbot) fast semantic search over conversation history without expensive API calls or GPU hardware.**

### The Problem

Running an AI assistant on a budget VPS, you need to search through memories, documents, and conversation history. Your options aren't great:

| Approach | Latency | Cost | Memory | Works on $5 VPS? |
|----------|---------|------|--------|------------------|
| OpenAI Embeddings API | 200-500ms | $0.0001/query | Minimal | ✅ but adds up |
| Local Sentence-Transformers | 800ms+ cold | Free | 2GB+ | ❌ too heavy |
| ChromaDB + local models | 500ms+ | Free | 1GB+ | ⚠️ marginal |
| **FastSearch (daemon)** | **4ms** | **Free** | **200-600MB** | **✅ built for this** |

### Why FastSearch Wins

With daemon mode, the embedding model stays loaded in memory. Searches that would take 850ms cold start complete in **4 milliseconds** — fast enough that your AI assistant can search its entire memory without noticeable delay.

```
Cold start:  ████████████████████████████████████████  850ms
Daemon mode: █                                          4ms
```

### Memory vs Quality Trade-off

Choose your model based on available RAM:

| VPS RAM | Recommended Model | FastSearch Memory | Search Quality |
|---------|-------------------|-------------------|----------------|
| 1GB | bge-small (384 dim) | ~200MB | Good |
| 2GB | bge-base (768 dim) | ~520MB | Better |
| 4GB+ | bge-base + reranker | ~610MB | Best |

The smaller model (bge-small) still delivers excellent results for most use cases — semantic search that understands meaning, not just keywords.

### Real-World Example

```python
from fastsearch import search

# AI assistant searching its memory before responding
relevant_context = search("user's preferred communication style", limit=5)

# Returns in ~4ms with daemon running
# [{"content": "User prefers concise responses...", "score": 0.89}, ...]
```

No API keys. No GPU. No waiting. Just fast, local semantic search.

---

## Quick Start

### Installation

```bash
# Basic installation
pip install fastsearch

# With reranking support
pip install "fastsearch[rerank]"

# From source
git clone https://github.com/your-username/fastsearch
cd fastsearch
pip install -e ".[all]"
```

### Index Documents

```bash
# Index a file
fastsearch index README.md

# Index a directory
fastsearch index ./docs --glob "*.md"

# Re-index (replace existing)
fastsearch index ./docs --reindex
```

### Search

```bash
# Hybrid search (default)
fastsearch search "how to configure"

# With reranking (more accurate)
fastsearch search "complex query" --rerank

# Specific mode
fastsearch search "exact phrase" --mode bm25
fastsearch search "semantic meaning" --mode vector
```

### Start the Daemon (Recommended)

```bash
# Start daemon (keeps models in memory)
fastsearch daemon start --detach

# Now searches are instant (~5ms vs ~800ms cold start)
fastsearch search "fast query"

# Check status
fastsearch daemon status

# Stop
fastsearch daemon stop
```

## Python API

```python
from fastsearch import FastSearchClient

# Connect to daemon
with FastSearchClient() as client:
    # Search
    results = client.search("query", limit=10)
    
    # Search with reranking
    results = client.search("query", rerank=True)
    
    # Get embeddings
    embeddings = client.embed(["text 1", "text 2"])
    
    # Rerank documents
    ranked = client.rerank("query", ["doc1", "doc2", "doc3"])
```

### Quick Functions

```python
from fastsearch import search, embed

# Uses daemon if available, falls back to direct
results = search("query")
vectors = embed(["text1", "text2"])
```

## Performance

| Operation | Cold Start | Daemon Mode |
|-----------|------------|-------------|
| Hybrid Search | ~850ms | **4ms** |
| Hybrid + Rerank | ~1100ms | **45ms** |
| BM25 Only | 2ms | 2ms |
| Vector Only | ~820ms | **3ms** |

Memory usage: ~500MB with embedder loaded, ~590MB with both models.

## Configuration

Create config at `~/.config/fastsearch/config.yaml`:

```yaml
daemon:
  socket_path: /tmp/fastsearch.sock
  log_level: INFO

models:
  embedder:
    name: "BAAI/bge-base-en-v1.5"
    keep_loaded: always

  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    keep_loaded: on_demand
    idle_timeout_seconds: 300

memory:
  max_ram_mb: 4000
  eviction_policy: lru
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────┐
│   CLI       │────▶│  FastSearch Daemon               │
└─────────────┘     │  ┌────────────────────────────┐  │
                    │  │  Model Manager (LRU)       │  │
┌─────────────┐     │  │  ├── Embedder (450MB)      │  │
│Python Client│────▶│  │  └── Reranker (90MB)       │  │
└─────────────┘     │  └────────────────────────────┘  │
       │            │                                  │
       │            │  ┌────────────────────────────┐  │
       │            │  │  Unix Socket Server        │  │
       │            │  │  JSON-RPC 2.0 Protocol     │  │
       │            │  └────────────────────────────┘  │
       │            └──────────────────────────────────┘
       │                          │
       └──────────────────────────┼─────────────────────┐
                                  │                     │
                            ┌─────▼─────┐         ┌─────▼─────┐
                            │ SQLite DB │         │ FTS5      │
                            │ sqlite-vec│         │ (BM25)    │
                            └───────────┘         └───────────┘
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and data flow
- [CLI Reference](docs/CLI.md) — Complete command line documentation
- [Python API](docs/API.md) — Python client and core classes
- [Configuration](docs/CONFIGURATION.md) — All configuration options
- [Deployment](docs/DEPLOYMENT.md) — Production setup guide
- [Performance](docs/PERFORMANCE.md) — Benchmarks and optimization
- [Integration](docs/INTEGRATION.md) — Using with other systems
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Common issues and solutions

## Requirements

- Python 3.10+
- ~500MB RAM for embedder model
- ~90MB additional for reranker
- Works on Linux, macOS, Windows (WSL2)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR.
