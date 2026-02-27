# VPS-FastSearch

**Fast hybrid search for CPU-only VPS using ONNX Runtime.**

VPS-FastSearch combines BM25 full-text search with vector similarity search using Reciprocal Rank Fusion (RRF). Designed for resource-constrained environments, it features a daemon mode that keeps models loaded for instant search latency.

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

## Quick Start

### Installation (Debian 13)

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/NarlySoftware/VPS-fastsearch.git ~/fastsearch
cd ~/fastsearch
./install.sh
source ~/.bashrc
```

### Installation (pip)

```bash
# Basic installation
pip install vps-fastsearch

# With reranking support
pip install "vps-fastsearch[rerank]"

# From source (development)
git clone https://github.com/NarlySoftware/VPS-fastsearch.git
cd VPS-fastsearch
pip install -e ".[all]"
```

See [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for the full setup walkthrough.

### Index Documents

```bash
# Index a file
vps-fastsearch index README.md

# Index a directory
vps-fastsearch index ./docs --glob "*.md"

# Re-index (replace existing)
vps-fastsearch index ./docs --reindex
```

### Search

```bash
# Hybrid search (default)
vps-fastsearch search "how to configure"

# With reranking (more accurate)
vps-fastsearch search "complex query" --rerank

# Specific mode
vps-fastsearch search "exact phrase" --mode bm25
vps-fastsearch search "semantic meaning" --mode vector
```

### Start the Daemon (Recommended)

```bash
# Start daemon (keeps models in memory)
vps-fastsearch daemon start --detach

# Now searches are instant (~5ms vs ~800ms cold start)
vps-fastsearch search "fast query"

# Check status
vps-fastsearch daemon status

# Stop
vps-fastsearch daemon stop
```

## Python API

```python
from vps_fastsearch import FastSearchClient

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
from vps_fastsearch import search, embed

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
│   CLI       │────▶│  VPS-FastSearch Daemon               │
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

## Author

Cliff Jones
