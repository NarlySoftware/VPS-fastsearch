# FastSearch Documentation

Welcome to the FastSearch documentation. FastSearch is a fast hybrid search system combining BM25 full-text search with vector similarity, designed for CPU-only environments.

## Quick Links

| Document | Description |
|----------|-------------|
| [README](../README.md) | Project overview and quick start |
| [Architecture](ARCHITECTURE.md) | System design, components, and data flow |
| [CLI Reference](CLI.md) | Complete command line documentation |
| [Python API](API.md) | Python client and core classes |
| [Configuration](CONFIGURATION.md) | All configuration options |
| [Deployment](DEPLOYMENT.md) | Production setup and systemd |
| [Performance](PERFORMANCE.md) | Benchmarks and optimization tips |
| [Integration](INTEGRATION.md) | Using with OpenClaw/Clawdbot |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |

## Getting Started

1. **Install FastSearch**
   ```bash
   pip install fastsearch
   ```

2. **Index your documents**
   ```bash
   fastsearch index ./docs --glob "*.md"
   ```

3. **Start the daemon** (recommended for production)
   ```bash
   fastsearch daemon start --detach
   ```

4. **Search**
   ```bash
   fastsearch search "your query"
   ```

## Documentation Structure

```
docs/
├── index.md           # This file - documentation index
├── ARCHITECTURE.md    # System design and internals
├── CLI.md             # Command line reference
├── API.md             # Python API documentation
├── CONFIGURATION.md   # Configuration reference
├── DEPLOYMENT.md      # Production deployment guide
├── PERFORMANCE.md     # Benchmarks and optimization
├── INTEGRATION.md     # Integration with other systems
└── TROUBLESHOOTING.md # Common problems and solutions
```

## Key Concepts

### Hybrid Search

FastSearch combines two search methods:

- **BM25** — Traditional full-text search using term frequency. Fast and good for exact matches.
- **Vector Search** — Semantic similarity using embeddings. Good for meaning-based queries.

Results are fused using **Reciprocal Rank Fusion (RRF)**, which combines rankings from both methods.

### Daemon Mode

The daemon keeps ML models loaded in memory, reducing search latency from ~800ms to ~5ms. It uses a Unix socket for fast IPC and supports multiple concurrent clients.

### Model Slots

| Slot | Model | Purpose | Memory |
|------|-------|---------|--------|
| `embedder` | bge-base-en-v1.5 | Generate embeddings | ~450MB |
| `reranker` | ms-marco-MiniLM | Rerank results | ~90MB |

## Version

Current version: **0.2.0**
