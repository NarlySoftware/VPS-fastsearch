# Frequently Asked Questions

## General

### What makes VPS-FastSearch different from other search libraries?

VPS-FastSearch is optimized for **CPU-only environments** with **daemon mode** that keeps models loaded. Most alternatives either require GPU, use expensive API calls, or have slow cold-start times. VPS-FastSearch gives you 4ms searches on a $5 VPS.

### Do I need a GPU?

No. VPS-FastSearch uses ONNX Runtime optimized for CPU inference. It runs efficiently on any modern x86 or ARM processor.

### What embedding model does it use?

By default, VPS-FastSearch uses [BGE (BAAI General Embedding)](https://huggingface.co/BAAI/bge-base-en-v1.5) models via FastEmbed:
- `bge-small-en-v1.5` — 384 dimensions, ~130MB
- `bge-base-en-v1.5` — 768 dimensions, ~450MB (default)
- `bge-large-en-v1.5` — 1024 dimensions, ~1.2GB

### What's "hybrid search"?

Hybrid search combines two approaches:
1. **BM25** — Traditional keyword matching (finds "error" when you search "error")
2. **Vector search** — Semantic similarity (finds "exception" when you search "error")

Results are merged using Reciprocal Rank Fusion (RRF) for best of both worlds.

---

## Performance

### Why is the first search slow?

The first search loads the embedding model into memory (~600-800ms). Use daemon mode to keep models loaded:

```bash
vps-fastsearch daemon start --detach
```

### How much RAM do I need?

| Configuration | Memory Required |
|---------------|-----------------|
| bge-small only | ~200MB |
| bge-base only | ~520MB |
| bge-base + reranker | ~610MB |

### How many documents can it handle?

VPS-FastSearch scales well:
- 1K chunks: <5ms search
- 10K chunks: <10ms search
- 100K chunks: <20ms search
- 1M+ chunks: May need tuning (see Performance docs)

---

## Usage

### How do I update indexed documents?

Re-index with the `--reindex` flag:

```bash
vps-fastsearch index ./docs --reindex
```

This replaces existing chunks from those files.

### Can I use multiple databases?

Yes, specify `--db` with any command:

```bash
vps-fastsearch index ./notes --db notes.db
vps-fastsearch index ./code --db code.db
vps-fastsearch search "query" --db notes.db
```

### How do I search from Python without the daemon?

```python
from vps_fastsearch import SearchDB, get_embedder

db = SearchDB("fastsearch.db")
embedder = get_embedder()

embedding = embedder.embed_single("my query")
results = db.search_hybrid("my query", embedding, limit=5)
```

### What file types can I index?

Currently optimized for text files:
- Markdown (`.md`) — with section-aware chunking
- Plain text (`.txt`)
- Any UTF-8 text file

For other formats, extract text first then index.

---

## Daemon

### How do I run the daemon on startup?

Use the provided systemd service file:

```bash
sudo cp fastsearch.service /etc/systemd/system/
sudo systemctl enable fastsearch
sudo systemctl start fastsearch
```

Or use your system's process manager (supervisord, launchd, etc.).

### How do I check if the daemon is running?

```bash
vps-fastsearch daemon status
```

### The daemon won't start. What do I check?

1. Check if socket exists: `ls -la /tmp/fastsearch.sock`
2. Check logs: `vps-fastsearch daemon logs`
3. Try foreground mode: `vps-fastsearch daemon start` (without `--detach`)
4. Check memory: daemon needs ~500MB free

---

## Troubleshooting

### "Connection refused" errors

The daemon isn't running. Start it:

```bash
vps-fastsearch daemon start --detach
```

### "Out of memory" errors

Use a smaller model:

```yaml
# ~/.config/fastsearch/config.yaml
models:
  embedder:
    name: "BAAI/bge-small-en-v1.5"
```

### Search results aren't relevant

Try:
1. Use `--rerank` for better accuracy
2. Check your chunk size (default 500 tokens is usually good)
3. Use `--mode bm25` for exact keyword matches
4. Re-index with `--reindex` if content changed

### Daemon uses too much memory

Configure idle timeout for the reranker:

```yaml
models:
  reranker:
    keep_loaded: on_demand
    idle_timeout_seconds: 300  # Unload after 5 min idle
```
