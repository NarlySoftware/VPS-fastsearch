# Performance Guide

This guide covers VPS-FastSearch performance characteristics, benchmarks, and optimization strategies.

## Benchmark Results

Tested on Apple M2 (MacBook Air, 8GB RAM) with 1000 document chunks (~500 tokens each).

### Search Latency

| Operation | Cold Start | Daemon Mode | Notes |
|-----------|------------|-------------|-------|
| Hybrid Search | 850ms | **4ms** | First search loads embedder |
| Hybrid + Rerank | 1100ms | **45ms** | Reranker adds ~40ms |
| BM25 Only | 2ms | **2ms** | No model loading needed |
| Vector Only | 820ms | **3ms** | Embedding + search |

### Embedding Performance

| Operation | Time | Throughput |
|-----------|------|------------|
| Single text | 8ms | 125 texts/sec |
| Batch of 10 | 25ms | 400 texts/sec |
| Batch of 100 | 150ms | 667 texts/sec |

*Batch processing is significantly more efficient due to model inference overhead amortization.*

### Reranking Performance

| Documents | Time | Throughput |
|-----------|------|------------|
| 5 docs | 10ms | 500 docs/sec |
| 10 docs | 20ms | 500 docs/sec |
| 20 docs | 40ms | 500 docs/sec |

*Linear scaling: ~2ms per document.*

### Database Operations

| Operation | 1K chunks | 10K chunks | 100K chunks |
|-----------|-----------|------------|-------------|
| BM25 search | 0.5ms | 1ms | 5ms |
| Vector search | 0.3ms | 1ms | 8ms |
| RRF fusion | 0.1ms | 0.2ms | 0.5ms |
| Index single | 0.1ms | 0.1ms | 0.1ms |
| Index batch (100) | 5ms | 5ms | 5ms |

---

## Speed Comparison

### Query Modes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Query Latency Comparison                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  QMD (cold):      ████████████████████████████████████████  850ms      │
│                                                                         │
│  Direct (cold):   ████████████████████████████████████████  850ms      │
│                                                                         │
│  Daemon (warm):   █                                          4ms       │
│                                                                         │
│  BM25 only:       █                                          2ms       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Daemon Mode is Fast

| Phase | Cold Start | Daemon Mode | Savings |
|-------|------------|-------------|---------|
| Python import | 200ms | 0ms | Already loaded |
| Model loading | 600ms | 0ms | Already in memory |
| Embedding | 8ms | 8ms | Same |
| Database query | 2ms | 2ms | Same |
| **Total** | **~850ms** | **~10ms** | **99%** |

---

## Memory Usage

### By Component

| Component | Memory | Notes |
|-----------|--------|-------|
| Python + libs | 50MB | Base overhead |
| FastEmbed runtime | 20MB | ONNX inference |
| bge-small embedder | 130MB | 384 dimensions |
| bge-base embedder | 450MB | 768 dimensions |
| bge-large embedder | 1200MB | 1024 dimensions |
| MiniLM-6 reranker | 90MB | Default reranker |
| MiniLM-12 reranker | 130MB | Larger reranker |

### Typical Configurations

| Configuration | Total Memory |
|---------------|--------------|
| Daemon idle (no models) | ~70MB |
| bge-small only | ~200MB |
| bge-base only | ~520MB |
| bge-base + reranker | ~610MB |
| bge-large + reranker | ~1400MB |

### Memory Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Memory Usage Over Time                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  600MB │                    ┌─────────────────┐                         │
│        │                    │                 │                         │
│  500MB │  ┌─────────────────┤    Reranker     │                         │
│        │  │                 │    loaded       │                         │
│  400MB │  │   Embedder      └─────────────────┤                         │
│        │  │   loaded                          │  Reranker               │
│  300MB │  │                                   │  unloaded               │
│        │  │                                   │  (idle timeout)         │
│  200MB │  │                                   └─────────────────────    │
│        │  │                                                             │
│  100MB │──┘                                                             │
│        │  Daemon start                                                  │
│    0MB │                                                                │
│        └────────────────────────────────────────────────────────────────│
│           0s     5s      1min     2min     5min     6min     10min     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Tips

### 1. Use Daemon Mode

**Impact: 100x faster queries**

```bash
# Start daemon once
vps-fastsearch daemon start --detach

# All subsequent searches are instant
vps-fastsearch search "query"  # ~4ms vs ~850ms
```

### 2. Batch Operations

**Impact: 5x faster indexing**

```python
# Instead of:
for text in texts:
    embedding = embedder.embed_single(text)  # Slow

# Do:
embeddings = embedder.embed(texts)  # 5x faster
```

### 3. Choose the Right Model

| Use Case | Recommended Model | Memory | Speed |
|----------|-------------------|--------|-------|
| Memory constrained (<1GB) | bge-small | 130MB | ★★★★★ |
| Balanced | bge-base | 450MB | ★★★★ |
| Maximum quality | bge-large | 1.2GB | ★★★ |

### 4. Tune Reranking

Reranking improves accuracy but adds latency. Use strategically:

```python
# Quick search (4ms)
results = client.search("simple query", rerank=False)

# High-accuracy search (45ms)
results = client.search("complex ambiguous query", rerank=True)
```

**When to use reranking:**
- Ambiguous queries
- When precision matters more than speed
- Low result volume (top 5-10)

**When to skip reranking:**
- Simple keyword queries
- High-volume requests
- Time-sensitive applications

### 5. Optimize Database

```python
# Periodic optimization
import apsw

conn = apsw.Connection("vps_fastsearch.db")
conn.execute("VACUUM")  # Reclaim space
conn.execute("ANALYZE")  # Update statistics
```

### 6. Use BM25 for Keywords

```bash
# For exact matches, skip vector search
vps-fastsearch search "error_code_123" --mode bm25  # 2ms
```

### 7. Limit Results Appropriately

```python
# Only fetch what you need
results = client.search("query", limit=5)  # Faster than limit=100
```

---

## When to Use Reranking

### Accuracy vs Speed Trade-off

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Accuracy vs Latency                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Accuracy                                                               │
│     │                                                                   │
│  95%│                                          ● Hybrid + Rerank        │
│     │                                                                   │
│  90%│                              ● Hybrid                             │
│     │                                                                   │
│  85%│                 ● Vector                                          │
│     │                                                                   │
│  80%│     ● BM25                                                        │
│     │                                                                   │
│     └───────────────────────────────────────────────────────────────    │
│           2ms        4ms         8ms         45ms        Latency        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Query Type | Volume | Recommendation |
|------------|--------|----------------|
| Simple keywords | Any | BM25 only |
| Semantic | High (>100/sec) | Hybrid, no rerank |
| Semantic | Medium | Hybrid + rerank |
| Complex/ambiguous | Low | Hybrid + rerank |
| Critical accuracy | Any | Hybrid + rerank |

---

## Scaling Considerations

### Vertical Scaling

| Bottleneck | Solution |
|------------|----------|
| Memory | Use smaller model, increase RAM |
| CPU | More cores (parallelizes embedding) |
| Disk I/O | SSD, increase page cache |

### Database Size Limits

| Chunks | DB Size | Search Time | Notes |
|--------|---------|-------------|-------|
| 1K | ~5MB | <5ms | Trivial |
| 10K | ~50MB | <10ms | Easy |
| 100K | ~500MB | <20ms | Comfortable |
| 1M | ~5GB | <100ms | May need tuning |
| 10M+ | ~50GB+ | Variable | Consider sharding |

### Tuning for Large Datasets

```yaml
# config.yaml for large datasets
memory:
  max_ram_mb: 8000  # More memory for caching

# Database tuning (via PRAGMA)
# fastsearch doesn't expose these directly yet
# but you can set them via apsw:
```

```python
import apsw
conn = apsw.Connection("vps_fastsearch.db")
conn.execute("PRAGMA cache_size = -100000")  # 100MB cache
conn.execute("PRAGMA mmap_size = 1073741824")  # 1GB mmap
```

---

## Profiling

### Enable Debug Logging

```yaml
# config.yaml
daemon:
  log_level: DEBUG
```

### Time Search Components

```python
import time
from vps_fastsearch import FastSearchClient

client = FastSearchClient()

# Measure search
start = time.perf_counter()
result = client.search("query", rerank=True)
total = time.perf_counter() - start

print(f"Total: {total*1000:.1f}ms")
print(f"Server reported: {result['search_time_ms']:.1f}ms")
print(f"Network overhead: {(total*1000 - result['search_time_ms']):.1f}ms")
```

### Profile Indexing

```python
import time
from vps_fastsearch import Embedder, SearchDB, chunk_text

content = open("large_file.md").read()
chunks = list(chunk_text(content))

# Embedding time
embedder = Embedder()
start = time.perf_counter()
embeddings = embedder.embed([c for c in chunks])
embed_time = time.perf_counter() - start

# Index time
db = SearchDB("test.db")
start = time.perf_counter()
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    db.index_document("test.md", i, chunk, embedding)
index_time = time.perf_counter() - start

print(f"Chunks: {len(chunks)}")
print(f"Embed: {embed_time:.2f}s ({embed_time/len(chunks)*1000:.1f}ms/chunk)")
print(f"Index: {index_time:.2f}s ({index_time/len(chunks)*1000:.1f}ms/chunk)")
```

---

## Performance Checklist

- [ ] Daemon running in production
- [ ] Embedder set to `keep_loaded: always`
- [ ] Batch operations for indexing
- [ ] Appropriate result limits
- [ ] Reranking used selectively
- [ ] Database on SSD
- [ ] Periodic VACUUM/ANALYZE
- [ ] Memory budget configured appropriately
