# VPS-VPS-FastSearch Architecture

This document describes the internal architecture of VPS-FastSearch, including component design, data flow, and the hybrid search algorithm.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VPS-FastSearch System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────────────────┐   │
│  │   CLI    │    │  Python  │    │        VPS-FastSearch Daemon         │   │
│  │          │    │  Client  │    │   ┌─────────────────────────┐   │   │
│  └────┬─────┘    └────┬─────┘    │   │     Model Manager       │   │   │
│       │               │          │   │  ┌──────────────────┐   │   │   │
│       │               │          │   │  │ Embedder (450MB) │   │   │   │
│       │               │          │   │  │ bge-base-en-v1.5 │   │   │   │
│       │               │          │   │  └──────────────────┘   │   │   │
│       │               │          │   │  ┌──────────────────┐   │   │   │
│       │               │          │   │  │ Reranker (90MB)  │   │   │   │
│       │               │          │   │  │ ms-marco-MiniLM  │   │   │   │
│       │               │          │   │  └──────────────────┘   │   │   │
│       └───────────────┴──────────┼───▶  LRU Eviction + Budget  │   │   │
│                                  │   └─────────────────────────┘   │   │
│                                  │                                 │   │
│                                  │   ┌─────────────────────────┐   │   │
│                                  │   │   Unix Socket Server    │   │   │
│                                  │   │   JSON-RPC 2.0 over     │   │   │
│                                  │   │   /tmp/vps_fastsearch.sock  │   │   │
│                                  │   └─────────────────────────┘   │   │
│                                  └─────────────────────────────────┘   │
│                                              │                         │
│                                              │                         │
│  ┌───────────────────────────────────────────┼───────────────────────┐ │
│  │                        SearchDB           │                       │ │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │ │
│  │  │    docs table    │  │    docs_fts     │  │    docs_vec      │ │ │
│  │  │   (main data)    │  │  (FTS5 index)   │  │  (vector index)  │ │ │
│  │  │                  │  │                 │  │                  │ │ │
│  │  │  id, source,     │  │   BM25 search   │  │ 768-dim vectors  │ │ │
│  │  │  chunk_index,    │  │   full-text     │  │ cosine distance  │ │ │
│  │  │  content,        │  │                 │  │                  │ │ │
│  │  │  metadata        │  │                 │  │                  │ │ │
│  │  └──────────────────┘  └─────────────────┘  └──────────────────┘ │ │
│  │                                                                   │ │
│  │                      SQLite + sqlite-vec                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Embedder

The `Embedder` class generates dense vector representations of text using FastEmbed with ONNX Runtime.

```python
class Embedder:
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DIMENSIONS = 768
```

**Key characteristics:**
- Uses `BAAI/bge-base-en-v1.5` (768 dimensions)
- ONNX Runtime for CPU-optimized inference
- Batch processing for efficiency
- ~450MB memory footprint
- ~8ms per single embedding, ~1.5ms per text in batches

**API:**
```python
embedder = Embedder()
vectors = embedder.embed(["text1", "text2"])  # List of 768-dim vectors
vector = embedder.embed_single("text")         # Single 768-dim vector
```

### 2. Reranker

The `Reranker` class uses a cross-encoder model to score query-document pairs for more accurate ranking.

```python
class Reranker:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Key characteristics:**
- Cross-encoder architecture (vs bi-encoder for embeddings)
- Processes (query, document) pairs together
- More accurate but slower than vector similarity
- ~90MB memory footprint
- ~2ms per document scoring

**Why cross-encoders are better for reranking:**
- Bi-encoders (embeddings) encode query and document independently
- Cross-encoders see both together, capturing interactions
- Trade-off: O(n) inference vs O(1) vector comparison

**API:**
```python
reranker = Reranker()
scores = reranker.rerank("query", ["doc1", "doc2", "doc3"])
indexed = reranker.rerank_with_indices("query", docs, top_k=5)
```

### 3. SearchDB

The `SearchDB` class manages the SQLite database with FTS5 and sqlite-vec extensions.

**Database Schema:**

```sql
-- Main document storage
CREATE TABLE docs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,           -- File path
    chunk_index INTEGER NOT NULL,   -- Chunk number within file
    content TEXT NOT NULL,          -- Chunk text
    metadata JSON,                  -- Section info, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual table for BM25 search
CREATE VIRTUAL TABLE docs_fts USING fts5(
    content,
    content='docs',
    content_rowid='id'
);

-- Vector virtual table for similarity search
CREATE VIRTUAL TABLE docs_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
```

**Triggers for FTS sync:**
```sql
-- Auto-sync FTS on insert/update/delete
CREATE TRIGGER docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO docs_fts(rowid, content) VALUES (new.id, new.content);
END;
```

### 4. Daemon

The `FastSearchDaemon` class provides a Unix socket server with JSON-RPC 2.0 protocol.

**Components:**
- **ModelManager** — LRU cache for models with memory budget
- **Unix Socket Server** — Async server at `/tmp/vps_fastsearch.sock`
- **Request Handler** — JSON-RPC 2.0 method dispatch

**Model lifecycle:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   UNLOADED  │────▶│   LOADING   │────▶│   LOADED    │
└─────────────┘     └─────────────┘     └─────────────┘
      ▲                                       │
      │                                       │ idle timeout
      │                                       │ or eviction
      │            ┌─────────────┐            │
      └────────────│  UNLOADING  │◀───────────┘
                   └─────────────┘
```

### 5. Client

The `FastSearchClient` class provides a Python interface to the daemon.

**Features:**
- Connection pooling (persistent socket)
- Automatic reconnection
- Context manager support
- Fallback to direct mode when daemon unavailable

## Data Flow

### Indexing Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  File    │───▶│  Chunker │───▶│ Embedder │───▶│ SearchDB │───▶│  Indexed │
│  Input   │    │          │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │               │               │
                     │               │               │
                     ▼               ▼               ▼
               ┌──────────┐    ┌──────────┐    ┌──────────┐
               │ ~500 char│    │ 768-dim  │    │  3 tables│
               │  chunks  │    │  vectors │    │ updated  │
               │ + overlap│    │          │    │          │
               └──────────┘    └──────────┘    └──────────┘
```

**Chunking strategy:**
1. Split text by paragraphs (double newlines)
2. Accumulate until ~500 tokens (~2000 chars)
3. Add ~50 token (~200 char) overlap for context
4. For markdown, preserve section headers in metadata

### Search Flow

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────┐    ┌──────────────────────────────────────────────┐
│ Embedder │    │                 Parallel Search               │
│          │    │  ┌────────────┐            ┌────────────┐    │
└────┬─────┘    │  │   BM25     │            │   Vector   │    │
     │          │  │  Search    │            │  Search    │    │
     └──────────┼──▶   FTS5     │            │ sqlite-vec │    │
                │  │            │            │            │    │
                │  └─────┬──────┘            └─────┬──────┘    │
                │        │                         │           │
                │        │    ┌─────────────┐     │           │
                │        └───▶│  RRF Fusion │◀────┘           │
                │             └──────┬──────┘                 │
                └────────────────────┼────────────────────────┘
                                     │
                                     ▼
                              ┌────────────┐
                              │ (Optional) │
                              │  Reranker  │
                              └─────┬──────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ Results  │
                              └──────────┘
```

## Hybrid Search Algorithm

VPS-FastSearch uses **Reciprocal Rank Fusion (RRF)** to combine BM25 and vector search results.

### RRF Formula

```
RRF_score(doc) = Σ (weight_i / (k + rank_i(doc)))
```

Where:
- `k` = 60 (constant to prevent division by small numbers)
- `rank_i(doc)` = rank of document in result list i (1-indexed)
- `weight_i` = weight for each search method (default: 1.0)

### Example

Given a query with these rankings:

| Document | BM25 Rank | Vector Rank |
|----------|-----------|-------------|
| Doc A | 1 | 5 |
| Doc B | 3 | 1 |
| Doc C | 2 | 3 |

RRF scores (k=60):
- Doc A: 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = **0.0318**
- Doc B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = **0.0323**
- Doc C: 1/(60+2) + 1/(60+3) = 0.0161 + 0.0159 = **0.0320**

Final ranking: **B > C > A**

### Why RRF?

- **No score normalization needed** — Works with ranks, not raw scores
- **Robust to outliers** — Penalizes documents missing from one list
- **Parameter-light** — Only k needs tuning
- **Fast** — Simple arithmetic after parallel searches

## Model Details

### Embedder: BAAI/bge-base-en-v1.5

| Property | Value |
|----------|-------|
| Architecture | BERT-based |
| Dimensions | 768 |
| Max tokens | 512 |
| Training data | MS MARCO, NQ, etc. |
| Performance | MTEB score ~63 |
| Memory | ~450MB |

**Strengths:**
- Good balance of speed and quality
- Optimized for retrieval tasks
- Works well with English text
- ONNX-optimized for CPU

### Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

| Property | Value |
|----------|-------|
| Architecture | MiniLM (6 layers) |
| Max tokens | 512 |
| Training data | MS MARCO |
| Memory | ~90MB |

**Strengths:**
- Fast for cross-encoder (~2ms per pair)
- Trained specifically for reranking
- Significant accuracy improvement over bi-encoder

## Memory Management

The daemon uses LRU eviction to stay within memory budget:

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Budget (4GB)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐   │
│  │    Embedder     │  │    Reranker     │  │  Overhead │   │
│  │     450MB       │  │      90MB       │  │    60MB   │   │
│  │   (always)      │  │  (on-demand)    │  │           │   │
│  └─────────────────┘  └─────────────────┘  └───────────┘   │
│                                                             │
│  Total: ~600MB with both models loaded                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Eviction rules:**
1. Check available memory before loading
2. Evict LRU on-demand models first
3. Never evict "always" loaded models
4. Auto-unload idle on-demand models after timeout

## Protocol: JSON-RPC 2.0

Messages are length-prefixed (4-byte big-endian) followed by JSON:

```
┌──────────────┬─────────────────────────────────────────┐
│ Length (4B)  │              JSON Body                  │
│  big-endian  │                                         │
└──────────────┴─────────────────────────────────────────┘
```

**Request format:**
```json
{
    "jsonrpc": "2.0",
    "method": "search",
    "params": {
        "query": "example query",
        "limit": 10,
        "mode": "hybrid",
        "rerank": false
    },
    "id": 1
}
```

**Response format:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "results": [...],
        "search_time_ms": 4.2
    },
    "id": 1
}
```

## File Structure

```
fastsearch/
├── __init__.py       # Package exports
├── cli.py            # Click CLI commands
├── core.py           # Embedder, Reranker, SearchDB
├── daemon.py         # Unix socket server, ModelManager
├── client.py         # VPS-FastSearchClient
├── config.py         # Configuration system
└── chunker.py        # Text chunking utilities
```

## Design Decisions

### Why SQLite?

- Single-file deployment
- No server to manage
- ACID transactions
- Works everywhere Python works
- sqlite-vec adds efficient vector search

### Why Unix Sockets?

- Lower latency than TCP (~10x faster)
- No network stack overhead
- Built-in access control (file permissions)
- Simple cleanup (just delete the file)

### Why ONNX Runtime?

- CPU-optimized inference
- 2-3x faster than vanilla PyTorch on CPU
- Smaller memory footprint
- Cross-platform compatibility

### Why RRF over learned fusion?

- No training required
- Works out-of-the-box
- Consistent across different query types
- Well-studied and predictable behavior
