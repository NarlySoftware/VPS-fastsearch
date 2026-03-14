# VPS-FastSearch Architecture

This document describes the internal architecture of VPS-FastSearch, including component
design, data flow, wire protocol, search algorithm, database schema, model management,
and configuration system. All values are taken directly from the source code.

---

## 1. System Overview

VPS-FastSearch is a hybrid search system combining BM25 full-text search with vector
similarity search, designed for CPU-only VPS environments. It uses SQLite with FTS5 and
sqlite-vec extensions, ONNX Runtime for embedding inference, and a persistent Unix socket
daemon for low-latency queries.

### Component Diagram

```
                          vps-fastsearch CLI (cli.py)
                                    |
                                    | Click commands: daemon, index, search, query,
                                    |   vector_search, config, stats, delete, update,
                                    |   embed, list, collection, migrate-paths
                                    v
                         FastSearchClient (client.py)
                                    |
                                    | Unix socket (length-prefixed JSON-RPC 2.0)
                                    v
                        FastSearchDaemon (daemon.py)
                           /                  \
                          v                    v
                   ModelManager           SearchDB (core.py)
                  /           \                |
                 v             v               v
           Embedder        Reranker        SQLite + APSW
        (fastembed)    (CrossEncoder)     /      |       \
     BAAI/bge-base    ms-marco-MiniLM   docs  docs_fts  docs_vec
       768-dim ONNX    L-6-v2            |    (FTS5)    (sqlite-vec)
                                         |       |          |
                                         +-------+----------+
                                              SQLite DB
                                            (WAL mode)
```

### Data Flow

**Indexing**: CLI reads files, chunks text (chunker.py), generates embeddings via daemon
or direct Embedder, then inserts into SearchDB (docs + docs_vec tables; FTS5 synced via
triggers).

**Searching**: Client sends JSON-RPC request over Unix socket to daemon. Daemon loads
embedder, generates query embedding, runs BM25 and/or vector search, fuses results with
RRF, optionally reranks with cross-encoder, and returns results.

**Fallback**: If the daemon is not running, the client library and CLI fall back to direct
mode -- loading models in-process and querying SQLite directly (higher latency due to cold
model loading).

### File Structure

```
vps_fastsearch/
    __init__.py       # Package exports, version (0.3.1)
    __main__.py       # python -m vps_fastsearch entry point
    cli.py            # Click CLI commands
    core.py           # Embedder, Reranker, SearchDB
    daemon.py         # Unix socket server, ModelManager, RateLimiter
    client.py         # FastSearchClient, convenience functions
    config.py         # YAML configuration system
    chunker.py        # Text chunking utilities
```

---

## 2. Daemon Protocol

### Wire Format

The daemon uses a **binary length-prefixed** framing protocol over a Unix domain socket
(`AF_UNIX`, `SOCK_STREAM`):

```
+-------------------+-----------------------------+
| 4 bytes (big-end) |       JSON payload          |
|   payload length  |  (UTF-8 encoded JSON-RPC)   |
+-------------------+-----------------------------+
```

- **Length prefix**: 4-byte unsigned integer in big-endian byte order
  (`int.to_bytes(4, "big")`).
- **Payload**: UTF-8 encoded JSON conforming to JSON-RPC 2.0.
- **Maximum message size**: 10 MB (10,485,760 bytes). Messages exceeding this limit cause
  the server to close the connection.
- **Socket buffers**: Both send and receive buffers are tuned to 2 MB
  (`SO_SNDBUF` / `SO_RCVBUF = 2,097,152`) on both client and server sides.
- **Zero-length messages**: A length prefix of 0 causes the server to close the connection.

### JSON-RPC 2.0 Request Format

```json
{
    "jsonrpc": "2.0",
    "method": "<method_name>",
    "params": { ... },
    "id": 1
}
```

The `params` field must be a JSON object (not an array). The `id` field is echoed back in
the response.

### JSON-RPC 2.0 Response Format

Success:

```json
{
    "jsonrpc": "2.0",
    "result": { ... },
    "id": 1
}
```

Error:

```json
{
    "jsonrpc": "2.0",
    "error": { "code": -32601, "message": "Method not found: foo" },
    "id": 1
}
```

### RPC Methods

#### `ping`

Health check.

- **Params**: `{}` (none)
- **Returns**: `{ "pong": true, "timestamp": <unix_float> }`

#### `status`

Get daemon status and model information.

- **Params**: `{}` (none)
- **Returns**:

```json
{
    "uptime_seconds": 3600.0,
    "request_count": 42,
    "socket_path": "/tmp/fastsearch.sock",
    "concurrent_slots_available": 64,
    "loaded_models": {
        "embedder": {
            "loaded_at": 1700000000.0,
            "last_used": 1700003600.0,
            "memory_mb": 450.0,
            "actual_memory_mb": 420.0,
            "idle_seconds": 5.0
        }
    },
    "total_memory_mb": 512.0,
    "max_memory_mb": 4000
}
```

#### `search`

Search indexed documents.

- **Params**:

| Param     | Type   | Default               | Description                                |
|-----------|--------|-----------------------|--------------------------------------------|
| `query`   | string | (required)            | Search query text (must be non-empty)       |
| `db_path` | string | `DEFAULT_DB_PATH`     | Path to SQLite database                     |
| `limit`   | int    | `10`                  | Max results (1--1000)                       |
| `mode`    | string | `"hybrid"`            | `"bm25"`, `"vector"`, or `"hybrid"`         |
| `rerank`  | bool   | `false`               | Apply cross-encoder reranking (hybrid only) |
| `metadata_filter` | object | `null`       | Key-value pairs for exact metadata match (AND logic) |

- **Returns**:

```json
{
    "query": "search terms",
    "mode": "hybrid",
    "reranked": false,
    "search_time_ms": 29.5,
    "results": [
        {
            "id": 1,
            "source": "/path/to/file.md",
            "chunk_index": 0,
            "content": "...",
            "metadata": {},
            "rrf_score": 0.032,
            "bm25_rank": 1,
            "vec_rank": 3,
            "rank": 1
        }
    ]
}
```

When `rerank=true`, result fields change: `rerank_score` replaces `rrf_score`, and the
`bm25_rank`, `vec_rank`, and `score` fields are removed.

#### `embed`

Generate embeddings for a list of texts.

- **Params**:

| Param   | Type     | Default    | Description                      |
|---------|----------|------------|----------------------------------|
| `texts` | string[] | (required) | Texts to embed (max 256 items)   |

- **Returns**:

```json
{
    "embeddings": [[0.012, -0.034, ...]],
    "count": 1,
    "embed_time_ms": 15.2
}
```

Each embedding is a 768-dimensional float array.

#### `rerank`

Score and rank documents against a query using the cross-encoder.

- **Params**:

| Param       | Type     | Default    | Description                       |
|-------------|----------|------------|-----------------------------------|
| `query`     | string   | (required) | Query text                        |
| `documents` | string[] | (required) | Documents to rerank (max 100)     |

- **Returns**:

```json
{
    "scores": [2.34, -1.56, 0.89],
    "ranked": [
        { "index": 0, "score": 2.34 },
        { "index": 2, "score": 0.89 },
        { "index": 1, "score": -1.56 }
    ],
    "rerank_time_ms": 45.0
}
```

#### `load_model`

Pre-load a model into memory.

- **Params**:

| Param  | Type   | Default    | Description                              |
|--------|--------|------------|------------------------------------------|
| `slot` | string | (required) | Model slot: `"embedder"` or `"reranker"` |

- **Returns**: `{ "slot": "reranker", "loaded": true, "memory_mb": 90.0 }`

The reference is released immediately after loading (the caller just wants the model warm).

#### `unload_model`

Unload a model from memory. Will not unload models with `keep_loaded: "always"` or models
with active references.

- **Params**:

| Param  | Type   | Default    | Description  |
|--------|--------|------------|--------------|
| `slot` | string | (required) | Model slot   |

- **Returns**: `{ "slot": "reranker", "unloaded": true }`

#### `reload_config`

Reload the daemon configuration from disk without restarting.

- **Params**:

| Param         | Type   | Default | Description                          |
|---------------|--------|---------|--------------------------------------|
| `config_path` | string | `null`  | Path to config file (.yaml/.yml)     |

- **Returns**: `{ "reloaded": true, "socket_path": "/tmp/fastsearch.sock" }`

Also triggered by sending `SIGHUP` to the daemon process.

#### `batch_index`

Batch index multiple documents into the database.

- **Params**:

| Param       | Type     | Default           | Description                                   |
|-------------|----------|-------------------|-----------------------------------------------|
| `db_path`   | string   | `DEFAULT_DB_PATH` | Path to SQLite database                       |
| `documents` | object[] | (required)        | List of document objects (max 1000)           |

Each document object must contain:

| Field         | Type     | Description                                  |
|---------------|----------|----------------------------------------------|
| `source`      | string   | Source file path or identifier (non-empty)   |
| `chunk_index` | int      | Zero-based chunk position within source      |
| `content`     | string   | Text content of the chunk                    |
| `embedding`   | float[]  | 768-dimensional embedding vector             |
| `metadata`    | object   | Optional JSON metadata                       |

- **Returns**:

```json
{
    "indexed": 5,
    "doc_ids": [1, 2, 3, 4, 5],
    "index_time_ms": 12.5
}
```

#### `delete`

Delete documents by source path or by individual document ID.

- **Params**:

| Param    | Type   | Default           | Description                                  |
|----------|--------|-------------------|----------------------------------------------|
| `db_path`| string | `DEFAULT_DB_PATH` | Path to SQLite database                      |
| `source` | string | `null`            | Source path to delete all chunks for          |
| `id`     | int    | `null`            | Single document ID to delete                 |

Exactly one of `source` or `id` must be provided.

- **Returns** (by source): `{ "deleted": 5, "source": "/path/to/file.md" }`
- **Returns** (by ID): `{ "deleted": 1, "id": 42 }` (deleted is 0 if not found)

#### `update_content`

Update the content and embedding for an existing document by ID. The daemon automatically
generates a new embedding for the provided content using the loaded embedder model.

- **Params**:

| Param    | Type   | Default           | Description                            |
|----------|--------|-------------------|----------------------------------------|
| `db_path`| string | `DEFAULT_DB_PATH` | Path to SQLite database                |
| `id`     | int    | (required)        | Document ID to update                  |
| `content`| string | (required)        | New text content (non-empty)           |

- **Returns**: `{ "updated": true, "id": 42 }` (updated is false if ID not found)

#### `list_sources`

List all indexed sources with chunk counts and ID ranges.

- **Params**:

| Param    | Type   | Default           | Description               |
|----------|--------|-------------------|---------------------------|
| `db_path`| string | `DEFAULT_DB_PATH` | Path to SQLite database   |

- **Returns**:

```json
{
    "sources": [
        { "source": "docs/guide.md", "chunks": 12, "min_id": 1, "max_id": 12 },
        { "source": "docs/api.md", "chunks": 8, "min_id": 13, "max_id": 20 }
    ],
    "count": 2
}
```

#### `shutdown`

Gracefully shut down the daemon.

- **Params**: `{}` (none)
- **Returns**: `{ "shutdown": true }`

### Error Codes

| Code    | Meaning              | When                                                             |
|---------|----------------------|------------------------------------------------------------------|
| -32700  | Parse error          | Malformed JSON in request payload                                |
| -32600  | Invalid Request      | Not a JSON object, or `method` missing/not a string              |
| -32601  | Method not found     | Unknown method name                                              |
| -32602  | Invalid params       | `params` not an object, or `ValueError` from handler validation  |
| -32000  | Server error         | Internal exception (type name only) or rate limit exceeded       |

### Rate Limiting

Each client connection has an independent sliding-window rate limiter
(`RateLimiter` class):

- **Window**: 1 second (`window_seconds=1.0`)
- **Limit**: 20 requests per window per connection (`max_requests=20`)
- **Behavior**: When rate-limited, the server still reads and discards the message body
  (to keep the binary stream in sync), then returns a `-32000` error response with
  `"Rate limited"` message.

### Concurrency and Timeouts

| Parameter                   | Value    | Description                                     |
|-----------------------------|----------|-------------------------------------------------|
| Max concurrent requests     | 64       | `asyncio.Semaphore(64)`                          |
| Idle connection timeout     | 300s     | No data from client for 5 minutes               |
| Data read timeout           | 30s      | For reading message body after length prefix     |
| Client socket timeout       | 30s      | Default `FastSearchClient` timeout               |
| Client retry attempts       | 2        | Auto-reconnect once on `TimeoutError`/`OSError`  |

### Daemon Lifecycle

**Startup**:
1. Check for existing PID file; verify process is still alive (cross-platform, with
   `/proc` check on Linux for process name validation).
2. Remove stale socket file.
3. Create Unix socket server with restrictive umask (`0o177` -- owner-only access).
4. Write PID file with `O_CREAT | O_EXCL` (prevents symlink attacks).
5. Pre-load all models with `keep_loaded: "always"`. If any fail, abort startup.
6. Wait for shutdown event.

**Shutdown** (triggered by `SIGTERM`, `SIGINT`, or `shutdown` RPC):
1. Close the server socket.
2. Unload all models (including "always" models), call `gc.collect()`.
3. Checkpoint and close all cached database connections.
4. Remove socket and PID files.

**Daemonization**: The `--detach` flag uses the Unix double-fork technique. A pipe
communicates the grandchild PID back to the original parent. The daemon process redirects
stdin/stdout/stderr to `/dev/null` and changes working directory to `/`.

**Signals**:
- `SIGTERM` / `SIGINT`: Graceful shutdown.
- `SIGHUP`: Reload configuration.

---

## 3. Search Algorithm

### BM25 Full-Text Search (FTS5)

BM25 search is implemented via SQLite's FTS5 extension using the `bm25()` ranking function.

**Query preprocessing**: The raw query is tokenized with `re.findall(r"\w+", query)` to
extract word tokens, then joined with `OR` operators and quoted for FTS5 safety:

```
"python" OR "search" OR "engine"
```

This prevents FTS5 syntax errors from special characters (hyphens, colons, etc. are column
operators in FTS5). If no word tokens are found, an empty result is returned.

**Tokenizer**: `porter unicode61` -- applies Porter stemming and Unicode-aware tokenization,
so "searching" matches "search" and accented characters are normalized.

**Scoring**: FTS5's `bm25()` function returns negative scores where more negative = more
relevant. Results are ordered by `score ASC` (most relevant first).

**Result fields**: `id`, `source`, `chunk_index`, `content`, `metadata`, `score`, `rank`.

**Limit**: Capped at `MAX_SEARCH_LIMIT = 10,000` in SearchDB.

### Vector Similarity Search (sqlite-vec)

Vector search uses the `sqlite-vec` extension for approximate nearest neighbor search on
768-dimensional float32 embeddings.

**Embedding model**: BAAI/bge-base-en-v1.5, a 768-dimensional bi-encoder model running via
ONNX Runtime through the FastEmbed library. Optimized for CPU inference with configurable
thread count (default: 2). Model download is approximately 130 MB on first use.

**Distance metric**: Cosine distance (lower = more similar). The `vec0` virtual table stores
embeddings as `float32[768]` (3,072 bytes per embedding) and supports `MATCH` queries with
a `k` parameter for top-k retrieval.

**Query**: Embeddings are serialized to float32 binary blobs via
`sqlite_vec.serialize_float32()` before passing to the `MATCH` operator.

**Result fields**: `id`, `source`, `chunk_index`, `content`, `metadata`, `distance`, `rank`.

### Reciprocal Rank Fusion (RRF)

Hybrid search combines BM25 and vector results using RRF. This rank-based fusion method is
robust to score scale differences between the two retrieval methods.

**Algorithm**:

1. Retrieve `limit * 3` candidates from both BM25 and vector search.
2. Assign ranks within each result set (1-indexed, best first).
3. For each unique document across both sets, compute:

```
rrf_score = weight_bm25 * 1/(k + bm25_rank) + weight_vec * 1/(k + vec_rank)
```

Where:
- `k = 60` (default, configurable) -- controls how much rank position affects the score.
  Higher k values reduce the gap between adjacent ranks.
- `weight_bm25 = 1.0` (default, configurable)
- `weight_vec = 1.0` (default, configurable)
- If a document appears in only one result set, its missing rank defaults to
  `fetch_limit + 1` (a penalty for absence).

4. Sort by RRF score descending, breaking ties by document ID ascending.
5. Return the top `limit` results with final rank assigned (1-indexed).

**Example** (k=60):

| Document | BM25 Rank | Vector Rank | RRF Score                               |
|----------|-----------|-------------|------------------------------------------|
| Doc A    | 1         | 5           | 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = **0.0318** |
| Doc B    | 3         | 1           | 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = **0.0323** |
| Doc C    | 2         | 3           | 1/(60+2) + 1/(60+3) = 0.0161 + 0.0159 = **0.0320** |

Final ranking: **B > C > A**

**Why RRF**:
- No score normalization needed -- works with ranks, not raw scores.
- Robust to outliers -- penalizes documents missing from one list.
- Parameter-light -- only k needs tuning.
- Fast -- simple arithmetic after parallel searches.

**Result fields**: `id`, `source`, `chunk_index`, `content`, `metadata`, `rrf_score`,
`bm25_rank` (null if absent from BM25), `vec_rank` (null if absent from vector), `rank`.

### Cross-Encoder Reranking (Optional)

When `rerank=true`, hybrid search results are refined with a cross-encoder model that scores
each (query, document) pair directly, producing more accurate relevance estimates at the cost
of additional latency.

**Algorithm**:

1. Run hybrid search to get `min(limit * 3, 30)` candidates.
2. Pass each (query, candidate_content) pair through the cross-encoder.
3. Sort by cross-encoder score descending.
4. Return the top `limit` results with final rank assigned.

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB), loaded on-demand via the
`sentence-transformers` library. Requires the optional `[rerank]` install extra.

**Why cross-encoders are better for reranking**: Bi-encoders (embeddings) encode query and
document independently, while cross-encoders see both together, capturing token-level
interactions. The trade-off is O(n) forward passes vs O(1) vector comparison.

**Result fields**: `id`, `source`, `chunk_index`, `content`, `metadata`, `rerank_score`,
`rank`. The intermediate RRF fields (`rrf_score`, `bm25_rank`, `vec_rank`, `score`) are
stripped after reranking since they are no longer meaningful.

---

## 4. Database Schema

The database uses SQLite via APSW bindings (not stdlib `sqlite3`) with the `sqlite-vec`
loadable extension. The schema version is tracked via `PRAGMA user_version` (currently `3`).

### PRAGMA Settings

| PRAGMA                | Value       | Purpose                                          |
|-----------------------|-------------|--------------------------------------------------|
| `journal_mode`        | `WAL`       | Write-Ahead Logging for concurrent readers/writer |
| `busy_timeout`        | `5000`      | Wait up to 5s if database is locked               |
| `cache_size`          | `-4000`     | 4 MB page cache (negative = KB)                   |
| `mmap_size`           | `268435456` | 256 MB memory-mapped I/O                          |
| `wal_autocheckpoint`  | `1000`      | Checkpoint every 1000 pages                       |
| `user_version`        | `3`         | Schema version for migration tracking              |

**Why WAL mode**: WAL allows concurrent read access while a single writer is active, which
is critical for a daemon serving multiple simultaneous search requests. Without WAL, readers
would block on writes and vice versa. The daemon also performs explicit
`PRAGMA wal_checkpoint(PASSIVE)` on shutdown and when evicting cached database connections.

### Table: `docs`

Primary document storage.

```sql
CREATE TABLE docs (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSON,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_docs_source ON docs(source);
CREATE UNIQUE INDEX idx_docs_source_chunk ON docs(source, chunk_index);
```

| Column       | Type      | Description                                      |
|--------------|-----------|--------------------------------------------------|
| `id`         | INTEGER   | Auto-incrementing primary key (rowid alias)       |
| `source`     | TEXT      | File path or document identifier                  |
| `chunk_index`| INTEGER   | Zero-based position of this chunk within source   |
| `content`    | TEXT      | The actual text content of the chunk              |
| `metadata`   | JSON      | JSON-encoded metadata (e.g., `{"section": "..."}`) |
| `created_at` | TIMESTAMP | Insertion timestamp (`CURRENT_TIMESTAMP`)         |

**Indexes**:
- `idx_docs_source` on `(source)` -- fast lookup and deletion by source file.
- `idx_docs_source_chunk` UNIQUE on `(source, chunk_index)` -- prevents duplicate chunks
  from the same source.

### Virtual Table: `docs_fts` (FTS5)

Full-text search index, kept in sync with `docs` via triggers.

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(
    content,
    content='docs',
    content_rowid='id',
    tokenize='porter unicode61'
);
```

| Setting          | Value                | Description                            |
|------------------|----------------------|----------------------------------------|
| `content`        | `'docs'`             | External content table (content-sync)  |
| `content_rowid`  | `'id'`               | Maps FTS rowid to `docs.id`            |
| `tokenize`       | `'porter unicode61'` | Porter stemming + Unicode tokenization |

**Sync Triggers**: Three triggers keep the FTS index synchronized with the `docs` table:

```sql
-- After insert: add new content to FTS
CREATE TRIGGER docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO docs_fts(rowid, content) VALUES (new.id, new.content);
END;

-- After delete: remove content from FTS (using FTS5 delete command)
CREATE TRIGGER docs_ad AFTER DELETE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;

-- After update: remove old content, add new content
CREATE TRIGGER docs_au AFTER UPDATE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO docs_fts(docs_fts, rowid, content)
        VALUES (new.id, new.content);
END;
```

### Virtual Table: `docs_vec` (sqlite-vec)

Vector similarity search index.

```sql
CREATE VIRTUAL TABLE docs_vec USING vec0(
    id        INTEGER PRIMARY KEY,
    embedding float32[768]
);
```

| Column      | Type           | Description                                     |
|-------------|----------------|-------------------------------------------------|
| `id`        | INTEGER        | Foreign key to `docs.id`                        |
| `embedding` | float32[768]   | 768-dimensional embedding vector (3,072 bytes)  |

**Distance metric**: Cosine distance (built into `vec0`).

**Query syntax**: `WHERE embedding MATCH ? AND k = ?` where the first parameter is a
serialized float32 vector and `k` is the number of nearest neighbors to return.

Note: Unlike the FTS5 table, the vector table is **not** auto-synced via triggers.
Insertions and deletions in `docs_vec` are managed explicitly in `index_document()`,
`index_batch()`, and `delete_source()`, all within explicit transactions.

### Table: `db_meta`

Database-level metadata stored as key-value pairs.

```sql
CREATE TABLE db_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

| Column | Type | Description                                |
|--------|------|------------------------------------------|
| `key`  | TEXT | Metadata key (primary key)                 |
| `value`| TEXT | Metadata value (JSON or plain text)       |

**Current keys**:
- `base_dir`: Base directory for resolving relative source paths (defaults to the directory
  containing the database file). Allows paths in the `source` column to be stored relative
  to a configurable root.

**Purpose**: Stores database-level settings and configuration that do not belong in individual
tables. Accessed via the `get_meta(key)` and `set_meta(key, value)` methods in `SearchDB`.

### Transactions

All write operations (`index_document`, `index_batch`, `delete_source`) use explicit
`BEGIN`/`COMMIT` transactions with `ROLLBACK` on error. The `index_document` method inserts
into `docs` (triggering FTS sync) and then `docs_vec` in a single transaction.

### Schema Versioning

Schema versioning uses `PRAGMA user_version` (currently `SCHEMA_VERSION = 3`). On database
open:
- If the stored version is less than the code's version, migrations are applied and the
  version is updated.
- If the stored version is greater, a `RuntimeError` is raised indicating an upgrade is
  needed.

### Integrity Checks

On every database open, a lightweight `PRAGMA quick_check(1)` is executed. If corruption is
detected, an error is logged with a suggestion to delete and re-index, and an exception is
raised.

### Daemon DB Connection Cache

The daemon caches up to 8 (`_DB_CACHE_MAX`) SearchDB connections. When the cache is full,
the oldest entry is evicted with a WAL checkpoint and close. Database paths are validated
against a base directory to prevent path traversal attacks.

---

## 5. Model Management

The `ModelManager` class in `daemon.py` handles the lifecycle of ML models within the daemon
process.

### Model Slots

| Slot        | Model                                 | Download | Default Policy | Estimated Memory |
|-------------|---------------------------------------|----------|----------------|------------------|
| `embedder`  | BAAI/bge-base-en-v1.5                 | ~130 MB  | `always`       | 450 MB           |
| `reranker`  | cross-encoder/ms-marco-MiniLM-L-6-v2  | ~80 MB   | `on_demand`    | 90 MB            |
| `summarizer`| (reserved for future 7B models)       | --       | --             | 4,000 MB         |

### Loading Policies

Each model slot has a `keep_loaded` policy:

- **`always`**: Loaded at daemon startup. Cannot be unloaded via `unload_model` (the
  request is rejected with a warning). Failure to load an `always` model prevents daemon
  startup with a `RuntimeError`.
- **`on_demand`**: Loaded on first use. Automatically scheduled for unload after
  `idle_timeout_seconds` of inactivity (default: 300 seconds for reranker).
- **`never`**: Not loaded (reserved for future use).

### Model Lifecycle

```
                              load_model()
    UNLOADED -----------------------------------------> LOADED
        ^                                                  |
        |                                                  | idle timeout
        |                                                  | or LRU eviction
        |                  unload_model()                   |
        +--------------------------------------------------+
                          (gc.collect())
```

### Lazy Loading

Models are loaded in a background thread pool (`loop.run_in_executor`) to avoid blocking
the async event loop. A single `asyncio.Lock` (`_load_lock`) serializes all load/unload
operations to prevent race conditions.

The loading sequence:

1. Cancel any pending unload task for this slot.
2. If already loaded, touch (update `last_used`), increment `ref_count`, move to end of
   OrderedDict (LRU update), and return.
3. Check memory budget, evicting if necessary.
4. Load model in thread pool executor.
5. Measure actual RSS delta during load (trusted if >= 10 MB).
6. Create `LoadedModel` tracking entry with `ref_count=1`.
7. Schedule auto-unload if policy is `on_demand`.

### Reference Counting

Each `load_model()` call increments `ref_count` on the `LoadedModel`. Each
`release_model()` decrements it (floor at 0). Models with `ref_count > 0` cannot be
evicted -- this prevents unloading a model while a search request is actively using it.

### LRU Eviction

Models are stored in a `collections.OrderedDict`. Each access moves the model to the end
(`move_to_end`). When memory budget is exceeded before loading a new model:

1. Iterate from the oldest (least recently used) entry.
2. Skip models with `keep_loaded == "always"`.
3. Attempt to unload the first eligible model; if it has `ref_count > 0`, stop trying.
4. Call `gc.collect()` after deletion.
5. Re-check memory. Repeat until the estimated memory for the new model fits within
   `max_ram_mb`, or no more models can be evicted.

### Memory Tracking

Two memory values are tracked per model:

- **`memory_mb`**: Used for budget calculations. Prefers the measured RSS delta; falls
  back to the hardcoded estimate if measurement is below 10 MB.
- **`actual_memory_mb`**: The raw RSS delta measured via `psutil.Process().memory_info().rss`
  before and after model loading (0.0 if measurement failed).

Total process memory is queried via `psutil.Process().memory_info().rss`.

### Auto-Unload

For `on_demand` models, after loading, an `asyncio.Task` is scheduled that:

1. Sleeps for `idle_timeout_seconds`.
2. Checks if the model's idle time (`time.time() - last_used`) exceeds the timeout.
3. If so, calls `unload_model()`.
4. If the model was used in the interim, the idle check fails and the task expires
   without action. A new unload task is scheduled on the next `load_model()` call.

### Shutdown

On daemon shutdown (`ModelManager.shutdown()`):

1. Cancel all pending unload tasks.
2. Pop all models from the OrderedDict regardless of `keep_loaded` policy or `ref_count`.
3. Delete all model instances.
4. Call `gc.collect()`.

---

## 6. Configuration

### File Location

Configuration is loaded from YAML files following the XDG Base Directory Specification.

**Load priority** (first match wins):

1. Explicit `path` argument to `load_config()`
2. `FASTSEARCH_CONFIG` environment variable
3. `$XDG_CONFIG_HOME/fastsearch/config.yaml` (default: `~/.config/fastsearch/config.yaml`)
4. Built-in defaults (no file needed)

### Default Paths

| Path            | With XDG_RUNTIME_DIR set                  | Without XDG_RUNTIME_DIR      |
|-----------------|-------------------------------------------|------------------------------|
| Config file     | `$XDG_CONFIG_HOME/fastsearch/config.yaml` | `~/.config/fastsearch/config.yaml` |
| Database        | `$XDG_DATA_HOME/fastsearch/fastsearch.db` | `~/.local/share/fastsearch/fastsearch.db` |
| Unix socket     | `$XDG_RUNTIME_DIR/fastsearch.sock`        | `/tmp/fastsearch.sock`       |
| PID file        | `$XDG_RUNTIME_DIR/fastsearch.pid`         | `/tmp/fastsearch.pid`        |

### Environment Variable Overrides

| Variable            | Overrides                          |
|---------------------|------------------------------------|
| `FASTSEARCH_CONFIG` | Config file path                   |
| `FASTSEARCH_DB`     | Database path (in CLI and client)  |
| `XDG_CONFIG_HOME`   | Base for config file path          |
| `XDG_DATA_HOME`     | Base for database path             |
| `XDG_RUNTIME_DIR`   | Base for socket and PID paths      |

### Config File Format

```yaml
daemon:
  socket_path: /tmp/fastsearch.sock    # Unix socket path
  pid_path: /tmp/fastsearch.pid        # PID file path
  log_level: INFO                      # DEBUG | INFO | WARNING | ERROR | CRITICAL

models:
  embedder:
    name: "BAAI/bge-base-en-v1.5"     # HuggingFace model identifier
    keep_loaded: always                # always | on_demand | never
    idle_timeout_seconds: 0            # 0 = no auto-unload (relevant for on_demand)
    threads: 2                         # ONNX Runtime thread count

  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    keep_loaded: on_demand
    idle_timeout_seconds: 300          # Unload after 5 min idle

memory:
  max_ram_mb: 4000                     # Maximum RSS before LRU eviction
  eviction_policy: lru                 # lru | fifo
```

### Config Dataclasses

The configuration is represented by four dataclasses in `config.py`:

- **`FastSearchConfig`**: Top-level container with `daemon`, `models`, and `memory` fields.
- **`DaemonConfig`**: `socket_path` (str), `pid_path` (str), `log_level` (str).
- **`ModelConfig`**: `name` (str), `keep_loaded` (Literal["always", "on_demand", "never"]),
  `idle_timeout_seconds` (int, default 300), `threads` (int, default 2).
- **`MemoryConfig`**: `max_ram_mb` (int, default 4000), `eviction_policy`
  (Literal["lru", "fifo"], default "lru").

### Validation

All configuration values are validated on load with fallback to defaults and a warning log:

| Field                  | Valid values                             | Default      |
|------------------------|------------------------------------------|--------------|
| `log_level`            | DEBUG, INFO, WARNING, ERROR, CRITICAL    | INFO         |
| `keep_loaded`          | always, on_demand, never                 | on_demand    |
| `idle_timeout_seconds` | Non-negative number                      | 300          |
| `threads`              | Positive integer                         | 2            |
| `max_ram_mb`           | Positive number                          | 4000         |
| `eviction_policy`      | lru, fifo                                | lru          |

Invalid values produce a warning log and fall back to the default. The daemon does not
crash on misconfiguration.

### YAML Dependency

The `pyyaml` library is preferred for parsing but optional. If not installed, a built-in
simple parser (`_simple_yaml_parse`) handles basic nested key-value structures (up to 3
levels of indentation). This allows the config system to work without additional
dependencies in minimal installations.

### Runtime Reload

Configuration can be reloaded without restarting the daemon:

- **Via RPC**: `reload_config` method (optionally with a new config file path).
- **Via signal**: Sending `SIGHUP` to the daemon process triggers a config reload.

Note: Reloading updates the config objects on `FastSearchDaemon` and `ModelManager` but
does not re-bind the socket or restart already-loaded models. Socket path changes require
a full restart.

---

## Appendix A: Text Chunking

The `chunker.py` module splits documents into overlapping chunks for indexing.

### Parameters

| Constant        | Value | Description                                       |
|-----------------|-------|---------------------------------------------------|
| `CHARS_PER_TOKEN` | 4   | Approximate characters per token (English)        |
| `TARGET_TOKENS` | 500   | Target chunk size in tokens                       |
| `TARGET_CHARS`  | 2000  | Target chunk size in characters                   |
| `OVERLAP_TOKENS`| 50    | Overlap between consecutive chunks in tokens      |
| `OVERLAP_CHARS` | 200   | Overlap in characters                             |

### Chunking Strategies

**`chunk_text(text, target_chars, overlap_chars)`**: Splits by paragraph boundaries
(`\n\n`), accumulates paragraphs until target size, includes overlap from the previous
chunk. Long paragraphs exceeding the target are split further by sentence boundaries using
a regex that handles common abbreviations (Dr., Mr., Ms., Mrs., Prof., etc.).

**`chunk_markdown(text, target_chars, overlap_chars)`**: Splits by markdown headers
(`# ... ######`), then applies `chunk_text()` within each section. Yields
`(text, metadata)` tuples where metadata contains `{"section": "<heading text>"}`.

**`estimate_tokens(text)`**: Returns `len(text) // CHARS_PER_TOKEN`.

## Appendix B: Design Decisions

**Why SQLite**: Single-file deployment, no server to manage, ACID transactions, works
everywhere Python works, and sqlite-vec adds efficient vector search.

**Why Unix Sockets**: Lower latency than TCP (no network stack overhead), built-in access
control via file permissions, simple cleanup.

**Why ONNX Runtime**: CPU-optimized inference, 2-3x faster than vanilla PyTorch on CPU,
smaller memory footprint, cross-platform.

**Why APSW over stdlib sqlite3**: APSW provides direct access to the SQLite C API, supports
loadable extensions (required for sqlite-vec), offers better error handling, and supports
`last_insert_rowid()` needed for transactional inserts.

**Why RRF over learned fusion**: No training required, works out-of-the-box, consistent
across different query types, well-studied and predictable behavior.

**Why orjson**: Used for daemon protocol serialization. Significantly faster than stdlib
`json` for encoding/decoding, important for large embedding responses.
