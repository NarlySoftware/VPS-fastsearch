# VPS-FastSearch — Improvement Notes
*Compiled from real-world install experience on OpenClaw 2026.3.12 / Debian 13 / ARM64*
*Date: 2026-03-13*

> **Note on shim architecture:** These notes cover the VPS-fastsearch repo itself.
> The OpenAI-compatible embedding shim (`embedding_shim.py`) used to bridge
> VPS-fastsearch to OpenClaw's memory search is a local implementation — not part
> of the repo. Shim issues (#S1–#S4 below) are documented separately.

---

## Bug Fixes (Confirmed)

### 1. Wrong venv path in service file

**File:** `vps-fastsearch.service`

**Problem:** The bundled systemd service file references the wrong venv path:
```ini
ExecStart=%h/venv/bin/vps-fastsearch daemon start
```

**Fix:**
```ini
ExecStart=%h/fastsearch/.venv/bin/vps-fastsearch daemon start
```

The install script correctly creates the venv at `~/fastsearch/.venv` but the service file doesn't match, causing an immediate `status=203/EXEC` failure on first start.

---

## Bug Fixes (Deep Code Review — Second Pass)

### 3. `_get_db` path traversal check rejects ALL custom `--db` paths

**File:** `daemon.py` — `FastSearchDaemon._get_db()`

**Problem:**
```python
allowed_base = Path(DEFAULT_DB_PATH).resolve().parent  # ~/.local/share/fastsearch/
if allowed_base not in resolved.parents:
    raise ValueError(f"db_path must be under {allowed_base}")
```
Any `db_path` outside `~/.local/share/fastsearch/` is rejected. This breaks all
users who use a custom `--db ~/myproject/search.db` — the client sends that path
in every RPC request (`search`, `batch_index`, `delete`) and the daemon rejects it.
The security intent is valid but the check is too narrow.

**Fix:** Replace the hardcoded `DEFAULT_DB_PATH.parent` with a configurable
allowed-paths list from config, and/or validate by checking path resolution safety
(no `..` traversal) rather than requiring a specific parent directory:
```python
allowed_bases = config.daemon.allowed_db_dirs or [Path(DEFAULT_DB_PATH).parent]
if not any(allowed_base in resolved.parents for allowed_base in allowed_bases):
    raise ValueError(f"db_path must be under one of: {allowed_bases}")
```

---

### 4. `batch_index` retry causes UNIQUE constraint violation

**File:** `client.py` — `FastSearchClient._send_request()`, `core.py` — `SearchDB.index_batch()`

**Problem:** The client auto-retries all requests on connection timeout (including
non-idempotent ones). For `batch_index`, if the server processed the request but
the response was lost, the retry tries to INSERT the same `(source, chunk_index)`
pairs again — hitting the `UNIQUE INDEX idx_docs_source_chunk` constraint and
raising a confusing `FastSearchError: RPC error -32000`.

**Fix:** Make `index_batch` idempotent by using `INSERT OR REPLACE` instead of
plain `INSERT` in `SearchDB.index_batch()`:
```sql
INSERT OR REPLACE INTO docs (source, chunk_index, content, metadata, content_hash)
VALUES (?, ?, ?, ?, ?)
```
This makes retry-on-timeout safe: re-inserting the same chunk just overwrites it.

---

### 5. `index` command sends entire file as one embed batch (OOM on ARM64)

**File:** `cli.py` — `index` command

**Problem:**
```python
texts = [c[0] for c in chunks]   # ALL chunks of the file — could be 60, 100+
result = client.embed(texts)      # Sent as one RPC call
```
The daemon caps at 256 but on ARM64 even 50 texts at once causes OOM kill (ONNX
Runtime allocates heavily). A large file producing 80 chunks will reliably OOM.

**Fix:** Batch in groups of `batch_size` (default 5 on ARM64, 20 elsewhere):
```python
SAFE_EMBED_BATCH = 5  # conservative for ARM64
all_embeddings = []
for i in range(0, len(texts), SAFE_EMBED_BATCH):
    batch = texts[i:i + SAFE_EMBED_BATCH]
    result = client.embed(batch)
    all_embeddings.extend(result["embeddings"])
```
Should also be a config option: `indexing.embed_batch_size`.

---

### 6. `stop_daemon` SIGKILL timeout too short — may leave WAL dirty

**File:** `daemon.py` — `stop_daemon()`

**Problem:** After SIGTERM, `stop_daemon()` waits only 5 seconds before escalating
to SIGKILL:
```python
for _ in range(50):  # 5 seconds
    try:
        os.kill(pid, 0)
        time.sleep(0.1)
    except ProcessLookupError:
        return True
os.kill(pid, signal.SIGKILL)
```
On first shutdown with a loaded ONNX model, the daemon can take 8-15 seconds to
unload the model and complete the WAL checkpoint. SIGKILL prevents the checkpoint,
leaving the SQLite DB in a WAL state that requires recovery on next open.

**Fix:** Increase timeout to 20 seconds, or better, check WAL state before SIGKILL:
```python
for _ in range(200):  # 20 seconds
    ...
```

---

### 7. Negative BM25 scores exposed to API callers

**File:** `core.py` — `SearchDB.search_bm25()`

**Problem:** SQLite FTS5's `bm25()` function returns negative values — lower (more
negative) = more relevant. The `BM25Result` dict exposes these raw:
```json
{"score": -0.42, "source": "memory/2026-03-13.md", ...}
```
Every caller must know SQLite FTS5's sign convention. Unintuitive and a common
source of confusion when building on the API.

**Fix:** Negate at the source so higher = more relevant:
```sql
SELECT d.id, -bm25(docs_fts) AS score, ...
```

---

### 8. `EMBEDDING_DIM = 768` hardcoded — no mismatch guard

**File:** `core.py` — `SearchDB`

**Problem:** The schema creates `docs_vec` with `float32[768]` hardcoded. If a
user switches models to bge-large (1024-dim), inserts silently fail or produce
corrupt vectors. There is no guard comparing stored dims to the current model.

**Fix:** Store embedding dim in `db_meta` on schema creation, check on open:
```python
def _init_schema(self):
    # ... existing schema ...
    # Store dims in metadata
    self._execute(
        "INSERT OR IGNORE INTO db_meta (key, value) VALUES ('embedding_dims', ?)",
        (str(self.EMBEDDING_DIM),)
    )

def _check_embedding_dims(self, model_dims: int):
    row = self._execute(
        "SELECT value FROM db_meta WHERE key = 'embedding_dims'"
    ).fetchone()
    if row and int(row[0]) != model_dims:
        raise RuntimeError(
            f"Embedding dimension mismatch: index has {row[0]}-dim vectors "
            f"but model produces {model_dims}-dim. "
            f"Run: vps-fastsearch index --reindex to rebuild."
        )
```

---

### 9. `rerank_top_k` hard-capped at 30 regardless of limit

**File:** `daemon.py` — `_handle_search()`

**Problem:**
```python
rerank_top_k=min(limit * 3, 30)
```
For `limit > 10`, the reranker only sees 30 candidates to pick the top N results.
With `limit=20`, you pick 20 from 30 — only 10 extras to work with, poor quality.
The cap is undocumented and surprising.

**Fix:** Raise cap or make configurable:
```python
max_rerank_candidates = config.search.max_rerank_candidates  # default 100
rerank_top_k = min(limit * 3, max_rerank_candidates)
```

---

## Recommended Improvements

### 10. ARM64 OOM — Document and default safely (install.sh + README)

**Problem:** ONNX Runtime allocates heavily on ARM64. Embedding 50 texts at once
caused OOM kill (2GB RAM + 8.9GB swap exhausted) during initial workspace indexing.
The daemon itself runs fine at ~480MB; the OOM happens during batch embed calls.

**Fixes:**
- Add an `ARM64` / `aarch64` warning to README and DEPLOYMENT.md
- Default `batch_size` in example configs to `5` for ARM64 (not 50)
- Add a note to the indexer script/example: skip raw conversation logs — they're large
  and redundant if lossless-claw or similar LCM is managing conversation history
- Consider auto-detecting `uname -m` in install.sh and writing a safe config default

**Safe config for ARM64:**
```yaml
memory:
  max_ram_mb: 2000
  eviction_policy: lru

# Not a real config key yet — but worth adding:
indexing:
  batch_size: 5  # Keep low on ARM64; default 50 causes OOM
```

---

### 11. loginctl enable-linger — Add to install script

**Problem:** User-level systemd services (vps-fastsearch, any shim) die at logout
and don't start at boot unless linger is enabled.

**Fix:** Add to `install.sh` after the systemd service step:

```bash
# ---- Step 8b: Enable linger for boot persistence ----
echo "[8b/8] Enabling systemd linger for boot persistence..."

if command -v loginctl >/dev/null 2>&1; then
    CURRENT_USER=$(whoami)
    if loginctl show-user "$CURRENT_USER" 2>/dev/null | grep -q "Linger=yes"; then
        echo "  Linger already enabled for $CURRENT_USER"
    else
        echo "  NOTE: To start VPS-FastSearch at boot (without login), run:"
        echo "    sudo loginctl enable-linger $CURRENT_USER"
        echo "  This is required if OpenClaw or other services start before you log in."
    fi
fi
```

Note: Can't run `sudo loginctl enable-linger` non-interactively without privilege
escalation, but at minimum the install script should prompt clearly.

---

### 16. Incremental indexer — Make it first-class

**Problem:** `DEPLOYMENT.md` references `examples/incremental_indexer.py` and the
bundled timer service files expect it, but it's just an example file. Users either
miss it or don't know where to put it.

**Fix:**
- Promote `incremental_indexer.py` to a proper installed script at `~/fastsearch/scripts/`
- Have `install.sh` copy it there and configure the timer service `ExecStart` to point
  at it automatically
- The timer service files currently have placeholder `ExecStart=` — fill them in during
  install with correct paths

---

### 17. sqlite-vec ARM64 fix — Make more visible

**Problem:** The fix is in `install.sh` (step 4b) but silent when it runs. Users
don't know if it triggered or not.

**Fix:** Improve output visibility — already partially there, but make it louder:
```
[4b/8] Checking sqlite-vec architecture...
  Architecture: aarch64
  Checking vec0.so... ELF 64-bit ✓ (native ARM64, no fix needed)
```
vs:
```
  Checking vec0.so... ELF 32-bit ✗ — rebuilding from source for ARM64...
  Replaced vec0.so with native ARM64 build ✓
```

---

## OpenClaw QMD Protocol — Complete Spec

This is the full protocol extracted directly from OpenClaw 2026.3.12 source
(`qmd-manager-CxdcBNp5.js`). Implementing this in VPS-fastsearch eliminates the
need for an OpenAI-compatible embedding shim entirely.

### Overview

OpenClaw's `memory.backend: "qmd"` delegates all memory search to an external CLI
command. The command is configured via `memory.qmd.command` in `openclaw.json`.
OpenClaw spawns the command as a subprocess and communicates via stdin/stdout.

### Configuration (openclaw.json)

```json
{
  "memory": {
    "backend": "qmd",
    "qmd": {
      "command": "vps-fastsearch",
      "searchMode": "vsearch",
      "includeDefaultMemory": true,
      "paths": [
        { "path": "~/.openclaw/workspace", "name": "memory", "pattern": "**/*.md" }
      ],
      "update": {
        "interval": "10m",
        "onBoot": true
      },
      "limits": {
        "maxResults": 10,
        "timeoutMs": 30000
      }
    }
  }
}
```

### Search Commands

OpenClaw calls one of three search modes based on `memory.qmd.searchMode`:

#### `query` — BM25 / keyword search (always supported, used as fallback)
```
vps-fastsearch query "<text>" --json -n <limit>
vps-fastsearch query "<text>" --json -n <limit> -c <collection-name>
```

#### `search` — Hybrid BM25+vector search
```
vps-fastsearch search "<text>" --json -n <limit>
vps-fastsearch search "<text>" --json -n <limit> -c <collection-name>
```

#### `vector_search` — Pure vector/semantic search (used when searchMode="vsearch")
```
vps-fastsearch vector_search "<text>" --json -n <limit>
vps-fastsearch vector_search "<text>" --json -n <limit> -c <collection-name>
```

**Fallback behaviour:** If `search` or `vector_search` returns an error containing
`unknown flag`, `unknown option`, `unrecognized option`, `flag provided but not defined`,
or `unexpected argument` — OpenClaw automatically retries with `query`. So implementing
`query` alone is enough to get basic integration working; `search` and `vector_search`
are progressive enhancements.

**Search result output** — JSON array on stdout, exit code 0:
```json
[
  {
    "file": "memory/2026-03-13.md",
    "collection": "memory",
    "docid": "optional-unique-id",
    "score": 0.87,
    "snippet": "...matched text excerpt..."
  }
]
```

**No results** — emit one of these on stdout or stderr (exit code 0):
```
no results found.
no results found
```
OpenClaw also accepts log-prefixed variants:
```
[info] qmd: no results found.
warn: no results found.
```

**Collection filter** `-c <name>` — restrict search to a named collection.
Multiple `-c` flags may be passed for multi-collection search (one per collection).

---

### Indexing Commands

#### `update` — Reindex all collections
```
vps-fastsearch update
```
- Called periodically (per `memory.qmd.update.interval`) and on boot if `onBoot: true`
- stdout/stderr output is discarded — exit code 0 = success
- Must complete within `memory.qmd.update.updateTimeoutMs` (default: varies)

#### `embed` — Run embedding pass after update
```
vps-fastsearch embed
```
- Called after `update` when `searchMode` ≠ `"search"`
- stdout/stderr discarded — exit code 0 = success
- Must complete within `memory.qmd.update.embedTimeoutMs`
- If `searchMode = "search"` (BM25 only), this command is never called

---

### Collection Management

OpenClaw manages named collections that map to filesystem paths + glob patterns.

#### `collection add`
```
vps-fastsearch collection add <path> --name <name> --mask <glob-pattern>
```
Example:
```
vps-fastsearch collection add /home/bot/.openclaw/workspace --name memory --mask **/*.md
```
- **Error if already exists:** exit non-zero with message containing `already exists` or `exists`
- OpenClaw handles rebinding conflicts automatically

#### `collection remove`
```
vps-fastsearch collection remove <name>
```
- **Error if missing:** exit non-zero with message containing `not found`, `does not exist`, or `missing`
- OpenClaw ignores missing-collection errors on remove

#### `collection list`
```
vps-fastsearch collection list --json
```

**Output** — JSON array (preferred format):
```json
[
  {
    "name": "memory",
    "path": "/home/bot/.openclaw/workspace",
    "pattern": "**/*.md"
  }
]
```

Also accepted — `mask` instead of `pattern`:
```json
[{ "name": "memory", "path": "/home/bot/.openclaw/workspace", "mask": "**/*.md" }]
```

Also accepted — simple string array (no path/pattern info):
```json
["memory", "docs"]
```

Also accepted — plain text fallback (OpenClaw parses this too):
```
memory (qmd://memory)
  pattern: **/*.md

docs (qmd://docs)
  pattern: **/*.md
```

---

### Error Detection Strings

OpenClaw uses string matching on error messages. Your CLI should use these exact
phrases where appropriate:

| Situation | Required string in error message |
|---|---|
| Unknown flag/option | `unknown flag` OR `unknown option` OR `unrecognized option` OR `flag provided but not defined` OR `unexpected argument` |
| Collection not found | `not found` OR `does not exist` OR `missing` |
| Collection already exists | `already exists` OR `exists` |

These trigger automatic repair/fallback behaviour in OpenClaw. Get them wrong and
OpenClaw will either crash or not recover gracefully.

---

### Implementation Mapping (VPS-fastsearch → QMD)

| QMD Command | VPS-fastsearch equivalent |
|---|---|
| `query "<text>" --json -n <n>` | `search "<text>" --mode bm25 --limit <n>` (output reformatted) |
| `search "<text>" --json -n <n>` | `search "<text>" --mode hybrid --limit <n>` (output reformatted) |
| `vector_search "<text>" --json -n <n>` | `search "<text>" --mode vector --limit <n>` (output reformatted) |
| `update` | `index <collection-paths> --reindex` |
| `embed` | no-op or trigger daemon embed pass |
| `collection add <path> --name <n> --mask <p>` | register path+pattern in internal state |
| `collection remove <name>` | deregister collection |
| `collection list --json` | list registered collections as JSON |

The main work is:
1. Adding a `qmd` subcommand group to the CLI
2. Reformatting search output to the QMD JSON schema (`file`, `collection`, `docid`, `score`, `snippet`)
3. Maintaining a collection registry (SQLite table or simple JSON file)
4. Wiring `update` to re-index registered collection paths

---

## Pluggable Embedding Model Architecture

### Current Limitation

VPS-fastsearch is currently locked to FastEmbed (ONNX Runtime) with
`BAAI/bge-base-en-v1.5`. The embedding dimension (768) is baked into the
sqlite-vec schema at index creation time — switching models requires a full reindex.
There is no guard against dimension mismatch, which would silently corrupt vector search.

---

### Proposed: Multi-Provider Embedding Architecture

#### Provider Config (config.yaml)

```yaml
models:
  embedder:
    # Option A — current default (FastEmbed/ONNX, CPU-optimised)
    provider: fastembed
    name: BAAI/bge-base-en-v1.5
    keep_loaded: always
    threads: 2

    # Option B — Ollama (recommended: free model choice, familiar to self-hosters)
    provider: ollama
    name: nomic-embed-text
    base_url: http://localhost:11434

    # Option C — any OpenAI-compatible HTTP endpoint
    provider: http
    name: bge-large-en-v1.5
    base_url: http://localhost:8080/v1
    api_key: local        # or real key for remote APIs

    # Option D — sentence-transformers (heavier, widest model selection)
    provider: sentence-transformers
    name: BAAI/bge-large-en-v1.5
```

---

### Available Models by Provider

#### FastEmbed (current, CPU-optimised ONNX)

| Model | Dims | RAM | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~180MB | Fastest, lowest RAM |
| `BAAI/bge-base-en-v1.5` | 768 | ~480MB | ← current default |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3GB | Better accuracy |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~550MB | Strong on long docs |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | ~1.3GB | State of art CPU |

#### Ollama (if installed)

Any model in the Ollama library with embedding support:
`nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `snowflake-arctic-embed`, etc.
No ONNX, no FastEmbed dependency — uses Ollama's existing infrastructure.

---

### Dimension Mismatch Guard (critical safety fix)

Currently a model change silently corrupts the vector index. Must add:

```python
# On daemon startup and before any index/search operation:
stored_dims = db.get_meta("embedding_dims")
current_dims = embedder.get_dims()

if stored_dims and int(stored_dims) != current_dims:
    raise RuntimeError(
        f"Embedding dimension mismatch: index has {stored_dims}-dim vectors "
        f"but current model produces {current_dims}-dim vectors. "
        f"Run: vps-fastsearch index --reindex to rebuild the index."
    )

# On first index creation, store dims:
db.set_meta("embedding_dims", str(current_dims))
```

---

### Instruction Prefix Support

BGE and many modern models support task instruction prefixes for better recall.
Expose as config:

```yaml
models:
  embedder:
    name: BAAI/bge-base-en-v1.5
    instruction_prefix:
      document: "Represent this document for retrieval: "
      query: "Represent this query for retrieval: "
```

Low implementation effort, measurable quality improvement especially for
agentic memory recall use cases.

---

### Matryoshka Dimension Truncation

Newer models (nomic-embed-text-v1.5, mxbai-embed-large-v1) support Matryoshka
embeddings — truncate to smaller dims for storage/speed tradeoff:

```yaml
models:
  embedder:
    name: nomic-ai/nomic-embed-text-v1.5
    output_dims: 256   # truncate from 768 → 256 (3x smaller index, small quality cost)
```

Good option for storage-constrained VPS deployments.

---

### Model Hot-Swap Command

Currently changing the model requires manually restarting the daemon and
running a full reindex. Add a first-class command:

```bash
vps-fastsearch model swap BAAI/bge-large-en-v1.5 --reindex
```

Which handles: unload old model → update config → load new model →
drop vector table → reindex all collections → done.

---

### Implementation Priority for Providers

| Priority | Provider | Effort | Why |
|---|---|---|---|
| 🔴 High | Dimension mismatch guard | Low | Prevents silent corruption today |
| 🔴 High | Ollama provider | Medium | Huge — free model choice, familiar ecosystem |
| 🟡 Medium | `http` provider (OpenAI-compatible) | Low | Maximum flexibility, works with any API |
| 🟡 Medium | Instruction prefix config | Low | Better recall quality, no model change needed |
| 🟢 Low | Matryoshka truncation | Medium | Storage savings on small VPS |
| 🟢 Low | `model swap` command | Medium | Better UX for model upgrades |
| 🟢 Low | sentence-transformers provider | Low | Widest model selection |

**Ollama is the highest-value addition.** It's already installed on most
self-hosted AI setups, handles model management transparently, and opens up
dozens of embedding models with zero additional infrastructure.

---

## Local Shim Issues (Already Fixed — 2026-03-13)

These were found in our `~/fastsearch/embedding_shim.py` implementation.
All resolved in the same session.

| # | Issue | Fix Applied |
|---|---|---|
| S1 | Boot race — shim accepted connections 25s before daemon model loaded | Added `wait_for_daemon()` startup loop — waits for `loaded_models.embedder` |
| S2 | Single-threaded `HTTPServer` blocked concurrent requests | Switched to `ThreadingHTTPServer` |
| S3 | New `FastSearchClient` socket per request — connection overhead | Replaced with persistent module-level client with reconnect-on-failure |
| S4 | `log_message` used unsafe `format % args` string interpolation | Fixed to `log.info(format, *args)` |
| S5 | `fastsearch-shim.service` missing `After=network.target` | Added to `[Unit]` section |
| S6 | Stale comment `# venv at ~/venv` in `vps-fastsearch.service` | Updated to `~/fastsearch/.venv` |

---

## Summary Priority List

| Priority | # | Fix | Impact |
|---|---|---|---|
| 🔴 Critical | 1 | Fix venv path in service file | Blocks all installs |
| 🔴 Critical | 5 | `index` batches entire file in one embed call | OOM on ARM64 for large files |
| 🔴 Critical | 10 | ARM64 embed batch size — document + safe defaults | Prevents indexer OOM |
| 🟡 High | 3 | `_get_db` path traversal too restrictive | Breaks all custom `--db` paths |
| 🟡 High | 4 | `batch_index` retry hits UNIQUE constraint | Confusing failures on timeout |
| 🟡 High | 8 | Embedding dim mismatch guard | Prevents silent index corruption |
| 🟡 High | 11 | Add loginctl linger prompt to install.sh | Prevents silent boot failures |
| 🟡 High | 12 | Implement QMD protocol | Eliminates shim entirely |
| 🟡 High | 13 | Ollama embedding provider | Free model choice |
| 🟡 High | 14 | `http` embedding provider | Maximum flexibility |
| 🟢 Medium | 6 | `stop_daemon` SIGKILL timeout too short | Dirty WAL on shutdown |
| 🟢 Medium | 7 | Negative BM25 scores | Confusing API output |
| 🟢 Medium | 9 | `rerank_top_k` hard cap at 30 | Poor rerank quality for limit > 10 |
| 🟢 Medium | 15 | Instruction prefix config | Better recall quality |
| 🟢 Medium | 16 | Promote incremental indexer to first-class | Better UX |
| 🟢 Medium | 17 | Improve sqlite-vec ARM64 fix visibility | Better diagnostics |
| 🟢 Low | 18 | Matryoshka truncation | Storage savings |
| 🟢 Low | 19 | `model swap` command | Better upgrade UX |
