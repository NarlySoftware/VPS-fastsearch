# CLI Reference

VPS-FastSearch provides a comprehensive command-line interface for indexing, searching, and managing the daemon.

## Global Options

```bash
vps-fastsearch [OPTIONS] COMMAND [ARGS]...
```

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| `--version` | | Show the version and exit |
| `--db PATH` | `FASTSEARCH_DB` | Database path (default: `fastsearch.db`) |
| `--config PATH` | `FASTSEARCH_CONFIG` | Config file path |
| `--help` | | Show help message |

## Commands Overview

| Command | Description |
|---------|-------------|
| `index` | Index files or directories |
| `search` | Search indexed documents |
| `stats` | Show database statistics |
| `delete` | Delete documents by source name or by ID |
| `list` | List all indexed sources with chunk counts |
| `migrate-paths` | Convert absolute or misaligned relative source paths |
| `daemon` | Manage the daemon server |
| `config` | Manage configuration |
| `query` | BM25/keyword search (QMD protocol) |
| `vector_search` | Vector/semantic search (QMD protocol) |
| `update` | Reindex all registered collections (QMD protocol) |
| `embed` | Run embedding pass (QMD protocol, no-op) |
| `collection` | Manage QMD collections |

---

## index

Index a file or directory of documents.

```bash
vps-fastsearch index PATH [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PATH` | Yes | File or directory to index |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-g, --glob PATTERN` | `*.md` | Glob pattern for directory indexing |
| `--reindex` | `false` | Delete existing chunks before indexing |
| `--base-dir DIRECTORY` | DB parent dir | Base directory for relative path storage |
| `--strict` | `false` | Reject files outside base_dir (portable mode) |

### Examples

```bash
# Index a single file
vps-fastsearch index README.md

# Index all markdown files in a directory
vps-fastsearch index ./docs

# Index with a different pattern
vps-fastsearch index ./docs --glob "*.txt"

# Index Python files
vps-fastsearch index ./src --glob "*.py"

# Re-index (delete and recreate)
vps-fastsearch index ./docs --reindex

# Index to a specific database
vps-fastsearch --db myproject.db index ./docs
```

### Output

```
Indexing 5 file(s)...
Using daemon for embedding...
  README.md: 3 chunks (embed: 0.05s, index: 0.01s)
  ARCHITECTURE.md: 12 chunks (embed: 0.15s, index: 0.02s)
  API.md: 8 chunks (embed: 0.10s, index: 0.01s)
  CLI.md: 6 chunks (embed: 0.08s, index: 0.01s)
  CONFIGURATION.md: 4 chunks (embed: 0.05s, index: 0.01s)

Indexed 33 chunks in 0.49s
```

### Notes

- If the daemon is running, indexing uses it for embedding (faster)
- If not, the CLI loads the model directly (first use is slow)
- Chunks are ~500 tokens (~2000 characters) with overlap
- Markdown files get section metadata attached to chunks

---

## search

Search indexed documents.

```bash
vps-fastsearch search QUERY [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY` | Yes | Search query text |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --limit N` | `5` | Number of results to return |
| `-m, --mode MODE` | `hybrid` | Search mode: `hybrid`, `bm25`, `vector` |
| `-r, --rerank` | `false` | Use cross-encoder reranking |
| `-f, --filter TEXT` | | Metadata filter as `key=value` (repeatable, AND logic) |
| `--no-daemon` | `false` | Force direct mode (skip daemon) |
| `--json` | `false` | Output as JSON |

### Search Modes

| Mode | Description | Speed | Best For |
|------|-------------|-------|----------|
| `hybrid` | BM25 + vector with RRF fusion | Fast | General queries |
| `bm25` | Full-text search only | Fastest | Exact phrase matching |
| `vector` | Semantic similarity only | Fast | Meaning-based queries |

### Examples

```bash
# Basic hybrid search
vps-fastsearch search "how to configure settings"

# Get more results
vps-fastsearch search "configuration" --limit 10

# BM25 only (keyword matching)
vps-fastsearch search "socket_path" --mode bm25

# Vector only (semantic)
vps-fastsearch search "how to set up the system" --mode vector

# With reranking (more accurate)
vps-fastsearch search "complex query about configuration" --rerank

# Force direct mode (no daemon)
vps-fastsearch search "query" --no-daemon

# JSON output for scripting
vps-fastsearch search "query" --json

# Search a specific database
vps-fastsearch --db project.db search "query"
```

### Output (Text)

```
Search: 'configuration' (hybrid [daemon], 4ms)

[1] CONFIGURATION.md (chunk 0) - RRF: 0.0323, BM25 #1, Vec #3
    # Configuration Reference VPS-FastSearch uses a YAML configuration file...

[2] README.md (chunk 5) - RRF: 0.0298, BM25 #2, Vec #7
    ## Configuration Create config at `~/.config/fastsearch/config.yaml`...

[3] ARCHITECTURE.md (chunk 8) - RRF: 0.0276, BM25 #5, Vec #2
    ### Configuration System The `FastSearchConfig` class manages all settings...
```

### Output (JSON)

```json
{
  "query": "configuration",
  "mode": "hybrid",
  "reranked": false,
  "daemon": true,
  "search_time_ms": 4.21,
  "results": [
    {
      "id": 15,
      "source": "/path/to/CONFIGURATION.md",
      "chunk_index": 0,
      "content": "# Configuration Reference...",
      "metadata": {"section": "Configuration Reference"},
      "rrf_score": 0.0323,
      "bm25_rank": 1,
      "vec_rank": 3,
      "rank": 1
    }
  ]
}
```

### Notes

- Uses daemon if available, falls back to direct mode
- First search without daemon loads the model (~5-10s)
- Reranking adds ~40ms but improves accuracy significantly
- BM25 mode doesn't require model loading

---

## stats

Show database statistics.

```bash
vps-fastsearch stats
```

### Output

```
Database: fastsearch.db
Size: 2.34 MB
Total chunks: 156
Total sources: 12

Top sources by chunks:
  ARCHITECTURE.md: 24 chunks
  API.md: 18 chunks
  CLI.md: 15 chunks
  CONFIGURATION.md: 12 chunks
  README.md: 10 chunks
```

---

## delete

Delete documents by source name or by ID.

```bash
vps-fastsearch delete [SOURCE] [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `SOURCE` | No | Source file path (partial match supported) |

### Options

| Option | Description |
|--------|-------------|
| `--id INTEGER` | Delete a single document by ID |

### Examples

```bash
# Delete by exact path
vps-fastsearch delete "/path/to/README.md"

# Delete by filename (partial match)
vps-fastsearch delete "README.md"

# Delete a single document by ID
vps-fastsearch delete --id 42
```

### Notes

- Provide a SOURCE name to delete all chunks for that source (supports partial match)
- Use `--id` to delete a single document by its numeric ID
- Deletion is immediate (no undo)

---

## list

List all indexed sources with chunk counts.

```bash
vps-fastsearch list [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### Examples

```bash
# List all indexed sources
vps-fastsearch list

# JSON output for scripting
vps-fastsearch list --json
```

---

## migrate-paths

Convert absolute or misaligned relative source paths to clean relative paths.

```bash
vps-fastsearch migrate-paths [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would change without modifying the DB |
| `--base-dir DIRECTORY` | Set/override base directory before migration |
| `--old-base-dir DIRECTORY` | Rebase relative paths: resolve against old base dir, re-relativize against current |
| `--force` | Allow migration even when paths fall outside base directory |

### Examples

```bash
# Preview what would change
vps-fastsearch migrate-paths --dry-run

# Convert absolute paths to relative
vps-fastsearch migrate-paths

# Rebase paths that were relative to a different directory
vps-fastsearch migrate-paths --old-base-dir /home/user/.local/share/fastsearch

# Set a new base directory and migrate
vps-fastsearch migrate-paths --base-dir /home/user/workspace
```

### Notes

- Without `--old-base-dir`, converts absolute paths to relative
- With `--old-base-dir`, also rebases relative paths computed against a different base directory
- Use `--dry-run` first to preview changes
- Use `--force` if some paths fall outside the base directory

---

## daemon

Manage the VPS-FastSearch daemon server.

### daemon start

Start the daemon server.

```bash
vps-fastsearch daemon start [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-d, --detach` | Run in background |
| `--config PATH` | Config file path |

```bash
# Start in foreground (Ctrl+C to stop)
vps-fastsearch daemon start

# Start in background
vps-fastsearch daemon start --detach

# Start with custom config
vps-fastsearch daemon start --config /etc/fastsearch/config.yaml
```

### daemon stop

Stop the running daemon.

```bash
vps-fastsearch daemon stop [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Config file path (to find socket) |

```bash
vps-fastsearch daemon stop
```

### daemon status

Show daemon status.

```bash
vps-fastsearch daemon status [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--config PATH` | Config file path |

**Output (text):**
```
VPS-FastSearch Daemon Status
========================================
Uptime:         2h 15m 30s
Requests:       1234
Memory:         520MB / 4000MB
Socket:         /tmp/fastsearch.sock

Loaded Models:
  embedder: 450MB (idle: 5s)
  reranker: 90MB (idle: 120s)
```

**Output (JSON):**
```json
{
  "uptime_seconds": 8130,
  "request_count": 1234,
  "socket_path": "/tmp/fastsearch.sock",
  "loaded_models": {
    "embedder": {
      "loaded_at": 1704067200,
      "last_used": 1704075325,
      "memory_mb": 450,
      "idle_seconds": 5
    },
    "reranker": {
      "loaded_at": 1704070000,
      "last_used": 1704075205,
      "memory_mb": 90,
      "idle_seconds": 120
    }
  },
  "total_memory_mb": 520,
  "max_memory_mb": 4000
}
```

### daemon reload

Reload configuration without restart.

```bash
vps-fastsearch daemon reload [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Config file path to reload |

```bash
# Reload default config
vps-fastsearch daemon reload

# Reload specific config
vps-fastsearch daemon reload --config /etc/fastsearch/config.yaml
```

---

## config

Manage configuration.

### config init

Create default configuration file.

```bash
vps-fastsearch config init [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--path PATH` | Custom config file path |

```bash
# Create at default location (~/.config/fastsearch/config.yaml)
vps-fastsearch config init

# Create at custom location
vps-fastsearch config init --path /etc/fastsearch/config.yaml
```

### config show

Show current configuration.

```bash
vps-fastsearch config show [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--path PATH` | Config file to show |

**Output:**
```yaml
daemon:
  socket_path: /tmp/fastsearch.sock
  pid_path: /tmp/fastsearch.pid
  log_level: INFO
models:
  embedder:
    name: BAAI/bge-base-en-v1.5
    keep_loaded: always
    idle_timeout_seconds: 0
  reranker:
    name: cross-encoder/ms-marco-MiniLM-L-6-v2
    keep_loaded: on_demand
    idle_timeout_seconds: 300
memory:
  max_ram_mb: 4000
  eviction_policy: lru
```

### config path

Show default config file path.

```bash
vps-fastsearch config path
```

**Output:**
```
/Users/username/.config/fastsearch/config.yaml
```

---

## QMD Protocol Commands

These commands implement the QMD (Query-Memory-Daemon) protocol used by OpenClaw and other integrations. They always output JSON.

### query

BM25/keyword search via the QMD protocol.

```bash
vps-fastsearch query QUERY_TEXT [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY_TEXT` | Yes | Search query text |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n` | `10` | Number of results to return |
| `-c` | | Collection name filter |
| `--json` | | JSON output (always on, hidden flag) |

#### Example

```bash
# Keyword search
vps-fastsearch query "configuration reference"

# Limit results and filter by collection
vps-fastsearch query "socket path" -n 5 -c docs
```

---

### vector_search

Vector/semantic search via the QMD protocol.

```bash
vps-fastsearch vector_search QUERY_TEXT [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY_TEXT` | Yes | Search query text |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n` | `10` | Number of results to return |
| `-c` | | Collection name filter |
| `--json` | | JSON output (always on, hidden flag) |

#### Example

```bash
# Semantic search
vps-fastsearch vector_search "how do I set up the system"

# Filter by collection
vps-fastsearch vector_search "deployment steps" -c workspace -n 3
```

---

### update

Reindex all registered collections. Walks each collection path, chunks files matching the collection pattern, generates embeddings (via daemon if available, otherwise direct), and indexes the results.

```bash
vps-fastsearch update
```

No arguments or options. Silently skips collections whose paths do not exist or contain no matching files.

#### Example

```bash
# Reindex everything
vps-fastsearch update
```

---

### embed

Run an embedding pass (QMD protocol). This is a no-op -- embeddings are generated during `update`. Exits with code 0.

```bash
vps-fastsearch embed
```

---

### QMD Search Output Format

Both `query` and `vector_search` output a JSON array of result objects:

```json
[
  {
    "file": "relative/path/to/file.md",
    "collection": "docs",
    "docid": "42",
    "score": 0.032258,
    "snippet": "First 500 characters of the chunk..."
  }
]
```

---

## collection

Manage QMD collections (OpenClaw integration). Collections register directory paths with glob patterns for automated indexing via the `update` command.

### collection add

Register a collection path for QMD indexing.

```bash
vps-fastsearch collection add PATH --name NAME --mask GLOB
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PATH` | Yes | Directory path to index |

#### Options

| Option | Required | Description |
|--------|----------|-------------|
| `--name NAME` | Yes | Collection name |
| `--mask GLOB` | Yes | Glob pattern for files (e.g., `**/*.md`) |

#### Example

```bash
# Register a docs collection
vps-fastsearch collection add ./docs --name docs --mask "**/*.md"

# Register a workspace
vps-fastsearch collection add ~/.openclaw/workspace --name workspace --mask "**/*.md"
```

### collection remove

Remove a registered collection.

```bash
vps-fastsearch collection remove NAME
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Collection name to remove |

#### Example

```bash
vps-fastsearch collection remove docs
```

### collection list

List all registered collections.

```bash
vps-fastsearch collection list [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

#### Example

```bash
# Human-readable list
vps-fastsearch collection list

# JSON output
vps-fastsearch collection list --json
```

#### Output (Text)

```
docs (qmd://docs)
  path: /home/user/project/docs
  pattern: **/*.md
```

#### Output (JSON)

```json
[
  {
    "name": "docs",
    "path": "/home/user/project/docs",
    "pattern": "**/*.md"
  }
]
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FASTSEARCH_DB` | Default database path |
| `FASTSEARCH_CONFIG` | Default config file path |

```bash
# Set defaults via environment
export FASTSEARCH_DB="/var/lib/fastsearch/main.db"
export FASTSEARCH_CONFIG="/etc/fastsearch/config.yaml"

# Now all commands use these defaults
vps-fastsearch index ./docs
vps-fastsearch search "query"
```

---

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
_VPS_FASTSEARCH_COMPLETE=bash_source vps-fastsearch > ~/.vps-fastsearch-complete.bash
echo "source ~/.vps-fastsearch-complete.bash" >> ~/.bashrc

# Zsh
_VPS_FASTSEARCH_COMPLETE=zsh_source vps-fastsearch > ~/.vps-fastsearch-complete.zsh
echo "source ~/.vps-fastsearch-complete.zsh" >> ~/.zshrc

# Fish
_VPS_FASTSEARCH_COMPLETE=fish_source vps-fastsearch > ~/.config/fish/completions/vps-fastsearch.fish
```
