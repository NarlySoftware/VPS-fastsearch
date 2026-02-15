# CLI Reference

FastSearch provides a comprehensive command-line interface for indexing, searching, and managing the daemon.

## Global Options

```bash
fastsearch [OPTIONS] COMMAND [ARGS]...
```

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| `--db PATH` | `FASTSEARCH_DB` | Database path (default: `fastsearch.db`) |
| `--config PATH` | `FASTSEARCH_CONFIG` | Config file path |
| `--help` | | Show help message |

## Commands Overview

| Command | Description |
|---------|-------------|
| `index` | Index files or directories |
| `search` | Search indexed documents |
| `stats` | Show database statistics |
| `delete` | Delete indexed source |
| `daemon` | Manage the daemon server |
| `config` | Manage configuration |

---

## index

Index a file or directory of documents.

```bash
fastsearch index PATH [OPTIONS]
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

### Examples

```bash
# Index a single file
fastsearch index README.md

# Index all markdown files in a directory
fastsearch index ./docs

# Index with a different pattern
fastsearch index ./docs --glob "*.txt"

# Index Python files
fastsearch index ./src --glob "*.py"

# Re-index (delete and recreate)
fastsearch index ./docs --reindex

# Index to a specific database
fastsearch --db myproject.db index ./docs
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
fastsearch search QUERY [OPTIONS]
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
fastsearch search "how to configure settings"

# Get more results
fastsearch search "configuration" --limit 10

# BM25 only (keyword matching)
fastsearch search "socket_path" --mode bm25

# Vector only (semantic)
fastsearch search "how to set up the system" --mode vector

# With reranking (more accurate)
fastsearch search "complex query about configuration" --rerank

# Force direct mode (no daemon)
fastsearch search "query" --no-daemon

# JSON output for scripting
fastsearch search "query" --json

# Search a specific database
fastsearch --db project.db search "query"
```

### Output (Text)

```
Search: 'configuration' (hybrid [daemon], 4ms)

[1] CONFIGURATION.md (chunk 0) - RRF: 0.0323, BM25 #1, Vec #3
    # Configuration Reference FastSearch uses a YAML configuration file...

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
fastsearch stats
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

Delete all chunks from a source file.

```bash
fastsearch delete SOURCE
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `SOURCE` | Yes | Source file path (partial match supported) |

### Examples

```bash
# Delete by exact path
fastsearch delete "/path/to/README.md"

# Delete by filename (partial match)
fastsearch delete "README.md"

# Delete by partial match
fastsearch delete "README"
```

### Output

```
Deleted 8 chunks from /path/to/README.md
```

### Notes

- Supports partial matching
- If multiple matches found, you'll be asked to be more specific
- Deletion is immediate (no undo)

---

## daemon

Manage the FastSearch daemon server.

### daemon start

Start the daemon server.

```bash
fastsearch daemon start [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-d, --detach` | Run in background |
| `--config PATH` | Config file path |

```bash
# Start in foreground (Ctrl+C to stop)
fastsearch daemon start

# Start in background
fastsearch daemon start --detach

# Start with custom config
fastsearch daemon start --config /etc/fastsearch/config.yaml
```

### daemon stop

Stop the running daemon.

```bash
fastsearch daemon stop [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Config file path (to find socket) |

```bash
fastsearch daemon stop
```

### daemon status

Show daemon status.

```bash
fastsearch daemon status [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--config PATH` | Config file path |

**Output (text):**
```
FastSearch Daemon Status
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
fastsearch daemon reload [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Config file path to reload |

```bash
# Reload default config
fastsearch daemon reload

# Reload specific config
fastsearch daemon reload --config /etc/fastsearch/config.yaml
```

---

## config

Manage configuration.

### config init

Create default configuration file.

```bash
fastsearch config init [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--path PATH` | Custom config file path |

```bash
# Create at default location (~/.config/fastsearch/config.yaml)
fastsearch config init

# Create at custom location
fastsearch config init --path /etc/fastsearch/config.yaml
```

### config show

Show current configuration.

```bash
fastsearch config show [OPTIONS]
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
fastsearch config path
```

**Output:**
```
/Users/username/.config/fastsearch/config.yaml
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
fastsearch index ./docs
fastsearch search "query"
```

---

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
_FASTSEARCH_COMPLETE=bash_source fastsearch > ~/.fastsearch-complete.bash
echo "source ~/.fastsearch-complete.bash" >> ~/.bashrc

# Zsh
_FASTSEARCH_COMPLETE=zsh_source fastsearch > ~/.fastsearch-complete.zsh
echo "source ~/.fastsearch-complete.zsh" >> ~/.zshrc

# Fish
_FASTSEARCH_COMPLETE=fish_source fastsearch > ~/.config/fish/completions/fastsearch.fish
```
