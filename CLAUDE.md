# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VPS-FastSearch is a hybrid search system combining BM25 full-text search with vector similarity search, designed for CPU-only VPS environments. It uses SQLite with FTS5 and sqlite-vec extensions, ONNX Runtime for inference, and a Unix socket daemon for low-latency queries.

## Build & Development Commands

```bash
# Install in development mode (all extras)
pip install -e ".[all,dev]"

# Run unit tests
pytest tests/ -v

# Run a single test file
pytest tests/test_core.py -v

# Run a single test
pytest tests/test_core.py::TestClassName::test_method -v

# Lint
ruff check vps_fastsearch/

# Type check
mypy vps_fastsearch/ --strict --ignore-missing-imports

# Format
ruff format vps_fastsearch/

# Integration tests (starts/stops daemon, requires models downloaded)
python run_tests.py
```

## Architecture

### Core Components (`vps_fastsearch/`)

- **core.py** — Three main classes:
  - `Embedder`: Wraps BAAI/bge-base-en-v1.5 (768-dim) via FastEmbed + ONNX Runtime
  - `Reranker`: Cross-encoder ms-marco-MiniLM via sentence-transformers (optional `[rerank]` extra)
  - `SearchDB`: SQLite database with FTS5 (BM25) and sqlite-vec (vector search), uses APSW bindings

- **daemon.py** — Unix socket server with JSON-RPC 2.0 protocol. `ModelManager` handles model lifecycle with LRU eviction and configurable memory budgets.

- **client.py** — `FastSearchClient` communicates with daemon over Unix socket. Auto-reconnect and context manager support.

- **cli.py** — Click-based CLI (`vps-fastsearch`). Subcommands: `daemon`, `index`, `search`, `config`, `stats`, `delete`, `list`.

- **config.py** — YAML configuration at `~/.config/fastsearch/config.yaml` with XDG compliance and environment variable overrides.

- **chunker.py** — Text chunking with configurable overlap for indexing.

### Search Algorithm

Uses **Reciprocal Rank Fusion (RRF)** to merge BM25 and vector results:
`score = weight_bm25/(k+bm25_rank) + weight_vec/(k+vec_rank)` where k=60.

### Database Schema

SQLite with WAL mode. Four key structures:
1. `docs` table (id, source, chunk_index, content, metadata, created_at)
2. `docs_fts` FTS5 virtual table with porter unicode61 tokenizer (auto-synced via triggers)
3. `docs_vec` sqlite-vec virtual table for 768-dim cosine similarity search
4. `db_meta` key-value table for database-level settings (e.g., `base_dir` for relative path resolution)

### Daemon Protocol

JSON-RPC 2.0 over Unix socket. 13 methods: `ping`, `status`, `search`, `embed`, `rerank`, `load_model`, `unload_model`, `reload_config`, `batch_index`, `delete`, `update_content`, `list_sources`, `shutdown`. Client sends 4-byte big-endian length header + JSON payload.

## Code Style

- **Line length**: 100 (ruff)
- **Type hints**: Required on all functions (mypy strict with `disallow_untyped_defs`)
- **Target Python**: 3.10+ (3.13 recommended)
- **Ruff rules**: E, W, F, I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade)
- **APSW** is used instead of stdlib `sqlite3` for SQLite bindings

## Notable Features

- **Metadata filtering**: Search supports `--filter key=value` (CLI) or `metadata_filter` param (RPC) for exact-match filtering on JSON metadata fields. Multiple filters use AND logic.
- **Batch indexing**: `batch_index` RPC method accepts up to 1000 documents per call with validation.
- **Delete/update**: `delete` (by source or ID) and `update_content` (by ID, auto-generates new embedding) available via CLI and RPC.
- **Relative paths**: Source paths stored relative to a configurable `base_dir` (stored in `db_meta` table). Supports `--base-dir` CLI option and `to_relative`/`to_absolute` methods on SearchDB.
- **List sources**: `list` CLI command and `list_sources` RPC method show all indexed sources with chunk counts and ID ranges.

## Deployment & Scheduling

### Systemd Units (in repo root)

All units are **user-level** (`~/.config/systemd/user/`). Use `%h` for home directory portability.

- **`vps-fastsearch.service`** — Main daemon (long-running, auto-restart)
- **`fastsearch-index-incremental.timer`** — Triggers incremental indexing every 15 minutes (+ 3 min after boot)
- **`fastsearch-index-incremental.service`** — Oneshot: runs the indexer script in `--mode incremental`
- **`fastsearch-index-full.timer`** — Triggers full reindex nightly at 02:17
- **`fastsearch-index-full.service`** — Oneshot: runs the indexer script in `--mode full`

### Installation

```bash
# Copy units to user systemd directory
cp vps-fastsearch.service fastsearch-index-*.service fastsearch-index-*.timer \
   ~/.config/systemd/user/

# Reload, enable, and start
systemctl --user daemon-reload
systemctl --user enable --now vps-fastsearch.service
systemctl --user enable --now fastsearch-index-incremental.timer
systemctl --user enable --now fastsearch-index-full.timer

# Enable lingering so user services run without login
loginctl enable-linger $USER
```

### Important Notes

- **Do NOT use crontab** for indexing — use only systemd timers (they have `After=vps-fastsearch.service` dependency)
- The indexer script path in the service files must be updated to match your deployment (default: `%h/.openclaw/workspace/scripts/fastsearch_index.py`)
- An example generic indexer is provided at `examples/incremental_indexer.py`
- Check timer status: `systemctl --user list-timers`
- Check indexer logs: `journalctl --user -u fastsearch-index-incremental.service`

## Key Dependencies

- `fastembed` — ONNX-based embedding inference
- `sqlite-vec` — Vector similarity extension for SQLite
- `apsw` — Advanced SQLite bindings (used instead of stdlib sqlite3)
- `click` — CLI framework
- `orjson` — Fast JSON serialization (used for daemon protocol)
- `sentence-transformers` — Optional, for reranking (`[rerank]` extra)
