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
mypy vps_fastsearch/ --ignore-missing-imports

# Format
black vps_fastsearch/

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

- **cli.py** — Click-based CLI (`vps-fastsearch`). Subcommands: `daemon`, `index`, `search`, `config`.

- **config.py** — YAML configuration at `~/.config/fastsearch/config.yaml` with XDG compliance and environment variable overrides.

- **chunker.py** — Text chunking with configurable overlap for indexing.

### Search Algorithm

Uses **Reciprocal Rank Fusion (RRF)** to merge BM25 and vector results:
`score = weight_bm25/(k+bm25_rank) + weight_vec/(k+vec_rank)` where k=60.

### Database Schema

SQLite with WAL mode. Three key structures:
1. `documents` table (id, source, chunk_index, content, embedding as float32 blob)
2. `documents_fts` FTS5 virtual table with porter unicode61 tokenizer (auto-synced via triggers)
3. sqlite-vec virtual table for 768-dim cosine similarity search

### Daemon Protocol

JSON-RPC 2.0 over Unix socket. Methods include `search`, `embed`, `rerank`, `status`, `reload`, `unload_model`. Client sends JSON + newline delimiter.

## Code Style

- **Line length**: 100 (black + ruff)
- **Type hints**: Required on all functions (mypy strict with `disallow_untyped_defs`)
- **Target Python**: 3.10+ (3.13 recommended)
- **Ruff rules**: E, W, F, I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade)
- **APSW** is used instead of stdlib `sqlite3` for SQLite bindings

## Key Dependencies

- `fastembed` — ONNX-based embedding inference
- `sqlite-vec` — Vector similarity extension for SQLite
- `apsw` — Advanced SQLite bindings (used instead of stdlib sqlite3)
- `click` — CLI framework
- `orjson` — Fast JSON serialization (used for daemon protocol)
- `sentence-transformers` — Optional, for reranking (`[rerank]` extra)
