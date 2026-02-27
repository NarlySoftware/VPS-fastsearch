# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-27

### Fixed
- **Memory optimization**: Limit ONNX Runtime threads to 2 during model load, reducing arena memory allocation
- **FTS5 query crash**: Sanitize search queries containing hyphens (e.g., "node-llama-cpp") that were parsed as FTS5 column operators
- **Embedding precision**: Changed vector storage from `FLOAT[768]` to `float64[768]` for improved precision
- **Reranker error handling**: Graceful ImportError with helpful install message when sentence-transformers is missing

### Added
- **Debian 13 (Trixie) support**: Install script updated for Python 3.13 (system default on Debian 13)
- **ARM64 (aarch64) support**: Verified all native dependencies have prebuilt aarch64 wheels
- **Native extension verification**: Install script now verifies sqlite-vec, apsw, and onnxruntime load correctly
- **Unit test suite**: pytest tests for chunker, config, and core SearchDB

### Changed
- Install script targets Python 3.13 instead of 3.12
- Removed Ubuntu-only deadsnakes PPA fallback (not needed on Debian 13)

## [0.2.0] - 2025-02-15

### Added
- Initial public release
- Hybrid search combining BM25 + vector similarity with RRF fusion
- Cross-encoder reranking with ms-marco-MiniLM
- Daemon mode with Unix socket server for instant search latency
- LRU memory management with configurable budget
- Markdown-aware chunking with section preservation
- CLI with `index`, `search`, and `daemon` commands
- Python client API
- Comprehensive documentation

### Performance
- 4ms hybrid search latency in daemon mode
- ~500MB memory footprint with bge-base embedder
- Supports 100K+ document chunks efficiently
