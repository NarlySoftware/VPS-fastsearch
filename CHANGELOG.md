# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
