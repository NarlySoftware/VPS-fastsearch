# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-03-01

### Added
- **`migrate-paths` CLI command**: Bulk-convert legacy absolute source paths to relative paths with `--dry-run` mode, `--base-dir` override, and collision detection
- **`--strict` flag on `index` command**: Opt-in portable mode that rejects files outside `base_dir`, preventing non-portable `../../` paths
- **`is_within_base_dir()` method**: Check whether a path falls within the configured base directory
- **Content hash deduplication**: SHA-256 `content_hash` column on `docs` table for integrity checking and optional dedup via `skip_duplicates` parameter on `index_document()` and `index_batch()`
- **CLI smoke tests**: New `tests/test_cli.py` with CliRunner tests for `--version`, `--help`, and `migrate-paths` scenarios
- **Schema v2→v3 migration test**: Regression test creating a real v2 database and verifying automatic upgrade
- **Cross-host migration test**: Validates path portability across different `base_dir` settings
- **Wheel smoke CI job**: Builds wheel, installs from it, and verifies CLI and imports work

### Changed
- **Database schema version 3**: Auto-migrates from v2 on first open; adds `content_hash` column and backfills existing rows with SHA-256 hashes
- **CI workflow**: Added `vps-fastsearch --version` and `--help` smoke test step

### Stats
- 231 unit tests (up from 208)
- Deployed and verified on Debian 13 VM (eva@100.123.195.35)

## [0.3.0] - 2026-02-27

### Changed
- Comprehensive security and robustness audit fixes
- Updated install guide and tarball reference for Debian 13
- Added explicit onnxruntime>=1.20 pin for ARM64 aarch64 wheel availability
- Fixed DEPLOYMENT.md Python version references for Debian 13
- Fixed INSTALL_GUIDE.md systemd service filename reference

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
