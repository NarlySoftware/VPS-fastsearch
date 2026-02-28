"""Shared fixtures and constants for vps_fastsearch test suite."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from vps_fastsearch.config import (
    DaemonConfig,
    FastSearchConfig,
    MemoryConfig,
    ModelConfig,
)
from vps_fastsearch.core import SearchDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 768
"""Dimensionality of the BGE-base-en-v1.5 embedding model."""

DUMMY_EMBEDDING: list[float] = [0.1] * EMBEDDING_DIM
"""A trivial 768-dim embedding vector for tests that don't care about values."""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: object) -> Generator[SearchDB, None, None]:
    """Create a temporary SearchDB instance backed by a file in tmp_path."""
    from pathlib import Path

    database = SearchDB(Path(str(tmp_path)) / "test.db")
    yield database
    database.close()


# ---------------------------------------------------------------------------
# Helpers (imported explicitly where needed)
# ---------------------------------------------------------------------------


def _make_config(
    max_ram_mb: int = 4000,
    socket_path: str = "/tmp/test_fastsearch.sock",
    pid_path: str = "/tmp/test_fastsearch.pid",
    include_models: bool = True,
) -> FastSearchConfig:
    """Build a minimal FastSearchConfig for testing.

    Parameters
    ----------
    max_ram_mb:
        Maximum RAM budget in MB.
    socket_path:
        Unix socket path for the daemon.
    pid_path:
        PID file path for the daemon.
    include_models:
        If True, include default embedder and reranker model configs.
        If False, use an empty models dict (useful for client-only tests).
    """
    models: dict[str, ModelConfig] = {}
    if include_models:
        models = {
            "embedder": ModelConfig(
                name="BAAI/bge-base-en-v1.5",
                keep_loaded="always",
                idle_timeout_seconds=0,
                threads=2,
            ),
            "reranker": ModelConfig(
                name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                keep_loaded="on_demand",
                idle_timeout_seconds=300,
                threads=2,
            ),
        }
    return FastSearchConfig(
        daemon=DaemonConfig(socket_path=socket_path, pid_path=pid_path),
        models=models,
        memory=MemoryConfig(max_ram_mb=max_ram_mb),
    )
