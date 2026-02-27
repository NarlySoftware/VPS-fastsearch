"""Tests for vps_fastsearch.core.SearchDB — uses apsw/sqlite-vec but no ML models."""

import pytest

from vps_fastsearch.core import SearchDB


@pytest.fixture
def db(tmp_path):
    """Create a temporary SearchDB instance."""
    database = SearchDB(tmp_path / "test.db")
    yield database
    database.close()


def test_init_schema(db):
    """Database should have docs, docs_fts, and docs_vec tables."""
    tables = [
        row[0]
        for row in db._execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    ]
    assert "docs" in tables
    assert "docs_fts" in tables
    assert "docs_vec" in tables


def test_index_and_search_bm25(db):
    """Indexed document should be findable via BM25 search."""
    embedding = [0.1] * 768
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("Python programming", limit=5)
    assert len(results) >= 1
    assert "Python" in results[0]["content"]


def test_index_and_search_vector(db):
    """Indexed document should be findable via vector search."""
    embedding = [1.0] * 768
    db.index_document("test.md", 0, "Vector search test document", embedding)

    results = db.search_vector([1.0] * 768, limit=5)
    assert len(results) >= 1
    assert results[0]["content"] == "Vector search test document"


def test_search_hybrid(db):
    """Hybrid search should return results combining BM25 and vector."""
    embedding = [0.5] * 768
    db.index_document("test.md", 0, "Hybrid search combines multiple methods", embedding)

    results = db.search_hybrid("hybrid search methods", [0.5] * 768, limit=5)
    assert len(results) >= 1
    assert "rrf_score" in results[0]


def test_delete_source(db):
    """Deleting a source should remove its documents."""
    embedding = [0.1] * 768
    db.index_document("deleteme.md", 0, "This will be deleted", embedding)
    db.index_document("keepme.md", 0, "This will remain", embedding)

    deleted = db.delete_source("deleteme.md")
    assert deleted == 1

    results = db.search_bm25("deleted", limit=5)
    assert len(results) == 0

    results = db.search_bm25("remain", limit=5)
    assert len(results) == 1


def test_get_stats(db):
    """Stats should reflect indexed documents."""
    embedding = [0.1] * 768
    db.index_document("file1.md", 0, "First document", embedding)
    db.index_document("file1.md", 1, "Second chunk", embedding)
    db.index_document("file2.md", 0, "Another file", embedding)

    stats = db.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_sources"] == 2
    assert "top_sources" in stats
    assert "db_size_bytes" in stats
    assert "db_size_mb" in stats
