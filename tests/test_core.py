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


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_search_bm25_empty_query(db):
    """BM25 search with an empty query string should return an empty list."""
    embedding = [0.1] * 768
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("", limit=5)
    assert results == []


def test_search_bm25_special_chars_only(db):
    """BM25 search with only special characters should return an empty list."""
    embedding = [0.1] * 768
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("@#$%^&", limit=5)
    assert results == []


def test_search_hybrid_empty_query_still_vector_searches(db):
    """Hybrid search with an empty query (no BM25 tokens) should still return
    vector results — BM25 is skipped but vector search proceeds."""
    embedding = [0.1] * 768
    db.index_document("test.md", 0, "Some content for vector search", embedding)

    results = db.search_hybrid("", [0.1] * 768, limit=5)
    # Vector leg should still find the document
    assert len(results) >= 1
    assert "rrf_score" in results[0]
    # BM25 rank should be absent (None) since no BM25 query was issued
    assert results[0]["bm25_rank"] is None


def test_index_document_wrong_dimensions(db):
    """Indexing a document with the wrong embedding dimension should raise ValueError."""
    bad_embedding = [0.1] * 100  # 100-dim instead of 768

    with pytest.raises(ValueError, match="768"):
        db.index_document("test.md", 0, "Some content", bad_embedding)


def test_index_batch_wrong_dimensions(db):
    """index_batch with one bad embedding should raise ValueError and leave no rows inserted."""
    good_embedding = [0.1] * 768
    bad_embedding = [0.1] * 100

    items = [
        ("file.md", 0, "Good chunk", good_embedding, None),
        ("file.md", 1, "Bad chunk", bad_embedding, None),
    ]

    with pytest.raises(ValueError, match="768"):
        db.index_batch(items)

    # Validation happens before any DB writes, so no rows should exist
    stats = db.get_stats()
    assert stats["total_chunks"] == 0


def test_search_vector_wrong_dimensions(db):
    """Vector search with wrong embedding dimensions should raise ValueError."""
    bad_embedding = [0.1] * 100  # 100-dim instead of 768

    with pytest.raises(ValueError, match="768"):
        db.search_vector(bad_embedding, limit=5)


def test_search_bm25_limit_zero(db):
    """BM25 search with limit=0 should return an empty list."""
    embedding = [0.1] * 768
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("Python", limit=0)
    assert results == []


def test_index_document_transaction_rollback(db):
    """After a failed index_document call (wrong dimensions), the database
    should have no orphaned rows in docs, docs_fts, or docs_vec."""
    bad_embedding = [0.1] * 100

    with pytest.raises(ValueError):
        db.index_document("test.md", 0, "Orphan content", bad_embedding)

    # docs table must be empty
    count = list(db._execute("SELECT COUNT(*) FROM docs"))[0][0]
    assert count == 0

    # docs_vec table must be empty
    vec_count = list(db._execute("SELECT COUNT(*) FROM docs_vec"))[0][0]
    assert vec_count == 0

    # FTS index must be empty (content table approach — query via docs_fts)
    fts_results = db.search_bm25("Orphan", limit=5)
    assert fts_results == []


def test_index_and_search_unicode(db):
    """Index a document containing Unicode (Japanese text and emoji) and
    confirm it is retrievable via BM25 using ASCII terms in the same content."""
    embedding = [0.1] * 768
    content = "Hello world \u3053\u3093\u306b\u3061\u306f \U0001f600 Python"
    db.index_document("unicode.md", 0, content, embedding)

    # BM25 tokenises ASCII words; "Python" and "Hello" should match
    results = db.search_bm25("Python", limit=5)
    assert len(results) >= 1
    assert "\u3053\u3093\u306b\u3061\u306f" in results[0]["content"]

    results2 = db.search_bm25("Hello", limit=5)
    assert len(results2) >= 1
