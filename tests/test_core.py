"""Tests for vps_fastsearch.core.SearchDB — uses apsw/sqlite-vec but no ML models."""

from pathlib import Path

import pytest

from tests.conftest import DUMMY_EMBEDDING
from vps_fastsearch.core import SearchDB


def test_init_schema(db) -> None:
    """Database should have docs, docs_fts, and docs_vec tables."""
    tables = [
        row[0]
        for row in db._execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    ]
    assert "docs" in tables
    assert "docs_fts" in tables
    assert "docs_vec" in tables


def test_index_and_search_bm25(db) -> None:
    """Indexed document should be findable via BM25 search."""
    embedding = DUMMY_EMBEDDING
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("Python programming", limit=5)
    assert len(results) >= 1
    assert "Python" in results[0]["content"]


def test_index_and_search_vector(db) -> None:
    """Indexed document should be findable via vector search."""
    embedding = [1.0] * 768
    db.index_document("test.md", 0, "Vector search test document", embedding)

    results = db.search_vector([1.0] * 768, limit=5)
    assert len(results) >= 1
    assert results[0]["content"] == "Vector search test document"


def test_search_hybrid(db) -> None:
    """Hybrid search should return results combining BM25 and vector."""
    embedding = [0.5] * 768
    db.index_document("test.md", 0, "Hybrid search combines multiple methods", embedding)

    results = db.search_hybrid("hybrid search methods", [0.5] * 768, limit=5)
    assert len(results) >= 1
    assert "rrf_score" in results[0]


def test_delete_source(db) -> None:
    """Deleting a source should remove its documents."""
    embedding = DUMMY_EMBEDDING
    db.index_document("deleteme.md", 0, "This will be deleted", embedding)
    db.index_document("keepme.md", 0, "This will remain", embedding)

    deleted = db.delete_source("deleteme.md")
    assert deleted == 1

    results = db.search_bm25("deleted", limit=5)
    assert len(results) == 0

    results = db.search_bm25("remain", limit=5)
    assert len(results) == 1


def test_get_stats(db) -> None:
    """Stats should reflect indexed documents."""
    embedding = DUMMY_EMBEDDING
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


def test_search_bm25_empty_query(db) -> None:
    """BM25 search with an empty query string should return an empty list."""
    embedding = DUMMY_EMBEDDING
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("", limit=5)
    assert results == []


def test_search_bm25_special_chars_only(db) -> None:
    """BM25 search with only special characters should return an empty list."""
    embedding = DUMMY_EMBEDDING
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("@#$%^&", limit=5)
    assert results == []


def test_search_hybrid_empty_query_still_vector_searches(db) -> None:
    """Hybrid search with an empty query (no BM25 tokens) should still return
    vector results — BM25 is skipped but vector search proceeds."""
    embedding = DUMMY_EMBEDDING
    db.index_document("test.md", 0, "Some content for vector search", embedding)

    results = db.search_hybrid("", DUMMY_EMBEDDING, limit=5)
    # Vector leg should still find the document
    assert len(results) >= 1
    assert "rrf_score" in results[0]
    # BM25 rank should be absent (None) since no BM25 query was issued
    assert results[0]["bm25_rank"] is None


def test_index_document_wrong_dimensions(db) -> None:
    """Indexing a document with the wrong embedding dimension should raise ValueError."""
    bad_embedding = [0.1] * 100  # 100-dim instead of 768

    with pytest.raises(ValueError, match="768"):
        db.index_document("test.md", 0, "Some content", bad_embedding)


def test_index_batch_wrong_dimensions(db) -> None:
    """index_batch with one bad embedding should raise ValueError and leave no rows inserted."""
    good_embedding = DUMMY_EMBEDDING
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


def test_search_vector_wrong_dimensions(db) -> None:
    """Vector search with wrong embedding dimensions should raise ValueError."""
    bad_embedding = [0.1] * 100  # 100-dim instead of 768

    with pytest.raises(ValueError, match="768"):
        db.search_vector(bad_embedding, limit=5)


def test_search_bm25_limit_zero(db) -> None:
    """BM25 search with limit=0 should return an empty list."""
    embedding = DUMMY_EMBEDDING
    db.index_document("test.md", 0, "Python is a programming language", embedding)

    results = db.search_bm25("Python", limit=0)
    assert results == []


def test_index_document_transaction_rollback(db) -> None:
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


def test_index_and_search_unicode(db) -> None:
    """Index a document containing Unicode (Japanese text and emoji) and
    confirm it is retrievable via BM25 using ASCII terms in the same content."""
    embedding = DUMMY_EMBEDDING
    content = "Hello world \u3053\u3093\u306b\u3061\u306f \U0001f600 Python"
    db.index_document("unicode.md", 0, content, embedding)

    # BM25 tokenises ASCII words; "Python" and "Hello" should match
    results = db.search_bm25("Python", limit=5)
    assert len(results) >= 1
    assert "\u3053\u3093\u306b\u3061\u306f" in results[0]["content"]

    results2 = db.search_bm25("Hello", limit=5)
    assert len(results2) >= 1


# ---------------------------------------------------------------------------
# Metadata table and portable path tests (Branch 1)
# ---------------------------------------------------------------------------


def test_db_meta_table_exists(db) -> None:
    """The db_meta table should be created during schema init."""
    tables = [row[0] for row in db._execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "db_meta" in tables


def test_get_set_meta(db) -> None:
    """get_meta/set_meta should round-trip key-value pairs."""
    assert db.get_meta("base_dir") is None

    db.set_meta("base_dir", "/home/eva/docs")
    assert db.get_meta("base_dir") == "/home/eva/docs"

    # Upsert: overwrite existing key
    db.set_meta("base_dir", "/home/eva/new_docs")
    assert db.get_meta("base_dir") == "/home/eva/new_docs"


def test_base_dir_defaults_to_home(db) -> None:
    """base_dir should default to the user's home directory."""
    assert db.base_dir == Path.home()


def test_base_dir_custom(db) -> None:
    """set_base_dir should persist a custom base directory."""
    custom = db.db_path.parent / "custom_base"
    custom.mkdir()
    db.set_base_dir(custom)
    assert db.base_dir == custom.resolve()


def test_to_relative_under_base_dir(db) -> None:
    """to_relative should produce a simple relative path for files under base_dir."""
    db.set_base_dir(str(db.db_path.parent))
    base = db.db_path.parent
    abs_path = base / "subdir" / "notes.md"
    rel = db.to_relative(abs_path)
    assert rel == str(Path("subdir") / "notes.md")
    assert not Path(rel).is_absolute()


def test_to_relative_outside_base_dir(db) -> None:
    """to_relative should produce a ../.. style path for files outside base_dir."""
    db.set_base_dir(str(db.db_path.parent))
    base = db.db_path.parent
    outside = base.parent / "elsewhere" / "file.txt"
    rel = db.to_relative(outside)
    assert not Path(rel).is_absolute()
    # Resolving back should give the original path
    resolved = (base / rel).resolve()
    assert resolved == outside.resolve()


def test_to_relative_already_relative(db) -> None:
    """to_relative should return an already-relative path unchanged."""
    assert db.to_relative("some/relative/path.md") == "some/relative/path.md"


def test_to_absolute_relative_path(db) -> None:
    """to_absolute should resolve a relative path against base_dir."""
    db.set_base_dir(str(db.db_path.parent))
    base = db.db_path.parent
    result = db.to_absolute("subdir/notes.md")
    assert Path(result).is_absolute()
    assert result == str((base / "subdir" / "notes.md").resolve())


def test_to_absolute_already_absolute(db) -> None:
    """to_absolute should pass through an already-absolute path unchanged."""
    abs_path = "/usr/local/share/data.txt"
    assert db.to_absolute(abs_path) == abs_path


def test_to_relative_to_absolute_roundtrip(db) -> None:
    """to_relative followed by to_absolute should return the original path."""
    base = db.db_path.parent
    original = base / "docs" / "readme.md"
    rel = db.to_relative(original)
    restored = db.to_absolute(rel)
    assert Path(restored).resolve() == original.resolve()


def test_backward_compat_absolute_source_paths(db) -> None:
    """Databases with absolute source paths (pre-relative-path era) should
    still work: to_absolute passes absolute paths through unchanged."""
    abs_source = "/home/eva/docs/old_file.md"
    embedding = DUMMY_EMBEDDING
    db.index_document(abs_source, 0, "Legacy absolute path content", embedding)

    results = db.search_bm25("Legacy", limit=5)
    assert len(results) == 1
    assert results[0]["source"] == abs_source
    # to_absolute should return the absolute path as-is
    assert db.to_absolute(results[0]["source"]) == abs_source


def test_index_with_relative_source_and_search(db) -> None:
    """Index a document with a relative source path and verify search returns it."""
    base = db.db_path.parent
    abs_file = base / "mydir" / "test.md"
    rel_source = db.to_relative(abs_file)

    embedding = [0.2] * 768
    db.index_document(rel_source, 0, "Relative path indexed content", embedding)

    results = db.search_bm25("Relative", limit=5)
    assert len(results) == 1
    assert results[0]["source"] == rel_source
    # Resolve back to absolute
    resolved = db.to_absolute(results[0]["source"])
    assert Path(resolved).resolve() == abs_file.resolve()


def test_schema_version_updated_to_2(db) -> None:
    """Schema version should be 2 after init (db_meta table addition)."""
    row = list(db._execute("PRAGMA user_version"))
    assert row[0][0] == 2


# ---------------------------------------------------------------------------
# Delete by ID tests (Branch 4)
# ---------------------------------------------------------------------------


def test_delete_by_id(db) -> None:
    """delete_by_id should remove a single document and keep others intact."""
    emb = DUMMY_EMBEDDING
    id1 = db.index_document("a.md", 0, "First document", emb)
    id2 = db.index_document("a.md", 1, "Second document", emb)

    assert db.delete_by_id(id1) is True
    assert db.get_stats()["total_chunks"] == 1

    # Remaining doc should still be searchable via BM25
    results = db.search_bm25("Second", limit=5)
    assert len(results) == 1
    assert results[0]["id"] == id2

    # Vector table should also be cleaned up
    vec_count = list(db._execute("SELECT COUNT(*) FROM docs_vec"))[0][0]
    assert vec_count == 1


def test_delete_by_id_nonexistent(db) -> None:
    """delete_by_id on a missing ID should return False."""
    assert db.delete_by_id(9999) is False


def test_delete_by_id_fts_sync(db) -> None:
    """After delete_by_id, FTS should not find the deleted document."""
    emb = DUMMY_EMBEDDING
    doc_id = db.index_document("x.md", 0, "Unique banana content", emb)
    db.index_document("y.md", 0, "Other apple content", emb)

    db.delete_by_id(doc_id)

    results = db.search_bm25("banana", limit=5)
    assert len(results) == 0

    results = db.search_bm25("apple", limit=5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Update content tests (Branch 4)
# ---------------------------------------------------------------------------


def test_update_content(db) -> None:
    """update_content should change content and embedding."""
    emb_old = [0.1] * 768
    emb_new = [0.9] * 768
    doc_id = db.index_document("u.md", 0, "Old content here", emb_old)

    assert db.update_content(doc_id, "New content replaced", emb_new) is True

    # BM25 should find the new content
    results = db.search_bm25("replaced", limit=5)
    assert len(results) == 1
    assert results[0]["content"] == "New content replaced"

    # BM25 should NOT find the old content
    results_old = db.search_bm25("Old", limit=5)
    assert len(results_old) == 0

    # Vector search with new embedding should return the doc
    results_vec = db.search_vector(emb_new, limit=5)
    assert len(results_vec) >= 1
    assert results_vec[0]["id"] == doc_id


def test_update_content_nonexistent(db) -> None:
    """update_content on a missing ID should return False."""
    assert db.update_content(9999, "content", DUMMY_EMBEDDING) is False


def test_update_content_wrong_dimensions(db) -> None:
    """update_content with wrong embedding dims should raise ValueError."""
    emb = DUMMY_EMBEDDING
    doc_id = db.index_document("u.md", 0, "Some content", emb)

    with pytest.raises(ValueError, match="768"):
        db.update_content(doc_id, "New content", [0.1] * 100)


# ---------------------------------------------------------------------------
# List sources tests (Branch 4)
# ---------------------------------------------------------------------------


def test_list_sources_empty(db) -> None:
    """list_sources on an empty database should return an empty list."""
    assert db.list_sources() == []


def test_list_sources(db) -> None:
    """list_sources should return all sources with correct chunk counts."""
    emb = DUMMY_EMBEDDING
    db.index_document("file1.md", 0, "Chunk 0", emb)
    db.index_document("file1.md", 1, "Chunk 1", emb)
    db.index_document("file2.md", 0, "Only chunk", emb)

    sources = db.list_sources()
    assert len(sources) == 2

    # Should be ordered by source name
    assert sources[0]["source"] == "file1.md"
    assert sources[0]["chunks"] == 2
    assert sources[1]["source"] == "file2.md"
    assert sources[1]["chunks"] == 1

    # ID ranges should be present
    assert "min_id" in sources[0]
    assert "max_id" in sources[0]


# ---------------------------------------------------------------------------
# Metadata filtering tests (Branch 5)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_metadata(tmp_path):
    """Create a SearchDB with documents that have metadata."""
    database = SearchDB(tmp_path / "meta_test.db")
    embedding = DUMMY_EMBEDDING

    database.index_document(
        "doc1.md",
        0,
        "Python is a programming language",
        embedding,
        metadata={"author": "alice", "category": "tech", "priority": 1},
    )
    database.index_document(
        "doc2.md",
        0,
        "Rust is a systems programming language",
        embedding,
        metadata={"author": "bob", "category": "tech", "priority": 2},
    )
    database.index_document(
        "doc3.md",
        0,
        "Cooking recipes for beginners",
        embedding,
        metadata={"author": "alice", "category": "food", "priority": 1},
    )
    database.index_document(
        "doc4.md",
        0,
        "Advanced Python web development",
        [0.9] * 768,
        metadata={"author": "bob", "category": "tech", "featured": True},
    )

    yield database
    database.close()


def test_bm25_single_metadata_filter(db_with_metadata) -> None:
    """BM25 search with a single metadata filter should only return matching docs."""
    results = db_with_metadata.search_bm25(
        "programming",
        limit=10,
        metadata_filter={"author": "alice"},
    )
    assert len(results) == 1
    assert results[0]["metadata"]["author"] == "alice"
    assert "Python" in results[0]["content"]


def test_bm25_multiple_metadata_filters(db_with_metadata) -> None:
    """BM25 search with multiple metadata filters should AND them together."""
    results = db_with_metadata.search_bm25(
        "programming",
        limit=10,
        metadata_filter={"author": "bob", "category": "tech"},
    )
    assert len(results) == 1
    assert results[0]["metadata"]["author"] == "bob"
    assert "Rust" in results[0]["content"]


def test_bm25_metadata_filter_no_match(db_with_metadata) -> None:
    """BM25 search with a filter that matches nothing should return empty list."""
    results = db_with_metadata.search_bm25(
        "programming",
        limit=10,
        metadata_filter={"author": "charlie"},
    )
    assert results == []


def test_bm25_no_metadata_filter_backward_compat(db_with_metadata) -> None:
    """BM25 search with no metadata filter should return all matching docs."""
    results = db_with_metadata.search_bm25("programming", limit=10)
    assert len(results) >= 2  # Both Python and Rust docs match


def test_bm25_metadata_filter_numeric(db_with_metadata) -> None:
    """BM25 search with numeric metadata filter should work."""
    results = db_with_metadata.search_bm25(
        "programming",
        limit=10,
        metadata_filter={"priority": 2},
    )
    assert len(results) == 1
    assert "Rust" in results[0]["content"]


def test_bm25_metadata_filter_boolean(db_with_metadata) -> None:
    """BM25 search with boolean metadata filter should work."""
    results = db_with_metadata.search_bm25(
        "Python",
        limit=10,
        metadata_filter={"featured": True},
    )
    assert len(results) == 1
    assert "Advanced" in results[0]["content"]


def test_vector_single_metadata_filter(db_with_metadata) -> None:
    """Vector search with metadata filter should only return matching docs."""
    results = db_with_metadata.search_vector(
        DUMMY_EMBEDDING,
        limit=10,
        metadata_filter={"author": "alice"},
    )
    # Only alice's docs should be returned
    assert all(r["metadata"]["author"] == "alice" for r in results)
    assert len(results) == 2  # doc1 and doc3


def test_vector_metadata_filter_no_match(db_with_metadata) -> None:
    """Vector search with filter matching nothing should return empty list."""
    results = db_with_metadata.search_vector(
        DUMMY_EMBEDDING,
        limit=10,
        metadata_filter={"author": "nobody"},
    )
    assert results == []


def test_vector_no_metadata_filter_backward_compat(db_with_metadata) -> None:
    """Vector search without metadata filter returns all docs."""
    results = db_with_metadata.search_vector(DUMMY_EMBEDDING, limit=10)
    assert len(results) == 4


def test_hybrid_metadata_filter(db_with_metadata) -> None:
    """Hybrid search with metadata filter should only return matching docs."""
    results = db_with_metadata.search_hybrid(
        "programming",
        DUMMY_EMBEDDING,
        limit=10,
        metadata_filter={"category": "tech"},
    )
    assert all(r["metadata"]["category"] == "tech" for r in results)
    # Should not include the "food" doc
    assert not any("Cooking" in r["content"] for r in results)


def test_hybrid_no_metadata_filter_backward_compat(db_with_metadata) -> None:
    """Hybrid search without metadata filter returns all relevant docs."""
    results = db_with_metadata.search_hybrid("programming", DUMMY_EMBEDDING, limit=10)
    assert len(results) >= 2


def test_metadata_filter_empty_dict(db_with_metadata) -> None:
    """Empty metadata filter dict should behave like no filter."""
    results_no_filter = db_with_metadata.search_bm25("programming", limit=10)
    results_empty = db_with_metadata.search_bm25(
        "programming",
        limit=10,
        metadata_filter={},
    )
    assert len(results_no_filter) == len(results_empty)


def test_metadata_filter_invalid_value_type(db_with_metadata) -> None:
    """Metadata filter with unsupported value type should raise ValueError."""
    with pytest.raises(ValueError, match="must be str, int, float, or bool"):
        db_with_metadata.search_bm25(
            "programming",
            limit=10,
            metadata_filter={"author": ["alice"]},
        )
