"""Core classes for VPS-FastSearch: Embedder, Reranker, and SearchDB."""

import hashlib
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

import apsw
import orjson
import sqlite_vec

logger = logging.getLogger(__name__)

# Schema version for migration tracking (via PRAGMA user_version)
SCHEMA_VERSION = 4

# Lazy import fastembed to speed up CLI startup
_embedder_instance = None
_reranker_instance = None


def _content_hash(content: str) -> str:
    """Compute SHA-256 hex digest for content deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# TypedDict result types for search methods
# ---------------------------------------------------------------------------


class _BaseResult(TypedDict):
    """Common fields present in every search result."""

    id: int
    source: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    rank: int


class BM25Result(_BaseResult):
    """Result from :meth:`SearchDB.search_bm25`."""

    score: float


class VectorResult(_BaseResult):
    """Result from :meth:`SearchDB.search_vector`."""

    distance: float


class HybridResult(_BaseResult):
    """Result from :meth:`SearchDB.search_hybrid`."""

    rrf_score: float
    bm25_rank: int | None
    vec_rank: int | None


class RerankResult(_BaseResult):
    """Result from :meth:`SearchDB.search_hybrid_reranked`."""

    rerank_score: float


class _FastEmbedBackend:
    """FastEmbed/ONNX Runtime embedding backend (default)."""

    def __init__(self, model_name: str, threads: int = 2) -> None:
        from fastembed import TextEmbedding

        logger.info(f"Loading embedding model {model_name} (first run may download ~130MB)")
        for attempt in range(3):
            try:
                self._model = TextEmbedding(model_name, threads=threads)
                break
            except Exception as e:
                if attempt < 2 and ("connection" in str(e).lower() or "timeout" in str(e).lower()):
                    wait = 3 * (2**attempt)
                    logger.warning(
                        f"Model download attempt {attempt + 1}/3 failed, retrying in {wait}s: {e}"
                    )
                    import time

                    time.sleep(wait)
                else:
                    raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]


class _OllamaBackend:
    """Ollama embedding backend — calls /api/embeddings."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434") -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        logger.info(f"Using Ollama embedder: {model_name} at {self._base_url}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        results: list[list[float]] = []
        for text in texts:
            payload = json.dumps({"model": self._model_name, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self._base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            results.append(data["embedding"])
        return results


class _HTTPBackend:
    """OpenAI-compatible HTTP embedding backend — calls {base_url}/embeddings."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "") -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        logger.info(f"Using HTTP embedder: {model_name} at {self._base_url}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        payload = json.dumps({"model": self._model_name, "input": texts}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        # OpenAI format: data[].embedding, sorted by index
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class Embedder:
    """
    Embedding generator with pluggable backends.

    Supports three providers:
    - ``fastembed`` (default): Local ONNX Runtime via FastEmbed
    - ``ollama``: Ollama API at configurable base_url
    - ``http``: Any OpenAI-compatible /v1/embeddings endpoint
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DIMENSIONS = 768

    _instance: "Embedder | None" = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        model_name: str | None = None,
        document_prefix: str = "",
        query_prefix: str = "",
        provider: str = "fastembed",
        base_url: str = "",
        api_key: str = "",
        threads: int = 2,
    ) -> "Embedder":
        """Get or create a thread-safe singleton Embedder instance.

        Uses double-checked locking to avoid acquiring the lock on every call.
        Direct instantiation via ``Embedder()`` still works normally.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name, document_prefix, query_prefix,
                        provider, base_url, api_key, threads,
                    )
        return cls._instance

    def __init__(
        self,
        model_name: str | None = None,
        document_prefix: str = "",
        query_prefix: str = "",
        provider: str = "fastembed",
        base_url: str = "",
        api_key: str = "",
        threads: int = 2,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.provider = provider

        self._backend: _FastEmbedBackend | _OllamaBackend | _HTTPBackend
        if provider == "ollama":
            self._backend = _OllamaBackend(
                self.model_name, base_url or "http://localhost:11434"
            )
        elif provider == "http":
            if not base_url:
                raise ValueError("HTTP embedding provider requires base_url")
            self._backend = _HTTPBackend(self.model_name, base_url, api_key)
        else:
            self._backend = _FastEmbedBackend(self.model_name, threads)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (document prefix applied)."""
        if self.document_prefix:
            texts = [self.document_prefix + t for t in texts]
        return self._backend.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text (query prefix applied)."""
        prefixed = self.query_prefix + text if self.query_prefix else text
        return self._backend.embed([prefixed])[0]

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single query text (query prefix applied)."""
        return self.embed_query(text)


def get_embedder() -> Embedder:
    """Get or create singleton embedder instance (thread-safe)."""
    return Embedder.get_instance()


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Uses ms-marco-MiniLM-L-6-v2 for fast CPU inference.
    Cross-encoders are more accurate than bi-encoders for reranking
    but slower (O(n) forward passes vs O(1) for embedding comparison).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    _instance: "Reranker | None" = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "Reranker":
        """Get or create a thread-safe singleton Reranker instance.

        Uses double-checked locking to avoid acquiring the lock on every call.
        Direct instantiation via ``Reranker()`` still works normally.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Reranker requires sentence-transformers. "
                "Install with: pip install vps-fastsearch[rerank]"
            ) from None

        self.model_name = model_name or self.MODEL_NAME
        logger.info(f"Loading reranker model {self.model_name} (first run may download ~80MB)")
        for attempt in range(3):
            try:
                self._model = CrossEncoder(self.model_name)
                break
            except Exception as e:
                if attempt < 2 and ("connection" in str(e).lower() or "timeout" in str(e).lower()):
                    wait = 3 * (2**attempt)  # 3, 6, 12 seconds
                    logger.warning(
                        f"Model download attempt {attempt + 1}/3 failed, retrying in {wait}s: {e}"
                    )
                    import time

                    time.sleep(wait)
                else:
                    raise

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Score documents against query using cross-encoder.

        Returns list of relevance scores (higher = more relevant).
        """
        if not documents:
            return []

        # Cross-encoder expects pairs of (query, document)
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)

        # Convert numpy array to list of floats
        return list(scores.tolist())

    def rerank_with_indices(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Rerank documents and return sorted (index, score) pairs.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Optional limit on results

        Returns:
            List of (original_index, score) sorted by score descending
        """
        scores = self.rerank(query, documents)

        # Pair indices with scores and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


def get_reranker() -> Reranker:
    """Get or create singleton reranker instance (thread-safe)."""
    return Reranker.get_instance()


class SearchDB:
    """
    SQLite database with FTS5 (BM25) and sqlite-vec (vector) search.

    Provides:
    - BM25 full-text search via FTS5
    - Vector similarity search via sqlite-vec
    - Hybrid search using RRF (Reciprocal Rank Fusion)
    """

    EMBEDDING_DIM = 768
    MAX_SEARCH_LIMIT = 10000

    @staticmethod
    def _build_metadata_filter(
        metadata_filter: dict[str, Any] | None,
    ) -> tuple[str, tuple[Any, ...]]:
        """Build SQL WHERE clause fragment for metadata filtering.

        Args:
            metadata_filter: Dict of key-value pairs for exact match on top-level
                metadata keys. Supported value types: str, int, float, bool.

        Returns:
            Tuple of (sql_fragment, params) where sql_fragment is empty string
            if no filters, or " AND json_extract(...) = ? AND ..." otherwise.

        Raises:
            ValueError: If a filter value has an unsupported type.
        """
        if not metadata_filter:
            return "", ()

        clauses: list[str] = []
        params: list[Any] = []

        for key, value in metadata_filter.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata filter key must be a string, got {type(key).__name__}")
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", key):
                raise ValueError(f"Invalid metadata filter key: {key!r}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"Metadata filter value for '{key}' must be str, int, float, or bool, "
                    f"got {type(value).__name__}"
                )
            clauses.append(f"json_extract(d.metadata, '$.{key}') = ?")
            # SQLite stores JSON booleans as 1/0
            if isinstance(value, bool):
                params.append(1 if value else 0)
            else:
                params.append(value)

        sql = " AND " + " AND ".join(clauses)
        return sql, tuple(params)

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from .config import DEFAULT_DB_PATH

            db_path = os.environ.get("FASTSEARCH_DB", DEFAULT_DB_PATH)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = apsw.Connection(str(self.db_path))

        # Load sqlite-vec extension
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        # Enable WAL mode for concurrent read/write access
        result = list(self._execute("PRAGMA journal_mode=WAL"))
        if result and result[0][0].lower() != "wal":
            logger.warning(
                f"WAL mode not available (got {result[0][0]}). Performance may be degraded on network/FUSE filesystems."
            )
        # Wait up to 5 seconds if database is locked
        self._execute("PRAGMA busy_timeout=5000")
        self._execute("PRAGMA cache_size = -4000")  # 4MB cache
        self._execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
        self._execute("PRAGMA wal_autocheckpoint = 1000")  # Checkpoint every 1000 pages

        # Lightweight corruption check
        try:
            list(self._execute("PRAGMA quick_check(1)"))
        except Exception as e:
            logger.error(
                f"Database may be corrupted: {e}. Consider deleting {self.db_path} and re-indexing."
            )
            raise

        self._init_schema()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> apsw.Cursor:
        """Execute SQL and return cursor."""
        return self.conn.execute(sql, params)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Context manager for SQLite transactions with automatic rollback on error."""
        self.conn.execute("BEGIN")
        try:
            yield
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass  # ROLLBACK can fail if disk is full
            raise

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Main chunks table (one row per text chunk)
        self._execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")

        self._execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_source_chunk"
            " ON chunks(source, chunk_index)"
        )

        # FTS5 virtual table for BM25 search
        self._execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS in sync
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
            END
        """)

        self._execute("DROP TRIGGER IF EXISTS chunks_au")
        self._execute("""
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        # Vector virtual table for embedding search
        self._execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float32[768]
            )
        """)

        # Key-value metadata table for DB-level settings (e.g. base_dir)
        self._execute("""
            CREATE TABLE IF NOT EXISTS db_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        self._check_schema_version()
        self._check_embedding_dims()

    def _check_schema_version(self) -> None:
        """Check and update database schema version using PRAGMA user_version."""
        row = list(self._execute("PRAGMA user_version"))
        current_version = row[0][0]

        if current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than "
                f"supported version {SCHEMA_VERSION}. "
                f"Please upgrade vps-fastsearch."
            )

        if current_version < SCHEMA_VERSION:
            if current_version > 0 and current_version <= 3:
                # v1-v3 used 'docs' table names; v4 renames to 'chunks'.
                # No auto-migration — user must delete and re-index.
                raise RuntimeError(
                    f"Database uses old schema v{current_version} (table 'docs'). "
                    f"VPS-FastSearch v0.3.2+ uses 'chunks' tables. "
                    f"Delete {self.db_path} and re-index: "
                    f"vps-fastsearch index <path> or vps-fastsearch update"
                )
            # New database (version 0) — set content_hash column and version
            cols = [r[1] for r in self._execute("PRAGMA table_info(chunks)")]
            if "content_hash" not in cols:
                self._execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
            self._execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)"
            )
            logger.info(f"Setting database schema version to {SCHEMA_VERSION}")
            self._execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _check_embedding_dims(self) -> None:
        """Verify stored embedding dimensions match EMBEDDING_DIM.

        On first use, stores the current EMBEDDING_DIM in db_meta.
        On subsequent opens, raises RuntimeError if there is a mismatch
        (e.g. user switched models without reindexing).
        """
        row = list(self._execute("SELECT value FROM db_meta WHERE key = 'embedding_dims'"))
        if row:
            stored_dims = int(row[0][0])
            if stored_dims != self.EMBEDDING_DIM:
                raise RuntimeError(
                    f"Embedding dimension mismatch: index has {stored_dims}-dim vectors "
                    f"but current model produces {self.EMBEDDING_DIM}-dim. "
                    f"Run: vps-fastsearch index --reindex to rebuild."
                )
        else:
            # First time — record current dims
            self._execute(
                "INSERT OR IGNORE INTO db_meta (key, value) VALUES ('embedding_dims', ?)",
                (str(self.EMBEDDING_DIM),),
            )

    def find_sources(self, pattern: str) -> list[str]:
        """Find sources matching a partial name pattern (LIKE query).

        LIKE wildcards (``%`` and ``_``) in *pattern* are escaped so they
        are matched literally.

        Args:
            pattern: Substring to search for in source names.

        Returns:
            List of distinct source strings that contain *pattern*.
        """
        escaped = pattern.replace("%", r"\%").replace("_", r"\_")
        cursor = self._execute(
            "SELECT DISTINCT source FROM chunks WHERE source LIKE ? ESCAPE '\\'",
            (f"%{escaped}%",),
        )
        return [row[0] for row in cursor]

    def index_document(
        self,
        source: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        skip_duplicates: bool = False,
    ) -> int:
        """
        Index a document chunk with its embedding.

        Args:
            skip_duplicates: When True, skip insertion if a row with the same
                content_hash already exists.  Returns -1 for skipped items.

        Returns the document ID, or -1 if skipped as a duplicate.
        """
        if chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
        if len(embedding) != self.EMBEDDING_DIM:
            raise ValueError(f"Expected {self.EMBEDDING_DIM}-dim embedding, got {len(embedding)}")

        content_hash_val = _content_hash(content)

        if skip_duplicates:
            existing = list(
                self._execute(
                    "SELECT id FROM chunks WHERE content_hash = ? LIMIT 1",
                    (content_hash_val,),
                )
            )
            if existing:
                return -1

        self.conn.execute("BEGIN")
        try:
            # Insert into chunks table (triggers handle FTS)
            self._execute(
                """
                INSERT INTO chunks (source, chunk_index, content, metadata, content_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source,
                    chunk_index,
                    content,
                    orjson.dumps(metadata or {}).decode(),
                    content_hash_val,
                ),
            )
            doc_id = self.conn.last_insert_rowid()

            # Insert embedding into vector table
            self._execute(
                """
                INSERT INTO chunks_vec (id, embedding)
                VALUES (?, ?)
                """,
                (doc_id, sqlite_vec.serialize_float32(embedding)),
            )

            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass  # ROLLBACK can fail if disk is full
            raise

        return doc_id

    def index_batch(
        self,
        items: list[tuple[str, int, str, list[float], dict[str, Any] | None]],
        skip_duplicates: bool = False,
    ) -> list[int]:
        """
        Batch index multiple document chunks.

        Each item is (source, chunk_index, content, embedding, metadata).

        Args:
            skip_duplicates: When True, skip items whose content_hash already
                exists.  Skipped items get -1 in the returned list.

        Returns list of document IDs (-1 for skipped duplicates).
        """
        for i, (_, chunk_index, _, embedding, _) in enumerate(items):
            if chunk_index < 0:
                raise ValueError(f"Item {i}: chunk_index must be non-negative, got {chunk_index}")
            if len(embedding) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Item {i}: expected {self.EMBEDDING_DIM}-dim embedding, got {len(embedding)}"
                )

        # Pre-compute hashes
        hashes = [_content_hash(content) for _, _, content, _, _ in items]

        # Collect existing hashes if dedup is requested
        existing_hashes: set[str] = set()
        if skip_duplicates:
            for h in hashes:
                rows = list(
                    self._execute("SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1", (h,))
                )
                if rows:
                    existing_hashes.add(h)

        doc_ids: list[int] = []

        self.conn.execute("BEGIN")
        try:
            for (source, chunk_index, content, embedding, metadata), h in zip(
                items, hashes, strict=True
            ):
                if skip_duplicates and h in existing_hashes:
                    doc_ids.append(-1)
                    continue

                # Use INSERT OR REPLACE for idempotent retries — if the same
                # (source, chunk_index) pair already exists (e.g. from a retried
                # request after timeout), it will be overwritten safely.
                # First check if a row already exists so we can clean up chunks_vec.
                existing = list(
                    self._execute(
                        "SELECT id FROM chunks WHERE source = ? AND chunk_index = ?",
                        (source, chunk_index),
                    )
                )
                if existing:
                    old_id = existing[0][0]
                    self._execute("DELETE FROM chunks_vec WHERE id = ?", (old_id,))

                self._execute(
                    """
                    INSERT OR REPLACE INTO chunks (source, chunk_index, content, metadata, content_hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        chunk_index,
                        content,
                        orjson.dumps(metadata or {}).decode(),
                        h,
                    ),
                )
                doc_id = self.conn.last_insert_rowid()
                doc_ids.append(doc_id)

                self._execute(
                    """
                    INSERT INTO chunks_vec (id, embedding)
                    VALUES (?, ?)
                    """,
                    (doc_id, sqlite_vec.serialize_float32(embedding)),
                )

            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass  # ROLLBACK can fail if disk is full
            raise

        return doc_ids

    def search_bm25(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[BM25Result]:
        """
        Full-text search using BM25 ranking.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            metadata_filter: Optional dict of key-value pairs for exact match
                on top-level metadata keys. All conditions are ANDed together.

        Returns list of results with id, source, content, score, rank.
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        if limit == 0:
            return []
        limit = min(limit, self.MAX_SEARCH_LIMIT)

        # Sanitize query for FTS5: extract word tokens and join with OR
        # to avoid crashes on special characters like hyphens (column operators).
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        meta_sql, meta_params = self._build_metadata_filter(metadata_filter)

        cursor = self._execute(
            f"""
            SELECT
                d.id,
                d.source,
                d.chunk_index,
                d.content,
                d.metadata,
                -bm25(chunks_fts) as score
            FROM chunks_fts f
            JOIN chunks d ON f.rowid = d.id
            WHERE chunks_fts MATCH ?{meta_sql}
            ORDER BY score DESC
            LIMIT ?
            """,
            (fts_query, *meta_params, limit),
        )

        results: list[BM25Result] = []
        for rank, row in enumerate(cursor, 1):
            doc_id, source, chunk_index, content, metadata, score = row
            results.append(
                {
                    "id": doc_id,
                    "source": source,
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": orjson.loads(metadata) if metadata else {},
                    "score": score,
                    "rank": rank,
                }
            )

        return results

    def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        """
        Vector similarity search using cosine distance.

        Args:
            embedding: Query embedding vector (768 dimensions).
            limit: Maximum number of results.
            metadata_filter: Optional dict of key-value pairs for exact match
                on top-level metadata keys. All conditions are ANDed together.
                Note: When filtering, more candidates are fetched from the vector
                index and then post-filtered, so results may be fewer than limit
                if few documents match the filter.

        Returns list of results with id, source, content, distance, rank.
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        if limit == 0:
            return []
        limit = min(limit, self.MAX_SEARCH_LIMIT)

        if len(embedding) != self.EMBEDDING_DIM:
            raise ValueError(f"Expected {self.EMBEDDING_DIM}-dim embedding, got {len(embedding)}")

        meta_sql, meta_params = self._build_metadata_filter(metadata_filter)

        # sqlite-vec virtual tables don't support additional WHERE clauses,
        # so when filtering we over-fetch and post-filter via a subquery.
        if meta_sql:
            fetch_limit = min(limit * 5, self.MAX_SEARCH_LIMIT)
            cursor = self._execute(
                f"""
                SELECT sub.id, sub.distance, d.source, d.chunk_index, d.content, d.metadata
                FROM (
                    SELECT v.id, v.distance
                    FROM chunks_vec v
                    WHERE embedding MATCH ?
                        AND k = ?
                    ORDER BY distance
                ) sub
                JOIN chunks d ON sub.id = d.id
                WHERE 1=1{meta_sql}
                ORDER BY sub.distance
                LIMIT ?
                """,
                (sqlite_vec.serialize_float32(embedding), fetch_limit, *meta_params, limit),
            )
        else:
            cursor = self._execute(
                """
                SELECT
                    v.id,
                    v.distance,
                    d.source,
                    d.chunk_index,
                    d.content,
                    d.metadata
                FROM chunks_vec v
                JOIN chunks d ON v.id = d.id
                WHERE embedding MATCH ?
                    AND k = ?
                ORDER BY distance
                """,
                (sqlite_vec.serialize_float32(embedding), limit),
            )

        results: list[VectorResult] = []
        for rank, row in enumerate(cursor, 1):
            doc_id, distance, source, chunk_index, content, metadata = row
            results.append(
                {
                    "id": doc_id,
                    "source": source,
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": orjson.loads(metadata) if metadata else {},
                    "distance": distance,
                    "rank": rank,
                }
            )

        return results

    def search_hybrid(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        k: int = 60,
        bm25_weight: float = 1.0,
        vec_weight: float = 1.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """
        Hybrid search combining BM25 and vector search using RRF.

        RRF score = weight_bm25 * 1/(k + bm25_rank) + weight_vec * 1/(k + vec_rank)

        Args:
            query: Search query text.
            embedding: Query embedding vector (768 dimensions).
            limit: Maximum number of results.
            k: RRF parameter (default 60).
            bm25_weight: Weight for BM25 scores in RRF fusion.
            vec_weight: Weight for vector scores in RRF fusion.
            metadata_filter: Optional dict of key-value pairs for exact match
                on top-level metadata keys. All conditions are ANDed together.

        Returns list of results sorted by RRF score (descending).
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        if limit == 0:
            return []
        limit = min(limit, self.MAX_SEARCH_LIMIT)

        if k < 1:
            raise ValueError(f"RRF k parameter must be >= 1, got {k}")

        # Get more results from each method for better fusion
        fetch_limit = limit * 3

        # Skip BM25 if query has no word tokens (symbols, emoji, etc.)
        tokens = re.findall(r"\w+", query)
        bm25_results: list[dict[str, Any]] = []
        if tokens:
            bm25_results = self.search_bm25(  # type: ignore[assignment]
                query, limit=fetch_limit, metadata_filter=metadata_filter
            )
        vec_results: list[dict[str, Any]] = self.search_vector(  # type: ignore[assignment]
            embedding, limit=fetch_limit, metadata_filter=metadata_filter
        )

        # Build rank maps
        bm25_ranks = {r["id"]: r["rank"] for r in bm25_results}
        vec_ranks = {r["id"]: r["rank"] for r in vec_results}

        # Collect all unique document IDs
        all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())

        # Build result lookup
        result_lookup: dict[int, dict[str, Any]] = {}
        for r in bm25_results + vec_results:
            if r["id"] not in result_lookup:
                result_lookup[r["id"]] = r

        # Calculate RRF scores
        rrf_results: list[dict[str, Any]] = []
        default_rank = fetch_limit + 1  # Penalty for not appearing in a list

        for doc_id in all_ids:
            bm25_rank = bm25_ranks.get(doc_id, default_rank)
            vec_rank = vec_ranks.get(doc_id, default_rank)

            rrf_score = bm25_weight * (1 / (k + bm25_rank)) + vec_weight * (1 / (k + vec_rank))

            result = result_lookup[doc_id].copy()
            result["rrf_score"] = rrf_score
            result["bm25_rank"] = bm25_rank if doc_id in bm25_ranks else None
            result["vec_rank"] = vec_rank if doc_id in vec_ranks else None
            rrf_results.append(result)

        # Sort by RRF score descending
        rrf_results.sort(key=lambda x: (-x["rrf_score"], x["id"]))

        # Assign final ranks
        for i, r in enumerate(rrf_results[:limit], 1):
            r["rank"] = i

        return rrf_results[:limit]  # type: ignore[return-value]

    def search_hybrid_reranked(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        rerank_top_k: int = 20,
        reranker: Any = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RerankResult]:
        """
        Hybrid search with cross-encoder reranking.

        1. Get top rerank_top_k candidates from hybrid search (fast)
        2. Rerank candidates with cross-encoder (accurate)
        3. Return top limit results

        Args:
            query: Search query text
            embedding: Query embedding vector
            limit: Final number of results to return
            rerank_top_k: Number of candidates to fetch for reranking
            reranker: Optional Reranker instance (uses singleton if None)
            metadata_filter: Optional key=value metadata filter dict

        Returns:
            List of results sorted by reranker score (descending).
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        if limit == 0:
            return []
        limit = min(limit, self.MAX_SEARCH_LIMIT)

        # Get candidates from hybrid search
        candidates: list[dict[str, Any]] = self.search_hybrid(  # type: ignore[assignment]
            query, embedding, limit=rerank_top_k, metadata_filter=metadata_filter
        )

        if not candidates:
            return []

        # Get or create reranker
        if reranker is None:
            reranker = get_reranker()

        # Extract document contents for reranking
        doc_contents = [c["content"] for c in candidates]

        # Rerank with cross-encoder
        rerank_scores = reranker.rerank(query, doc_contents)

        # Attach scores to candidates
        for candidate, score in zip(candidates, rerank_scores, strict=True):
            candidate["rerank_score"] = score

        # Sort by reranker score (descending)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Assign final ranks and return top limit
        for i, r in enumerate(candidates[:limit], 1):
            r["rank"] = i

        # Remove stale RRF fields that are no longer meaningful after reranking
        for result in candidates[:limit]:
            result.pop("rrf_score", None)
            result.pop("bm25_rank", None)
            result.pop("vec_rank", None)
            result.pop("score", None)

        return candidates[:limit]  # type: ignore[return-value]

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a source. Returns count deleted."""
        count = list(self._execute("SELECT COUNT(*) FROM chunks WHERE source = ?", (source,)))[0][0]

        if count:
            self.conn.execute("BEGIN")
            try:
                # Delete from vector table via subquery (must precede chunks DELETE)
                self._execute(
                    "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE source = ?)",
                    (source,),
                )
                # Delete from chunks (triggers handle FTS)
                self._execute("DELETE FROM chunks WHERE source = ?", (source,))
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass  # ROLLBACK can fail if disk is full
                raise

        return int(count)

    def delete_by_id(self, doc_id: int) -> bool:
        """Delete a single document by ID. Returns True if a row was deleted."""
        self.conn.execute("BEGIN")
        try:
            # Delete from vector table first (no trigger for this)
            self._execute("DELETE FROM chunks_vec WHERE id = ?", (doc_id,))
            # Delete from chunks (triggers handle FTS sync)
            self._execute("DELETE FROM chunks WHERE id = ?", (doc_id,))
            affected = self.conn.changes()
            if affected == 0:
                self.conn.execute("ROLLBACK")
                return False
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        return True

    def update_content(self, doc_id: int, content: str, embedding: list[float]) -> bool:
        """Update content and embedding for a document. Returns True if updated."""
        if len(embedding) != self.EMBEDDING_DIM:
            raise ValueError(f"Expected {self.EMBEDDING_DIM}-dim embedding, got {len(embedding)}")

        self.conn.execute("BEGIN")
        try:
            # Update chunks table (trigger handles FTS delete-old + insert-new)
            self._execute(
                "UPDATE chunks SET content = ?, content_hash = ? WHERE id = ?",
                (content, _content_hash(content), doc_id),
            )
            affected = self.conn.changes()
            if affected == 0:
                self.conn.execute("ROLLBACK")
                return False
            # Update vector table — sqlite-vec vec0 doesn't support UPDATE, so
            # delete + re-insert.  Use positional VALUES (not named columns)
            # to work around a vec0 quirk after DELETE in the same transaction.
            self._execute("DELETE FROM chunks_vec WHERE id = ?", (doc_id,))
            self._execute(
                "INSERT INTO chunks_vec VALUES (?, ?)",
                (doc_id, sqlite_vec.serialize_float32(embedding)),
            )
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        return True

    def list_sources(self) -> list[dict[str, Any]]:
        """List all unique sources with chunk counts and ID ranges."""
        cursor = self._execute(
            "SELECT source, COUNT(*) as chunks, MIN(id) as min_id, MAX(id) as max_id "
            "FROM chunks GROUP BY source ORDER BY source"
        )
        return [
            {
                "source": row[0],
                "chunks": row[1],
                "min_id": row[2],
                "max_id": row[3],
            }
            for row in cursor
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        doc_count = list(self._execute("SELECT COUNT(*) FROM chunks"))[0][0]
        source_count = list(self._execute("SELECT COUNT(DISTINCT source) FROM chunks"))[0][0]

        top_sources = [
            {"source": row[0], "chunks": row[1]}
            for row in self._execute(
                "SELECT source, COUNT(*) as chunk_count FROM chunks GROUP BY source ORDER BY chunk_count DESC LIMIT 10"
            )
        ]

        # Database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_chunks": doc_count,
            "total_sources": source_count,
            "top_sources": top_sources,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Get a value from the db_meta table, or None if not set."""
        rows = list(self._execute("SELECT value FROM db_meta WHERE key = ?", (key,)))
        if rows:
            result: str = rows[0][0]
            return result
        return None

    def set_meta(self, key: str, value: str) -> None:
        """Set a key-value pair in the db_meta table (upsert)."""
        self._execute(
            "INSERT INTO db_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------
    # Portable path helpers
    # ------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        """Return the base directory for resolving relative source paths.

        Defaults to the user's home directory (``$HOME``) for portability
        across users and machines.  Can be overridden by storing a
        ``base_dir`` value in the ``db_meta`` table.
        """
        stored = self.get_meta("base_dir")
        if stored:
            return Path(stored)
        return Path.home()

    def set_base_dir(self, directory: str | Path) -> None:
        """Persist a custom base directory for relative path resolution."""
        self.set_meta("base_dir", str(Path(directory).resolve()))

    def is_within_base_dir(self, path: str | Path) -> bool:
        """Check whether *path* (resolved) is within :attr:`base_dir`."""
        return Path(path).resolve().is_relative_to(self.base_dir.resolve())

    def to_relative(self, abs_path: str | Path) -> str:
        """Convert an absolute path to a path relative to *base_dir*.

        If the path is already relative or cannot be made relative to
        *base_dir*, it is returned unchanged.
        """
        p = Path(abs_path)
        if not p.is_absolute():
            return str(abs_path)
        try:
            return str(p.relative_to(self.base_dir))
        except ValueError:
            # abs_path is not under base_dir — try os.path.relpath which
            # always succeeds and produces ``../../`` style paths.
            return os.path.relpath(str(p), str(self.base_dir))

    def to_absolute(self, rel_path: str | Path) -> str:
        """Resolve a stored (possibly relative) path to an absolute path.

        Absolute paths pass through unchanged for backward compatibility
        with databases that already contain absolute source paths.
        """
        p = Path(rel_path)
        if p.is_absolute():
            return str(p)
        return str((self.base_dir / p).resolve())

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
