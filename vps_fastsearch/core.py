"""Core classes for VPS-FastSearch: Embedder, Reranker, and SearchDB."""

from pathlib import Path
from typing import Any

import apsw
import orjson
import sqlite_vec

# Lazy import fastembed to speed up CLI startup
_embedder_instance = None
_reranker_instance = None


class Embedder:
    """
    Embedding generator using FastEmbed with ONNX Runtime.
    
    Uses bge-base-en-v1.5 (768 dimensions) for fast CPU inference.
    """
    
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DIMENSIONS = 768
    
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding
        
        self.model_name = model_name or self.MODEL_NAME
        self._model = TextEmbedding(self.model_name)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # fastembed returns a generator, convert to list
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def get_embedder() -> Embedder:
    """Get or create singleton embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.
    
    Uses ms-marco-MiniLM-L-6-v2 for fast CPU inference.
    Cross-encoders are more accurate than bi-encoders for reranking
    but slower (O(n) forward passes vs O(1) for embedding comparison).
    """
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name or self.MODEL_NAME
        self._model = CrossEncoder(self.model_name)
    
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
        return scores.tolist()
    
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
        
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores


def get_reranker() -> Reranker:
    """Get or create singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


class SearchDB:
    """
    SQLite database with FTS5 (BM25) and sqlite-vec (vector) search.
    
    Provides:
    - BM25 full-text search via FTS5
    - Vector similarity search via sqlite-vec
    - Hybrid search using RRF (Reciprocal Rank Fusion)
    """
    
    def __init__(self, db_path: str | Path = "fastsearch.db"):
        self.db_path = Path(db_path)
        self.conn = apsw.Connection(str(self.db_path))
        
        # Load sqlite-vec extension
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)
        
        self._init_schema()
    
    def _execute(self, sql: str, params: tuple = ()) -> apsw.Cursor:
        """Execute SQL and return cursor."""
        return self.conn.execute(sql, params)
    
    def _init_schema(self):
        """Initialize database schema."""
        # Main docs table
        self._execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._execute("CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source)")
        
        # FTS5 virtual table
        self._execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                content,
                content='docs',
                content_rowid='id'
            )
        """)
        
        # Triggers to keep FTS in sync
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
        
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, content) VALUES('delete', old.id, old.content);
            END
        """)
        
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO docs_fts(docs_fts, rowid, content) VALUES (new.id, new.content);
            END
        """)
        
        # Vector virtual table
        self._execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)
    
    def index_document(
        self,
        source: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> int:
        """
        Index a document chunk with its embedding.
        
        Returns the document ID.
        """
        # Insert into main docs table (triggers handle FTS)
        self._execute(
            """
            INSERT INTO docs (source, chunk_index, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (source, chunk_index, content, orjson.dumps(metadata or {}).decode()),
        )
        doc_id = self.conn.last_insert_rowid()
        
        # Insert embedding into vector table using serialize_float32
        self._execute(
            """
            INSERT INTO docs_vec (id, embedding)
            VALUES (?, ?)
            """,
            (doc_id, sqlite_vec.serialize_float32(embedding)),
        )
        
        return doc_id
    
    def index_batch(
        self,
        items: list[tuple[str, int, str, list[float], dict | None]],
    ) -> list[int]:
        """
        Batch index multiple document chunks.
        
        Each item is (source, chunk_index, content, embedding, metadata).
        Returns list of document IDs.
        """
        doc_ids = []
        
        for source, chunk_index, content, embedding, metadata in items:
            self._execute(
                """
                INSERT INTO docs (source, chunk_index, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (source, chunk_index, content, orjson.dumps(metadata or {}).decode()),
            )
            doc_id = self.conn.last_insert_rowid()
            doc_ids.append(doc_id)
            
            self._execute(
                """
                INSERT INTO docs_vec (id, embedding)
                VALUES (?, ?)
                """,
                (doc_id, sqlite_vec.serialize_float32(embedding)),
            )
        
        return doc_ids
    
    def search_bm25(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Full-text search using BM25 ranking.
        
        Returns list of results with id, source, content, score, rank.
        """
        cursor = self._execute(
            """
            SELECT 
                d.id,
                d.source,
                d.chunk_index,
                d.content,
                d.metadata,
                bm25(docs_fts) as score
            FROM docs_fts f
            JOIN docs d ON f.rowid = d.id
            WHERE docs_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        )
        
        results = []
        for rank, row in enumerate(cursor, 1):
            doc_id, source, chunk_index, content, metadata, score = row
            results.append({
                "id": doc_id,
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": orjson.loads(metadata) if metadata else {},
                "score": score,
                "rank": rank,
            })
        
        return results
    
    def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Vector similarity search using cosine distance.
        
        Returns list of results with id, source, content, distance, rank.
        """
        cursor = self._execute(
            """
            SELECT 
                v.id,
                v.distance,
                d.source,
                d.chunk_index,
                d.content,
                d.metadata
            FROM docs_vec v
            JOIN docs d ON v.id = d.id
            WHERE embedding MATCH ?
                AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(embedding), limit),
        )
        
        results = []
        for rank, row in enumerate(cursor, 1):
            doc_id, distance, source, chunk_index, content, metadata = row
            results.append({
                "id": doc_id,
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": orjson.loads(metadata) if metadata else {},
                "distance": distance,
                "rank": rank,
            })
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        k: int = 60,
        bm25_weight: float = 1.0,
        vec_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining BM25 and vector search using RRF.
        
        RRF score = weight_bm25 * 1/(k + bm25_rank) + weight_vec * 1/(k + vec_rank)
        
        Returns list of results sorted by RRF score (descending).
        """
        # Get more results from each method for better fusion
        fetch_limit = limit * 3
        
        bm25_results = self.search_bm25(query, limit=fetch_limit)
        vec_results = self.search_vector(embedding, limit=fetch_limit)
        
        # Build rank maps
        bm25_ranks = {r["id"]: r["rank"] for r in bm25_results}
        vec_ranks = {r["id"]: r["rank"] for r in vec_results}
        
        # Collect all unique document IDs
        all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
        
        # Build result lookup
        result_lookup = {}
        for r in bm25_results + vec_results:
            if r["id"] not in result_lookup:
                result_lookup[r["id"]] = r
        
        # Calculate RRF scores
        rrf_results = []
        default_rank = fetch_limit + 1  # Penalty for not appearing in a list
        
        for doc_id in all_ids:
            bm25_rank = bm25_ranks.get(doc_id, default_rank)
            vec_rank = vec_ranks.get(doc_id, default_rank)
            
            rrf_score = (
                bm25_weight * (1 / (k + bm25_rank)) +
                vec_weight * (1 / (k + vec_rank))
            )
            
            result = result_lookup[doc_id].copy()
            result["rrf_score"] = rrf_score
            result["bm25_rank"] = bm25_rank if doc_id in bm25_ranks else None
            result["vec_rank"] = vec_rank if doc_id in vec_ranks else None
            rrf_results.append(result)
        
        # Sort by RRF score descending
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Assign final ranks
        for i, r in enumerate(rrf_results[:limit], 1):
            r["rank"] = i
        
        return rrf_results[:limit]
    
    def search_hybrid_reranked(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        rerank_top_k: int = 20,
        reranker: "Reranker | None" = None,
    ) -> list[dict[str, Any]]:
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
            
        Returns:
            List of results sorted by reranker score (descending).
        """
        # Get candidates from hybrid search
        candidates = self.search_hybrid(query, embedding, limit=rerank_top_k)
        
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
        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = score
        
        # Sort by reranker score (descending)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Assign final ranks and return top limit
        for i, r in enumerate(candidates[:limit], 1):
            r["rank"] = i
        
        return candidates[:limit]
    
    def delete_source(self, source: str) -> int:
        """Delete all chunks from a source. Returns count deleted."""
        # Get IDs to delete from vector table
        ids = [row[0] for row in self._execute(
            "SELECT id FROM docs WHERE source = ?", (source,)
        )]
        
        if ids:
            # Delete from vector table
            for doc_id in ids:
                self._execute("DELETE FROM docs_vec WHERE id = ?", (doc_id,))
            # Delete from docs (triggers handle FTS)
            self._execute("DELETE FROM docs WHERE source = ?", (source,))
        
        return len(ids)
    
    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        doc_count = list(self._execute("SELECT COUNT(*) FROM docs"))[0][0]
        source_count = list(self._execute("SELECT COUNT(DISTINCT source) FROM docs"))[0][0]
        
        top_sources = [
            {"source": row[0], "chunks": row[1]}
            for row in self._execute(
                "SELECT source, COUNT(*) as chunks FROM docs GROUP BY source ORDER BY chunks DESC LIMIT 10"
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
    
    def close(self):
        """Close database connection."""
        self.conn.close()
