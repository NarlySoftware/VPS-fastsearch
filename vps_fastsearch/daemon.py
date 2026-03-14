"""VPS-FastSearch daemon with Unix socket server and model management."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import logging.handlers
import os
import signal
import socket
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import SearchDB

import orjson
import psutil

from .config import DEFAULT_DB_PATH, FastSearchConfig, load_config

# Configure logging
logger = logging.getLogger("vps_fastsearch.daemon")


@dataclass
class LoadedModel:
    """Tracks a loaded model and its metadata."""

    slot: str
    instance: Any
    loaded_at: float
    last_used: float
    memory_mb: float = 0.0
    actual_memory_mb: float = 0.0
    ref_count: int = 0

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()


class _RerankerAdapter:
    """Lightweight adapter that wraps a raw CrossEncoder instance to match the
    ``Reranker.rerank(query, documents)`` interface expected by
    ``SearchDB.search_hybrid_reranked``.  This avoids creating a second
    ``Reranker`` singleton (~90 MB) when the daemon already has the model
    loaded via ``ModelManager``.
    """

    def __init__(self, cross_encoder: Any) -> None:
        self._model = cross_encoder

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return list(scores.tolist())


class ModelManager:
    """
    Manages model lifecycle with memory budgets and LRU eviction.

    Model slots:
    - embedder: BAAI/bge-base-en-v1.5 (always loaded by default)
    - reranker: cross-encoder/ms-marco-MiniLM (on-demand)
    - summarizer: (future) 7B models
    """

    def __init__(self, config: FastSearchConfig) -> None:
        self.config = config
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._load_lock = asyncio.Lock()
        self._unload_tasks: dict[str, asyncio.Task[None]] = {}

    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return float(process.memory_info().rss) / (1024 * 1024)

    def estimate_model_memory(self, slot: str) -> float:
        """Estimate memory for a model slot (MB)."""
        estimates = {
            "embedder": 450,  # bge-base-en-v1.5
            "reranker": 90,  # ms-marco-MiniLM-L-6-v2
            "summarizer": 4000,  # 7B model (future)
        }
        return estimates.get(slot, 500)

    def _load_model_sync(self, slot: str) -> tuple[Any, float]:
        """Synchronously load a model (run in executor).

        Returns (instance, actual_memory_mb) where actual_memory_mb is the
        measured RSS delta during load, or 0.0 if measurement failed.
        """
        model_config = self.config.models.get(slot)
        if not model_config:
            raise ValueError(f"Unknown model slot: {slot}")

        logger.info(f"Loading model: {slot} ({model_config.name})")
        start = time.perf_counter()

        # Measure RSS before load
        try:
            rss_before = psutil.Process().memory_info().rss
        except Exception:
            rss_before = None

        instance: Any
        if slot == "embedder":
            from .core import Embedder

            # Use Embedder class which supports fastembed, ollama, and http providers
            instance = Embedder(
                model_name=model_config.name,
                document_prefix=model_config.document_prefix,
                query_prefix=model_config.query_prefix,
                provider=model_config.provider,
                base_url=model_config.base_url,
                api_key=model_config.api_key,
                threads=model_config.threads,
            )
        elif slot == "reranker":
            from sentence_transformers import CrossEncoder

            instance = CrossEncoder(model_config.name)
        else:
            raise ValueError(f"Unknown model slot: {slot}")

        # Measure RSS after load
        actual_memory_mb = 0.0
        if rss_before is not None:
            try:
                rss_after = psutil.Process().memory_info().rss
                delta_mb = (rss_after - rss_before) / (1024 * 1024)
                # Only trust the measurement if it's meaningful (>10MB)
                if delta_mb >= 10:
                    actual_memory_mb = delta_mb
            except Exception:
                pass

        elapsed = time.perf_counter() - start
        logger.info(f"Loaded {slot} in {elapsed:.2f}s")

        return instance, actual_memory_mb

    async def load_model(self, slot: str) -> LoadedModel:
        """Load a model into memory."""
        async with self._load_lock:
            # Cancel any pending unload before checking if loaded
            if slot in self._unload_tasks:
                self._unload_tasks[slot].cancel()
                del self._unload_tasks[slot]

            # Already loaded?
            if slot in self._models:
                model = self._models[slot]
                model.touch()
                model.ref_count += 1
                # Move to end for LRU
                self._models.move_to_end(slot)
                return model

            # Check memory budget
            await self._ensure_memory_budget(slot)

            # Load model in thread pool
            loop = asyncio.get_running_loop()
            instance, actual_memory_mb = await loop.run_in_executor(
                None, self._load_model_sync, slot
            )

            # Create tracking entry — prefer measured memory over estimate
            now = time.time()
            estimated_mb = self.estimate_model_memory(slot)
            memory_mb = actual_memory_mb if actual_memory_mb > 0 else estimated_mb

            model = LoadedModel(
                slot=slot,
                instance=instance,
                loaded_at=now,
                last_used=now,
                memory_mb=memory_mb,
                actual_memory_mb=actual_memory_mb,
                ref_count=1,
            )

            self._models[slot] = model
            logger.info(f"Model {slot} loaded. Memory: {self.get_memory_usage():.0f}MB")

            # Schedule unload if on-demand
            self._schedule_unload(slot)

            return model

    async def release_model(self, slot: str) -> None:
        """Release a reference to a loaded model."""
        async with self._load_lock:
            if slot in self._models:
                model = self._models[slot]
                model.ref_count = max(0, model.ref_count - 1)

    async def _ensure_memory_budget(self, slot: str) -> None:
        """Evict models if needed to fit new model."""
        needed_mb = self.estimate_model_memory(slot)
        max_ram = self.config.memory.max_ram_mb

        current_usage = self.get_memory_usage()

        # Keep evicting until we have room
        while current_usage + needed_mb > max_ram and self._models:
            # Find eviction candidate (LRU, excluding "always" models)
            evict_slot = None

            for s in self._models:
                model_config = self.config.models.get(s)
                if model_config and model_config.keep_loaded != "always":
                    evict_slot = s
                    break

            if evict_slot is None:
                logger.warning("Cannot evict: all models are 'always' loaded")
                break

            await self._unload_model_unlocked(evict_slot)
            # If model wasn't actually unloaded (ref_count > 0), stop trying
            if evict_slot in self._models:
                logger.warning(f"Cannot evict {evict_slot}: still has active references")
                break
            current_usage = self.get_memory_usage()

    async def unload_model(self, slot: str) -> None:
        """Unload a model from memory (acquires _load_lock)."""
        async with self._load_lock:
            await self._unload_model_unlocked(slot)

    async def _unload_model_unlocked(self, slot: str) -> None:
        """Unload a model from memory. Caller must hold _load_lock."""
        if slot not in self._models:
            return

        model = self._models[slot]

        # Don't unload models with active references
        if model.ref_count > 0:
            logger.info(f"Skipping unload of {slot}: {model.ref_count} active refs")
            return

        # Don't unload "always" models
        model_config = self.config.models.get(slot)
        if model_config and model_config.keep_loaded == "always":
            logger.warning(f"Cannot unload 'always' model: {slot}")
            return

        logger.info(f"Unloading model: {slot}")

        # Remove from tracking
        del self._models[slot]

        # Delete instance to free memory
        del model.instance

        # Force garbage collection
        import gc

        gc.collect()

        logger.info(f"Model {slot} unloaded. Memory: {self.get_memory_usage():.0f}MB")

    def _schedule_unload(self, slot: str) -> None:
        """Schedule auto-unload for on-demand models."""
        model_config = self.config.models.get(slot)
        if not model_config:
            return

        if model_config.keep_loaded != "on_demand":
            return

        timeout = model_config.idle_timeout_seconds
        if timeout <= 0:
            return

        async def _delayed_unload() -> None:
            await asyncio.sleep(timeout)
            if slot in self._models:
                model = self._models[slot]
                idle_time = time.time() - model.last_used
                if idle_time >= timeout:
                    await self.unload_model(slot)

        # Cancel existing task
        if slot in self._unload_tasks:
            self._unload_tasks[slot].cancel()

        self._unload_tasks[slot] = asyncio.create_task(_delayed_unload())

    def get_model(self, slot: str) -> LoadedModel | None:
        """Get a loaded model without loading it."""
        return self._models.get(slot)

    def is_loaded(self, slot: str) -> bool:
        """Check if a model is loaded."""
        return slot in self._models

    def get_status(self) -> dict[str, Any]:
        """Get current model status."""
        return {
            "loaded_models": {
                slot: {
                    "loaded_at": model.loaded_at,
                    "last_used": model.last_used,
                    "memory_mb": model.memory_mb,
                    "actual_memory_mb": model.actual_memory_mb,
                    "idle_seconds": time.time() - model.last_used,
                }
                for slot, model in self._models.items()
            },
            "total_memory_mb": self.get_memory_usage(),
            "max_memory_mb": self.config.memory.max_ram_mb,
        }

    async def shutdown(self) -> None:
        """Shutdown and unload all models."""
        # Cancel all unload tasks
        for task in self._unload_tasks.values():
            task.cancel()
        self._unload_tasks.clear()

        # Unload all models (even "always" ones)
        slots = list(self._models.keys())
        for slot in slots:
            if slot in self._models:
                model = self._models.pop(slot)
                del model.instance

        import gc

        gc.collect()


class RateLimiter:
    """Per-connection sliding window rate limiter."""

    def __init__(self, max_requests: int = 20, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def check(self) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.monotonic()
        while self._timestamps and self._timestamps[0] < now - self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True


class FastSearchDaemon:
    """
    Unix socket server for FastSearch operations.

    JSON-RPC 2.0 protocol for requests/responses.
    """

    def __init__(self, config: FastSearchConfig | None = None) -> None:
        self.config = config or load_config()
        self.model_manager = ModelManager(self.config)
        self._server: asyncio.Server | None = None
        self._start_time: float | None = None
        self._request_count = 0
        self._shutdown_event = asyncio.Event()
        self._db_cache: dict[str, tuple[SearchDB, threading.Lock]] = {}
        self._concurrent_sem = asyncio.Semaphore(64)

        # Handler registry
        self._handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            "ping": self._handle_ping,
            "status": self._handle_status,
            "search": self._handle_search,
            "embed": self._handle_embed,
            "rerank": self._handle_rerank,
            "load_model": self._handle_load_model,
            "unload_model": self._handle_unload_model,
            "reload_config": self._handle_reload_config,
            "batch_index": self._handle_batch_index,
            "delete": self._handle_delete,
            "update_content": self._handle_update_content,
            "list_sources": self._handle_list_sources,
            "shutdown": self._handle_shutdown,
        }

    async def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Simple ping handler."""
        return {"pong": True, "timestamp": time.time()}

    async def _handle_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get daemon status."""
        model_status = self.model_manager.get_status()

        return {
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "request_count": self._request_count,
            "socket_path": self.config.daemon.socket_path,
            "concurrent_slots_available": self._concurrent_sem._value,
            **model_status,
        }

    _DB_CACHE_MAX: int = 8

    def _get_db(self, db_path: str) -> tuple[SearchDB, threading.Lock]:
        """Get or create a cached SearchDB connection with its associated lock.

        Validates that db_path resolves to a location under the allowed base
        directory (parent of DEFAULT_DB_PATH) to prevent path traversal attacks.
        Caps the connection cache at _DB_CACHE_MAX entries.

        Returns a (SearchDB, threading.Lock) tuple. The lock MUST be held when
        performing any operation on the SearchDB from an executor thread to
        prevent concurrent threads from interleaving transactions on the same
        apsw.Connection.
        """
        resolved = Path(db_path).resolve()
        # Security: reject paths containing '..' components after resolution
        # to prevent path traversal, but allow any absolute path that resolves
        # cleanly. This replaces the overly restrictive single-directory check.
        try:
            resolved.relative_to(resolved.anchor)
        except ValueError:
            raise ValueError(f"db_path could not be resolved: {db_path}") from None
        if ".." in resolved.parts:
            raise ValueError(f"db_path must not contain '..': {db_path}")
        key = str(resolved)
        if key not in self._db_cache:
            from .core import SearchDB

            # Evict oldest entry if cache is full
            if len(self._db_cache) >= self._DB_CACHE_MAX:
                oldest_key = next(iter(self._db_cache))
                evicted_db, _evicted_lock = self._db_cache.pop(oldest_key)
                try:
                    evicted_db.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    logger.debug(f"WAL checkpoint completed for evicted DB {oldest_key}")
                except Exception as e:
                    logger.warning(f"WAL checkpoint failed for evicted DB {oldest_key}: {e}")
                try:
                    evicted_db.close()
                except Exception as e:
                    logger.warning(f"Error closing evicted DB {oldest_key}: {e}")
            self._db_cache[key] = (SearchDB(str(resolved)), threading.Lock())
        return self._db_cache[key]

    async def _handle_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle search request."""
        query = params.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("Missing or invalid 'query' parameter (must be non-empty string)")

        db_path = params.get("db_path", DEFAULT_DB_PATH)
        limit = params.get("limit", 10)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if limit > 1000:
            raise ValueError("limit must not exceed 1000")
        mode = params.get("mode", "hybrid")
        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"Invalid mode: {mode!r}, must be 'bm25', 'vector', or 'hybrid'")
        rerank = params.get("rerank", False)
        metadata_filter = params.get("metadata_filter")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise ValueError("metadata_filter must be a JSON object (dict)")

        # Get or load embedder
        embedder_model = await self.model_manager.load_model("embedder")
        reranker_model = None

        db, db_lock = self._get_db(db_path)

        try:
            start_time = time.perf_counter()
            loop = asyncio.get_running_loop()

            if mode == "bm25":

                def _search_bm25() -> list[Any]:
                    with db_lock:
                        return db.search_bm25(query, limit=limit, metadata_filter=metadata_filter)

                results = await loop.run_in_executor(None, _search_bm25)
            elif mode == "vector":
                embedding = embedder_model.instance.embed_query(query)

                def _search_vector() -> list[Any]:
                    with db_lock:
                        return db.search_vector(
                            embedding, limit=limit, metadata_filter=metadata_filter
                        )

                results = await loop.run_in_executor(None, _search_vector)
            else:  # hybrid
                embedding = embedder_model.instance.embed_query(query)

                if rerank:
                    # Load reranker on-demand
                    reranker_model = await self.model_manager.load_model("reranker")

                    def _search_hybrid_reranked() -> list[Any]:
                        with db_lock:
                            return db.search_hybrid_reranked(
                                query,
                                embedding,
                                limit=limit,
                                rerank_top_k=min(limit * 3, 100),
                                reranker=_RerankerAdapter(reranker_model.instance),
                                metadata_filter=metadata_filter,
                            )

                    results = await loop.run_in_executor(None, _search_hybrid_reranked)
                else:

                    def _search_hybrid() -> list[Any]:
                        with db_lock:
                            return db.search_hybrid(
                                query,
                                embedding,
                                limit=limit,
                                metadata_filter=metadata_filter,
                            )

                    results = await loop.run_in_executor(None, _search_hybrid)

            search_time = time.perf_counter() - start_time

            return {
                "query": query,
                "mode": mode,
                "reranked": rerank,
                "search_time_ms": round(search_time * 1000, 2),
                "results": results,
            }
        finally:
            await self.model_manager.release_model("embedder")
            if reranker_model is not None:
                await self.model_manager.release_model("reranker")

    async def _handle_embed(self, params: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings for texts."""
        texts = params.get("texts", [])
        if not texts:
            raise ValueError("Missing 'texts' parameter")

        MAX_BATCH_SIZE = 256
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(f"Too many texts: {len(texts)} (max {MAX_BATCH_SIZE})")

        embedder_model = await self.model_manager.load_model("embedder")

        try:
            start_time = time.perf_counter()
            embeddings = embedder_model.instance.embed(texts)
            embed_time = time.perf_counter() - start_time

            return {
                "embeddings": embeddings,
                "count": len(embeddings),
                "embed_time_ms": round(embed_time * 1000, 2),
            }
        finally:
            await self.model_manager.release_model("embedder")

    async def _handle_rerank(self, params: dict[str, Any]) -> dict[str, Any]:
        """Rerank documents against query."""
        query = params.get("query")
        documents = params.get("documents", [])

        if not query:
            raise ValueError("Missing 'query' parameter")
        if not documents:
            raise ValueError("Missing 'documents' parameter")

        MAX_RERANK_SIZE = 100
        if len(documents) > MAX_RERANK_SIZE:
            raise ValueError(f"Too many documents: {len(documents)} (max {MAX_RERANK_SIZE})")

        reranker_model = await self.model_manager.load_model("reranker")

        try:
            start_time = time.perf_counter()

            # Cross-encoder expects pairs
            pairs = [[query, doc] for doc in documents]
            scores = reranker_model.instance.predict(pairs).tolist()

            rerank_time = time.perf_counter() - start_time

            # Return sorted indices with scores
            indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            return {
                "scores": scores,
                "ranked": [{"index": idx, "score": score} for idx, score in indexed_scores],
                "rerank_time_ms": round(rerank_time * 1000, 2),
            }
        finally:
            await self.model_manager.release_model("reranker")

    async def _handle_load_model(self, params: dict[str, Any]) -> dict[str, Any]:
        """Load a model slot."""
        slot = params.get("slot")
        if not slot:
            raise ValueError("Missing 'slot' parameter")

        model = await self.model_manager.load_model(slot)
        # Release immediately — caller just wants the model loaded
        await self.model_manager.release_model(slot)

        return {
            "slot": slot,
            "loaded": True,
            "memory_mb": model.memory_mb,
        }

    async def _handle_unload_model(self, params: dict[str, Any]) -> dict[str, Any]:
        """Unload a model slot."""
        slot = params.get("slot")
        if not slot:
            raise ValueError("Missing 'slot' parameter")

        await self.model_manager.unload_model(slot)

        return {
            "slot": slot,
            "unloaded": True,
        }

    async def _handle_reload_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reload configuration."""
        config_path = params.get("config_path")

        if config_path is not None:
            resolved = Path(config_path).resolve()
            if resolved.suffix not in (".yaml", ".yml"):
                raise ValueError(f"Config path must end with .yaml or .yml: {config_path}")
            if not resolved.is_file():
                raise ValueError(f"Config file does not exist: {resolved}")
            config_path = str(resolved)

        new_config = load_config(config_path)
        self.config = new_config
        self.model_manager.config = new_config

        return {
            "reloaded": True,
            "socket_path": new_config.daemon.socket_path,
        }

    async def _handle_batch_index(self, params: dict[str, Any]) -> dict[str, Any]:
        """Batch index documents into the database."""
        db_path = params.get("db_path", DEFAULT_DB_PATH)
        documents = params.get("documents")

        if not isinstance(documents, list):
            raise ValueError("Missing or invalid 'documents' parameter (must be a list)")

        MAX_BATCH_SIZE = 1000
        if len(documents) > MAX_BATCH_SIZE:
            raise ValueError(f"Too many documents: {len(documents)} (max {MAX_BATCH_SIZE})")

        # Validate each document has required fields
        required_fields = {"source", "chunk_index", "content", "embedding"}
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise ValueError(f"Document at index {i} must be an object")
            missing = required_fields - doc.keys()
            if missing:
                raise ValueError(
                    f"Document at index {i} missing required fields: {sorted(missing)}"
                )
            if not isinstance(doc["source"], str) or not doc["source"]:
                raise ValueError(f"Document at index {i}: 'source' must be a non-empty string")
            if not isinstance(doc["chunk_index"], int):
                raise ValueError(f"Document at index {i}: 'chunk_index' must be an integer")
            if doc["chunk_index"] < 0:
                raise ValueError(f"Document at index {i}: 'chunk_index' must be non-negative")
            if not isinstance(doc["content"], str):
                raise ValueError(f"Document at index {i}: 'content' must be a string")
            if not doc["content"]:
                raise ValueError(f"Document at index {i}: 'content' must not be empty")
            if not isinstance(doc["embedding"], list):
                raise ValueError(f"Document at index {i}: 'embedding' must be a list of floats")

        db, db_lock = self._get_db(db_path)

        start_time = time.perf_counter()

        # Build items tuple list for SearchDB.index_batch
        items: list[tuple[str, int, str, list[float], dict[str, Any] | None]] = [
            (
                doc["source"],
                doc["chunk_index"],
                doc["content"],
                doc["embedding"],
                doc.get("metadata"),
            )
            for doc in documents
        ]

        # Run in executor with lock to prevent concurrent transaction interleaving
        loop = asyncio.get_running_loop()

        def _index_batch() -> list[int]:
            with db_lock:
                return db.index_batch(items)

        doc_ids = await loop.run_in_executor(None, _index_batch)

        index_time = time.perf_counter() - start_time

        return {
            "indexed": len(doc_ids),
            "doc_ids": doc_ids,
            "index_time_ms": round(index_time * 1000, 2),
        }

    async def _handle_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        """Delete documents by source or by ID."""
        source = params.get("source")
        doc_id = params.get("id")

        if source is not None and doc_id is not None:
            raise ValueError("Specify either 'source' or 'id', not both")
        if source is None and doc_id is None:
            raise ValueError("Missing 'source' or 'id' parameter")
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("'source' must be a non-empty string")
        if doc_id is not None and not isinstance(doc_id, int):
            raise ValueError("'id' must be an integer")

        db_path = params.get("db_path", DEFAULT_DB_PATH)
        db, db_lock = self._get_db(db_path)

        loop = asyncio.get_running_loop()

        if source is not None:

            def _delete_source() -> int:
                with db_lock:
                    return db.delete_source(source)

            count = await loop.run_in_executor(None, _delete_source)
            return {"deleted": count, "source": source}
        else:
            assert isinstance(doc_id, int)  # validated above

            def _delete_by_id() -> bool:
                with db_lock:
                    return db.delete_by_id(doc_id)

            found = await loop.run_in_executor(None, _delete_by_id)
            return {"deleted": 1 if found else 0, "id": doc_id}

    async def _handle_update_content(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update content and embedding for a document by ID."""
        doc_id = params.get("id")
        content = params.get("content")

        if doc_id is None:
            raise ValueError("Missing 'id' parameter")
        if not isinstance(doc_id, int):
            raise ValueError("'id' must be an integer")
        if not isinstance(content, str) or not content:
            raise ValueError("Missing or invalid 'content' parameter (must be non-empty string)")

        db_path = params.get("db_path", DEFAULT_DB_PATH)
        db, db_lock = self._get_db(db_path)

        # Generate embedding for the new content
        embedder_model = await self.model_manager.load_model("embedder")
        try:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None, lambda: embedder_model.instance.embed([content])[0]
            )
        finally:
            await self.model_manager.release_model("embedder")

        def _update_content() -> bool:
            with db_lock:
                return db.update_content(doc_id, content, embedding)

        updated = await asyncio.get_running_loop().run_in_executor(None, _update_content)
        return {"updated": updated, "id": doc_id}

    async def _handle_list_sources(self, params: dict[str, Any]) -> dict[str, Any]:
        """List all indexed sources with chunk counts."""
        db_path = params.get("db_path", DEFAULT_DB_PATH)
        db, db_lock = self._get_db(db_path)

        loop = asyncio.get_running_loop()

        def _list_sources() -> list[dict[str, Any]]:
            with db_lock:
                return db.list_sources()

        sources = await loop.run_in_executor(None, _list_sources)
        return {"sources": sources, "count": len(sources)}

    async def _handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        """Shutdown the daemon."""
        self._shutdown_event.set()
        return {"shutdown": True}

    async def _handle_request(self, data: bytes) -> bytes:
        """Process a JSON-RPC request."""
        try:
            request = orjson.loads(data)
        except (orjson.JSONDecodeError, ValueError) as e:
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None,
                }
            )

        if not isinstance(request, dict):
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Invalid Request: expected JSON object"},
                    "id": None,
                }
            )

        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "method must be a string"},
                    "id": request_id,
                }
            )
        params = request.get("params", {})
        if not isinstance(params, dict):
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "params must be an object"},
                    "id": request_id,
                }
            )

        self._request_count += 1

        if method not in self._handlers:
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id,
                }
            )

        try:
            result = await self._handlers[method](params)
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id,
                }
            )
        except ValueError as e:
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": str(e)},
                    "id": request_id,
                }
            )
        except Exception as e:
            logger.exception(f"Error handling {method}")
            # Return generic message to client; details are in the daemon log
            error_type = type(e).__name__
            return orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": f"Internal error: {error_type}"},
                    "id": request_id,
                }
            )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        # Tune socket buffers for large embed batches
        try:
            sock = writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2_097_152)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2_097_152)
        except (AttributeError, OSError):
            pass

        limiter = RateLimiter()
        try:
            while True:
                # Read length prefix (idle timeout: 300s)
                length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=300.0)
                length = int.from_bytes(length_bytes, "big")

                if length == 0:
                    logger.warning("Zero-length message received, closing connection")
                    break

                if length > 10 * 1024 * 1024:  # 10MB limit
                    logger.warning(f"Message too large: {length} bytes")
                    break

                # Check per-connection rate limit before reading body
                if not limiter.check():
                    logger.warning("Client rate-limited (>20 req/s)")
                    error_response = orjson.dumps(
                        {
                            "jsonrpc": "2.0",
                            "error": {"code": -32001, "message": "Rate limited"},
                            "id": None,
                        }
                    )
                    # Still need to consume the message body to stay in sync
                    await asyncio.wait_for(reader.readexactly(length), timeout=30.0)
                    writer.write(len(error_response).to_bytes(4, "big"))
                    writer.write(error_response)
                    await writer.drain()
                    continue

                # Read message body (data timeout: 30s)
                data = await asyncio.wait_for(reader.readexactly(length), timeout=30.0)

                # Acquire concurrency slot, then process and respond
                async with self._concurrent_sem:
                    response = await self._handle_request(data)
                    # Send length-prefixed response
                    writer.write(len(response).to_bytes(4, "big"))
                    writer.write(response)
                    await writer.drain()

        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        except TimeoutError:
            logger.info("Client connection timed out")
        except Exception:
            logger.exception("Error handling client")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self, foreground: bool = True) -> None:
        """Start the daemon server."""
        socket_path = self.config.daemon.socket_path

        # Check if a daemon is already running
        pid_path = self.config.daemon.pid_path
        try:
            pid = int(Path(pid_path).read_text().strip())
            # Cross-platform PID reuse guard
            try:
                os.kill(pid, 0)  # Check if process exists (signal 0 = no signal sent)
                is_fastsearch = True  # Process exists; assume it's fastsearch
                try:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_text()
                    is_fastsearch = "fastsearch" in cmdline
                except (FileNotFoundError, PermissionError):
                    pass  # /proc not available (macOS), keep assumption
            except ProcessLookupError:
                is_fastsearch = False  # Process doesn't exist, stale PID
            except PermissionError:
                is_fastsearch = True  # Process exists but we can't signal it
            if is_fastsearch:
                raise RuntimeError(
                    f"Daemon already running (PID {pid}). "
                    "Stop it first with 'vps-fastsearch daemon stop'."
                )
            # PID reused by a different process — treat PID file as stale
            logger.warning(f"Stale PID file: process {pid} exists but is not fastsearch")
        except (FileNotFoundError, ValueError):
            pass  # No PID file or invalid content

        # Warn if XDG_RUNTIME_DIR is not set
        if not os.environ.get("XDG_RUNTIME_DIR"):
            logger.warning("XDG_RUNTIME_DIR not set; using /tmp for socket and PID file")

        # Remove existing socket (catch FileNotFoundError instead of TOCTOU check)
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass

        # Create server with restrictive umask to avoid world-accessible window
        old_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=socket_path,
            )
        finally:
            os.umask(old_umask)

        self._start_time = time.time()

        # Write PID file
        # Remove any existing PID file first (don't follow symlinks)
        try:
            os.unlink(pid_path)
        except FileNotFoundError:
            pass
        # Create PID file with exclusive creation to prevent symlink attacks
        fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)

        logger.info(f"VPS-FastSearch daemon started on {socket_path}")

        # Pre-load "always" models (fail fast if required models can't load)
        for slot, model_config in self.config.models.items():
            if model_config.keep_loaded == "always":
                try:
                    await self.model_manager.load_model(slot)
                    # Release the ref from pre-loading
                    await self.model_manager.release_model(slot)
                except Exception as e:
                    logger.error(f"Failed to pre-load required model {slot}: {e}")
                    raise RuntimeError(
                        f"Cannot start daemon: required model '{slot}' failed to load"
                    ) from e

        if foreground:
            try:
                # Wait for shutdown signal
                await self._shutdown_event.wait()
            finally:
                await self.stop()

    async def stop(self) -> None:
        """Stop the daemon server."""
        logger.info("Shutting down VPS-FastSearch daemon...")

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        await self.model_manager.shutdown()

        # Checkpoint and close cached database connections
        for db_key, (db, _lock) in self._db_cache.items():
            try:
                db.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                logger.debug(f"WAL checkpoint completed for DB {db_key}")
            except Exception as e:
                logger.warning(f"WAL checkpoint failed for DB {db_key}: {e}")
            try:
                db.close()
            except Exception as e:
                logger.warning(f"Error closing DB {db_key}: {e}")
        self._db_cache.clear()

        # Clean up socket and PID files
        socket_path = self.config.daemon.socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        pid_path = self.config.daemon.pid_path
        if os.path.exists(pid_path):
            os.unlink(pid_path)

        logger.info("VPS-FastSearch daemon stopped")


def run_daemon(
    config_path: str | None = None, foreground: bool = True, detach: bool = False
) -> None:
    """Run the VPS-FastSearch daemon."""
    # Set up logging
    config = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.daemon.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if detach:
        # Double-fork daemonization (Unix standard technique).
        # A pipe communicates the final daemon PID back to the original parent.
        read_fd, write_fd = os.pipe()

        # --- First fork: detach from parent ---
        try:
            pid = os.fork()
        except OSError as e:
            os.close(read_fd)
            os.close(write_fd)
            print(f"First fork failed: {e}", file=sys.stderr)
            sys.exit(1)

        if pid > 0:
            # Original parent: read the grandchild PID from the pipe, then exit.
            os.close(write_fd)
            try:
                data = b""
                while True:
                    chunk = os.read(read_fd, 64)
                    if not chunk:
                        break
                    data += chunk
                grandchild_pid = int(data.decode().strip())
                print(f"VPS-FastSearch daemon started (PID: {grandchild_pid})")
            except (ValueError, OSError):
                print("VPS-FastSearch daemon started (PID unknown)")
            finally:
                os.close(read_fd)
            os._exit(0)

        # --- First child: create new session ---
        os.close(read_fd)
        os.setsid()

        # --- Second fork: prevent reacquiring a controlling terminal ---
        try:
            pid2 = os.fork()
        except OSError as e:
            os.close(write_fd)
            print(f"Second fork failed: {e}", file=sys.stderr)
            os._exit(1)

        if pid2 > 0:
            # First child exits; grandchild continues as the daemon.
            os.close(write_fd)
            os._exit(0)

        # --- Grandchild (final daemon process) ---
        # Send our PID back to the original parent via the pipe.
        try:
            os.write(write_fd, str(os.getpid()).encode())
        except OSError:
            pass
        os.close(write_fd)

        # Change working directory to / so we don't hold any mountpoint busy
        os.chdir("/")

        # Reset file creation mask
        os.umask(0o022)

        # Redirect stdin/stdout/stderr to /dev/null
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if devnull > 2:
            os.close(devnull)

        # Configure file-based logging since stdout/stderr are now /dev/null
        _xdg_state = os.environ.get("XDG_STATE_HOME", os.path.join(Path.home(), ".local", "state"))
        log_dir = Path(_xdg_state) / "fastsearch"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "daemon.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info("Daemon started in detached mode, logging to %s", log_file)

    daemon = FastSearchDaemon(config)

    # Register atexit handler to checkpoint/close DBs on non-SIGKILL exits
    def cleanup() -> None:
        for db, _lock in daemon._db_cache.values():
            try:
                db.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                db.close()
            except Exception:
                pass

    atexit.register(cleanup)

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler() -> None:
        daemon._shutdown_event.set()

    def sighup_handler() -> None:
        logger.info("SIGHUP received, reloading configuration")
        asyncio.ensure_future(daemon._handle_reload_config({}))

    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGHUP, sighup_handler)

    try:
        loop.run_until_complete(daemon.start(foreground=True))
    finally:
        loop.close()


def stop_daemon(config_path: str | None = None) -> bool:
    """Stop the running daemon."""
    config = load_config(config_path)
    pid_path = config.daemon.pid_path

    if not os.path.exists(pid_path):
        return False

    try:
        pid = int(Path(pid_path).read_text().strip())
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit
        for _ in range(200):  # 20 seconds — allow time for ONNX model unload + WAL checkpoint
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(0.1)
            except ProcessLookupError:
                return True

        # Process didn't exit after SIGTERM, escalate to SIGKILL
        try:
            logger.warning(f"Daemon (PID {pid}) didn't exit after SIGTERM, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        except ProcessLookupError:
            pass

        # Clean up stale PID file
        if os.path.exists(pid_path):
            os.unlink(pid_path)
        return True
    except (ValueError, ProcessLookupError):
        # Clean up stale PID file
        if os.path.exists(pid_path):
            os.unlink(pid_path)
        return False


def get_daemon_status(config_path: str | None = None) -> dict[str, Any] | None:
    """Get status of running daemon."""
    config = load_config(config_path)
    socket_path = config.daemon.socket_path

    if not os.path.exists(socket_path):
        return None

    # Try to connect and get status
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(socket_path)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2_097_152)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2_097_152)
        except OSError:
            pass

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "status",
                "params": {},
                "id": 1,
            }
        ).encode()

        sock.sendall(len(request).to_bytes(4, "big"))
        sock.sendall(request)

        length_bytes = b""
        while len(length_bytes) < 4:
            chunk = sock.recv(4 - len(length_bytes))
            if not chunk:
                return None
            length_bytes += chunk

        length = int.from_bytes(length_bytes, "big")

        response = b""
        while len(response) < length:
            chunk = sock.recv(min(8192, length - len(response)))
            if not chunk:
                return None
            response += chunk

        result = json.loads(response)
        return dict(result["result"]) if "result" in result else None
    except Exception:
        return None
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass
