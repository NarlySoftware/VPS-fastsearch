"""Unit tests for vps_fastsearch.daemon — no real server, no model downloads."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import orjson

from vps_fastsearch.config import (
    DaemonConfig,
    FastSearchConfig,
    MemoryConfig,
    ModelConfig,
)
from vps_fastsearch.daemon import (
    FastSearchDaemon,
    LoadedModel,
    ModelManager,
    RateLimiter,
    _RerankerAdapter,
    get_daemon_status,
    stop_daemon,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    max_ram_mb: int = 4000,
    socket_path: str = "/tmp/test_fastsearch.sock",
    pid_path: str = "/tmp/test_fastsearch.pid",
) -> FastSearchConfig:
    """Build a minimal FastSearchConfig for testing."""
    return FastSearchConfig(
        daemon=DaemonConfig(socket_path=socket_path, pid_path=pid_path),
        models={
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
        },
        memory=MemoryConfig(max_ram_mb=max_ram_mb),
    )


def _make_loaded_model(slot: str = "embedder", ref_count: int = 0) -> LoadedModel:
    """Create a LoadedModel with a mock instance."""
    now = time.time()
    return LoadedModel(
        slot=slot,
        instance=MagicMock(),
        loaded_at=now,
        last_used=now,
        memory_mb=450.0,
        actual_memory_mb=420.0,
        ref_count=ref_count,
    )


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for the sliding-window RateLimiter."""

    def test_allows_up_to_max_requests(self) -> None:
        """Requests up to max_requests should be allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            assert limiter.check() is True

    def test_rejects_over_limit(self) -> None:
        """The (max_requests + 1)-th request within the window is rejected."""
        limiter = RateLimiter(max_requests=3, window_seconds=1.0)
        for _ in range(3):
            limiter.check()
        assert limiter.check() is False

    def test_allows_again_after_window_passes(self) -> None:
        """After the window slides past old timestamps, new requests are allowed."""
        limiter = RateLimiter(max_requests=2, window_seconds=1.0)
        limiter.check()
        limiter.check()
        # Both slots used — next request rejected
        assert limiter.check() is False

        # Fake the passage of time by backdating all timestamps
        for i in range(len(limiter._timestamps)):
            limiter._timestamps[i] = time.monotonic() - 2.0

        # Now the window has slid; request should be allowed
        assert limiter.check() is True

    def test_single_request_allowed(self) -> None:
        """A limiter with max_requests=1 allows exactly one request per window."""
        limiter = RateLimiter(max_requests=1, window_seconds=60.0)
        assert limiter.check() is True
        assert limiter.check() is False

    def test_large_window_multiple_requests(self) -> None:
        """Requests spread over a large window are all allowed."""
        limiter = RateLimiter(max_requests=100, window_seconds=60.0)
        for _ in range(100):
            assert limiter.check() is True
        assert limiter.check() is False

    def test_empty_limiter_always_allows_first_request(self) -> None:
        """A fresh limiter always allows the first request."""
        limiter = RateLimiter(max_requests=20, window_seconds=1.0)
        assert limiter.check() is True


# ---------------------------------------------------------------------------
# _RerankerAdapter tests
# ---------------------------------------------------------------------------


class TestRerankerAdapter:
    """Tests for _RerankerAdapter."""

    def test_rerank_calls_predict(self) -> None:
        """rerank() should call model.predict with correct pairs."""
        mock_model = MagicMock()
        scores_array = MagicMock()
        scores_array.tolist.return_value = [0.9, 0.3, 0.7]
        mock_model.predict.return_value = scores_array

        adapter = _RerankerAdapter(mock_model)
        result = adapter.rerank("my query", ["doc1", "doc2", "doc3"])

        mock_model.predict.assert_called_once_with(
            [["my query", "doc1"], ["my query", "doc2"], ["my query", "doc3"]]
        )
        assert result == [0.9, 0.3, 0.7]

    def test_rerank_empty_documents_returns_empty(self) -> None:
        """Empty document list should return [] without calling predict."""
        mock_model = MagicMock()
        adapter = _RerankerAdapter(mock_model)
        result = adapter.rerank("query", [])
        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_single_document(self) -> None:
        """Single document produces a single-element score list."""
        mock_model = MagicMock()
        scores_array = MagicMock()
        scores_array.tolist.return_value = [0.5]
        mock_model.predict.return_value = scores_array

        adapter = _RerankerAdapter(mock_model)
        result = adapter.rerank("test", ["only doc"])
        assert result == [0.5]


# ---------------------------------------------------------------------------
# LoadedModel tests
# ---------------------------------------------------------------------------


class TestLoadedModel:
    """Tests for LoadedModel.touch()."""

    def test_touch_updates_last_used(self) -> None:
        """touch() should update the last_used timestamp to approximately now."""
        now = time.time()
        model = LoadedModel(
            slot="embedder",
            instance=MagicMock(),
            loaded_at=now - 100,
            last_used=now - 100,
            memory_mb=450.0,
            actual_memory_mb=0.0,
            ref_count=0,
        )
        before = model.last_used
        time.sleep(0.01)
        model.touch()
        assert model.last_used > before

    def test_touch_does_not_change_loaded_at(self) -> None:
        """touch() must not modify the loaded_at field."""
        now = time.time()
        loaded_at = now - 50
        model = LoadedModel(
            slot="embedder",
            instance=MagicMock(),
            loaded_at=loaded_at,
            last_used=loaded_at,
            memory_mb=450.0,
            actual_memory_mb=0.0,
            ref_count=0,
        )
        model.touch()
        assert model.loaded_at == loaded_at


# ---------------------------------------------------------------------------
# ModelManager tests
# ---------------------------------------------------------------------------


class TestModelManager:
    """Tests for ModelManager without loading real models."""

    def _make_manager(self) -> ModelManager:
        return ModelManager(_make_config())

    def test_is_loaded_false_initially(self) -> None:
        """A freshly created manager has no models loaded."""
        manager = self._make_manager()
        assert manager.is_loaded("embedder") is False
        assert manager.is_loaded("reranker") is False

    def test_get_model_returns_none_when_not_loaded(self) -> None:
        """get_model() returns None for slots that have not been loaded."""
        manager = self._make_manager()
        assert manager.get_model("embedder") is None

    def test_get_model_returns_instance_after_manual_insert(self) -> None:
        """After manually inserting a LoadedModel, get_model returns it."""
        manager = self._make_manager()
        lm = _make_loaded_model("embedder")
        manager._models["embedder"] = lm
        assert manager.get_model("embedder") is lm

    def test_is_loaded_true_after_manual_insert(self) -> None:
        manager = self._make_manager()
        manager._models["embedder"] = _make_loaded_model("embedder")
        assert manager.is_loaded("embedder") is True

    def test_get_status_structure(self) -> None:
        """get_status() should return the expected top-level keys."""
        manager = self._make_manager()
        status = manager.get_status()
        assert "loaded_models" in status
        assert "total_memory_mb" in status
        assert "max_memory_mb" in status
        assert isinstance(status["loaded_models"], dict)
        assert status["max_memory_mb"] == 4000

    def test_get_status_with_loaded_model(self) -> None:
        """get_status() should include loaded model metadata."""
        manager = self._make_manager()
        lm = _make_loaded_model("embedder")
        manager._models["embedder"] = lm
        status = manager.get_status()
        assert "embedder" in status["loaded_models"]
        info = status["loaded_models"]["embedder"]
        assert "loaded_at" in info
        assert "last_used" in info
        assert "memory_mb" in info
        assert "idle_seconds" in info

    def test_estimate_model_memory_known_slots(self) -> None:
        """estimate_model_memory() returns reasonable values for known slots."""
        manager = self._make_manager()
        assert manager.estimate_model_memory("embedder") == 450
        assert manager.estimate_model_memory("reranker") == 90
        assert manager.estimate_model_memory("summarizer") == 4000

    def test_estimate_model_memory_unknown_slot(self) -> None:
        """Unknown slot returns a fallback estimate of 500 MB."""
        manager = self._make_manager()
        assert manager.estimate_model_memory("unknown_slot") == 500

    def test_unload_model_noop_when_not_loaded(self) -> None:
        """unload_model() on a non-existent slot should complete without error."""
        manager = self._make_manager()
        asyncio.run(manager.unload_model("nonexistent"))  # Should not raise

    def test_unload_model_removes_slot(self) -> None:
        """unload_model() removes the slot from _models when ref_count == 0."""
        manager = self._make_manager()
        lm = _make_loaded_model("reranker", ref_count=0)
        manager._models["reranker"] = lm
        asyncio.run(manager.unload_model("reranker"))
        assert "reranker" not in manager._models

    def test_unload_model_skips_always_models(self) -> None:
        """unload_model() refuses to unload models with keep_loaded='always'."""
        manager = self._make_manager()
        lm = _make_loaded_model("embedder", ref_count=0)
        manager._models["embedder"] = lm
        asyncio.run(manager.unload_model("embedder"))
        # Should still be present
        assert "embedder" in manager._models

    def test_unload_model_skips_models_with_active_refs(self) -> None:
        """unload_model() skips slots that still have active references."""
        manager = self._make_manager()
        lm = _make_loaded_model("reranker", ref_count=2)
        manager._models["reranker"] = lm
        asyncio.run(manager.unload_model("reranker"))
        assert "reranker" in manager._models


# ---------------------------------------------------------------------------
# FastSearchDaemon._handle_request tests
# ---------------------------------------------------------------------------


class TestHandleRequest:
    """Tests for JSON-RPC protocol handling in FastSearchDaemon._handle_request."""

    def _make_daemon(self) -> FastSearchDaemon:
        config = _make_config()
        return FastSearchDaemon(config)

    def _run(self, coro: Any) -> Any:
        return asyncio.run(coro)

    # -- Parse errors --

    def test_parse_error_invalid_json(self) -> None:
        """Invalid JSON returns error code -32700."""
        daemon = self._make_daemon()
        response = orjson.loads(self._run(daemon._handle_request(b"not json")))
        assert response["error"]["code"] == -32700
        assert response["id"] is None

    def test_parse_error_empty_bytes(self) -> None:
        """Empty bytes (which is invalid JSON) returns a parse error."""
        daemon = self._make_daemon()
        response = orjson.loads(self._run(daemon._handle_request(b"")))
        assert response["error"]["code"] == -32700

    # -- Invalid request --

    def test_invalid_request_array(self) -> None:
        """JSON array instead of object returns -32600."""
        daemon = self._make_daemon()
        response = orjson.loads(self._run(daemon._handle_request(orjson.dumps([1, 2, 3]))))
        assert response["error"]["code"] == -32600

    def test_invalid_request_null(self) -> None:
        """JSON null returns -32600."""
        daemon = self._make_daemon()
        response = orjson.loads(self._run(daemon._handle_request(orjson.dumps(None))))
        assert response["error"]["code"] == -32600

    def test_invalid_request_method_not_string(self) -> None:
        """Non-string method returns -32600."""
        daemon = self._make_daemon()
        payload = orjson.dumps({"jsonrpc": "2.0", "method": 42, "params": {}, "id": 1})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32600

    def test_invalid_request_params_not_dict(self) -> None:
        """Array params returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "ping", "params": [1, 2], "id": 1})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602

    # -- Method not found --

    def test_method_not_found(self) -> None:
        """Unknown method returns -32601."""
        daemon = self._make_daemon()
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "nonexistent", "params": {}, "id": 7})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32601
        assert response["id"] == 7

    # -- Valid ping --

    def test_valid_ping_returns_result(self) -> None:
        """A well-formed ping request returns a result with pong=True."""
        daemon = self._make_daemon()
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert "result" in response
        assert response["result"]["pong"] is True
        assert response["id"] == 1

    # -- Handler ValueError -> -32602 --

    def test_handler_value_error_returns_32602(self) -> None:
        """A ValueError raised in a handler maps to -32602."""
        daemon = self._make_daemon()

        async def _bad_handler(params: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("bad param")

        daemon._handlers["bad_method"] = _bad_handler
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "bad_method", "params": {}, "id": 99})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "bad param" in response["error"]["message"]
        assert response["id"] == 99

    # -- Handler unexpected exception -> -32000 --

    def test_handler_unexpected_exception_returns_32000(self) -> None:
        """An unexpected exception in a handler maps to -32000."""
        daemon = self._make_daemon()

        async def _crashing_handler(params: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        daemon._handlers["crash"] = _crashing_handler
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "crash", "params": {}, "id": 5})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32000
        assert response["id"] == 5

    # -- id preservation --

    def test_response_id_preserved(self) -> None:
        """Response id must match request id."""
        daemon = self._make_daemon()
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "abc-123"})
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["id"] == "abc-123"

    # -- Search validation --

    def test_handle_search_missing_query(self) -> None:
        """Search with missing query returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {"jsonrpc": "2.0", "method": "search", "params": {"limit": 5}, "id": 1}
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602

    def test_handle_search_invalid_mode(self) -> None:
        """Search with invalid mode returns -32602 before touching the model manager."""
        # The mode validation happens before load_model is called, so we don't
        # need to mock the model manager at all.
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "search",
                "params": {"query": "test", "mode": "invalid_mode"},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602

    def test_handle_search_limit_too_large(self) -> None:
        """Search with limit > 1000 returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "search",
                "params": {"query": "test", "limit": 9999},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602

    def test_handle_search_limit_zero(self) -> None:
        """Search with limit=0 returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "search",
                "params": {"query": "test", "limit": 0},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602

    # -- request_count increment --

    def test_request_count_incremented_on_valid_request(self) -> None:
        """_request_count increments for valid (parseable, known method) requests."""
        daemon = self._make_daemon()
        assert daemon._request_count == 0
        payload = orjson.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        self._run(daemon._handle_request(payload))
        assert daemon._request_count == 1

    def test_request_count_not_incremented_on_parse_error(self) -> None:
        """_request_count should NOT increment when JSON is malformed."""
        daemon = self._make_daemon()
        self._run(daemon._handle_request(b"bad json"))
        assert daemon._request_count == 0


# ---------------------------------------------------------------------------
# batch_index handler tests
# ---------------------------------------------------------------------------


class TestHandleBatchIndex:
    """Tests for the batch_index RPC handler."""

    def _make_daemon(self) -> FastSearchDaemon:
        config = _make_config()
        return FastSearchDaemon(config)

    def _run(self, coro: Any) -> Any:
        return asyncio.run(coro)

    def _make_document(
        self,
        source: str = "test.txt",
        chunk_index: int = 0,
        content: str = "hello world",
    ) -> dict[str, Any]:
        """Create a valid document dict with a 768-dim embedding."""
        return {
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": [0.1] * 768,
            "metadata": {"tag": "test"},
        }

    def test_batch_index_valid_documents(self, tmp_path: Any) -> None:
        """batch_index with valid documents returns indexed count."""
        daemon = self._make_daemon()
        db_path = str(tmp_path / "test.db")

        # Patch _get_db to return a real SearchDB at the tmp_path
        from vps_fastsearch.core import SearchDB

        db = SearchDB(db_path)
        db_lock = threading.Lock()
        daemon._db_cache[str(tmp_path / "test.db")] = (db, db_lock)
        with patch.object(daemon, "_get_db", return_value=(db, db_lock)):
            docs = [
                self._make_document("a.txt", 0, "first document"),
                self._make_document("b.txt", 0, "second document"),
            ]
            payload = orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "batch_index",
                    "params": {"db_path": db_path, "documents": docs},
                    "id": 1,
                }
            )
            response = orjson.loads(self._run(daemon._handle_request(payload)))

        assert "result" in response
        assert response["result"]["indexed"] == 2
        assert len(response["result"]["doc_ids"]) == 2
        assert "index_time_ms" in response["result"]
        db.close()

    def test_batch_index_empty_documents(self, tmp_path: Any) -> None:
        """batch_index with empty documents list returns indexed=0."""
        daemon = self._make_daemon()
        db_path = str(tmp_path / "test.db")

        from vps_fastsearch.core import SearchDB

        db = SearchDB(db_path)
        db_lock = threading.Lock()
        daemon._db_cache[str(tmp_path / "test.db")] = (db, db_lock)
        with patch.object(daemon, "_get_db", return_value=(db, db_lock)):
            payload = orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "batch_index",
                    "params": {"db_path": db_path, "documents": []},
                    "id": 1,
                }
            )
            response = orjson.loads(self._run(daemon._handle_request(payload)))

        assert "result" in response
        assert response["result"]["indexed"] == 0
        assert response["result"]["doc_ids"] == []
        db.close()

    def test_batch_index_missing_documents_param(self) -> None:
        """batch_index without documents param returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {"db_path": "/tmp/test.db"},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "documents" in response["error"]["message"]

    def test_batch_index_documents_not_list(self) -> None:
        """batch_index with documents as string returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {"documents": "not a list"},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "documents" in response["error"]["message"]

    def test_batch_index_document_not_dict(self) -> None:
        """batch_index with a non-dict document returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {"documents": ["not a dict"]},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "index 0" in response["error"]["message"]

    def test_batch_index_missing_required_fields(self) -> None:
        """batch_index with missing required fields returns -32602."""
        daemon = self._make_daemon()
        # Document missing 'content' and 'embedding'
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {
                    "documents": [{"source": "a.txt", "chunk_index": 0}],
                },
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "missing required fields" in response["error"]["message"]

    def test_batch_index_invalid_source_type(self) -> None:
        """batch_index with non-string source returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {
                    "documents": [
                        {
                            "source": 123,
                            "chunk_index": 0,
                            "content": "text",
                            "embedding": [0.1] * 768,
                        }
                    ],
                },
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "source" in response["error"]["message"]

    def test_batch_index_invalid_chunk_index_type(self) -> None:
        """batch_index with non-integer chunk_index returns -32602."""
        daemon = self._make_daemon()
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {
                    "documents": [
                        {
                            "source": "a.txt",
                            "chunk_index": "zero",
                            "content": "text",
                            "embedding": [0.1] * 768,
                        }
                    ],
                },
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "chunk_index" in response["error"]["message"]

    def test_batch_index_too_many_documents(self) -> None:
        """batch_index with >1000 documents returns -32602."""
        daemon = self._make_daemon()
        docs = [self._make_document(f"doc{i}.txt", i) for i in range(1001)]
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": "batch_index",
                "params": {"documents": docs},
                "id": 1,
            }
        )
        response = orjson.loads(self._run(daemon._handle_request(payload)))
        assert response["error"]["code"] == -32602
        assert "Too many documents" in response["error"]["message"]

    def test_batch_index_with_null_metadata(self, tmp_path: Any) -> None:
        """batch_index with null metadata succeeds."""
        daemon = self._make_daemon()
        db_path = str(tmp_path / "test.db")

        from vps_fastsearch.core import SearchDB

        db = SearchDB(db_path)
        db_lock = threading.Lock()
        daemon._db_cache[str(tmp_path / "test.db")] = (db, db_lock)
        with patch.object(daemon, "_get_db", return_value=(db, db_lock)):
            doc = self._make_document()
            doc["metadata"] = None
            payload = orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "batch_index",
                    "params": {"db_path": db_path, "documents": [doc]},
                    "id": 1,
                }
            )
            response = orjson.loads(self._run(daemon._handle_request(payload)))

        assert "result" in response
        assert response["result"]["indexed"] == 1
        db.close()


# ---------------------------------------------------------------------------
# get_daemon_status / stop_daemon tests (no real socket)
# ---------------------------------------------------------------------------


class TestDaemonHelpers:
    """Tests for module-level helper functions."""

    def test_get_daemon_status_no_socket(self, tmp_path: Any) -> None:
        """get_daemon_status returns None when the socket file does not exist."""
        socket_path = str(tmp_path / "missing.sock")
        with patch(
            "vps_fastsearch.daemon.load_config",
            return_value=_make_config(socket_path=socket_path),
        ):
            result = get_daemon_status()
        assert result is None

    def test_stop_daemon_no_pid_file(self, tmp_path: Any) -> None:
        """stop_daemon returns False when the PID file does not exist."""
        pid_path = str(tmp_path / "missing.pid")
        with patch(
            "vps_fastsearch.daemon.load_config",
            return_value=_make_config(pid_path=pid_path),
        ):
            result = stop_daemon()
        assert result is False
