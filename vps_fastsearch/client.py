"""VPS-FastSearch client library for connecting to the daemon."""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Embedder

from .config import DEFAULT_DB_PATH, DEFAULT_SOCKET_PATH, load_config

logger = logging.getLogger(__name__)


class FastSearchError(Exception):
    """FastSearch client error."""

    pass


class DaemonNotRunningError(FastSearchError):
    """Daemon is not running or unreachable."""

    pass


class FastSearchClient:
    """
    Python client for FastSearch daemon.

    Usage:
        from vps_fastsearch import FastSearchClient

        client = FastSearchClient()
        results = client.search("query")
        results = client.search("query", rerank=True)
        status = client.status()

    Note: This class is NOT thread-safe. Use a separate instance per thread,
    or synchronize access externally.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        config_path: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize FastSearch client.

        Args:
            socket_path: Path to Unix socket (default: from config or /tmp/fastsearch.sock)
            config_path: Path to config file (optional)
            timeout: Socket timeout in seconds
        """
        if socket_path:
            self.socket_path = socket_path
        else:
            config = load_config(config_path)
            self.socket_path = config.daemon.socket_path

        self.timeout = timeout
        self._sock: socket.socket | None = None

    def _connect(self) -> None:
        """Establish connection to daemon."""
        if self._sock is not None:
            return

        if not os.path.exists(self.socket_path):
            raise DaemonNotRunningError(f"Daemon socket not found: {self.socket_path}")

        logger.debug("Connecting to %s", self.socket_path)
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout)
            self._sock.connect(self.socket_path)
            # Tune socket buffers for large embed batches
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2_097_152)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2_097_152)
            except OSError:
                pass
        except (ConnectionRefusedError, FileNotFoundError):
            self._sock = None
            raise DaemonNotRunningError(f"Cannot connect to daemon at {self.socket_path}") from None
        except Exception as e:
            self._sock = None
            raise FastSearchError(f"Connection error: {e}") from e

    def _disconnect(self) -> None:
        """Close connection."""
        if self._sock:
            logger.debug("Disconnected")
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Send JSON-RPC request and get response.

        This method implements automatic retry on connection failures (TimeoutError, OSError).
        On the first failure, it closes the connection and retries once after reconnecting.

        WARNING: Non-idempotent operations may execute twice:
        - If the first request times out AFTER being processed by the daemon but BEFORE
          the response is received, the retry will execute the operation again.
        - This affects: batch_index, delete, update_content
        - Idempotent operations (search, embed, status) are safe to retry.

        For non-idempotent operations, ensure your application layer handles potential
        duplicates (e.g., via database constraints, idempotency keys, or duplicate detection).
        """
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

        logger.debug("Request: method=%s", request.get("method"))
        data = json.dumps(request).encode()

        # Validate message size before sending
        MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB, matches daemon limit
        if len(data) > MAX_MESSAGE_SIZE:
            raise FastSearchError(f"Request too large: {len(data)} bytes (max 10MB)")

        for attempt in range(2):
            self._connect()
            assert self._sock is not None
            try:
                # Send length-prefixed message
                self._sock.sendall(len(data).to_bytes(4, "big"))
                self._sock.sendall(data)

                # Receive length-prefixed response
                length_bytes = b""
                while len(length_bytes) < 4:
                    chunk = self._sock.recv(4 - len(length_bytes))
                    if not chunk:
                        raise FastSearchError("Connection closed by daemon")
                    length_bytes += chunk

                length = int.from_bytes(length_bytes, "big")

                # Validate response size
                if length > MAX_MESSAGE_SIZE:
                    self._disconnect()
                    raise FastSearchError(f"Response too large: {length} bytes (max 10MB)")

                # Receive full response
                response_data = bytearray(length)
                bytes_read = 0
                while bytes_read < length:
                    chunk = self._sock.recv(min(8192, length - bytes_read))
                    if not chunk:
                        raise FastSearchError("Connection closed while receiving response")
                    response_data[bytes_read : bytes_read + len(chunk)] = chunk
                    bytes_read += len(chunk)

                response = json.loads(response_data)

                if not isinstance(response, dict):
                    raise FastSearchError(
                        f"Invalid JSON-RPC response: expected dict, got {type(response).__name__}"
                    )

                if "error" in response:
                    error = response["error"]
                    raise FastSearchError(f"RPC error {error.get('code')}: {error.get('message')}")

                if "result" not in response:
                    raise FastSearchError(
                        "Invalid JSON-RPC response from daemon: missing 'result' and 'error'"
                    )

                result = response["result"]
                if not isinstance(result, dict):
                    raise FastSearchError(f"Unexpected result type: {type(result).__name__}")
                return dict(result)

            except (TimeoutError, OSError) as e:
                self._disconnect()
                if attempt == 0:
                    logger.warning("Connection lost, retrying: %s", e)
                    continue  # retry once after reconnect
                raise FastSearchError(f"Connection lost: {e}") from e
            except json.JSONDecodeError as e:
                self._disconnect()
                raise FastSearchError(f"Invalid response: {e}") from e
        raise FastSearchError("Request failed after 2 attempts")

    def ping(self) -> bool:
        """Check if daemon is responding."""
        try:
            result = self._send_request("ping")
            return bool(result.get("pong", False))
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        """
        Get daemon status.

        Returns:
            dict with:
            - uptime_seconds: Daemon uptime
            - request_count: Total requests handled
            - loaded_models: Dict of loaded model info
            - total_memory_mb: Current memory usage
            - max_memory_mb: Memory budget
        """
        return self._send_request("status")

    def search(
        self,
        query: str,
        db_path: str | None = None,
        limit: int = 10,
        mode: str = "hybrid",
        rerank: bool = False,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search indexed documents.

        Args:
            query: Search query text
            db_path: Path to database file
            limit: Maximum results to return
            mode: Search mode (hybrid, bm25, vector)
            rerank: Apply cross-encoder reranking
            metadata_filter: Optional dict of key-value pairs for exact match
                on top-level metadata keys (AND logic). Example:
                ``{"author": "alice", "category": "tech"}``

        Returns:
            dict with:
            - query: Original query
            - mode: Search mode used
            - reranked: Whether reranking was applied
            - search_time_ms: Search latency
            - results: List of result dicts
        """
        if db_path is None:
            db_path = os.environ.get("FASTSEARCH_DB", DEFAULT_DB_PATH)
        params: dict[str, Any] = {
            "query": query,
            "db_path": db_path,
            "limit": limit,
            "mode": mode,
            "rerank": rerank,
        }
        if metadata_filter:
            params["metadata_filter"] = metadata_filter
        return self._send_request("search", params)

    def embed(self, texts: list[str]) -> dict[str, Any]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            dict with:
            - embeddings: List of embedding vectors
            - count: Number of embeddings
            - embed_time_ms: Embedding latency
        """
        return self._send_request("embed", {"texts": texts})

    def rerank(self, query: str, documents: list[str]) -> dict[str, Any]:
        """
        Rerank documents against query.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            dict with:
            - scores: Raw relevance scores
            - ranked: Sorted list of {index, score}
            - rerank_time_ms: Reranking latency
        """
        return self._send_request(
            "rerank",
            {
                "query": query,
                "documents": documents,
            },
        )

    def load_model(self, slot: str) -> dict[str, Any]:
        """
        Load a model slot.

        Args:
            slot: Model slot name (embedder, reranker)

        Returns:
            dict with slot, loaded status, memory_mb
        """
        return self._send_request("load_model", {"slot": slot})

    def unload_model(self, slot: str) -> dict[str, Any]:
        """
        Unload a model slot.

        Args:
            slot: Model slot name

        Returns:
            dict with slot, unloaded status
        """
        return self._send_request("unload_model", {"slot": slot})

    def batch_index(
        self,
        documents: list[dict[str, Any]],
        db_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Batch index documents into the database.

        This operation is NOT idempotent: if the daemon processes the request but the
        response is lost due to connection timeout, the retry will index the documents
        again, potentially creating duplicates. Use database constraints or application
        logic to detect and handle duplicate indexing.

        Args:
            db_path: Path to database file
            documents: List of document dicts, each with keys:
                - source (str): Document source identifier
                - chunk_index (int): Chunk position within the source
                - content (str): Document text content
                - embedding (list[float]): 768-dim embedding vector
                - metadata (dict|None): Optional metadata

        Returns:
            dict with:
            - indexed: Count of documents indexed
            - doc_ids: List of assigned document IDs
            - index_time_ms: Indexing latency

        Raises:
            FastSearchError: On indexing failure or connection loss after 2 attempts
        """
        if db_path is None:
            db_path = os.environ.get("FASTSEARCH_DB", DEFAULT_DB_PATH)
        return self._send_request(
            "batch_index",
            {
                "db_path": db_path,
                "documents": documents,
            },
        )

    def delete(
        self,
        source: str | None = None,
        doc_id: int | None = None,
        db_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete documents by source name or by document ID.

        This operation is NOT idempotent: if the daemon processes the request but the
        response is lost due to connection timeout, the retry will attempt deletion again.
        Deleting by doc_id is safe (delete fails gracefully if ID doesn't exist), but
        deleting by source may have implications if source records are recreated. Ensure
        idempotency checks at the application level if needed.

        Args:
            source: Source name to delete all chunks for
            doc_id: Single document ID to delete
            db_path: Optional database path

        Returns:
            dict with deleted count and source/id

        Raises:
            FastSearchError: On deletion failure or connection loss after 2 attempts
        """
        params: dict[str, Any] = {}
        if source is not None:
            params["source"] = source
        if doc_id is not None:
            params["id"] = doc_id
        if db_path is not None:
            params["db_path"] = db_path
        return self._send_request("delete", params)

    def update_content(
        self,
        doc_id: int,
        content: str,
        db_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Update content and embedding for a document by ID.

        The daemon re-embeds the new content automatically.

        This operation is NOT idempotent: if the daemon processes the request but the
        response is lost due to connection timeout, the retry will update the content again,
        which should be safe due to overwrite semantics (same content written twice).
        However, ensure your application logs or tracks updates if audit trails are needed.

        Args:
            doc_id: Document ID to update
            content: New content text
            db_path: Optional database path

        Returns:
            dict with updated status and id

        Raises:
            FastSearchError: On update failure or connection loss after 2 attempts
        """
        params: dict[str, Any] = {"id": doc_id, "content": content}
        if db_path is not None:
            params["db_path"] = db_path
        return self._send_request("update_content", params)

    def list_sources(self, db_path: str | None = None) -> dict[str, Any]:
        """
        List all indexed sources with chunk counts.

        Args:
            db_path: Optional database path

        Returns:
            dict with sources list and count
        """
        params: dict[str, Any] = {}
        if db_path is not None:
            params["db_path"] = db_path
        return self._send_request("list_sources", params)

    def reload_config(self, config_path: str | None = None) -> dict[str, Any]:
        """
        Reload daemon configuration.

        Args:
            config_path: Optional config file path

        Returns:
            dict with reload status
        """
        params = {}
        if config_path:
            params["config_path"] = config_path
        return self._send_request("reload_config", params)

    def shutdown(self) -> dict[str, Any]:
        """
        Shutdown the daemon.

        Returns:
            dict with shutdown status
        """
        try:
            return self._send_request("shutdown")
        finally:
            self._disconnect()

    def close(self) -> None:
        """Close the client connection."""
        self._disconnect()

    def __enter__(self) -> FastSearchClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def is_daemon_running(socket_path: str | None = None) -> bool:
        """Check if daemon is running."""
        if socket_path is None:
            socket_path = DEFAULT_SOCKET_PATH

        if not os.path.exists(socket_path):
            return False

        try:
            with FastSearchClient(socket_path=socket_path, timeout=2.0) as client:
                return client.ping()
        except Exception:
            return False


def _get_embedder_with_config() -> Embedder:
    """Get embedder singleton with full config (provider, prefixes, etc.)."""
    from .core import Embedder

    config = load_config()
    embedder_config = config.models.get("embedder")
    if embedder_config:
        return Embedder.get_instance(
            model_name=embedder_config.name or None,
            document_prefix=embedder_config.document_prefix,
            query_prefix=embedder_config.query_prefix,
            provider=embedder_config.provider,
            base_url=embedder_config.base_url,
            api_key=embedder_config.api_key,
            threads=embedder_config.threads,
        )
    return Embedder.get_instance()


# Convenience functions for quick usage
def search(query: str, **kwargs: Any) -> list[Any]:
    """Quick search using daemon (falls back to direct if unavailable)."""
    try:
        with FastSearchClient(timeout=10.0) as client:
            result = client.search(query, **kwargs)
            return list(result.get("results", []))
    except (DaemonNotRunningError, FastSearchError):
        # Fall back to direct search
        from .core import SearchDB

        db_path = kwargs.get("db_path", os.environ.get("FASTSEARCH_DB", DEFAULT_DB_PATH))
        limit = kwargs.get("limit", 10)
        mode = kwargs.get("mode", "hybrid")
        rerank = kwargs.get("rerank", False)

        metadata_filter = kwargs.get("metadata_filter")
        db = SearchDB(db_path)
        try:
            search_results: list[Any]
            if mode == "bm25":
                search_results = db.search_bm25(query, limit=limit, metadata_filter=metadata_filter)
            else:
                embedder = _get_embedder_with_config()
                embedding = embedder.embed_single(query)
                if rerank:
                    search_results = db.search_hybrid_reranked(
                        query, embedding, limit=limit, metadata_filter=metadata_filter
                    )
                else:
                    search_results = db.search_hybrid(
                        query, embedding, limit=limit, metadata_filter=metadata_filter
                    )
        finally:
            db.close()

        return search_results


def embed(texts: list[str]) -> list[list[float]]:
    """Quick embed using daemon (falls back to direct if unavailable)."""
    try:
        with FastSearchClient(timeout=10.0) as client:
            result = client.embed(texts)
            return list(result.get("embeddings", []))
    except (DaemonNotRunningError, FastSearchError):
        embedder = _get_embedder_with_config()
        return embedder.embed(texts)
