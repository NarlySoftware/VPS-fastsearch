"""VPS-FastSearch client library for connecting to the daemon."""

import json
import os
import socket
from typing import Any

from .config import DEFAULT_DB_PATH, DEFAULT_SOCKET_PATH, load_config


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
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send JSON-RPC request and get response."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

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
                    raise FastSearchError(
                        f"Response too large: {length} bytes (max 10MB)"
                    )

                # Receive full response
                response_data = bytearray(length)
                bytes_read = 0
                while bytes_read < length:
                    chunk = self._sock.recv(min(8192, length - bytes_read))
                    if not chunk:
                        raise FastSearchError(
                            "Connection closed while receiving response"
                        )
                    response_data[bytes_read:bytes_read + len(chunk)] = chunk
                    bytes_read += len(chunk)

                response = json.loads(response_data)

                if not isinstance(response, dict):
                    raise FastSearchError(
                        f"Invalid JSON-RPC response: expected dict,"
                        f" got {type(response).__name__}"
                    )

                if "error" in response:
                    error = response["error"]
                    raise FastSearchError(
                        f"RPC error {error.get('code')}: {error.get('message')}"
                    )

                if "result" not in response:
                    raise FastSearchError(
                        "Invalid JSON-RPC response from daemon:"
                        " missing 'result' and 'error'"
                    )

                return dict(response["result"])

            except (TimeoutError, OSError) as e:
                self._disconnect()
                if attempt == 0:
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
    ) -> dict[str, Any]:
        """
        Search indexed documents.

        Args:
            query: Search query text
            db_path: Path to database file
            limit: Maximum results to return
            mode: Search mode (hybrid, bm25, vector)
            rerank: Apply cross-encoder reranking

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
        return self._send_request("search", {
            "query": query,
            "db_path": db_path,
            "limit": limit,
            "mode": mode,
            "rerank": rerank,
        })

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
        return self._send_request("rerank", {
            "query": query,
            "documents": documents,
        })

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

    def __enter__(self) -> "FastSearchClient":
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


# Convenience functions for quick usage
def search(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Quick search using daemon (falls back to direct if unavailable)."""
    try:
        with FastSearchClient(timeout=10.0) as client:
            result = client.search(query, **kwargs)
            return list(result.get("results", []))
    except (DaemonNotRunningError, FastSearchError):
        # Fall back to direct search
        from .core import SearchDB, get_embedder

        db_path = kwargs.get("db_path", os.environ.get("FASTSEARCH_DB", DEFAULT_DB_PATH))
        limit = kwargs.get("limit", 10)
        mode = kwargs.get("mode", "hybrid")
        rerank = kwargs.get("rerank", False)

        db = SearchDB(db_path)
        try:
            if mode == "bm25":
                results = db.search_bm25(query, limit=limit)
            else:
                embedder = get_embedder()
                embedding = embedder.embed_single(query)
                if rerank:
                    results = db.search_hybrid_reranked(
                        query, embedding, limit=limit
                    )
                else:
                    results = db.search_hybrid(query, embedding, limit=limit)
        finally:
            db.close()

        return results


def embed(texts: list[str]) -> list[list[float]]:
    """Quick embed using daemon (falls back to direct if unavailable)."""
    try:
        with FastSearchClient(timeout=10.0) as client:
            result = client.embed(texts)
            return list(result.get("embeddings", []))
    except (DaemonNotRunningError, FastSearchError):
        from .core import get_embedder
        embedder = get_embedder()
        return embedder.embed(texts)
