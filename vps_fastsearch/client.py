"""VPS-FastSearch client library for connecting to the daemon."""

import json
import socket
import os
from pathlib import Path
from typing import Any

from .config import load_config, DEFAULT_SOCKET_PATH


class FastSearchError(Exception):
    """VPS-FastSearch client error."""
    pass


class DaemonNotRunningError(FastSearchError):
    """Daemon is not running or unreachable."""
    pass


class FastSearchClient:
    """
    Python client for VPS-VPS-FastSearch daemon.
    
    Usage:
        from vps_fastsearch import FastSearchClient
        
        client = FastSearchClient()
        results = client.search("query")
        results = client.search("query", rerank=True)
        status = client.status()
    """
    
    def __init__(
        self,
        socket_path: str | None = None,
        config_path: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize VPS-FastSearch client.
        
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
    
    def _connect(self):
        """Establish connection to daemon."""
        if self._sock is not None:
            return
        
        if not os.path.exists(self.socket_path):
            raise DaemonNotRunningError(f"Daemon socket not found: {self.socket_path}")
        
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout)
            self._sock.connect(self.socket_path)
        except ConnectionRefusedError:
            self._sock = None
            raise DaemonNotRunningError(f"Cannot connect to daemon at {self.socket_path}")
        except Exception as e:
            self._sock = None
            raise FastSearchError(f"Connection error: {e}")
    
    def _disconnect(self):
        """Close connection."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
    
    def _send_request(self, method: str, params: dict[str, Any] = None) -> dict[str, Any]:
        """Send JSON-RPC request and get response."""
        self._connect()
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }
        
        data = json.dumps(request).encode()
        
        try:
            # Send length-prefixed message
            self._sock.sendall(len(data).to_bytes(4, "big"))
            self._sock.sendall(data)
            
            # Receive length-prefixed response
            length_bytes = self._sock.recv(4)
            if not length_bytes:
                raise FastSearchError("Connection closed by daemon")
            
            length = int.from_bytes(length_bytes, "big")
            
            # Receive full response
            response_data = b""
            while len(response_data) < length:
                chunk = self._sock.recv(min(8192, length - len(response_data)))
                if not chunk:
                    raise FastSearchError("Connection closed while receiving response")
                response_data += chunk
            
            response = json.loads(response_data.decode())
            
            if "error" in response:
                error = response["error"]
                raise FastSearchError(f"RPC error {error.get('code')}: {error.get('message')}")
            
            return response.get("result", {})
            
        except (BrokenPipeError, ConnectionResetError) as e:
            self._disconnect()
            raise FastSearchError(f"Connection lost: {e}")
        except json.JSONDecodeError as e:
            raise FastSearchError(f"Invalid response: {e}")
    
    def ping(self) -> bool:
        """Check if daemon is responding."""
        try:
            result = self._send_request("ping")
            return result.get("pong", False)
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
        db_path: str = "fastsearch.db",
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
    
    def close(self):
        """Close the client connection."""
        self._disconnect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @staticmethod
    def is_daemon_running(socket_path: str | None = None) -> bool:
        """Check if daemon is running."""
        if socket_path is None:
            socket_path = DEFAULT_SOCKET_PATH
        
        if not os.path.exists(socket_path):
            return False
        
        try:
            client = FastSearchClient(socket_path=socket_path, timeout=2.0)
            result = client.ping()
            client.close()
            return result
        except Exception:
            return False


# Convenience functions for quick usage
def search(query: str, **kwargs) -> list[dict]:
    """Quick search using daemon (falls back to direct if unavailable)."""
    try:
        client = FastSearchClient(timeout=10.0)
        result = client.search(query, **kwargs)
        client.close()
        return result.get("results", [])
    except DaemonNotRunningError:
        # Fall back to direct search
        from .core import SearchDB, get_embedder
        
        db_path = kwargs.get("db_path", "fastsearch.db")
        limit = kwargs.get("limit", 10)
        
        db = SearchDB(db_path)
        embedder = get_embedder()
        embedding = embedder.embed_single(query)
        
        results = db.search_hybrid(query, embedding, limit=limit)
        db.close()
        
        return results


def embed(texts: list[str]) -> list[list[float]]:
    """Quick embed using daemon (falls back to direct if unavailable)."""
    try:
        client = FastSearchClient(timeout=10.0)
        result = client.embed(texts)
        client.close()
        return result.get("embeddings", [])
    except DaemonNotRunningError:
        from .core import get_embedder
        embedder = get_embedder()
        return embedder.embed(texts)
