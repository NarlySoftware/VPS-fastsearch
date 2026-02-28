"""Unit tests for vps_fastsearch.client — no real daemon, no model downloads."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import DUMMY_EMBEDDING, _make_config
from vps_fastsearch.client import (
    DaemonNotRunningError,
    FastSearchClient,
    FastSearchError,
    embed,
    search,
)


def _short_sock_path(name: str) -> str:
    """Return a short socket path in /tmp to avoid the 104-char AF_UNIX limit on macOS."""
    return f"/tmp/{name}.sock"


def _send_length_prefixed(conn: socket.socket, data: bytes) -> None:
    """Helper: send a length-prefixed message over a raw socket."""
    conn.sendall(struct.pack(">I", len(data)))
    conn.sendall(data)


def _recv_length_prefixed(conn: socket.socket) -> bytes:
    """Helper: receive a length-prefixed message from a raw socket."""
    raw_len = b""
    while len(raw_len) < 4:
        chunk = conn.recv(4 - len(raw_len))
        if not chunk:
            raise RuntimeError("Connection closed")
        raw_len += chunk
    length = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < length:
        chunk = conn.recv(min(4096, length - len(data)))
        if not chunk:
            raise RuntimeError("Connection closed")
        data += chunk
    return data


class _FakeServer:
    """A minimal Unix socket server for testing _send_request.

    Accepts one connection, receives one length-prefixed JSON-RPC request,
    and sends back the ``response`` supplied at construction time.
    """

    def __init__(self, socket_path: str, response: dict[str, Any]) -> None:
        self.socket_path = socket_path
        self.response = response
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Remove stale socket file if present
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        self._server_sock.bind(socket_path)
        self._server_sock.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._server_sock.accept()
            with conn:
                _recv_length_prefixed(conn)  # consume the request
                payload = json.dumps(self.response).encode()
                _send_length_prefixed(conn, payload)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._server_sock.close()
        except Exception:
            pass
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# FastSearchClient init
# ---------------------------------------------------------------------------


class TestFastSearchClientInit:
    """Tests for __init__ path resolution."""

    def test_explicit_socket_path_used(self) -> None:
        """When socket_path is supplied, it should be stored directly."""
        client = FastSearchClient(socket_path="/tmp/explicit.sock")
        assert client.socket_path == "/tmp/explicit.sock"

    def test_default_socket_path_from_config(self) -> None:
        """When no socket_path is given, the path comes from config."""
        fake_config = _make_config(socket_path="/tmp/config_provided.sock", include_models=False)
        with patch("vps_fastsearch.client.load_config", return_value=fake_config):
            client = FastSearchClient()
        assert client.socket_path == "/tmp/config_provided.sock"

    def test_custom_timeout_stored(self) -> None:
        """Custom timeout is stored on the instance."""
        client = FastSearchClient(socket_path="/tmp/x.sock", timeout=5.0)
        assert client.timeout == 5.0

    def test_default_timeout(self) -> None:
        """Default timeout is 30.0 seconds."""
        client = FastSearchClient(socket_path="/tmp/x.sock")
        assert client.timeout == 30.0

    def test_no_socket_on_init(self) -> None:
        """Socket is not opened during __init__."""
        client = FastSearchClient(socket_path="/tmp/x.sock")
        assert client._sock is None


# ---------------------------------------------------------------------------
# FastSearchClient._connect
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for the lazy _connect method."""

    def test_connect_missing_socket_raises(self) -> None:
        """_connect raises DaemonNotRunningError when the socket file is absent."""
        client = FastSearchClient(socket_path="/tmp/fastsearch_test_nonexistent_xyz.sock")
        with pytest.raises(DaemonNotRunningError, match="not found"):
            client._connect()

    def test_connect_is_idempotent(self) -> None:
        """Calling _connect twice when already connected should not open a new socket."""
        sock_path = _short_sock_path("fsc_idempotent")
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        server_sock.bind(sock_path)
        server_sock.listen(1)

        client = FastSearchClient(socket_path=sock_path, timeout=2.0)
        client._connect()
        first_sock = client._sock

        # Accept the connection on the server side to prevent blocking
        server_conn, _ = server_sock.accept()

        client._connect()  # Should be a no-op
        assert client._sock is first_sock

        server_conn.close()
        server_sock.close()
        client._disconnect()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# FastSearchClient._disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    """Tests for _disconnect."""

    def test_disconnect_when_not_connected_is_safe(self) -> None:
        """_disconnect() on a fresh client (no socket) must not raise."""
        client = FastSearchClient(socket_path="/tmp/fsc_x.sock")
        client._disconnect()  # Should not raise

    def test_disconnect_clears_sock(self) -> None:
        """After _disconnect(), _sock should be None."""
        sock_path = _short_sock_path("fsc_disconnect")
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        server_sock.bind(sock_path)
        server_sock.listen(1)

        client = FastSearchClient(socket_path=sock_path, timeout=2.0)
        client._connect()
        server_conn, _ = server_sock.accept()

        assert client._sock is not None
        client._disconnect()
        assert client._sock is None

        server_conn.close()
        server_sock.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

    def test_disconnect_idempotent(self) -> None:
        """Calling _disconnect() twice on a fresh client must not raise."""
        client = FastSearchClient(socket_path="/tmp/fsc_x.sock")
        client._disconnect()
        client._disconnect()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for __enter__ / __exit__."""

    def test_context_manager_returns_self(self) -> None:
        """__enter__ returns the client instance itself."""
        client = FastSearchClient(socket_path="/tmp/x.sock")
        result = client.__enter__()
        assert result is client
        client.__exit__(None, None, None)

    def test_context_manager_closes_on_exit(self) -> None:
        """__exit__ should call close() / _disconnect()."""
        client = FastSearchClient(socket_path="/tmp/x.sock")
        mock_sock = MagicMock()
        client._sock = mock_sock

        with client:
            pass  # __exit__ should fire here

        assert client._sock is None
        mock_sock.close.assert_called_once()


# ---------------------------------------------------------------------------
# is_daemon_running
# ---------------------------------------------------------------------------


class TestIsDaemonRunning:
    """Tests for the static is_daemon_running helper."""

    def test_returns_false_when_no_socket(self) -> None:
        """Returns False when the socket file does not exist."""
        assert (
            FastSearchClient.is_daemon_running(socket_path="/tmp/fastsearch_test_absent_xyz.sock")
            is False
        )

    def test_returns_false_when_socket_exists_but_no_listener(self) -> None:
        """Returns False when socket file exists but nothing is listening."""
        sock_path = _short_sock_path("fsc_ghost")
        # Create a socket file without binding a listening server
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        try:
            s.bind(sock_path)
            # Do NOT listen — connect should be refused
            result = FastSearchClient.is_daemon_running(socket_path=sock_path)
            assert result is False
        finally:
            s.close()
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    """Tests for ping()."""

    def test_ping_returns_false_when_not_connected(self) -> None:
        """ping() returns False without a running daemon."""
        client = FastSearchClient(socket_path="/tmp/nonexistent_abc.sock")
        assert client.ping() is False


# ---------------------------------------------------------------------------
# _send_request — size validation
# ---------------------------------------------------------------------------


class TestSendRequestValidation:
    """Tests for _send_request size guard."""

    def test_request_too_large_raises(self) -> None:
        """Requests exceeding 10 MB raise FastSearchError before sending."""
        client = FastSearchClient(socket_path="/tmp/x.sock")
        # Build a large params dict
        large_text = "x" * (10 * 1024 * 1024 + 100)
        with pytest.raises(FastSearchError, match="too large"):
            client._send_request("embed", {"texts": [large_text]})

    def test_none_params_treated_as_empty_dict(self) -> None:
        """params=None should be serialized as {} and produce a valid request."""
        sock_path = _short_sock_path("fsc_none_params")
        fake_response = {"jsonrpc": "2.0", "result": {"pong": True}, "id": 1}
        server = _FakeServer(sock_path, fake_response)

        try:
            client = FastSearchClient(socket_path=sock_path, timeout=5.0)
            result = client._send_request("ping", None)
            assert result == {"pong": True}
        finally:
            client._disconnect()
            server.close()


# ---------------------------------------------------------------------------
# _send_request — response handling
# ---------------------------------------------------------------------------


class TestSendRequestResponse:
    """Tests for _send_request response parsing."""

    def test_rpc_error_response_raises_fast_search_error(self) -> None:
        """A JSON-RPC error response raises FastSearchError."""
        sock_path = _short_sock_path("fsc_rpc_error")
        error_response = {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": 1,
        }
        server = _FakeServer(sock_path, error_response)

        try:
            client = FastSearchClient(socket_path=sock_path, timeout=5.0)
            with pytest.raises(FastSearchError, match="RPC error"):
                client._send_request("unknown_method")
        finally:
            client._disconnect()
            server.close()

    def test_valid_result_returned(self) -> None:
        """A well-formed JSON-RPC result is returned as a dict."""
        sock_path = _short_sock_path("fsc_valid_result")
        ok_response = {
            "jsonrpc": "2.0",
            "result": {"pong": True, "timestamp": 12345.0},
            "id": 1,
        }
        server = _FakeServer(sock_path, ok_response)

        try:
            client = FastSearchClient(socket_path=sock_path, timeout=5.0)
            result = client._send_request("ping")
            assert result["pong"] is True
        finally:
            client._disconnect()
            server.close()


# ---------------------------------------------------------------------------
# Convenience functions: search() and embed()
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level search() and embed() fallback behaviour."""

    def test_search_falls_back_to_direct_when_daemon_unavailable(self) -> None:
        """search() falls back to direct SearchDB path when daemon is down.

        The SearchDB and get_embedder imports inside the fallback path are
        lazy (inside the except block), so we patch them at the core module.
        """
        mock_db = MagicMock()
        mock_db.search_bm25.return_value = [{"content": "hello", "rank": 1}]
        mock_db.close.return_value = None

        with (
            patch("vps_fastsearch.client.FastSearchClient") as MockClient,
            patch("vps_fastsearch.core.SearchDB", return_value=mock_db),
        ):
            MockClient.return_value.__enter__.side_effect = DaemonNotRunningError("no daemon")

            results = search("test query", mode="bm25", db_path="/tmp/fake.db")

        assert isinstance(results, list)

    def test_embed_falls_back_to_direct_when_daemon_unavailable(self) -> None:
        """embed() falls back to direct Embedder path when daemon is down."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [DUMMY_EMBEDDING]

        with (
            patch("vps_fastsearch.client.FastSearchClient") as MockClient,
            patch("vps_fastsearch.core.get_embedder", return_value=mock_embedder),
        ):
            MockClient.return_value.__enter__.side_effect = DaemonNotRunningError("no daemon")

            result = embed(["hello world"])

        assert isinstance(result, list)
        mock_embedder.embed.assert_called_once_with(["hello world"])

    def test_search_returns_list_of_results_from_daemon(self, tmp_path: Any) -> None:
        """search() returns the results list from the daemon response."""
        fake_results = [{"content": "doc1", "rank": 1}, {"content": "doc2", "rank": 2}]

        with patch("vps_fastsearch.client.FastSearchClient") as MockClient:
            instance = MagicMock()
            instance.search.return_value = {"results": fake_results}
            MockClient.return_value.__enter__.return_value = instance

            results = search("my query")

        assert results == fake_results

    def test_embed_returns_embeddings_from_daemon(self) -> None:
        """embed() returns the embeddings list from the daemon response."""
        fake_embeddings = [DUMMY_EMBEDDING, [0.2] * 768]

        with patch("vps_fastsearch.client.FastSearchClient") as MockClient:
            instance = MagicMock()
            instance.embed.return_value = {"embeddings": fake_embeddings}
            MockClient.return_value.__enter__.return_value = instance

            result = embed(["text1", "text2"])

        assert result == fake_embeddings
