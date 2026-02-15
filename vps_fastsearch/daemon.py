"""VPS-FastSearch daemon with Unix socket server and model management."""

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

from .config import FastSearchConfig, load_config, ModelConfig

# Configure logging
logger = logging.getLogger("fastsearch.daemon")


@dataclass
class LoadedModel:
    """Tracks a loaded model and its metadata."""
    slot: str
    instance: Any
    loaded_at: float
    last_used: float
    memory_mb: float = 0.0
    
    def touch(self):
        """Update last used timestamp."""
        self.last_used = time.time()


class ModelManager:
    """
    Manages model lifecycle with memory budgets and LRU eviction.
    
    Model slots:
    - embedder: BAAI/bge-base-en-v1.5 (always loaded by default)
    - reranker: cross-encoder/ms-marco-MiniLM (on-demand)
    - summarizer: (future) 7B models
    """
    
    def __init__(self, config: FastSearchConfig):
        self.config = config
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._load_lock = asyncio.Lock()
        self._unload_tasks: dict[str, asyncio.Task] = {}
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def estimate_model_memory(self, slot: str) -> float:
        """Estimate memory for a model slot (MB)."""
        estimates = {
            "embedder": 450,   # bge-base-en-v1.5
            "reranker": 90,    # ms-marco-MiniLM-L-6-v2
            "summarizer": 4000, # 7B model (future)
        }
        return estimates.get(slot, 500)
    
    def _load_model_sync(self, slot: str) -> Any:
        """Synchronously load a model (run in executor)."""
        model_config = self.config.models.get(slot)
        if not model_config:
            raise ValueError(f"Unknown model slot: {slot}")
        
        logger.info(f"Loading model: {slot} ({model_config.name})")
        start = time.perf_counter()
        
        if slot == "embedder":
            from fastembed import TextEmbedding
            instance = TextEmbedding(model_config.name)
        elif slot == "reranker":
            from sentence_transformers import CrossEncoder
            instance = CrossEncoder(model_config.name)
        else:
            raise ValueError(f"Unknown model slot: {slot}")
        
        elapsed = time.perf_counter() - start
        logger.info(f"Loaded {slot} in {elapsed:.2f}s")
        
        return instance
    
    async def load_model(self, slot: str) -> LoadedModel:
        """Load a model into memory."""
        async with self._load_lock:
            # Already loaded?
            if slot in self._models:
                model = self._models[slot]
                model.touch()
                # Move to end for LRU
                self._models.move_to_end(slot)
                return model
            
            # Cancel any pending unload
            if slot in self._unload_tasks:
                self._unload_tasks[slot].cancel()
                del self._unload_tasks[slot]
            
            # Check memory budget
            await self._ensure_memory_budget(slot)
            
            # Load model in thread pool
            loop = asyncio.get_event_loop()
            instance = await loop.run_in_executor(None, self._load_model_sync, slot)
            
            # Create tracking entry
            now = time.time()
            memory_mb = self.estimate_model_memory(slot)
            
            model = LoadedModel(
                slot=slot,
                instance=instance,
                loaded_at=now,
                last_used=now,
                memory_mb=memory_mb,
            )
            
            self._models[slot] = model
            logger.info(f"Model {slot} loaded. Memory: {self.get_memory_usage():.0f}MB")
            
            # Schedule unload if on-demand
            self._schedule_unload(slot)
            
            return model
    
    async def _ensure_memory_budget(self, slot: str):
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
                logger.warning(f"Cannot evict: all models are 'always' loaded")
                break
            
            await self.unload_model(evict_slot)
            current_usage = self.get_memory_usage()
    
    async def unload_model(self, slot: str):
        """Unload a model from memory."""
        if slot not in self._models:
            return
        
        model = self._models[slot]
        
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
    
    def _schedule_unload(self, slot: str):
        """Schedule auto-unload for on-demand models."""
        model_config = self.config.models.get(slot)
        if not model_config:
            return
        
        if model_config.keep_loaded != "on_demand":
            return
        
        timeout = model_config.idle_timeout_seconds
        if timeout <= 0:
            return
        
        async def _delayed_unload():
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
                    "idle_seconds": time.time() - model.last_used,
                }
                for slot, model in self._models.items()
            },
            "total_memory_mb": self.get_memory_usage(),
            "max_memory_mb": self.config.memory.max_ram_mb,
        }
    
    async def shutdown(self):
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


class FastSearchDaemon:
    """
    Unix socket server for VPS-FastSearch operations.
    
    JSON-RPC 2.0 protocol for requests/responses.
    """
    
    def __init__(self, config: FastSearchConfig | None = None):
        self.config = config or load_config()
        self.model_manager = ModelManager(self.config)
        self._server: asyncio.Server | None = None
        self._start_time: float | None = None
        self._request_count = 0
        self._shutdown_event = asyncio.Event()
        
        # Handler registry
        self._handlers: dict[str, Callable] = {
            "ping": self._handle_ping,
            "status": self._handle_status,
            "search": self._handle_search,
            "embed": self._handle_embed,
            "rerank": self._handle_rerank,
            "load_model": self._handle_load_model,
            "unload_model": self._handle_unload_model,
            "reload_config": self._handle_reload_config,
            "shutdown": self._handle_shutdown,
        }
    
    async def _handle_ping(self, params: dict) -> dict:
        """Simple ping handler."""
        return {"pong": True, "timestamp": time.time()}
    
    async def _handle_status(self, params: dict) -> dict:
        """Get daemon status."""
        model_status = self.model_manager.get_status()
        
        return {
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "request_count": self._request_count,
            "socket_path": self.config.daemon.socket_path,
            **model_status,
        }
    
    async def _handle_search(self, params: dict) -> dict:
        """Handle search request."""
        from .core import SearchDB
        
        query = params.get("query")
        if not query:
            raise ValueError("Missing 'query' parameter")
        
        db_path = params.get("db_path", "fastsearch.db")
        limit = params.get("limit", 10)
        mode = params.get("mode", "hybrid")
        rerank = params.get("rerank", False)
        
        # Get or load embedder
        embedder_model = await self.model_manager.load_model("embedder")
        embedder_model.touch()
        
        db = SearchDB(db_path)
        
        try:
            start_time = time.perf_counter()
            
            if mode == "bm25":
                results = db.search_bm25(query, limit=limit)
            elif mode == "vector":
                embedding = list(embedder_model.instance.embed([query]))[0].tolist()
                results = db.search_vector(embedding, limit=limit)
            else:  # hybrid
                embedding = list(embedder_model.instance.embed([query]))[0].tolist()
                
                if rerank:
                    # Load reranker on-demand
                    reranker_model = await self.model_manager.load_model("reranker")
                    reranker_model.touch()
                    
                    results = db.search_hybrid_reranked(
                        query, embedding, limit=limit,
                        rerank_top_k=min(limit * 3, 30),
                    )
                else:
                    results = db.search_hybrid(query, embedding, limit=limit)
            
            search_time = time.perf_counter() - start_time
            
            return {
                "query": query,
                "mode": mode,
                "reranked": rerank,
                "search_time_ms": round(search_time * 1000, 2),
                "results": results,
            }
        finally:
            db.close()
    
    async def _handle_embed(self, params: dict) -> dict:
        """Generate embeddings for texts."""
        texts = params.get("texts", [])
        if not texts:
            raise ValueError("Missing 'texts' parameter")
        
        embedder_model = await self.model_manager.load_model("embedder")
        embedder_model.touch()
        
        start_time = time.perf_counter()
        embeddings = list(embedder_model.instance.embed(texts))
        embed_time = time.perf_counter() - start_time
        
        return {
            "embeddings": [e.tolist() for e in embeddings],
            "count": len(embeddings),
            "embed_time_ms": round(embed_time * 1000, 2),
        }
    
    async def _handle_rerank(self, params: dict) -> dict:
        """Rerank documents against query."""
        query = params.get("query")
        documents = params.get("documents", [])
        
        if not query:
            raise ValueError("Missing 'query' parameter")
        if not documents:
            raise ValueError("Missing 'documents' parameter")
        
        reranker_model = await self.model_manager.load_model("reranker")
        reranker_model.touch()
        
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
    
    async def _handle_load_model(self, params: dict) -> dict:
        """Load a model slot."""
        slot = params.get("slot")
        if not slot:
            raise ValueError("Missing 'slot' parameter")
        
        model = await self.model_manager.load_model(slot)
        
        return {
            "slot": slot,
            "loaded": True,
            "memory_mb": model.memory_mb,
        }
    
    async def _handle_unload_model(self, params: dict) -> dict:
        """Unload a model slot."""
        slot = params.get("slot")
        if not slot:
            raise ValueError("Missing 'slot' parameter")
        
        await self.model_manager.unload_model(slot)
        
        return {
            "slot": slot,
            "unloaded": True,
        }
    
    async def _handle_reload_config(self, params: dict) -> dict:
        """Reload configuration."""
        config_path = params.get("config_path")
        
        new_config = load_config(config_path)
        self.config = new_config
        self.model_manager.config = new_config
        
        return {
            "reloaded": True,
            "socket_path": new_config.daemon.socket_path,
        }
    
    async def _handle_shutdown(self, params: dict) -> dict:
        """Shutdown the daemon."""
        self._shutdown_event.set()
        return {"shutdown": True}
    
    async def _handle_request(self, data: bytes) -> bytes:
        """Process a JSON-RPC request."""
        try:
            request = json.loads(data.decode())
        except json.JSONDecodeError as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": None,
            }).encode()
        
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        self._request_count += 1
        
        if method not in self._handlers:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id,
            }).encode()
        
        try:
            result = await self._handlers[method](params)
            return json.dumps({
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }).encode()
        except Exception as e:
            logger.exception(f"Error handling {method}")
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": request_id,
            }).encode()
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection."""
        try:
            while True:
                # Read length-prefixed message
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, "big")
                
                if length > 10 * 1024 * 1024:  # 10MB limit
                    logger.warning(f"Message too large: {length} bytes")
                    break
                
                data = await reader.readexactly(length)
                response = await self._handle_request(data)
                
                # Send length-prefixed response
                writer.write(len(response).to_bytes(4, "big"))
                writer.write(response)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        except Exception as e:
            logger.exception("Error handling client")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def start(self, foreground: bool = True):
        """Start the daemon server."""
        socket_path = self.config.daemon.socket_path
        
        # Remove existing socket
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        
        # Create server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=socket_path,
        )
        
        # Set socket permissions
        os.chmod(socket_path, 0o600)
        
        self._start_time = time.time()
        
        # Write PID file
        pid_path = self.config.daemon.pid_path
        Path(pid_path).write_text(str(os.getpid()))
        
        logger.info(f"VPS-FastSearch daemon started on {socket_path}")
        
        # Pre-load "always" models
        for slot, model_config in self.config.models.items():
            if model_config.keep_loaded == "always":
                try:
                    await self.model_manager.load_model(slot)
                except Exception as e:
                    logger.error(f"Failed to load {slot}: {e}")
        
        if foreground:
            try:
                # Wait for shutdown signal
                await self._shutdown_event.wait()
            finally:
                await self.stop()
    
    async def stop(self):
        """Stop the daemon server."""
        logger.info("Shutting down VPS-FastSearch daemon...")
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        await self.model_manager.shutdown()
        
        # Clean up socket and PID files
        socket_path = self.config.daemon.socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        
        pid_path = self.config.daemon.pid_path
        if os.path.exists(pid_path):
            os.unlink(pid_path)
        
        logger.info("VPS-FastSearch daemon stopped")


def run_daemon(config_path: str | None = None, foreground: bool = True, detach: bool = False):
    """Run the VPS-FastSearch daemon."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    config = load_config(config_path)
    
    if detach:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent - exit
            print(f"VPS-FastSearch daemon started (PID: {pid})")
            sys.exit(0)
        
        # Child - create new session
        os.setsid()
        
        # Redirect stdio
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    daemon = FastSearchDaemon(config)
    
    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def signal_handler():
        daemon._shutdown_event.set()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
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
        for _ in range(50):  # 5 seconds
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(0.1)
            except ProcessLookupError:
                break
        
        return True
    except (ValueError, ProcessLookupError):
        # Clean up stale PID file
        if os.path.exists(pid_path):
            os.unlink(pid_path)
        return False


def get_daemon_status(config_path: str | None = None) -> dict | None:
    """Get status of running daemon."""
    config = load_config(config_path)
    socket_path = config.daemon.socket_path
    
    if not os.path.exists(socket_path):
        return None
    
    # Try to connect and get status
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(socket_path)
        
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "status",
            "params": {},
            "id": 1,
        }).encode()
        
        sock.sendall(len(request).to_bytes(4, "big"))
        sock.sendall(request)
        
        length_bytes = sock.recv(4)
        length = int.from_bytes(length_bytes, "big")
        response = sock.recv(length)
        
        sock.close()
        
        result = json.loads(response)
        return result.get("result")
    except Exception:
        return None
