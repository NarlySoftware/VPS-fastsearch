"""VPS-FastSearch CLI - Index and search documents with optional daemon mode."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
import orjson

from . import __version__
from .chunker import chunk_markdown, chunk_text
from .client import DaemonNotRunningError, FastSearchClient
from .config import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH, create_default_config, load_config
from .core import Embedder, SearchDB

logger = logging.getLogger(__name__)


def _is_qmd_mode() -> bool:
    """Detect if we're being called by OpenClaw as a QMD subprocess."""
    return "QMD_CONFIG_DIR" in os.environ


def _qmd_db_path() -> str:
    """Return the QMD-expected database path: $XDG_CACHE_HOME/qmd/index.sqlite."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(Path.home(), ".cache"))
    db_dir = Path(xdg_cache) / "qmd"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "index.sqlite")


def _qmd_config_path() -> str | None:
    """Return config path for QMD mode.

    OpenClaw sets XDG_CONFIG_HOME to a sandboxed dir where config.yaml
    won't exist. Fall back to the real user config.
    """
    # Check sandboxed config first
    sandboxed = os.environ.get("QMD_CONFIG_DIR") or os.environ.get("XDG_CONFIG_HOME")
    if sandboxed:
        sandboxed_config = Path(sandboxed) / "fastsearch" / "config.yaml"
        if sandboxed_config.exists():
            return str(sandboxed_config)

    # Fall back to real user config
    real_config = Path.home() / ".config" / "fastsearch" / "config.yaml"
    if real_config.exists():
        return str(real_config)

    return None


def _setup_qmd_env() -> None:
    """Set environment variables for QMD sandbox compatibility.

    Points fastembed model cache at the QMD models directory so OpenClaw's
    symlinked models are found.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(Path.home(), ".cache"))
    fastembed_cache = str(Path(xdg_cache) / "qmd")
    # fastembed respects FASTEMBED_CACHE_PATH
    if "FASTEMBED_CACHE_PATH" not in os.environ:
        os.environ["FASTEMBED_CACHE_PATH"] = fastembed_cache


def _get_embedder(config_path: str | None = None) -> Embedder:
    """Get embedder singleton with full config (provider, prefixes, etc.)."""
    cfg = load_config(config_path)
    embedder_config = cfg.models.get("embedder")
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


@click.group()
@click.version_option(version=__version__, prog_name="vps-fastsearch")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database path", envvar="FASTSEARCH_DB")
@click.option(
    "--config", "config_path", default=None, help="Config file path", envvar="FASTSEARCH_CONFIG"
)
@click.pass_context
def cli(ctx: click.Context, db: str, config_path: str | None) -> None:
    """VPS-FastSearch - Fast memory/vector search for CPU-only VPS."""
    ctx.ensure_object(dict)

    # QMD sandbox: override paths when called by OpenClaw
    if _is_qmd_mode():
        _setup_qmd_env()
        # Use QMD db path unless explicitly overridden by --db or FASTSEARCH_DB
        if db == DEFAULT_DB_PATH and "FASTSEARCH_DB" not in os.environ:
            db = _qmd_db_path()
        # Use QMD config path unless explicitly provided
        if config_path is None and "FASTSEARCH_CONFIG" not in os.environ:
            config_path = _qmd_config_path()

    ctx.obj["db_path"] = db
    ctx.obj["config_path"] = config_path

    # Load embedding_dim from config for SearchDB construction
    cfg = load_config(config_path)
    embedder_cfg = cfg.models.get("embedder")
    ctx.obj["embedding_dim"] = embedder_cfg.embedding_dim if embedder_cfg else 768


def _make_searchdb(ctx: click.Context, db_path: str | None = None) -> SearchDB:
    """Create a SearchDB with the configured embedding_dim."""
    return SearchDB(
        db_path or ctx.obj["db_path"],
        embedding_dim=ctx.obj.get("embedding_dim", 768),
    )


# ============================================================================
# Daemon Commands
# ============================================================================


@cli.group()
def daemon() -> None:
    """Manage the VPS-FastSearch daemon."""
    pass


@daemon.command("start")
@click.option("--detach", "-d", is_flag=True, help="Run in background")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.pass_context
def daemon_start(ctx: click.Context, detach: bool, config_path: str | None) -> None:
    """Start the VPS-FastSearch daemon."""
    from .daemon import get_daemon_status, run_daemon

    config_path = config_path or ctx.obj.get("config_path")

    # Check if already running
    status = get_daemon_status(config_path)
    if status:
        click.echo(f"Daemon is already running (uptime: {status['uptime_seconds']:.0f}s)")
        return

    if detach:
        click.echo("Starting VPS-FastSearch daemon in background...")
        run_daemon(config_path=config_path, foreground=False, detach=True)
    else:
        click.echo("Starting VPS-FastSearch daemon (press Ctrl+C to stop)...")
        run_daemon(config_path=config_path, foreground=True, detach=False)


@daemon.command("stop")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.pass_context
def daemon_stop(ctx: click.Context, config_path: str | None) -> None:
    """Stop the VPS-FastSearch daemon."""
    from .daemon import stop_daemon

    config_path = config_path or ctx.obj.get("config_path")

    if stop_daemon(config_path):
        click.echo("VPS-FastSearch daemon stopped")
    else:
        click.echo("Daemon is not running", err=True)


@daemon.command("status")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def daemon_status(ctx: click.Context, config_path: str | None, output_json: bool) -> None:
    """Show daemon status."""
    from .daemon import get_daemon_status

    config_path = config_path or ctx.obj.get("config_path")
    status = get_daemon_status(config_path)

    if not status:
        if output_json:
            click.echo(orjson.dumps({"running": False}).decode())
        else:
            click.echo("Daemon is not running")
        return

    if output_json:
        click.echo(orjson.dumps(status, option=orjson.OPT_INDENT_2).decode())
    else:
        uptime = status.get("uptime_seconds", 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)

        click.echo("VPS-FastSearch Daemon Status")
        click.echo("=" * 40)
        click.echo(f"Uptime:         {hours}h {minutes}m {seconds}s")
        click.echo(f"Requests:       {status.get('request_count', 0)}")
        click.echo(
            f"Memory:         {status.get('total_memory_mb', 0):.0f}MB / {status.get('max_memory_mb', 0)}MB"
        )
        click.echo(f"Socket:         {status.get('socket_path', 'N/A')}")
        click.echo()

        loaded_models = status.get("loaded_models", {})
        if loaded_models:
            click.echo("Loaded Models:")
            for slot, info in loaded_models.items():
                idle = info.get("idle_seconds", 0)
                click.echo(f"  {slot}: {info.get('memory_mb', 0):.0f}MB (idle: {idle:.0f}s)")
        else:
            click.echo("No models loaded")


@daemon.command("reload")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.pass_context
def daemon_reload(ctx: click.Context, config_path: str | None) -> None:
    """Reload daemon configuration without restart."""
    config_path = config_path or ctx.obj.get("config_path")

    try:
        client = FastSearchClient(config_path=config_path)
        try:
            client.reload_config(config_path)
            click.echo("Configuration reloaded")
        finally:
            client.close()
    except DaemonNotRunningError:
        click.echo("Daemon is not running", err=True)
        sys.exit(1)


# ============================================================================
# Model Commands
# ============================================================================


@cli.group()
def model() -> None:
    """Manage embedding models."""
    pass


@model.command("swap")
@click.argument("model_name")
@click.option("--reindex", is_flag=True, help="Re-embed all documents after swapping")
@click.pass_context
def model_swap(ctx: click.Context, model_name: str, reindex: bool) -> None:
    """Swap the embedding model and optionally re-embed all documents.

    Updates the config file, notifies the daemon (if running), and optionally
    re-indexes all documents with the new model.
    """
    config_path = ctx.obj.get("config_path")
    db_path = ctx.obj["db_path"]

    # Load and update config
    cfg = load_config(config_path)
    old_model = cfg.models.get("embedder")
    old_name = old_model.name if old_model else "none"

    if old_name == model_name and not reindex:
        click.echo(f"Already using model '{model_name}'. Use --reindex to force re-embedding.")
        return

    click.echo(f"Swapping model: {old_name} -> {model_name}")

    # Update config file on disk
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if target_path.exists():
        try:
            import yaml

            raw = yaml.safe_load(target_path.read_text()) or {}
        except ImportError:
            raw = {}
    else:
        raw = {}

    raw.setdefault("models", {}).setdefault("embedder", {})["name"] = model_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        target_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
    except ImportError:
        # Fallback: write minimal config
        target_path.write_text(f"models:\n  embedder:\n    name: {model_name}\n")
    click.echo(f"  Config updated: {target_path}")

    # Notify daemon if running
    try:
        client = FastSearchClient(config_path=config_path, timeout=10.0)
        try:
            if client.ping():
                client.unload_model("embedder")
                click.echo("  Daemon: unloaded old model")
                client.reload_config(config_path)
                click.echo("  Daemon: config reloaded")
                client.load_model("embedder")
                click.echo("  Daemon: loaded new model")
        finally:
            client.close()
    except (OSError, ConnectionError, TimeoutError, DaemonNotRunningError):
        click.echo("  Daemon not running (will use new model on next start)")

    if not reindex:
        click.echo(
            "\nModel swapped. Run 'vps-fastsearch model swap "
            f"{model_name} --reindex' to re-embed existing documents."
        )
        return

    # Re-index all documents with the new model
    if not Path(db_path).exists():
        click.echo("\nNo database found — nothing to reindex.")
        return

    click.echo("\nRe-indexing all documents with new model...")

    # Reset the embedder singleton so it picks up the new model
    Embedder._instance = None

    db = _make_searchdb(ctx)
    try:
        # Get all documents
        rows = list(
            db._execute("SELECT id, source, chunk_index, content, metadata FROM chunks ORDER BY id")
        )

        if not rows:
            click.echo("  No documents to reindex.")
            return

        click.echo(f"  {len(rows)} chunks to re-embed")

        # Load new embedder and validate dimensions
        embedder = _get_embedder(config_path)
        click.echo("  Checking new model dimensions...", nl=False)
        test_embedding = embedder.embed(["dimension check"])
        new_dims = len(test_embedding[0])
        click.echo(f" {new_dims}-dim")

        if new_dims != db.EMBEDDING_DIM:
            click.echo(
                f"\n  ERROR: Model '{model_name}' produces {new_dims}-dim embeddings, "
                f"but VPS-FastSearch requires {db.EMBEDDING_DIM}-dim.\n"
                f"  The vector table schema is fixed at {db.EMBEDDING_DIM} dimensions.\n"
                f"  Choose a compatible model (e.g., BAAI/bge-base-en-v1.5 for 768-dim).",
                err=True,
            )
            sys.exit(1)

        # Clear vector table and update embedding dims
        db._execute("DELETE FROM chunks_vec")
        db._execute(
            "INSERT OR REPLACE INTO db_meta (key, value) VALUES ('embedding_dims', ?)",
            (str(new_dims),),
        )

        # Re-embed in batches with progress
        import sqlite_vec

        BATCH_SIZE = 10
        done = 0
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch = rows[batch_start : batch_start + BATCH_SIZE]
            texts = [row[3] for row in batch]  # content column
            embeddings = embedder.embed(texts)

            with db._transaction():
                for (doc_id, _source, _chunk_idx, _content, _metadata), embedding in zip(
                    batch, embeddings, strict=True
                ):
                    db._execute(
                        "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                        (doc_id, sqlite_vec.serialize_float32(embedding)),
                    )

            done += len(batch)
            pct = done * 100 // len(rows)
            click.echo(f"  [{pct:3d}%] {done}/{len(rows)} chunks re-embedded", nl=True)

        click.echo(f"\nDone — {len(rows)} chunks re-embedded with '{model_name}'.")
    finally:
        db.close()


# ============================================================================
# Config Commands
# ============================================================================


@cli.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command("init")
@click.option("--path", default=None, help="Config file path")
def config_init(path: str | None) -> None:
    """Create default configuration file."""
    config_path = create_default_config(path)
    click.echo(f"Created config at: {config_path}")


@config.command("show")
@click.option("--path", default=None, help="Config file path")
def config_show(path: str | None) -> None:
    """Show current configuration."""
    cfg = load_config(path)
    click.echo(cfg.to_yaml())


@config.command("path")
def config_path() -> None:
    """Show default config file path."""
    click.echo(DEFAULT_CONFIG_PATH)


# ============================================================================
# Index Commands
# ============================================================================


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--glob", "-g", default="*.md", help="Glob pattern for directory indexing")
@click.option("--reindex", is_flag=True, help="Delete existing chunks before indexing")
@click.option(
    "--base-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Base directory for relative path storage (default: DB file's parent directory)",
)
@click.option("--strict", is_flag=True, help="Reject files outside base_dir (portable mode)")
@click.pass_context
def index(
    ctx: click.Context, path: str, glob: str, reindex: bool, base_dir: str | None, strict: bool
) -> None:
    """Index a file or directory of documents."""
    index_path = Path(path).resolve()
    db = _make_searchdb(ctx)

    try:
        # Set up base directory for portable relative path storage
        if base_dir is not None:
            db.set_base_dir(base_dir)

        # Collect files to index
        if index_path.is_file():
            files = [index_path]
        else:
            files = list(index_path.glob(glob))
            if not files:
                click.echo(f"No files matching '{glob}' in {index_path}", err=True)
                return

        logger.info("Indexing %d files from %s", len(files), index_path)
        click.echo(f"Indexing {len(files)} file(s)...")

        # Try to use daemon for embedding, fall back to direct
        use_daemon = False
        client = None

        try:
            client = FastSearchClient(config_path=ctx.obj.get("config_path"), timeout=60.0)
            if client.ping():
                use_daemon = True
                click.echo("Using daemon for embedding...")
        except (OSError, ConnectionError, TimeoutError):
            logger.warning("Daemon not available, falling back to direct embedding")

        try:
            if not use_daemon:
                click.echo("Loading embedding model...", nl=False)
                t0 = time.perf_counter()
                embedder = _get_embedder(ctx.obj.get("config_path"))
                model_time = time.perf_counter() - t0
                click.echo(f" done ({model_time:.2f}s)")

            total_chunks = 0
            total_time = 0.0
            skipped_strict = 0

            for file_path in files:
                if strict and not db.is_within_base_dir(file_path):
                    click.echo(
                        f"  Skipping {file_path}: outside base_dir ({db.base_dir})",
                        err=True,
                    )
                    skipped_strict += 1
                    continue

                source = db.to_relative(file_path.resolve())

                # Delete existing if reindexing
                if reindex:
                    deleted = db.delete_source(source)
                    if deleted:
                        click.echo(f"  Deleted {deleted} existing chunks from {file_path.name}")

                # Read file
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    click.echo(f"  Error reading {file_path}: {e}", err=True)
                    continue

                # Chunk based on file type
                if file_path.suffix.lower() == ".md":
                    chunks = list(chunk_markdown(content))
                else:
                    chunks = [(c, {}) for c in chunk_text(content)]

                if not chunks:
                    click.echo(f"  Skipping {file_path.name} (no content)")
                    continue

                # Generate embeddings in safe-sized batches to avoid OOM on ARM64
                EMBED_BATCH_SIZE = 10
                t0 = time.perf_counter()
                texts = [c[0] for c in chunks]
                embeddings: list[list[float]] = []

                for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
                    batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
                    if use_daemon:
                        assert client is not None
                        result = client.embed(batch)
                        embeddings.extend(result.get("embeddings", []))
                    else:
                        embeddings.extend(embedder.embed(batch))

                embed_time = time.perf_counter() - t0

                # Index chunks
                t0 = time.perf_counter()
                items: list[tuple[str, int, str, list[float], dict[str, Any] | None]] = []
                for i, ((text, metadata), embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                ):
                    items.append((source, i, text, embedding, metadata))

                db.index_batch(items)
                index_time = time.perf_counter() - t0

                total_chunks += len(chunks)
                total_time += embed_time + index_time

                click.echo(
                    f"  {file_path.name}: {len(chunks)} chunks "
                    f"(embed: {embed_time:.2f}s, index: {index_time:.3f}s)"
                )

            logger.info("Indexed %d chunks from %d files", total_chunks, len(files))
            click.echo(f"\nIndexed {total_chunks} chunks in {total_time:.2f}s")
            if skipped_strict:
                click.echo(
                    f"Skipped {skipped_strict} file(s) outside base_dir (--strict)",
                    err=True,
                )
        finally:
            if client:
                client.close()
    finally:
        db.close()


# ============================================================================
# Search Helpers
# ============================================================================


def _parse_metadata_filters(filters: tuple[str, ...]) -> dict[str, Any] | None:
    """Parse ``--filter key=value`` CLI options into a metadata filter dict.

    Coerces ``true``/``false`` to booleans, numeric strings to ``int`` or
    ``float``, and leaves everything else as a string.  Returns ``None``
    when *filters* is empty.
    """
    if not filters:
        return None

    metadata_filter: dict[str, Any] = {}
    for f in filters:
        if "=" not in f:
            click.echo(f"Invalid filter format: '{f}' (expected key=value)", err=True)
            sys.exit(1)
        key, value = f.split("=", 1)
        if not key:
            click.echo(f"Invalid filter: empty key in '{f}'", err=True)
            sys.exit(1)
        # Try to parse numeric/boolean values
        if value.lower() == "true":
            metadata_filter[key] = True
        elif value.lower() == "false":
            metadata_filter[key] = False
        else:
            try:
                metadata_filter[key] = int(value)
            except ValueError:
                try:
                    metadata_filter[key] = float(value)
                except ValueError:
                    metadata_filter[key] = value

    return metadata_filter


def _resolve_source_paths(
    results: list[dict[str, Any]], db_path: str, use_daemon: bool, db: Any
) -> None:
    """Resolve relative ``source`` paths to absolute and store as ``source_abs``.

    When *use_daemon* is ``True`` we avoid opening a full ``SearchDB`` (with
    all its PRAGMAs and schema init) just to read ``base_dir``.  Instead we
    query the ``db_meta`` table directly with a lightweight apsw connection.

    When *use_daemon* is ``False``, *db* must be a live ``SearchDB`` instance
    whose ``to_absolute`` method is used.

    Mutates *results* in-place.
    """
    if use_daemon:
        base_dir: Path | None = None
        if Path(db_path).exists():
            import apsw

            _conn = apsw.Connection(str(db_path))
            try:
                _cur = _conn.execute("SELECT value FROM db_meta WHERE key = 'base_dir'")
                _row = _cur.fetchone()
                base_dir = Path(_row[0]) if _row else Path(db_path).parent
            except apsw.SQLError:
                # db_meta table may not exist in older schema
                base_dir = Path(db_path).parent
            finally:
                _conn.close()

        if base_dir is not None:
            for r in results:
                p = Path(r["source"])
                r["source_abs"] = str(p) if p.is_absolute() else str((base_dir / p).resolve())
    else:
        for r in results:
            r["source_abs"] = db.to_absolute(r["source"])


def _display_results(
    results: list[dict[str, Any]],
    query: str,
    mode: str,
    rerank: bool,
    use_daemon: bool,
    search_time: float,
    output_json: bool,
) -> None:
    """Format and print search results to stdout.

    *search_time* is in seconds (converted to milliseconds for display).
    When *output_json* is ``True``, emits a single JSON object; otherwise
    prints a human-readable summary with per-result score details.
    """
    if output_json:
        output: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "reranked": rerank,
            "daemon": use_daemon,
            "search_time_ms": round(search_time * 1000, 2),
            "results": results,
        }
        click.echo(orjson.dumps(output, option=orjson.OPT_INDENT_2).decode())
    else:
        daemon_info = " [daemon]" if use_daemon else ""
        rerank_info = " +rerank" if rerank else ""
        click.echo(
            f"Search: '{query}' ({mode}{rerank_info}{daemon_info}, {search_time * 1000:.0f}ms)\n"
        )

        for r in results:
            source = Path(r["source"]).name
            preview = r["content"][:200].replace("\n", " ")
            if len(r["content"]) > 200:
                preview += "..."

            if rerank and "rerank_score" in r:
                score_info = f"Rerank: {r['rerank_score']:.4f}"
            elif mode == "hybrid":
                score_info = f"RRF: {r['rrf_score']:.4f}"
                if r.get("bm25_rank"):
                    score_info += f", BM25 #{r['bm25_rank']}"
                if r.get("vec_rank"):
                    score_info += f", Vec #{r['vec_rank']}"
            elif mode == "bm25":
                score_info = f"BM25: {r['score']:.2f}"
            else:
                score_info = f"Dist: {r['distance']:.4f}"

            click.echo(f"[{r['rank']}] {source} (chunk {r['chunk_index']}) - {score_info}")
            click.echo(f"    {preview}\n")


# ============================================================================
# Search Command
# ============================================================================


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid")
@click.option("--rerank", "-r", is_flag=True, help="Use cross-encoder reranking")
@click.option("--no-daemon", is_flag=True, help="Force direct mode (no daemon)")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--filter",
    "-f",
    "filters",
    multiple=True,
    help="Metadata filter as key=value (repeatable, AND logic)",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    mode: str,
    rerank: bool,
    no_daemon: bool,
    output_json: bool,
    filters: tuple[str, ...],
) -> None:
    """Search indexed documents."""
    logger.debug("Search query=%r mode=%s limit=%d", query, mode, limit)
    db_path = ctx.obj["db_path"]
    config_path = ctx.obj.get("config_path")
    metadata_filter = _parse_metadata_filters(filters)

    # Try daemon first unless --no-daemon
    use_daemon = False
    client = None

    if not no_daemon:
        try:
            client = FastSearchClient(config_path=config_path, timeout=30.0)
            if client.ping():
                use_daemon = True
        except (OSError, ConnectionError, TimeoutError):
            logger.warning("Daemon not available, falling back to direct search")

    t0 = time.perf_counter()

    try:
        if use_daemon:
            # Use daemon for search
            assert client is not None
            try:
                result = client.search(
                    query=query,
                    db_path=db_path,
                    limit=limit,
                    mode=mode,
                    rerank=rerank,
                    metadata_filter=metadata_filter,
                )
                results = result.get("results", [])
            finally:
                client.close()
                client = None
        else:
            # Direct search
            if not Path(db_path).exists():
                click.echo(
                    f"Database not found at {db_path}. "
                    "Run 'vps-fastsearch index <path>' to create it, "
                    "or start the daemon with 'vps-fastsearch daemon start'.",
                    err=True,
                )
                sys.exit(1)
            db = _make_searchdb(ctx)

            try:
                if mode == "bm25":
                    results = db.search_bm25(query, limit=limit, metadata_filter=metadata_filter)
                elif mode == "vector":
                    embedder = _get_embedder(config_path)
                    embedding = embedder.embed_single(query)
                    results = db.search_vector(
                        embedding, limit=limit, metadata_filter=metadata_filter
                    )
                else:  # hybrid
                    embedder = _get_embedder(config_path)
                    embedding = embedder.embed_single(query)

                    if rerank:
                        try:
                            results = db.search_hybrid_reranked(
                                query,
                                embedding,
                                limit=limit,
                                rerank_top_k=min(limit * 3, 100),
                                metadata_filter=metadata_filter,
                            )
                        except ImportError as e:
                            click.echo(f"Error: {e}", err=True)
                            sys.exit(1)
                    else:
                        results = db.search_hybrid(
                            query,
                            embedding,
                            limit=limit,
                            metadata_filter=metadata_filter,
                        )
            except BaseException:
                db.close()
                raise

            # Keep db reference for path resolution below; close after display.

        search_time = time.perf_counter() - t0

        _resolve_source_paths(results, db_path, use_daemon, db=None if use_daemon else db)

        try:
            _display_results(results, query, mode, rerank, use_daemon, search_time, output_json)
        finally:
            # Close DB connection (only exists in direct/non-daemon path)
            if not use_daemon:
                db.close()
    finally:
        # Clean up client if not closed yet (e.g., ping succeeded but search skipped)
        if client is not None:
            client.close()


# ============================================================================
# Stats Commands
# ============================================================================


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show index statistics."""
    db_path = Path(ctx.obj["db_path"])

    if not db_path.exists():
        click.echo(f"Database not found: {db_path}", err=True)
        return

    db = _make_searchdb(ctx)
    try:
        stats = db.get_stats()

        click.echo(f"Database: {db_path}")
        click.echo(f"Size: {stats['db_size_mb']} MB")
        click.echo(f"Total chunks: {stats['total_chunks']}")
        click.echo(f"Total sources: {stats['total_sources']}")

        if stats["top_sources"]:
            click.echo("\nTop sources by chunks:")
            for s in stats["top_sources"]:
                name = Path(s["source"]).name
                click.echo(f"  {name}: {s['chunks']} chunks")
    finally:
        db.close()


@cli.command()
@click.argument("source", required=False)
@click.option("--id", "doc_id", type=int, default=None, help="Delete a single document by ID")
@click.pass_context
def delete(ctx: click.Context, source: str | None, doc_id: int | None) -> None:
    """Delete documents by source name or by ID.

    Provide a SOURCE name to delete all chunks for that source (supports partial match),
    or use --id to delete a single document by its numeric ID.
    """
    if source is None and doc_id is None:
        click.echo("Provide a SOURCE argument or --id option.", err=True)
        sys.exit(1)

    db = _make_searchdb(ctx)

    try:
        if doc_id is not None:
            found = db.delete_by_id(doc_id)
            if found:
                click.echo(f"Deleted document ID {doc_id}")
            else:
                click.echo(f"No document with ID {doc_id}", err=True)
            return

        assert source is not None  # for type checker

        # Support partial match
        matches = db.find_sources(source)

        if not matches:
            click.echo(f"No sources matching '{source}'", err=True)
            return

        if len(matches) > 1:
            click.echo(f"Multiple matches for '{source}':")
            for m in matches:
                click.echo(f"  {m}")
            click.echo("Be more specific.", err=True)
            return

        source_path = matches[0]
        deleted = db.delete_source(source_path)
        click.echo(f"Deleted {deleted} chunks from {source_path}")
    finally:
        db.close()


@cli.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def list_sources(ctx: click.Context, output_json: bool) -> None:
    """List all indexed sources with chunk counts."""
    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        click.echo(f"Database not found: {db_path}", err=True)
        return

    db = _make_searchdb(ctx)
    try:
        sources = db.list_sources()

        if output_json:
            click.echo(
                orjson.dumps(
                    {"sources": sources, "count": len(sources)}, option=orjson.OPT_INDENT_2
                ).decode()
            )
        else:
            if not sources:
                click.echo("No indexed sources.")
            else:
                click.echo(f"{'Source':<60} {'Chunks':>6}  {'ID Range':>12}")
                click.echo("-" * 82)
                for s in sources:
                    name = s["source"]
                    if len(name) > 58:
                        name = "..." + name[-55:]
                    click.echo(f"{name:<60} {s['chunks']:>6}  {s['min_id']}-{s['max_id']:>6}")
                click.echo(f"\nTotal: {len(sources)} source(s)")
    finally:
        db.close()


@cli.command("migrate-paths")
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying the DB")
@click.option(
    "--base-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Set/override base directory before migration",
)
@click.option(
    "--old-base-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Rebase relative paths: resolve against OLD base dir, re-relativize against current",
)
@click.option(
    "--force",
    is_flag=True,
    help="Allow migration even when paths fall outside base directory",
)
@click.pass_context
def migrate_paths(
    ctx: click.Context,
    dry_run: bool,
    base_dir: str | None,
    old_base_dir: str | None,
    force: bool,
) -> None:
    """Convert absolute or misaligned relative source paths to clean relative paths.

    Without --old-base-dir, converts absolute paths to relative.
    With --old-base-dir, also rebases relative paths that were computed against
    a different base directory (e.g. ../../../ style paths).
    """
    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        click.echo(f"Database not found: {db_path}", err=True)
        sys.exit(1)

    db = _make_searchdb(ctx)
    try:
        if base_dir is not None:
            db.set_base_dir(base_dir)

        click.echo(f"Base directory: {db.base_dir}")
        if old_base_dir is not None:
            click.echo(f"Old base directory: {old_base_dir}")

        # Get all distinct source values
        rows = list(db._execute("SELECT DISTINCT source FROM chunks"))
        sources = [row[0] for row in rows]

        if not sources:
            click.echo("No documents found in database.")
            return

        absolute: list[str] = []
        already_relative: list[str] = []
        for src in sources:
            if Path(src).is_absolute():
                absolute.append(src)
            else:
                already_relative.append(src)

        # Compute migrations for absolute paths
        migrations: list[tuple[str, str]] = []
        collisions: list[tuple[str, str]] = []
        seen_targets: dict[str, str] = {}
        # Track which relative paths will be rewritten (so they don't block collisions)
        rebase_old_paths: set[str] = set()

        # Phase 1: Rebase misaligned relative paths (--old-base-dir)
        if old_base_dir is not None:
            old_base = Path(old_base_dir).resolve()
            dotdot = [s for s in already_relative if s.startswith("..")]
            if dotdot:
                click.echo(f"\nRelative paths to rebase: {len(dotdot)}")
            for rel_src in dotdot:
                # Resolve against old base dir to recover the correct absolute path
                correct_abs = str((old_base / rel_src).resolve())
                new_rel = db.to_relative(correct_abs)
                if new_rel == rel_src:
                    continue  # No change needed
                if new_rel in seen_targets:
                    collisions.append((rel_src, new_rel))
                else:
                    seen_targets[new_rel] = rel_src
                    rebase_old_paths.add(rel_src)
                    migrations.append((rel_src, new_rel))
            # Remove rebased paths from already_relative for collision checks
            already_relative = [s for s in already_relative if s not in rebase_old_paths]

        # Phase 2: Convert absolute paths
        existing_relative = set(already_relative)
        if not absolute and not migrations:
            click.echo(
                f"\nAll {len(sources)} source(s) are already relative. Nothing to migrate."
            )
            return

        for abs_src in absolute:
            rel = db.to_relative(abs_src)
            if rel in existing_relative:
                collisions.append((abs_src, rel))
            elif rel in seen_targets:
                collisions.append((abs_src, rel))
            else:
                seen_targets[rel] = abs_src
                migrations.append((abs_src, rel))

        # Detect paths outside base_dir (produce ../ prefixes)
        outside = [(a, r) for a, r in migrations if r.startswith("..")]
        if outside and not force:
            click.echo(
                f"\nError: {len(outside)} path(s) fall outside base directory "
                f"({db.base_dir}):",
                err=True,
            )
            for abs_src, rel in outside:
                click.echo(f"  {abs_src} -> {rel}", err=True)
            click.echo(
                "\nThis usually means --base-dir is wrong. "
                "Use --force to migrate anyway.",
                err=True,
            )
            sys.exit(1)

        if not migrations and not collisions:
            click.echo(
                f"\nAll {len(sources)} source(s) are already relative. Nothing to migrate."
            )
            return

        # Display plan
        click.echo(f"\nTotal sources: {len(sources)}")
        click.echo(f"To convert: {len(migrations)}")
        click.echo(f"Collisions (skipped): {len(collisions)}")

        if migrations:
            click.echo("\nConversions:")
            for old_src, new_rel in migrations:
                click.echo(f"  {old_src} -> {new_rel}")

        if collisions:
            click.echo("\nCollisions (would overwrite existing):")
            for old_src, new_rel in collisions:
                click.echo(f"  {old_src} -> {new_rel} (SKIPPED)")

        if dry_run:
            click.echo("\n[dry-run] No changes made.")
            return

        # Apply migrations
        converted = 0
        with db._transaction():
            for old_src, new_rel in migrations:
                db._execute(
                    "UPDATE chunks SET source = ? WHERE source = ?",
                    (new_rel, old_src),
                )
                converted += 1

        click.echo(f"\nMigrated {converted} source(s).")
    finally:
        db.close()


# ============================================================================
# QMD Protocol Commands (OpenClaw integration)
# ============================================================================

_COLLECTIONS_META_KEY = "qmd_collections"


def _load_collections(db: SearchDB) -> list[dict[str, str]]:
    """Load registered QMD collections from db_meta."""
    raw = db.get_meta(_COLLECTIONS_META_KEY)
    if raw:
        return list(orjson.loads(raw))
    return []


def _save_collections(db: SearchDB, collections: list[dict[str, str]]) -> None:
    """Save QMD collections to db_meta."""
    db.set_meta(_COLLECTIONS_META_KEY, orjson.dumps(collections).decode())


def _qmd_init_schema(db: SearchDB) -> None:
    """Create QMD-specific tables if they don't exist.

    Called from QMD command handlers before first use. Does not affect
    databases used only via direct CLI or Python client.
    """
    db._execute("""
        CREATE TABLE IF NOT EXISTS documents (
            collection  TEXT NOT NULL,
            path        TEXT NOT NULL,
            hash        TEXT,
            active      INTEGER NOT NULL DEFAULT 1,
            indexed_at  INTEGER,
            UNIQUE(collection, path)
        )
    """)


def _qmd_get_snippet(db: SearchDB, query_text: str, doc_id: int, mode: str) -> str:
    """Get a snippet for a search result using FTS5 snippet() when possible.

    For BM25/hybrid modes, uses FTS5 snippet() to highlight matching terms.
    For pure vector mode, falls back to first 200 chars of content.
    """
    if mode in ("bm25", "hybrid"):
        import re

        tokens = re.findall(r"\w+", query_text)
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            try:
                rows = list(db._execute(
                    """
                    SELECT snippet(chunks_fts, 0, '**', '**', '...', 12)
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND rowid = ?
                    """,
                    (fts_query, doc_id),
                ))
                if rows and rows[0][0]:
                    return str(rows[0][0])
            except Exception:
                pass

    # Fallback: first 200 chars of content
    rows = list(db._execute("SELECT content FROM chunks WHERE id = ?", (doc_id,)))
    if rows:
        content = rows[0][0]
        return str(content[:200])
    return ""


def _qmd_search(
    ctx: click.Context,
    query_text: str,
    limit: int,
    collection: str | None,
    mode: str,
    rerank: bool = False,
) -> None:
    """Shared implementation for QMD search commands."""
    db_path = ctx.obj["db_path"]
    config_path = ctx.obj.get("config_path")

    # Build metadata filter for collection
    metadata_filter: dict[str, Any] | None = None
    if collection:
        metadata_filter = {"collection": collection}

    # Try daemon first
    use_daemon = False
    client = None
    try:
        client = FastSearchClient(config_path=config_path, timeout=30.0)
        if client.ping():
            use_daemon = True
    except (OSError, ConnectionError, TimeoutError):
        pass

    try:
        if use_daemon:
            assert client is not None
            try:
                result = client.search(
                    query=query_text,
                    db_path=db_path,
                    limit=limit,
                    mode=mode,
                    rerank=rerank,
                    metadata_filter=metadata_filter,
                )
                results = result.get("results", [])
            finally:
                client.close()
                client = None
        else:
            if not Path(db_path).exists():
                click.echo("no results found.")
                return
            db = _make_searchdb(ctx)
            try:
                if mode == "bm25":
                    results = db.search_bm25(
                        query_text, limit=limit, metadata_filter=metadata_filter
                    )
                elif mode == "hybrid" and rerank:
                    embedder = _get_embedder(config_path)
                    embedding = embedder.embed_single(query_text)
                    try:
                        results = db.search_hybrid_reranked(
                            query_text,
                            embedding,
                            limit=limit,
                            rerank_top_k=min(limit * 3, 100),
                            metadata_filter=metadata_filter,
                        )
                    except ImportError:
                        # Reranker not installed — fall back to hybrid without rerank
                        results = db.search_hybrid(
                            query_text, embedding, limit=limit,
                            metadata_filter=metadata_filter,
                        )
                elif mode == "vector":
                    embedder = _get_embedder(config_path)
                    embedding = embedder.embed_single(query_text)
                    results = db.search_vector(
                        embedding, limit=limit, metadata_filter=metadata_filter
                    )
                else:  # hybrid without rerank
                    embedder = _get_embedder(config_path)
                    embedding = embedder.embed_single(query_text)
                    results = db.search_hybrid(
                        query_text, embedding, limit=limit, metadata_filter=metadata_filter
                    )
            finally:
                db.close()
    finally:
        if client is not None:
            client.close()

    if not results:
        click.echo("no results found.")
        return

    # Load collections and generate snippets
    collections: list[dict[str, str]] = []
    snippets: dict[int, str] = {}
    if Path(db_path).exists():
        _db = _make_searchdb(ctx)
        try:
            collections = _load_collections(_db)
            for r in results:
                r["_abs_source"] = _db.to_absolute(r["source"])
                snippets[r["id"]] = _qmd_get_snippet(_db, query_text, r["id"], mode)
        finally:
            _db.close()
    else:
        for r in results:
            r["_abs_source"] = r["source"]

    # Format as QMD JSON output
    qmd_results: list[dict[str, Any]] = []
    for r in results:
        # Normalize score to 0-1
        if rerank and "rerank_score" in r:
            # Cross-encoder scores are typically -10 to +10; sigmoid normalize
            import math

            raw = float(r["rerank_score"])
            score = 1.0 / (1.0 + math.exp(-raw))
        elif "rrf_score" in r:
            # RRF scores are small positives; scale relative to top result
            score = float(r["rrf_score"])
        elif "score" in r:
            score = float(r["score"])
        elif "distance" in r:
            score = max(0.0, 1.0 - float(r["distance"]))
        else:
            score = 0.0

        # Resolve file path relative to collection root
        abs_path = Path(r["_abs_source"])
        file_path = str(abs_path)
        result_collection = r.get("metadata", {}).get("collection", collection or "")

        for coll in collections:
            coll_root = Path(coll["path"])
            try:
                file_path = str(abs_path.relative_to(coll_root))
                if not result_collection:
                    result_collection = coll["name"]
                break
            except ValueError:
                continue

        # Build docid: <collection>/<path>:<chunk_index>
        chunk_idx = r.get("chunk_index", 0)
        docid = f"{result_collection}/{file_path}:{chunk_idx}" if result_collection else file_path

        snippet = snippets.get(r["id"], r.get("content", "")[:200])

        qmd_results.append({
            "file": file_path,
            "collection": result_collection,
            "docid": docid,
            "score": round(score, 6),
            "snippet": snippet,
        })

    click.echo(orjson.dumps(qmd_results, option=orjson.OPT_INDENT_2).decode())


@cli.command("query")
@click.argument("query_text")
@click.option("-n", "limit", default=10, help="Number of results")
@click.option("-c", "collection", default=None, help="Collection name filter")
@click.option("--json", "output_json", is_flag=True, hidden=True, help="JSON output (always on)")
@click.pass_context
def qmd_query(
    ctx: click.Context, query_text: str, limit: int, collection: str | None, output_json: bool
) -> None:
    """BM25/keyword search (QMD protocol)."""
    _qmd_search(ctx, query_text, limit, collection, mode="bm25")


@cli.command("vsearch")
@click.argument("query_text")
@click.option("-n", "limit", default=10, help="Number of results")
@click.option("-c", "collection", default=None, help="Collection name filter")
@click.option("--json", "output_json", is_flag=True, hidden=True, help="JSON output (always on)")
@click.pass_context
def qmd_vsearch(
    ctx: click.Context, query_text: str, limit: int, collection: str | None, output_json: bool
) -> None:
    """Hybrid search with cross-encoder reranking (QMD protocol)."""
    _qmd_search(ctx, query_text, limit, collection, mode="hybrid", rerank=True)


@cli.command("vector_search")
@click.argument("query_text")
@click.option("-n", "limit", default=10, help="Number of results")
@click.option("-c", "collection", default=None, help="Collection name filter")
@click.option("--json", "output_json", is_flag=True, hidden=True, help="JSON output (always on)")
@click.pass_context
def qmd_vector_search(
    ctx: click.Context, query_text: str, limit: int, collection: str | None, output_json: bool
) -> None:
    """Vector/semantic search (QMD protocol)."""
    _qmd_search(ctx, query_text, limit, collection, mode="vector")


@cli.command("update")
@click.pass_context
def qmd_update(ctx: click.Context) -> None:
    """Reindex all registered collections (QMD protocol)."""
    import hashlib

    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        return

    db = _make_searchdb(ctx)
    try:
        _qmd_init_schema(db)
        collections = _load_collections(db)
    finally:
        db.close()

    if not collections:
        return

    for coll in collections:
        coll_path = Path(coll["path"]).expanduser()
        pattern = coll.get("pattern", coll.get("mask", "**/*.md"))
        name = coll["name"]

        if not coll_path.is_dir():
            continue

        files = list(coll_path.rglob(pattern))
        if not files:
            continue

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Change detection: skip unchanged files
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            rel_path = str(file_path.relative_to(coll_path))

            db = _make_searchdb(ctx)
            try:
                _qmd_init_schema(db)
                rows = list(db._execute(
                    "SELECT hash FROM documents WHERE collection = ? AND path = ?",
                    (name, rel_path),
                ))
                if rows and rows[0][0] == file_hash:
                    continue  # Unchanged — skip
            finally:
                db.close()

            if file_path.suffix.lower() == ".md":
                chunks = list(chunk_markdown(content))
            else:
                chunks = [(c, {}) for c in chunk_text(content)]

            if not chunks:
                continue

            # Embed: daemon if available, else direct
            try:
                client = FastSearchClient(config_path=ctx.obj.get("config_path"), timeout=60.0)
                try:
                    if client.ping():
                        texts = [c[0] for c in chunks]
                        EMBED_BATCH_SIZE = 10
                        embeddings: list[list[float]] = []
                        for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
                            batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
                            result = client.embed(batch)
                            embeddings.extend(result.get("embeddings", []))
                    else:
                        raise ConnectionError
                finally:
                    client.close()
            except (OSError, ConnectionError, TimeoutError, Exception):
                embedder = _get_embedder(ctx.obj.get("config_path"))
                texts = [c[0] for c in chunks]
                embeddings = []
                EMBED_BATCH_SIZE = 10
                for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
                    batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
                    embeddings.extend(embedder.embed(batch))

            # Index chunks and update documents table
            db = _make_searchdb(ctx)
            try:
                _qmd_init_schema(db)
                source = db.to_relative(file_path.resolve())
                items: list[tuple[str, int, str, list[float], dict[str, Any] | None]] = []
                for i, ((text, metadata), embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                ):
                    meta = dict(metadata) if metadata else {}
                    meta["collection"] = name
                    items.append((source, i, text, embedding, meta))
                db.index_batch(items)

                # Upsert into documents table
                now = int(time.time())
                db._execute(
                    """
                    INSERT INTO documents (collection, path, hash, active, indexed_at)
                    VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT(collection, path) DO UPDATE SET
                        hash = excluded.hash, active = 1, indexed_at = excluded.indexed_at
                    """,
                    (name, rel_path, file_hash, now),
                )
            finally:
                db.close()


@cli.command("embed")
@click.pass_context
def qmd_embed(ctx: click.Context) -> None:
    """Run embedding pass (QMD protocol). No-op — embeddings are generated during update."""
    pass


# ============================================================================
# QMD Collection Management
# ============================================================================


@cli.group()
def collection() -> None:
    """Manage QMD collections (OpenClaw integration)."""
    pass


@collection.command("add")
@click.argument("path")
@click.option("--name", required=True, help="Collection name")
@click.option("--mask", required=True, help="Glob pattern for files")
@click.pass_context
def collection_add(ctx: click.Context, path: str, name: str, mask: str) -> None:
    """Register a collection path for QMD indexing."""
    db = _make_searchdb(ctx)
    try:
        _qmd_init_schema(db)
        collections = _load_collections(db)

        for coll in collections:
            if coll["name"] == name:
                click.echo(f"Collection '{name}' already exists", err=True)
                sys.exit(1)

        collections.append({"name": name, "path": str(Path(path).resolve()), "pattern": mask})
        _save_collections(db, collections)
        click.echo(f"Added collection '{name}': {path} ({mask})")
    finally:
        db.close()


@collection.command("remove")
@click.argument("name")
@click.pass_context
def collection_remove(ctx: click.Context, name: str) -> None:
    """Remove a registered collection."""
    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        click.echo(f"Collection '{name}' not found", err=True)
        sys.exit(1)

    db = _make_searchdb(ctx)
    try:
        _qmd_init_schema(db)
        collections = _load_collections(db)
        new_collections = [c for c in collections if c["name"] != name]

        if len(new_collections) == len(collections):
            click.echo(f"Collection '{name}' does not exist", err=True)
            sys.exit(1)

        # Soft-delete documents for this collection
        db._execute(
            "UPDATE documents SET active = 0 WHERE collection = ?",
            (name,),
        )

        _save_collections(db, new_collections)
        click.echo(f"Removed collection '{name}'")
    finally:
        db.close()


@collection.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def collection_list(ctx: click.Context, output_json: bool) -> None:
    """List registered collections."""
    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        if output_json:
            click.echo("[]")
        else:
            click.echo("No collections registered.")
        return

    db = _make_searchdb(ctx)
    try:
        collections = _load_collections(db)

        if output_json:
            click.echo(orjson.dumps(collections, option=orjson.OPT_INDENT_2).decode())
        else:
            if not collections:
                click.echo("No collections registered.")
            else:
                for coll in collections:
                    pattern = coll.get("pattern", coll.get("mask", ""))
                    click.echo(f"{coll['name']} (qmd://{coll['name']})")
                    click.echo(f"  path: {coll['path']}")
                    click.echo(f"  pattern: {pattern}")
                    click.echo()
    finally:
        db.close()


if __name__ == "__main__":
    cli()
