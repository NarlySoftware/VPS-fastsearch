"""VPS-FastSearch CLI - Index and search documents with optional daemon mode."""

import logging
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
    ctx.obj["db_path"] = db
    ctx.obj["config_path"] = config_path


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
    db = SearchDB(ctx.obj["db_path"])

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
                embedder = Embedder.get_instance()
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

                # Generate embeddings
                t0 = time.perf_counter()
                texts = [c[0] for c in chunks]

                if use_daemon:
                    assert client is not None
                    result = client.embed(texts)
                    embeddings = result.get("embeddings", [])
                else:
                    embeddings = embedder.embed(texts)

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
            db = SearchDB(db_path)

            try:
                if mode == "bm25":
                    results = db.search_bm25(query, limit=limit, metadata_filter=metadata_filter)
                elif mode == "vector":
                    embedder = Embedder.get_instance()
                    embedding = embedder.embed_single(query)
                    results = db.search_vector(
                        embedding, limit=limit, metadata_filter=metadata_filter
                    )
                else:  # hybrid
                    embedder = Embedder.get_instance()
                    embedding = embedder.embed_single(query)

                    if rerank:
                        try:
                            results = db.search_hybrid_reranked(
                                query,
                                embedding,
                                limit=limit,
                                rerank_top_k=min(limit * 3, 30),
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

    db = SearchDB(db_path)
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

    db = SearchDB(ctx.obj["db_path"])

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

    db = SearchDB(db_path)
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

    db = SearchDB(db_path)
    try:
        if base_dir is not None:
            db.set_base_dir(base_dir)

        click.echo(f"Base directory: {db.base_dir}")
        if old_base_dir is not None:
            click.echo(f"Old base directory: {old_base_dir}")

        # Get all distinct source values
        rows = list(db._execute("SELECT DISTINCT source FROM docs"))
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
                    "UPDATE docs SET source = ? WHERE source = ?",
                    (new_rel, old_src),
                )
                converted += 1

        click.echo(f"\nMigrated {converted} source(s).")
    finally:
        db.close()


if __name__ == "__main__":
    cli()
