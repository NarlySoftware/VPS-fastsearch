"""VPS-FastSearch CLI - Index and search documents with optional daemon mode."""

import sys
import time
from pathlib import Path

import click
import orjson

from .chunker import chunk_markdown, chunk_text
from .core import SearchDB, get_embedder
from .config import load_config, create_default_config, DEFAULT_CONFIG_PATH
from .client import FastSearchClient, DaemonNotRunningError


DEFAULT_DB = "fastsearch.db"


@click.group()
@click.option("--db", default=DEFAULT_DB, help="Database path", envvar="FASTSEARCH_DB")
@click.option("--config", "config_path", default=None, help="Config file path", envvar="FASTSEARCH_CONFIG")
@click.pass_context
def cli(ctx, db, config_path):
    """VPS-FastSearch - Fast memory/vector search for CPU-only VPS."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["config_path"] = config_path


# ============================================================================
# Daemon Commands
# ============================================================================

@cli.group()
def daemon():
    """Manage the VPS-VPS-FastSearch daemon."""
    pass


@daemon.command("start")
@click.option("--detach", "-d", is_flag=True, help="Run in background")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.pass_context
def daemon_start(ctx, detach, config_path):
    """Start the VPS-VPS-FastSearch daemon."""
    from .daemon import run_daemon, get_daemon_status
    
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
def daemon_stop(ctx, config_path):
    """Stop the VPS-VPS-FastSearch daemon."""
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
def daemon_status(ctx, config_path, output_json):
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
        click.echo(f"Memory:         {status.get('total_memory_mb', 0):.0f}MB / {status.get('max_memory_mb', 0)}MB")
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
def daemon_reload(ctx, config_path):
    """Reload daemon configuration without restart."""
    config_path = config_path or ctx.obj.get("config_path")
    
    try:
        client = FastSearchClient(config_path=config_path)
        result = client.reload_config(config_path)
        client.close()
        click.echo("Configuration reloaded")
    except DaemonNotRunningError:
        click.echo("Daemon is not running", err=True)
        sys.exit(1)


# ============================================================================
# Config Commands
# ============================================================================

@cli.group()
def config():
    """Manage configuration."""
    pass


@config.command("init")
@click.option("--path", default=None, help="Config file path")
def config_init(path):
    """Create default configuration file."""
    config_path = create_default_config(path)
    click.echo(f"Created config at: {config_path}")


@config.command("show")
@click.option("--path", default=None, help="Config file path")
def config_show(path):
    """Show current configuration."""
    cfg = load_config(path)
    click.echo(cfg.to_yaml())


@config.command("path")
def config_path():
    """Show default config file path."""
    click.echo(DEFAULT_CONFIG_PATH)


# ============================================================================
# Index Commands
# ============================================================================

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--glob", "-g", default="*.md", help="Glob pattern for directory indexing")
@click.option("--reindex", is_flag=True, help="Delete existing chunks before indexing")
@click.pass_context
def index(ctx, path, glob, reindex):
    """Index a file or directory of documents."""
    path = Path(path)
    db = SearchDB(ctx.obj["db_path"])
    
    # Collect files to index
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob(glob))
        if not files:
            click.echo(f"No files matching '{glob}' in {path}", err=True)
            return
    
    click.echo(f"Indexing {len(files)} file(s)...")
    
    # Try to use daemon for embedding, fall back to direct
    use_daemon = False
    client = None
    
    try:
        client = FastSearchClient(config_path=ctx.obj.get("config_path"), timeout=60.0)
        if client.ping():
            use_daemon = True
            click.echo("Using daemon for embedding...")
    except Exception:
        pass
    
    if not use_daemon:
        click.echo("Loading embedding model...", nl=False)
        t0 = time.perf_counter()
        embedder = get_embedder()
        model_time = time.perf_counter() - t0
        click.echo(f" done ({model_time:.2f}s)")
    
    total_chunks = 0
    total_time = 0
    
    for file_path in files:
        source = str(file_path)
        
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
            result = client.embed(texts)
            embeddings = result.get("embeddings", [])
        else:
            embeddings = embedder.embed(texts)
        
        embed_time = time.perf_counter() - t0
        
        # Index chunks
        t0 = time.perf_counter()
        items = []
        for i, ((text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            items.append((source, i, text, embedding, metadata))
        
        db.index_batch(items)
        index_time = time.perf_counter() - t0
        
        total_chunks += len(chunks)
        total_time += embed_time + index_time
        
        click.echo(
            f"  {file_path.name}: {len(chunks)} chunks "
            f"(embed: {embed_time:.2f}s, index: {index_time:.3f}s)"
        )
    
    if client:
        client.close()
    
    db.close()
    click.echo(f"\nIndexed {total_chunks} chunks in {total_time:.2f}s")


# ============================================================================
# Search Commands
# ============================================================================

@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid")
@click.option("--rerank", "-r", is_flag=True, help="Use cross-encoder reranking")
@click.option("--no-daemon", is_flag=True, help="Force direct mode (no daemon)")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx, query, limit, mode, rerank, no_daemon, output_json):
    """Search indexed documents."""
    db_path = ctx.obj["db_path"]
    config_path = ctx.obj.get("config_path")
    
    # Try daemon first unless --no-daemon
    use_daemon = False
    client = None
    
    if not no_daemon:
        try:
            client = FastSearchClient(config_path=config_path, timeout=30.0)
            if client.ping():
                use_daemon = True
        except Exception:
            pass
    
    t0 = time.perf_counter()
    
    if use_daemon:
        # Use daemon for search
        result = client.search(
            query=query,
            db_path=db_path,
            limit=limit,
            mode=mode,
            rerank=rerank,
        )
        results = result.get("results", [])
        client.close()
    else:
        # Direct search
        db = SearchDB(db_path)
        
        if mode == "bm25":
            results = db.search_bm25(query, limit=limit)
        elif mode == "vector":
            embedder = get_embedder()
            embedding = embedder.embed_single(query)
            results = db.search_vector(embedding, limit=limit)
        else:  # hybrid
            embedder = get_embedder()
            embedding = embedder.embed_single(query)
            
            if rerank:
                results = db.search_hybrid_reranked(
                    query, embedding, limit=limit,
                    rerank_top_k=min(limit * 3, 30),
                )
            else:
                results = db.search_hybrid(query, embedding, limit=limit)
        
        db.close()
    
    search_time = time.perf_counter() - t0
    
    if output_json:
        output = {
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
        click.echo(f"Search: '{query}' ({mode}{rerank_info}{daemon_info}, {search_time*1000:.0f}ms)\n")
        
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
# Stats Commands
# ============================================================================

@cli.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    db_path = Path(ctx.obj["db_path"])
    
    if not db_path.exists():
        click.echo(f"Database not found: {db_path}", err=True)
        return
    
    db = SearchDB(db_path)
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
    
    db.close()


@cli.command()
@click.argument("source")
@click.pass_context
def delete(ctx, source):
    """Delete all chunks from a source file."""
    db = SearchDB(ctx.obj["db_path"])
    
    # Support partial match
    cursor = db._execute("SELECT DISTINCT source FROM docs WHERE source LIKE ?", (f"%{source}%",))
    matches = [row[0] for row in cursor]
    
    if not matches:
        click.echo(f"No sources matching '{source}'", err=True)
        db.close()
        return
    
    if len(matches) > 1:
        click.echo(f"Multiple matches for '{source}':")
        for m in matches:
            click.echo(f"  {m}")
        click.echo("Be more specific.", err=True)
        db.close()
        return
    
    source_path = matches[0]
    deleted = db.delete_source(source_path)
    click.echo(f"Deleted {deleted} chunks from {source_path}")
    
    db.close()


if __name__ == "__main__":
    cli()
