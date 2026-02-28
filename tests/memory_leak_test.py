#!/usr/bin/env python3
"""Memory leak test for VPS-FastSearch daemon.

Runs on the VM to determine if the daemon leaks memory over time
with repeated indexing, searching, and deletion of growing data.

Usage:
    ~/fastsearch/.venv/bin/python ~/fastsearch/tests/memory_leak_test.py
"""

import json
import os
import random
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add parent dir so we can import the client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vps_fastsearch.client import FastSearchClient


def get_daemon_pid() -> int | None:
    """Get daemon PID from config-aware PID file path."""
    from vps_fastsearch.config import load_config
    config = load_config()
    pid_path = Path(config.daemon.pid_path)
    if not pid_path.exists():
        return None
    return int(pid_path.read_text().strip())


def get_daemon_rss_kb() -> int | None:
    """Get daemon RSS in KB from PID file."""
    pid = get_daemon_pid()
    if pid is None:
        return None
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def generate_documents(count: int, avg_words: int = 50) -> list[str]:
    """Generate random text documents."""
    words = [
        "search", "vector", "database", "index", "query", "embedding",
        "machine", "learning", "neural", "network", "deep", "model",
        "python", "server", "client", "daemon", "socket", "protocol",
        "memory", "performance", "cache", "buffer", "thread", "async",
        "token", "rank", "fusion", "hybrid", "score", "result",
        "document", "chunk", "text", "content", "source", "file",
        "algorithm", "optimization", "latency", "throughput", "batch",
        "sqlite", "virtual", "table", "column", "row", "insert",
    ]
    docs = []
    for i in range(count):
        length = random.randint(avg_words // 2, avg_words * 2)
        text = " ".join(random.choices(words, k=length))
        docs.append(f"Document {i}: {text}")
    return docs


def write_temp_files(docs: list[str], prefix: str) -> Path:
    """Write documents to temp files for CLI indexing."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f"fastsearch_leak_{prefix}_"))
    for i, doc in enumerate(docs):
        (tmpdir / f"doc_{i:04d}.txt").write_text(doc)
    return tmpdir


def run_search_burst(client: FastSearchClient, queries: list[str], mode: str = "hybrid") -> float:
    """Run a burst of searches and return average time in ms."""
    times = []
    for q in queries:
        for attempt in range(3):
            try:
                result = client.search(q, mode=mode, limit=10)
                times.append(result.get("search_time_ms", 0))
                break
            except Exception as e:
                if "Rate limited" in str(e):
                    time.sleep(0.2 * (attempt + 1))
                else:
                    raise
        time.sleep(0.06)  # Stay under 20 req/s limit
    return sum(times) / len(times) if times else 0


def main() -> None:
    cli = Path.home() / "fastsearch/.venv/bin/vps-fastsearch"
    print("=" * 70)
    print("VPS-FastSearch Memory Leak Test")
    print("=" * 70)

    # Check daemon is running
    rss = get_daemon_rss_kb()
    if rss is None:
        print("ERROR: Daemon not running. Start with: vps-fastsearch daemon start --detach")
        sys.exit(1)

    client = FastSearchClient()
    status = client.status()
    print(f"Daemon uptime: {status.get('uptime_seconds', '?')}s")
    print(f"Baseline RSS: {rss:,} KB ({rss / 1024:.1f} MB)")
    print()

    # Data collection
    checkpoints: list[dict] = []

    def record(phase: str, detail: str = "") -> None:
        rss_now = get_daemon_rss_kb()
        entry = {
            "phase": phase,
            "detail": detail,
            "rss_kb": rss_now,
            "rss_mb": round(rss_now / 1024, 1) if rss_now else None,
            "timestamp": time.time(),
        }
        checkpoints.append(entry)
        label = f"{phase} {detail}".strip()
        print(f"  [{len(checkpoints):3d}] {label:40s} RSS: {rss_now:>8,} KB ({rss_now / 1024:.1f} MB)")

    record("baseline")

    # ── Phase 1: Repeated index+search cycles with growing data ──
    print("\n── Phase 1: Index + Search (20 rounds, growing data) ──")
    search_queries = ["search vector database", "machine learning model", "hybrid rank fusion",
                      "python daemon socket", "memory performance cache", "algorithm optimization"]

    for round_num in range(1, 21):
        doc_count = 20 + round_num * 10  # 30, 40, ..., 220 docs
        docs = generate_documents(doc_count)
        tmpdir = write_temp_files(docs, f"r{round_num}")

        # Index via CLI (uses daemon)
        subprocess.run(
            [str(cli), "index", str(tmpdir), "--reindex"],
            capture_output=True, text=True, timeout=120,
        )

        # Search burst
        avg_ms = run_search_burst(client, search_queries)
        record("round", f"{round_num:2d} indexed={doc_count:3d} docs, search={avg_ms:.0f}ms")

        # Cleanup temp files (but indexed data remains in DB)
        for f in tmpdir.iterdir():
            f.unlink()
        tmpdir.rmdir()

    # ── Phase 2: Search-only workload (no indexing) ──
    print("\n── Phase 2: Search-only (10 rounds, 50 searches each) ──")
    for round_num in range(1, 11):
        queries = random.choices(search_queries, k=50)
        avg_ms = run_search_burst(client, queries)
        record("search-only", f"{round_num:2d} (50 queries, avg={avg_ms:.0f}ms)")

    # ── Phase 3: Delete all indexed data ──
    print("\n── Phase 3: Delete indexed data ──")
    record("pre-delete")

    # Get all sources
    stats_result = subprocess.run(
        [str(cli), "stats"], capture_output=True, text=True, timeout=30,
    )
    print(f"  Stats before delete: {stats_result.stdout.strip()[:200]}")

    # Delete via CLI — get sources from DB
    db_path = Path.home() / ".local/share/fastsearch/fastsearch.db"
    if db_path.exists():
        # Use the CLI delete for each source prefix
        import glob as glob_mod
        # Just delete all sources by finding them
        try:
            import apsw
            conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
            sources = [row[0] for row in conn.cursor().execute("SELECT DISTINCT source FROM documents")]
            conn.close()
            print(f"  Found {len(sources)} sources to delete")
            for src in sources:
                subprocess.run(
                    [str(cli), "delete", src],
                    capture_output=True, text=True, timeout=30,
                )
        except Exception as e:
            print(f"  Could not enumerate sources: {e}")

    record("post-delete")

    # ── Phase 4: Post-delete search (should return empty) ──
    print("\n── Phase 4: Post-delete search + re-index ──")
    for round_num in range(1, 6):
        avg_ms = run_search_burst(client, search_queries)
        record("post-del-search", f"{round_num} (avg={avg_ms:.0f}ms)")

    # Re-index a small set
    docs = generate_documents(50)
    tmpdir = write_temp_files(docs, "reindex")
    subprocess.run(
        [str(cli), "index", str(tmpdir), "--reindex"],
        capture_output=True, text=True, timeout=120,
    )
    record("re-indexed", "50 docs")
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()

    # Final searches
    for round_num in range(1, 6):
        avg_ms = run_search_burst(client, search_queries)
        record("final-search", f"{round_num} (avg={avg_ms:.0f}ms)")

    record("final")

    # ── Analysis ──
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline_rss = checkpoints[0]["rss_kb"]
    peak_rss = max(c["rss_kb"] for c in checkpoints if c["rss_kb"])
    final_rss = checkpoints[-1]["rss_kb"]
    post_delete_entries = [c for c in checkpoints if c["phase"] == "post-delete"]
    post_delete_rss = post_delete_entries[0]["rss_kb"] if post_delete_entries else None

    print(f"Baseline RSS:     {baseline_rss:>8,} KB ({baseline_rss / 1024:.1f} MB)")
    print(f"Peak RSS:         {peak_rss:>8,} KB ({peak_rss / 1024:.1f} MB)")
    print(f"Post-delete RSS:  {post_delete_rss:>8,} KB ({post_delete_rss / 1024:.1f} MB)" if post_delete_rss else "Post-delete RSS:  N/A")
    print(f"Final RSS:        {final_rss:>8,} KB ({final_rss / 1024:.1f} MB)")
    print(f"Growth from base: {final_rss - baseline_rss:>+8,} KB ({(final_rss - baseline_rss) / 1024:+.1f} MB)")
    print(f"Peak growth:      {peak_rss - baseline_rss:>+8,} KB ({(peak_rss - baseline_rss) / 1024:+.1f} MB)")

    # Check for monotonic growth in Phase 1 (leak indicator)
    phase1 = [c for c in checkpoints if c["phase"] == "round"]
    if len(phase1) >= 5:
        # Compare first 5 vs last 5 averages
        first5_avg = sum(c["rss_kb"] for c in phase1[:5]) / 5
        last5_avg = sum(c["rss_kb"] for c in phase1[-5:]) / 5
        growth_pct = ((last5_avg - first5_avg) / first5_avg) * 100
        print(f"\nPhase 1 trend:    first 5 avg={first5_avg / 1024:.1f} MB, last 5 avg={last5_avg / 1024:.1f} MB ({growth_pct:+.1f}%)")

        # Count consecutive increases
        increases = sum(1 for i in range(1, len(phase1)) if phase1[i]["rss_kb"] > phase1[i - 1]["rss_kb"])
        print(f"Consecutive rises: {increases}/{len(phase1) - 1}")

    # Memory returned after delete?
    if post_delete_rss and baseline_rss:
        returned = peak_rss - post_delete_rss
        print(f"\nMemory returned after delete: {returned:>+,} KB ({returned / 1024:+.1f} MB)")
        if post_delete_rss <= baseline_rss * 1.2:
            print("VERDICT: Memory returns to near baseline after deletion — no significant leak")
        else:
            retained = post_delete_rss - baseline_rss
            print(f"VERDICT: {retained:,} KB retained after delete — possible leak")

    # Overall verdict
    print()
    if (final_rss - baseline_rss) < baseline_rss * 0.3:
        print("OVERALL: RSS growth < 30% of baseline — daemon appears stable")
        print("RECOMMENDATION: No periodic restart needed")
    elif (final_rss - baseline_rss) < baseline_rss * 0.5:
        print("OVERALL: RSS growth 30-50% of baseline — moderate growth")
        print("RECOMMENDATION: Monitor in production; restart weekly as precaution")
    else:
        print("OVERALL: RSS growth > 50% of baseline — significant growth detected")
        print("RECOMMENDATION: Investigate leak sources; consider periodic restart")

    # Save raw data
    out_path = Path.home() / "fastsearch_memleak_results.json"
    out_path.write_text(json.dumps(checkpoints, indent=2))
    print(f"\nRaw data saved to: {out_path}")


if __name__ == "__main__":
    main()
