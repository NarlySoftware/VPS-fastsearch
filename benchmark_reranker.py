#!/usr/bin/env python3
"""Benchmark comparing hybrid search with and without cross-encoder reranking."""

import time
from vps_fastsearch import SearchDB, get_embedder, get_reranker

# Test queries with expected top results
TEST_QUERIES = [
    {
        "query": "VMC insider selling stock",
        "expected_keywords": ["VMC", "insider", "sell", "stock", "transaction"],
    },
    {
        "query": "portfolio holdings positions", 
        "expected_keywords": ["portfolio", "holdings", "position", "shares", "ticker"],
    },
    {
        "query": "database views earnings proximity",
        "expected_keywords": ["earnings", "proximity", "database", "view", "days"],
    },
    {
        "query": "VPS-FastSearch memory project ONNX",
        "expected_keywords": ["VPS-FastSearch", "memory", "ONNX", "search", "embed"],
    },
    {
        "query": "technical documentation setup installation",
        "expected_keywords": ["documentation", "setup", "install", "guide", "configuration"],
    },
]


def score_result(content: str, keywords: list[str]) -> float:
    """Score a result based on keyword matches (case-insensitive)."""
    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw.lower() in content_lower)
    return matches / len(keywords)


def benchmark_query(db: SearchDB, embedder, reranker, query_info: dict, limit: int = 5):
    """Benchmark a single query with and without reranking."""
    query = query_info["query"]
    keywords = query_info["expected_keywords"]
    
    # Generate embedding
    embed_start = time.perf_counter()
    embedding = embedder.embed_single(query)
    embed_time = time.perf_counter() - embed_start
    
    # Hybrid search (no reranking)
    hybrid_start = time.perf_counter()
    hybrid_results = db.search_hybrid(query, embedding, limit=limit)
    hybrid_time = time.perf_counter() - hybrid_start
    
    # Hybrid search with reranking
    rerank_start = time.perf_counter()
    reranked_results = db.search_hybrid_reranked(
        query, embedding, limit=limit, rerank_top_k=20, reranker=reranker
    )
    rerank_time = time.perf_counter() - rerank_start
    
    # Score results
    hybrid_scores = [score_result(r["content"], keywords) for r in hybrid_results]
    rerank_scores = [score_result(r["content"], keywords) for r in reranked_results]
    
    # Calculate accuracy metrics
    hybrid_top1 = hybrid_scores[0] if hybrid_scores else 0
    hybrid_avg = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0
    rerank_top1 = rerank_scores[0] if rerank_scores else 0
    rerank_avg = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
    
    return {
        "query": query,
        "embed_time_ms": embed_time * 1000,
        "hybrid_time_ms": hybrid_time * 1000,
        "rerank_time_ms": rerank_time * 1000,
        "hybrid_top1_score": hybrid_top1,
        "hybrid_avg_score": hybrid_avg,
        "rerank_top1_score": rerank_top1,
        "rerank_avg_score": rerank_avg,
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_results,
    }


def print_comparison_table(results: list[dict]):
    """Print a comparison table of results."""
    print("\n" + "=" * 100)
    print("ACCURACY COMPARISON (Top-1 Score = keyword match ratio in top result)")
    print("=" * 100)
    print(f"{'Query':<45} {'Hybrid Top1':>12} {'Rerank Top1':>12} {'Change':>10}")
    print("-" * 100)
    
    total_hybrid = 0
    total_rerank = 0
    
    for r in results:
        query_short = r["query"][:42] + "..." if len(r["query"]) > 45 else r["query"]
        change = r["rerank_top1_score"] - r["hybrid_top1_score"]
        change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        if change > 0:
            change_str = f"\033[92m{change_str}\033[0m"  # Green
        elif change < 0:
            change_str = f"\033[91m{change_str}\033[0m"  # Red
        
        print(f"{query_short:<45} {r['hybrid_top1_score']:>12.2f} {r['rerank_top1_score']:>12.2f} {change_str:>10}")
        total_hybrid += r["hybrid_top1_score"]
        total_rerank += r["rerank_top1_score"]
    
    print("-" * 100)
    avg_hybrid = total_hybrid / len(results)
    avg_rerank = total_rerank / len(results)
    avg_change = avg_rerank - avg_hybrid
    print(f"{'AVERAGE':<45} {avg_hybrid:>12.2f} {avg_rerank:>12.2f} {avg_change:>+10.2f}")
    
    print("\n" + "=" * 100)
    print("SPEED COMPARISON")
    print("=" * 100)
    print(f"{'Query':<45} {'Hybrid (ms)':>12} {'Rerank (ms)':>12} {'Overhead':>10}")
    print("-" * 100)
    
    total_hybrid_time = 0
    total_rerank_time = 0
    
    for r in results:
        query_short = r["query"][:42] + "..." if len(r["query"]) > 45 else r["query"]
        overhead = r["rerank_time_ms"] - r["hybrid_time_ms"]
        print(f"{query_short:<45} {r['hybrid_time_ms']:>12.1f} {r['rerank_time_ms']:>12.1f} {overhead:>+10.1f}")
        total_hybrid_time += r["hybrid_time_ms"]
        total_rerank_time += r["rerank_time_ms"]
    
    print("-" * 100)
    avg_hybrid_time = total_hybrid_time / len(results)
    avg_rerank_time = total_rerank_time / len(results)
    avg_overhead = avg_rerank_time - avg_hybrid_time
    print(f"{'AVERAGE':<45} {avg_hybrid_time:>12.1f} {avg_rerank_time:>12.1f} {avg_overhead:>+10.1f}")


def print_result_changes(results: list[dict]):
    """Print which results changed between hybrid and reranked."""
    print("\n" + "=" * 100)
    print("RESULT CHANGES (comparing top 5 results)")
    print("=" * 100)
    
    for r in results:
        print(f"\n\033[1mQuery: {r['query']}\033[0m")
        
        hybrid_ids = [res["id"] for res in r["hybrid_results"]]
        rerank_ids = [res["id"] for res in r["reranked_results"]]
        
        # Check if order changed
        if hybrid_ids == rerank_ids:
            print("  → No changes (same results, same order)")
            continue
        
        # Find differences
        hybrid_set = set(hybrid_ids)
        rerank_set = set(rerank_ids)
        
        added = rerank_set - hybrid_set
        removed = hybrid_set - rerank_set
        
        if added:
            print(f"  + Added to top-5: {list(added)}")
        if removed:
            print(f"  - Dropped from top-5: {list(removed)}")
        
        # Show reranking effect
        print("  Hybrid order → Reranked order:")
        for i, (h_id, r_id) in enumerate(zip(hybrid_ids[:5], rerank_ids[:5]), 1):
            if h_id != r_id:
                h_content = next((x["content"][:50] for x in r["hybrid_results"] if x["id"] == h_id), "?")
                r_content = next((x["content"][:50] for x in r["reranked_results"] if x["id"] == r_id), "?")
                print(f"    #{i}: {h_id} ({h_content}...) → {r_id} ({r_content}...)")


def main():
    print("Loading models...")
    db = SearchDB("benchmark.db")
    embedder = get_embedder()
    
    # Warm up reranker
    print("Warming up reranker (first load)...")
    reranker = get_reranker()
    _ = reranker.rerank("test", ["warmup doc"])
    print("Models ready.\n")
    
    # Get DB stats
    stats = db.get_stats()
    print(f"Database: {stats['total_chunks']} chunks from {stats['total_sources']} sources")
    print(f"Size: {stats['db_size_mb']} MB\n")
    
    # Run benchmarks
    print("Running benchmarks...")
    results = []
    for query_info in TEST_QUERIES:
        print(f"  Testing: {query_info['query'][:50]}...")
        result = benchmark_query(db, embedder, reranker, query_info)
        results.append(result)
    
    # Print comparisons
    print_comparison_table(results)
    print_result_changes(results)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    avg_hybrid_acc = sum(r["hybrid_top1_score"] for r in results) / len(results)
    avg_rerank_acc = sum(r["rerank_top1_score"] for r in results) / len(results)
    avg_hybrid_time = sum(r["hybrid_time_ms"] for r in results) / len(results)
    avg_rerank_time = sum(r["rerank_time_ms"] for r in results) / len(results)
    
    acc_improvement = ((avg_rerank_acc - avg_hybrid_acc) / avg_hybrid_acc * 100) if avg_hybrid_acc > 0 else 0
    time_overhead = ((avg_rerank_time - avg_hybrid_time) / avg_hybrid_time * 100) if avg_hybrid_time > 0 else 0
    
    print(f"Accuracy improvement: {acc_improvement:+.1f}%")
    print(f"Latency overhead: {time_overhead:+.1f}%")
    print(f"Average hybrid time: {avg_hybrid_time:.1f}ms")
    print(f"Average rerank time: {avg_rerank_time:.1f}ms")
    print(f"Overhead per query: {avg_rerank_time - avg_hybrid_time:.1f}ms")
    
    # Recommendation
    print("\n" + "-" * 50)
    if acc_improvement > 10 and time_overhead < 200:
        print("✅ RECOMMENDATION: Use reranker for accuracy-critical searches")
        print("   Good accuracy gains with acceptable latency overhead.")
    elif acc_improvement > 5:
        print("🤔 RECOMMENDATION: Use reranker selectively")
        print("   Moderate gains - consider for important queries only.")
    else:
        print("⚠️  RECOMMENDATION: Skip reranker for most use cases")
        print("   Gains don't justify the latency cost.")
    
    db.close()


if __name__ == "__main__":
    main()
