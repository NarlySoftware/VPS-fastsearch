# Getting Started

This guide will have you running semantic searches in under 5 minutes.

## Installation

```bash
pip install fastsearch
```

For reranking support (improves accuracy):
```bash
pip install "fastsearch[rerank]"
```

## Step 1: Index Some Documents

Create a few test files or use existing markdown/text files:

```bash
# Index a single file
fastsearch index README.md

# Index a directory of markdown files
fastsearch index ./docs --glob "*.md"

# Index with a custom database location
fastsearch index ./notes --db ~/mydata/notes.db
```

## Step 2: Search

```bash
# Basic search
fastsearch search "how to configure logging"

# Get more results
fastsearch search "error handling" --limit 10

# Use specific search mode
fastsearch search "exact error message" --mode bm25      # keyword matching
fastsearch search "what does this mean" --mode vector    # semantic meaning
fastsearch search "configure settings" --mode hybrid     # both (default)
```

## Step 3: Start the Daemon (Recommended)

The daemon keeps models loaded in memory for instant searches:

```bash
# Start in background
fastsearch daemon start --detach

# Check it's running
fastsearch daemon status

# Now searches are ~4ms instead of ~850ms
fastsearch search "fast query"
```

## Step 4: Use from Python

```python
from fastsearch import search, embed

# Search (uses daemon if running, otherwise loads models directly)
results = search("how to handle errors", limit=5)

for r in results:
    print(f"{r['score']:.2f} - {r['source']}:{r['chunk_id']}")
    print(f"  {r['content'][:100]}...")
    print()

# Get embeddings for your own use
vectors = embed(["text one", "text two"])
print(f"Embedding dimensions: {len(vectors[0])}")
```

## What's Next?

- **[Use Cases](Use-Cases)** — See real-world examples
- **[Configuration](https://github.com/NarlySoftware/fastsearch/blob/main/docs/CONFIGURATION.md)** — Customize models, memory limits, etc.
- **[Performance Tuning](https://github.com/NarlySoftware/fastsearch/blob/main/docs/PERFORMANCE.md)** — Optimize for your workload

## Quick Reference

| Command | Description |
|---------|-------------|
| `fastsearch index <path>` | Index files |
| `fastsearch search "query"` | Search indexed content |
| `fastsearch daemon start` | Start background daemon |
| `fastsearch daemon status` | Check daemon status |
| `fastsearch daemon stop` | Stop daemon |
| `fastsearch stats` | Show database statistics |
