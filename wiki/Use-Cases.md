# Use Cases

Real-world examples of FastSearch in action.

---

## 🤖 AI Assistant Memory Search

The primary use case: give your AI assistant instant access to conversation history and documents.

### The Problem

AI assistants like [OpenClaw](https://github.com/clawdbot/clawdbot) need to search through:
- Past conversations
- User preferences
- Documentation
- Notes and memories

API-based embeddings add 200-500ms latency per search. On a budget VPS, you can't afford heavy local models.

### The Solution

```python
from fastsearch import search

def get_relevant_context(user_query: str) -> list[str]:
    """Search assistant memory before responding."""
    results = search(user_query, limit=5)
    return [r["content"] for r in results]

# In your assistant's response flow:
context = get_relevant_context("What's the user's preferred tone?")
# Returns in ~4ms with daemon running
```

### Setup for AI Assistants

```bash
# Index your memory/knowledge files
fastsearch index ~/assistant/memories --glob "*.md"
fastsearch index ~/assistant/docs --glob "*.txt"

# Start daemon for instant searches
fastsearch daemon start --detach
```

---

## 📚 Documentation Search

Build a local search engine for your docs.

```python
from fastsearch import FastSearchClient

client = FastSearchClient()

def search_docs(query: str):
    results = client.search(query, limit=10, rerank=True)
    
    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['source']} (score: {r['score']:.2f})")
        print(f"   {r['content'][:150]}...\n")

# Example
search_docs("how to configure authentication")
```

---

## 🔍 Code Search

Search through code comments and docstrings.

```bash
# Index Python files
fastsearch index ./src --glob "*.py"

# Search for functionality
fastsearch search "parse JSON response"
fastsearch search "handle database connection errors"
```

---

## 📝 Note-Taking Apps

Add semantic search to your notes.

```python
import os
from pathlib import Path
from fastsearch import SearchDB, get_embedder, chunk_text

def index_notes(notes_dir: str, db_path: str = "notes.db"):
    """Index all markdown notes."""
    db = SearchDB(db_path)
    embedder = get_embedder()
    
    for md_file in Path(notes_dir).glob("**/*.md"):
        content = md_file.read_text()
        chunks = list(chunk_text(content, source=str(md_file)))
        
        for i, chunk in enumerate(chunks):
            embedding = embedder.embed_single(chunk)
            db.index_document(str(md_file), i, chunk, embedding)
    
    db.close()
    print(f"Indexed {notes_dir}")

def search_notes(query: str, db_path: str = "notes.db"):
    """Search notes semantically."""
    db = SearchDB(db_path)
    embedder = get_embedder()
    
    embedding = embedder.embed_single(query)
    results = db.search_hybrid(query, embedding, limit=5)
    
    db.close()
    return results
```

---

## 🗄️ Log Analysis

Search through application logs semantically.

```bash
# Index recent logs
fastsearch index /var/log/myapp --glob "*.log"

# Find relevant errors
fastsearch search "connection timeout database" --mode hybrid
fastsearch search "memory allocation failed" --limit 20
```

---

## 💬 Chat History Search

Search through exported chat histories (Slack, Discord, etc.).

```python
from fastsearch import search

# After indexing exported chat JSON/text files
results = search("discussion about deployment strategy")

for r in results:
    print(f"[{r['source']}] {r['content'][:200]}")
```

---

## Tips for All Use Cases

1. **Start the daemon** for any interactive use case
2. **Use reranking** for complex queries where accuracy matters
3. **Use BM25 mode** for exact keyword/error message searches
4. **Batch indexing** is much faster than one-at-a-time
5. **Chunk size matters** — default 500 tokens works well for most cases
