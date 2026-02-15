# Integration Guide

This guide covers integrating VPS-FastSearch with other systems, including Python applications, OpenClaw/Clawdbot, and custom clients.

## Python Integration

### Basic Usage

```python
from vps_fastsearch import FastSearchClient, search, embed

# Option 1: Client object (recommended for multiple operations)
with VPS-FastSearchClient() as client:
    results = client.search("query")
    embeddings = client.embed(["text1", "text2"])

# Option 2: Convenience functions (one-off operations)
results = search("query")
vectors = embed(["text"])
```

### Error Handling

```python
from vps_fastsearch import (
    FastSearchClient,
    DaemonNotRunningError,
    FastSearchError
)

def search_with_fallback(query: str) -> list:
    """Search with graceful fallback."""
    try:
        with VPS-FastSearchClient(timeout=10.0) as client:
            result = client.search(query)
            return result["results"]
    except DaemonNotRunningError:
        # Fall back to direct mode
        from vps_fastsearch import SearchDB, Embedder
        db = SearchDB("vps_fastsearch.db")
        embedder = Embedder()
        embedding = embedder.embed_single(query)
        results = db.search_hybrid(query, embedding)
        db.close()
        return results
    except FastSearchError as e:
        print(f"Search failed: {e}")
        return []
```

### Async Integration

VPS-FastSearch uses blocking I/O. For async applications, use a thread executor:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from vps_fastsearch import FastSearchClient

executor = ThreadPoolExecutor(max_workers=4)

async def async_search(query: str) -> dict:
    """Run search in thread pool."""
    loop = asyncio.get_event_loop()
    
    def _search():
        with VPS-FastSearchClient() as client:
            return client.search(query)
    
    return await loop.run_in_executor(executor, _search)

# Usage
async def main():
    results = await async_search("query")
    print(results)
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from vps_fastsearch import FastSearchClient, DaemonNotRunningError

app = Flask(__name__)

@app.route("/search")
def search():
    query = request.args.get("q", "")
    limit = request.args.get("limit", 10, type=int)
    rerank = request.args.get("rerank", "false").lower() == "true"
    
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    try:
        with VPS-FastSearchClient(timeout=30.0) as client:
            result = client.search(query, limit=limit, rerank=rerank)
            return jsonify(result)
    except DaemonNotRunningError:
        return jsonify({"error": "Search service unavailable"}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from vps_fastsearch import FastSearchClient, DaemonNotRunningError
from typing import List, Optional

app = FastAPI()

class SearchResult(BaseModel):
    id: int
    source: str
    content: str
    rank: int
    rrf_score: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    rerank: bool = Query(False),
):
    try:
        # Use sync client in thread pool
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def _search():
            with VPS-FastSearchClient(timeout=30.0) as client:
                return client.search(q, limit=limit, rerank=rerank)
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, _search)
        
        return SearchResponse(
            query=result["query"],
            results=[SearchResult(**r) for r in result["results"]],
            search_time_ms=result["search_time_ms"],
        )
    except DaemonNotRunningError:
        raise HTTPException(503, "Search service unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## OpenClaw/Clawdbot Integration

VPS-FastSearch integrates seamlessly with Clawdbot for semantic search over memory files, documentation, and knowledge bases.

### Setup

```bash
# Install VPS-FastSearch in Clawdbot environment
pip install vps-fastsearch[rerank]

# Index Clawdbot workspace
cd ~/clawd
vps-fastsearch index . --glob "**/*.md" --db ~/.clawdbot/search.db

# Start daemon
vps-fastsearch daemon start --detach
```

### Clawdbot Tool Configuration

Add to Clawdbot's TOOLS.md:

```markdown
### VPS-FastSearch

- **Database:** ~/.clawdbot/search.db
- **Socket:** /tmp/vps_fastsearch.sock

**Commands:**
- Search: `vps-fastsearch search "query" --limit 5`
- Index new files: `vps-fastsearch index /path/to/file.md`
- Check status: `vps-fastsearch daemon status`
```

### Search Memory Files

```python
#!/usr/bin/env python3
"""Search Clawdbot memory files."""

from pathlib import Path
from vps_fastsearch import FastSearchClient

def search_memory(query: str, limit: int = 5):
    """Search Clawdbot memory files."""
    db_path = Path.home() / ".clawdbot" / "search.db"
    
    with VPS-FastSearchClient() as client:
        result = client.search(
            query=query,
            db_path=str(db_path),
            limit=limit,
            rerank=True,  # Higher accuracy for memory recall
        )
    
    return result["results"]

# Usage
results = search_memory("What did we discuss about the stock database?")
for r in results:
    print(f"- {r['source']}: {r['content'][:100]}...")
```

### Index New Content

```python
#!/usr/bin/env python3
"""Index new content into Clawdbot's search database."""

from pathlib import Path
from vps_fastsearch import SearchDB, FastSearchClient, chunk_markdown

def index_file(file_path: str):
    """Index a file into the search database."""
    db_path = Path.home() / ".clawdbot" / "search.db"
    
    content = Path(file_path).read_text()
    chunks = list(chunk_markdown(content))
    
    if not chunks:
        return 0
    
    # Use daemon for embedding
    with VPS-FastSearchClient() as client:
        texts = [c[0] for c in chunks]
        result = client.embed(texts)
        embeddings = result["embeddings"]
    
    # Index into database
    db = SearchDB(db_path)
    db.delete_source(file_path)  # Remove old version
    
    for i, ((text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
        db.index_document(
            source=file_path,
            chunk_index=i,
            content=text,
            embedding=embedding,
            metadata=metadata,
        )
    
    db.close()
    return len(chunks)

# Usage
chunks = index_file("/path/to/new_memory.md")
print(f"Indexed {chunks} chunks")
```

---

## Unix Socket Protocol

VPS-FastSearch daemon uses JSON-RPC 2.0 over a Unix socket with length-prefixed framing.

### Message Format

```
┌──────────────────┬────────────────────────────────────────────┐
│  Length (4 bytes)│              JSON Body                     │
│   big-endian     │                                            │
└──────────────────┴────────────────────────────────────────────┘
```

### Request Format

```json
{
    "jsonrpc": "2.0",
    "method": "search",
    "params": {
        "query": "example query",
        "db_path": "vps_fastsearch.db",
        "limit": 10,
        "mode": "hybrid",
        "rerank": false
    },
    "id": 1
}
```

### Response Format

**Success:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "query": "example query",
        "mode": "hybrid",
        "reranked": false,
        "search_time_ms": 4.21,
        "results": [...]
    },
    "id": 1
}
```

**Error:**
```json
{
    "jsonrpc": "2.0",
    "error": {
        "code": -32000,
        "message": "Error description"
    },
    "id": 1
}
```

### Available Methods

| Method | Params | Description |
|--------|--------|-------------|
| `ping` | `{}` | Health check |
| `status` | `{}` | Get daemon status |
| `search` | `{query, db_path, limit, mode, rerank}` | Search documents |
| `embed` | `{texts}` | Generate embeddings |
| `rerank` | `{query, documents}` | Rerank documents |
| `load_model` | `{slot}` | Load a model |
| `unload_model` | `{slot}` | Unload a model |
| `reload_config` | `{config_path?}` | Reload configuration |
| `shutdown` | `{}` | Stop daemon |

---

## Building Custom Clients

### Bash Client

```bash
#!/bin/bash
# fastsearch-search.sh - Simple bash client

SOCKET="/tmp/vps_fastsearch.sock"
QUERY="$1"

# Build JSON-RPC request
REQUEST=$(cat <<EOF
{"jsonrpc":"2.0","method":"search","params":{"query":"$QUERY","limit":5},"id":1}
EOF
)

# Get length (4 bytes, big-endian)
LEN=${#REQUEST}
LEN_HEX=$(printf '%08x' $LEN)
LEN_BYTES=$(echo $LEN_HEX | sed 's/../\\x&/g')

# Send request
(echo -ne "$LEN_BYTES"; echo -n "$REQUEST") | nc -U "$SOCKET" | tail -c +5 | jq .
```

### Node.js Client

```javascript
// fastsearch-client.js
const net = require('net');
const path = require('path');

class FastSearchClient {
    constructor(socketPath = '/tmp/vps_fastsearch.sock') {
        this.socketPath = socketPath;
    }

    async request(method, params = {}) {
        return new Promise((resolve, reject) => {
            const client = net.createConnection(this.socketPath);
            const request = JSON.stringify({
                jsonrpc: '2.0',
                method,
                params,
                id: 1
            });

            client.on('connect', () => {
                // Send length-prefixed message
                const lenBuf = Buffer.alloc(4);
                lenBuf.writeUInt32BE(request.length);
                client.write(lenBuf);
                client.write(request);
            });

            let data = Buffer.alloc(0);
            client.on('data', (chunk) => {
                data = Buffer.concat([data, chunk]);
                
                if (data.length >= 4) {
                    const msgLen = data.readUInt32BE(0);
                    if (data.length >= 4 + msgLen) {
                        const response = JSON.parse(data.slice(4, 4 + msgLen));
                        client.end();
                        if (response.error) {
                            reject(new Error(response.error.message));
                        } else {
                            resolve(response.result);
                        }
                    }
                }
            });

            client.on('error', reject);
        });
    }

    async search(query, options = {}) {
        return this.request('search', { query, ...options });
    }

    async embed(texts) {
        return this.request('embed', { texts });
    }

    async status() {
        return this.request('status');
    }
}

// Usage
(async () => {
    const client = new FastSearchClient();
    const results = await client.search('test query');
    console.log(results);
})();
```

### Go Client

```go
// fastsearch/client.go
package fastsearch

import (
    "encoding/binary"
    "encoding/json"
    "fmt"
    "net"
)

type Client struct {
    socketPath string
}

type Request struct {
    JSONRPC string      `json:"jsonrpc"`
    Method  string      `json:"method"`
    Params  interface{} `json:"params"`
    ID      int         `json:"id"`
}

type Response struct {
    JSONRPC string          `json:"jsonrpc"`
    Result  json.RawMessage `json:"result,omitempty"`
    Error   *RPCError       `json:"error,omitempty"`
    ID      int             `json:"id"`
}

type RPCError struct {
    Code    int    `json:"code"`
    Message string `json:"message"`
}

func NewClient(socketPath string) *Client {
    if socketPath == "" {
        socketPath = "/tmp/vps_fastsearch.sock"
    }
    return &Client{socketPath: socketPath}
}

func (c *Client) Call(method string, params interface{}) (json.RawMessage, error) {
    conn, err := net.Dial("unix", c.socketPath)
    if err != nil {
        return nil, err
    }
    defer conn.Close()

    // Build request
    req := Request{
        JSONRPC: "2.0",
        Method:  method,
        Params:  params,
        ID:      1,
    }
    reqBytes, _ := json.Marshal(req)

    // Send length-prefixed
    length := make([]byte, 4)
    binary.BigEndian.PutUint32(length, uint32(len(reqBytes)))
    conn.Write(length)
    conn.Write(reqBytes)

    // Read response
    respLen := make([]byte, 4)
    conn.Read(respLen)
    msgLen := binary.BigEndian.Uint32(respLen)

    respBytes := make([]byte, msgLen)
    conn.Read(respBytes)

    var resp Response
    json.Unmarshal(respBytes, &resp)

    if resp.Error != nil {
        return nil, fmt.Errorf("RPC error: %s", resp.Error.Message)
    }

    return resp.Result, nil
}

func (c *Client) Search(query string, limit int) (json.RawMessage, error) {
    params := map[string]interface{}{
        "query": query,
        "limit": limit,
    }
    return c.Call("search", params)
}
```

---

## Webhook Patterns

### Post-Index Webhook

```python
#!/usr/bin/env python3
"""Index with webhook notification."""

import requests
from vps_fastsearch import FastSearchClient, chunk_markdown
from pathlib import Path

WEBHOOK_URL = "https://your-app.com/webhooks/fastsearch"

def index_with_webhook(file_path: str):
    """Index a file and notify via webhook."""
    content = Path(file_path).read_text()
    chunks = list(chunk_markdown(content))
    
    with VPS-FastSearchClient() as client:
        # Index
        texts = [c[0] for c in chunks]
        embeddings = client.embed(texts)["embeddings"]
        
        # ... index into database ...
        
        # Notify webhook
        requests.post(WEBHOOK_URL, json={
            "event": "indexed",
            "file": file_path,
            "chunks": len(chunks),
        })
```

### Search Event Logging

```python
#!/usr/bin/env python3
"""Log search events for analytics."""

import json
from datetime import datetime
from vps_fastsearch import FastSearchClient

def search_with_logging(query: str, **kwargs):
    """Search with event logging."""
    with VPS-FastSearchClient() as client:
        result = client.search(query, **kwargs)
        
        # Log event
        event = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": len(result["results"]),
            "search_time_ms": result["search_time_ms"],
            "reranked": result.get("reranked", False),
        }
        
        with open("search_events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
        
        return result
```

---

## Health Check Integration

### Kubernetes Liveness Probe

```yaml
# k8s deployment
spec:
  containers:
  - name: fastsearch
    livenessProbe:
      exec:
        command:
        - /bin/sh
        - -c
        - "vps-fastsearch daemon status --json | jq -e '.uptime_seconds > 0'"
      initialDelaySeconds: 30
      periodSeconds: 10
```

### Prometheus Metrics

```python
#!/usr/bin/env python3
"""Export VPS-FastSearch metrics for Prometheus."""

from prometheus_client import start_http_server, Gauge
from vps_fastsearch import FastSearchClient
import time

# Define metrics
uptime = Gauge('fastsearch_uptime_seconds', 'Daemon uptime')
requests = Gauge('fastsearch_requests_total', 'Total requests')
memory = Gauge('fastsearch_memory_mb', 'Memory usage in MB')
models_loaded = Gauge('fastsearch_models_loaded', 'Number of loaded models')

def collect_metrics():
    """Collect metrics from daemon."""
    try:
        with VPS-FastSearchClient(timeout=5.0) as client:
            status = client.status()
            
            uptime.set(status.get('uptime_seconds', 0))
            requests.set(status.get('request_count', 0))
            memory.set(status.get('total_memory_mb', 0))
            models_loaded.set(len(status.get('loaded_models', {})))
    except Exception:
        pass

if __name__ == '__main__':
    start_http_server(9090)
    while True:
        collect_metrics()
        time.sleep(15)
```
