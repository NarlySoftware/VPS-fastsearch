# OpenClaw Integration Guide for VPS-FastSearch

Drop-in replacement for QMD's semantic search backend. One config change, sub-second memory search.

## TL;DR (Quick Start)

```bash
# Install
python3 -m venv ~/.openclaw/fastsearch-venv
~/.openclaw/fastsearch-venv/bin/pip install git+https://github.com/NarlySoftware/VPS-fastsearch.git

# Download the QMD wrapper
mkdir -p ~/.openclaw/bin
curl -fsSL https://raw.githubusercontent.com/NarlySoftware/VPS-fastsearch/main/docs/qmd-wrapper.py \
  -o ~/.openclaw/bin/fastsearch
chmod +x ~/.openclaw/bin/fastsearch

# Configure OpenClaw
openclaw config set memory.qmd.command ~/.openclaw/bin/fastsearch
openclaw config set memory.qmd.limits.timeoutMs 10000

# Index your memory
~/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db ~/.cache/fastsearch/index.db \
  index ~/.openclaw/workspace/memory --glob "*.md"

# Start daemon (optional but 2600x faster)
~/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db ~/.cache/fastsearch/index.db daemon start --detach

# Restart OpenClaw
openclaw gateway restart
```

---

## Why

OpenClaw's default memory search (QMD) spawns a fresh process per query, loading 2.1GB of GGUF models through node-llama-cpp each time. On a typical VPS:

| Backend | Search Latency | Model Load | RAM (idle) |
|---------|---------------|------------|------------|
| QMD (query) | **82 seconds** | 3 models, 2.1GB | 0 (loads/dumps each time) |
| QMD (BM25 only) | 0.3s | None | 0 |
| **VPS-FastSearch (daemon)** | **30ms** | 1 model, 110MB ONNX | ~480MB |
| **VPS-FastSearch (cold)** | **3-4s** | 1 model, 110MB ONNX | 0 |

## Prerequisites

- Python 3.10+
- `python3-venv` package (`sudo apt install python3-venv` or `python3.XX-venv`)
- OpenClaw installed and running
- **VPS-FastSearch >= 0.2.1** (includes critical memory and FTS5 fixes)

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv ~/.openclaw/fastsearch-venv
```

### 2. Install VPS-FastSearch

```bash
~/.openclaw/fastsearch-venv/bin/pip install git+https://github.com/NarlySoftware/VPS-fastsearch.git
```

Verify installation:

```bash
~/.openclaw/fastsearch-venv/bin/vps-fastsearch --version
# Should show: vps-fastsearch 0.2.1 or higher
```

### 3. Create the QMD-compatible wrapper

OpenClaw's memory system calls QMD CLI commands (`query`, `search`, `collection add`, etc.). This wrapper translates those into VPS-FastSearch calls with matching JSON output.

Create `~/.openclaw/bin/fastsearch`:

```python
#!/usr/bin/env python3
"""
QMD-compatible wrapper for VPS-FastSearch.
Translates QMD CLI commands into VPS-FastSearch calls.
OpenClaw calls this as if it were QMD.
"""
import glob as globmod
import hashlib
import json
import os
import re
import subprocess
import sys

VENV = os.path.expanduser("~/.openclaw/fastsearch-venv")
FASTSEARCH = os.path.join(VENV, "bin", "vps-fastsearch")
DB_PATH = os.path.expanduser("~/.cache/fastsearch/index.db")
COLLECTIONS_FILE = os.path.expanduser("~/.config/fastsearch/collections.json")


def run_fastsearch(args, capture=True):
    """Run vps-fastsearch with the venv Python."""
    cmd = [FASTSEARCH, "--db", DB_PATH] + args
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = VENV
    env["PATH"] = os.path.join(VENV, "bin") + ":" + env.get("PATH", "")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return result.stdout, result.stderr, result.returncode
    else:
        return subprocess.run(cmd, env=env).returncode


def docid_hash(filepath):
    """Generate a short document ID from filepath."""
    return hashlib.md5(filepath.encode()).hexdigest()[:6]


def load_collections():
    """Load registered collections from config."""
    if os.path.exists(COLLECTIONS_FILE):
        with open(COLLECTIONS_FILE) as f:
            return json.load(f)
    return {}


def save_collections(data):
    """Save collections to config."""
    os.makedirs(os.path.dirname(COLLECTIONS_FILE), exist_ok=True)
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def make_qmd_path(source, collections):
    """Convert absolute file path to qmd:// URI."""
    for name, info in collections.items():
        base = info["path"]
        if source.startswith(base):
            rel = os.path.relpath(source, base)
            return f"qmd://{name}/{rel}"
    return f"qmd://unknown/{os.path.basename(source)}"


def extract_title(content):
    """Extract first heading as title."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def format_snippet(content, query):
    """Format snippet in QMD style with context markers."""
    lines = content.split("\n")
    query_terms = query.lower().split()
    best_line = 0
    best_score = -1
    for i, line in enumerate(lines):
        lower = line.lower()
        score = sum(1 for t in query_terms if t in lower)
        if score > best_score:
            best_score = score
            best_line = i
    start = max(0, best_line)
    end = min(len(lines), start + 4)
    before = start
    after = len(lines) - end
    header = f"@@ -{start + 1},{end - start} @@ ({before} before, {after} after)"
    return header + "\n" + "\n".join(lines[start:end])


def convert_results(raw_json, query, mode="hybrid"):
    """Convert VPS-FastSearch JSON output to QMD format."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return "[]"

    results = data.get("results", [])
    collections = load_collections()

    qmd_results = []
    for r in results:
        source = r.get("source", "")
        content = r.get("content", "")
        score = r.get("rrf_score", r.get("distance", r.get("score", 0)))
        qmd_results.append({
            "docid": "#" + docid_hash(source),
            "score": round(abs(score) if score else 0, 4),
            "file": make_qmd_path(source, collections),
            "title": extract_title(content) or r.get("metadata", {}).get("section", os.path.basename(source)),
            "snippet": format_snippet(content, query),
        })
    return json.dumps(qmd_results, indent=2)


def parse_search_args(args):
    """Parse common search arguments."""
    text = ""
    limit = 10
    json_output = False
    i = 0
    while i < len(args):
        if args[i] == "--json":
            json_output = True
        elif args[i] == "-n" and i + 1 < len(args):
            i += 1
            limit = int(args[i])
        elif args[i] in ("-c", "--collection", "--min-score"):
            i += 1  # skip value
        elif args[i] == "--all":
            limit = 500
        elif not args[i].startswith("-"):
            text = args[i] if not text else text + " " + args[i]
        i += 1
    return text, limit, json_output


def cmd_search(args, mode="hybrid"):
    """Handle search commands."""
    text, limit, json_output = parse_search_args(args)
    if not text:
        print("[]" if json_output else "No query provided.")
        return

    stdout, stderr, rc = run_fastsearch([
        "search", text, "--mode", mode, "--json", "-n", str(limit)
    ])

    if rc != 0 and mode == "hybrid":
        # Fallback to BM25 if hybrid fails
        stdout, stderr, rc = run_fastsearch([
            "search", text, "--mode", "bm25", "--json", "-n", str(limit)
        ])

    if json_output:
        print(convert_results(stdout, text, mode) if stdout.strip() else "[]")
    else:
        print(stdout if stdout else "No results found.")


def cmd_collection(args):
    """Handle collection commands."""
    if not args:
        return
    sub = args[0]

    if sub == "list":
        for name, info in load_collections().items():
            print(f"  {name}: {info['path']} (pattern: {info['pattern']}, files: {info.get('file_count', '?')})")
        return

    if sub == "add":
        path = name = None
        pattern = "**/*.md"
        i = 1
        while i < len(args):
            if args[i] == "--name" and i + 1 < len(args):
                i += 1
                name = args[i]
            elif args[i] == "--mask" and i + 1 < len(args):
                i += 1
                pattern = args[i]
            elif not args[i].startswith("-"):
                path = os.path.abspath(args[i])
            i += 1
        if path and name:
            colls = load_collections()
            if name not in colls:
                colls[name] = {"path": path, "pattern": pattern, "file_count": 0}
                save_collections(colls)
            print(f"Added collection '{name}': {path} ({pattern})")
        return

    if sub == "remove" and len(args) > 1:
        colls = load_collections()
        if args[1] in colls:
            del colls[args[1]]
            save_collections(colls)
        return

    if sub == "rename" and len(args) > 2:
        colls = load_collections()
        if args[1] in colls:
            colls[args[2]] = colls.pop(args[1])
            save_collections(colls)
        return


def cmd_update(args):
    """Re-scan and re-index all collections."""
    collections = load_collections()
    for name, info in collections.items():
        full_pattern = os.path.join(info["path"], info["pattern"])
        files = globmod.glob(full_pattern, recursive=True)
        if files:
            stdout, stderr, rc = run_fastsearch([
                "index", info["path"],
                "--glob", info["pattern"].replace("**/", ""),
                "--reindex"
            ])
            if stdout:
                print(stdout, end="")
        info["file_count"] = len(files)
    save_collections(collections)


def cmd_status(args):
    """Show status information."""
    stdout, _, _ = run_fastsearch(["stats"])
    if stdout:
        print("FastSearch Status (QMD-compatible)\n")
        print(stdout)
    colls = load_collections()
    if colls:
        print("\nCollections:")
        for name, info in colls.items():
            print(f"  {name} ({info['path']})")
            print(f"    Pattern: {info['pattern']}")
            print(f"    Files: {info.get('file_count', '?')}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h", "help"):
        print("FastSearch (QMD-compatible wrapper)\n")
        print("Commands: query, search, vsearch, collection, update, embed, status")
        return

    cmd, rest = args[0], args[1:]

    commands = {
        "query": lambda: cmd_search(rest, "hybrid"),
        "search": lambda: cmd_search(rest, "bm25"),
        "vsearch": lambda: cmd_search(rest, "vector"),
        "collection": lambda: cmd_collection(rest),
        "update": lambda: cmd_update(rest),
        "embed": lambda: cmd_update(rest),  # embedding happens during index
        "status": lambda: cmd_status(rest),
    }

    if cmd == "mcp":
        if "--daemon" in rest or "start" in rest:
            run_fastsearch(["daemon", "start", "--detach"], capture=False)
        elif "stop" in rest:
            run_fastsearch(["daemon", "stop"], capture=False)
        return

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
mkdir -p ~/.openclaw/bin
chmod +x ~/.openclaw/bin/fastsearch
```

### 4. Configure OpenClaw

One config change — tell OpenClaw to use your wrapper instead of QMD:

```json
{
  "memory": {
    "backend": "qmd",
    "qmd": {
      "command": "~/.openclaw/bin/fastsearch",
      "limits": {
        "timeoutMs": 10000
      }
    }
  }
}
```

Or via CLI:

```bash
openclaw config set memory.qmd.command ~/.openclaw/bin/fastsearch
openclaw config set memory.qmd.limits.timeoutMs 10000
openclaw gateway restart
```

OpenClaw still thinks it's talking to QMD — the wrapper translates commands and matches the expected JSON output format.

### 5. Index your memory files

```bash
# Index your workspace memory files
~/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db ~/.cache/fastsearch/index.db \
  index ~/.openclaw/workspace/memory --glob "*.md"

# Index MEMORY.md
~/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db ~/.cache/fastsearch/index.db \
  index ~/.openclaw/workspace/MEMORY.md

# Register collections so qmd:// paths resolve correctly
~/.openclaw/bin/fastsearch collection add ~/.openclaw/workspace/memory \
  --name memory-dir --mask "**/*.md"
~/.openclaw/bin/fastsearch collection add ~/.openclaw/workspace \
  --name memory-root --mask "MEMORY.md"
```

Verify indexing:

```bash
~/.openclaw/fastsearch-venv/bin/vps-fastsearch --db ~/.cache/fastsearch/index.db stats
# Should show document count > 0
```

### 6. Start the daemon (recommended)

```bash
~/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db ~/.cache/fastsearch/index.db \
  daemon start --detach
```

Check status:

```bash
~/.openclaw/fastsearch-venv/bin/vps-fastsearch daemon status
```

Expected output:

```
VPS-FastSearch Daemon Status
========================================
Uptime:         0h 5m 12s
Requests:       15
Memory:         480MB / 4000MB
Socket:         /tmp/fastsearch.sock

Loaded Models:
  embedder: 450MB (idle: 3s)
```

### 7. Test the integration

```bash
# Test search through the wrapper
~/.openclaw/bin/fastsearch query "test query" --json -n 5

# Should return QMD-format JSON with results
```

### 8. Optional: systemd service for auto-start

Create `/etc/systemd/system/fastsearch-daemon.service`:

```ini
[Unit]
Description=VPS-FastSearch Daemon
After=network.target

[Service]
Type=simple
User=%USER%
ExecStart=%HOME%/.openclaw/fastsearch-venv/bin/vps-fastsearch \
  --db %HOME%/.cache/fastsearch/index.db daemon start
Restart=on-failure
RestartSec=5
Environment="HOME=%HOME%"

[Install]
WantedBy=multi-user.target
```

Replace `%USER%` and `%HOME%` with your actual username and home directory, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fastsearch-daemon
sudo systemctl start fastsearch-daemon
```

---

## How It Works

```
OpenClaw (memory_search tool)
  │
  │  spawns: fastsearch query "some query" --json -n 10
  │
  ▼
QMD Wrapper (~/.openclaw/bin/fastsearch)
  │
  │  translates to: vps-fastsearch search "some query" --mode hybrid --json -n 10
  │
  ▼
VPS-FastSearch CLI
  │
  ├── Daemon running? ──yes──▶ Unix socket (30ms)
  │                                │
  └── No daemon? ──────────▶ Direct mode (3-4s cold)
                                   │
                                   ▼
                            ┌──────────────┐
                            │  BM25 (FTS5) │──┐
                            └──────────────┘  │  RRF Fusion
                            ┌──────────────┐  │
                            │ Vector Search │──┘
                            │ (sqlite-vec)  │
                            └──────────────┘
                                   │
                                   ▼
                          QMD-format JSON response
                                   │
                                   ▼
                          OpenClaw displays results
```

## QMD Command Compatibility

| QMD Command | Wrapper Translation | Notes |
|---|---|---|
| `query <text> --json -n N` | `search <text> --mode hybrid --json -n N` | Primary search command |
| `search <text> --json -n N` | `search <text> --mode bm25 --json -n N` | Keyword only |
| `vsearch <text> --json -n N` | `search <text> --mode vector --json -n N` | Semantic only |
| `collection add <path> --name X --mask Y` | Stored in collections.json | Path mapping for qmd:// URIs |
| `collection list` | Reads collections.json | |
| `update` | `index <path> --glob <pattern> --reindex` | Per collection |
| `embed` | Same as update | Embedding happens during indexing |
| `status` | `stats` | |

## JSON Output Format

VPS-FastSearch output is converted to match QMD's expected format:

```json
[
  {
    "docid": "#a1b2c3",
    "score": 0.0328,
    "file": "qmd://memory-dir/2026-02-15.md",
    "title": "QMD Performance Breakthrough",
    "snippet": "@@ -10,4 @@ (9 before, 20 after)\n## QMD Performance Breakthrough\n- Root cause found..."
  }
]
```

## Benchmarks

Tested on 4-core AMD EPYC 9354P, 16GB RAM, no GPU:

| Operation | QMD | VPS-FastSearch | Speedup |
|---|---|---|---|
| Hybrid search (cold) | 82,000ms | 3,900ms | **21x** |
| Hybrid search (daemon) | 82,000ms | 31ms | **2,645x** |
| BM25 search | 300ms | 118ms | **2.5x** |
| Model RAM | 0 (load/dump) | 480MB (daemon) | Constant |
| Model disk | 2.1GB (3 GGUF) | 110MB (1 ONNX) | **19x smaller** |

## Troubleshooting

### Search returns empty results
- Check the database exists: `ls ~/.cache/fastsearch/index.db`
- Re-index: `vps-fastsearch --db ~/.cache/fastsearch/index.db index ~/.openclaw/workspace/memory --glob "*.md"`

### Daemon using too much memory (>1GB)
- Ensure VPS-FastSearch >= 0.2.1 with the `threads=2` fix
- Without this fix, ONNX arena allocates ~4GB; with it, stays at ~480MB
- Upgrade: `pip install --upgrade git+https://github.com/NarlySoftware/VPS-fastsearch.git`

### Hyphens in searches cause errors
- Ensure VPS-FastSearch >= 0.2.1 with FTS5 query sanitization
- Hyphens like "node-llama-cpp" are now properly quoted for FTS5

### OpenClaw timeout errors
- Set `memory.qmd.limits.timeoutMs` to at least 10000 (10s)
- With daemon running, searches complete in <100ms so 10s is very generous
- Without daemon, cold start takes 3-4s

### Daemon not starting
- Check socket: `ls -la /tmp/fastsearch.sock`
- Check logs: `journalctl -u fastsearch-daemon` (if using systemd)
- Manual start: `~/.openclaw/fastsearch-venv/bin/vps-fastsearch --db ~/.cache/fastsearch/index.db daemon start`

---

## See Also

- [VPS-FastSearch README](../README.md)
- [CLI Reference](CLI.md)
- [API Reference](API.md)
- [Performance Tuning](Performance.md)
