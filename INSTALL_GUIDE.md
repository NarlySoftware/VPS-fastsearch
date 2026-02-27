# VPS-FastSearch Installation & Setup Guide

## 1. Install

The VPS-FastSearch tarball is at `~/vps-fastsearch-0.2.0.tar.gz`. Extract it and run the installer:

```bash
cd ~
tar xzf vps-fastsearch-0.2.0.tar.gz
mv vps-fastsearch-0.2.0 fastsearch
cd ~/fastsearch
chmod +x install.sh
./install.sh
```

After install, add `~/.local/bin` to your PATH if it isn't already:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify it works:

```bash
vps-fastsearch --help
```

## 2. Start the daemon

Always run the daemon — it keeps the embedding model in memory so searches take ~5ms instead of ~800ms:

```bash
vps-fastsearch daemon start --detach
```

To check it's running:

```bash
vps-fastsearch daemon status
```

To start the daemon automatically on boot, install the systemd service:

```bash
sudo cp ~/fastsearch/fastsearch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fastsearch
```

**Note:** Edit the service file first if you're not running as a `fastsearch` user — change `User=` and `ExecStart=` paths to match your setup.

## 3. Index your documents

Index your memory files:

```bash
vps-fastsearch index ~/memory/ --glob "*.md"
```

Index a document library:

```bash
vps-fastsearch index ~/docs/ --glob "*.md"
vps-fastsearch index ~/docs/ --glob "*.txt"
```

To re-index after changes:

```bash
vps-fastsearch index ~/memory/ --glob "*.md" --reindex
```

Check what's indexed:

```bash
vps-fastsearch stats
```

## 4. Search from the command line

```bash
vps-fastsearch search "what did cliff say about email config"
vps-fastsearch search "database connection details" --rerank
vps-fastsearch search "exact phrase match" --mode bm25
```

## 5. Search from Python

For use in your own code:

```python
from vps_fastsearch import search, embed

# Quick search (uses daemon automatically)
results = search("what is the database password")
for r in results:
    print(f"{r['source']}: {r['content'][:100]}")

# With more control
from vps_fastsearch import FastSearchClient

with FastSearchClient() as client:
    # Hybrid search
    result = client.search("query", limit=5)

    # With reranking (more accurate, slower)
    result = client.search("query", rerank=True)

    # Generate embeddings for new text
    vectors = client.embed(["some new text"])

    # Rerank candidate documents
    ranked = client.rerank("query", ["doc1", "doc2", "doc3"])
```

## 6. Keep memory fresh

After you create or update memory files, re-index them:

```bash
vps-fastsearch index ~/memory/ --glob "*.md" --reindex
```

You can wrap this in a cron job or call it after writing new files.

## Quick Reference

| Command | What it does |
|---------|-------------|
| `vps-fastsearch daemon start --detach` | Start daemon (do this first) |
| `vps-fastsearch daemon status` | Check daemon health |
| `vps-fastsearch index <path> --glob "*.md"` | Index files |
| `vps-fastsearch index <path> --glob "*.md" --reindex` | Re-index after changes |
| `vps-fastsearch search "query"` | Search from CLI |
| `vps-fastsearch search "query" --rerank` | Search with better accuracy |
| `vps-fastsearch stats` | See what's indexed |
| `vps-fastsearch daemon stop` | Stop daemon |
