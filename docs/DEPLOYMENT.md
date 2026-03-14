# Deployment Guide

This guide covers deploying VPS-FastSearch in production environments, including VPS setup, systemd configuration, and operational best practices.

## System Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 1GB | 2GB+ |
| CPU | 1 core | 2+ cores |
| Disk | 100MB + data | 1GB+ |
| Python | 3.10+ | 3.11+ |
| OS | Linux (glibc) | Debian 12/13, Ubuntu 22.04+ |

### ARM64 / aarch64 Notes

ARM64 VPS instances (e.g. Oracle Cloud, Hetzner CAX, AWS Graviton) work well but
have two caveats:

- **Embedding batch size:** ONNX Runtime allocates heavily on ARM64. The CLI
  automatically batches embed calls in groups of 10, but if you write a custom
  indexer, keep batches small (5–10 texts). Sending 50+ texts at once can exhaust
  RAM on 2GB instances.
- **sqlite-vec:** The `sqlite-vec` 0.1.6 pip wheel ships a 32-bit `vec0.so` on
  ARM64 (upstream bug #211). `install.sh` detects and rebuilds it automatically.
  If installing manually, use `sqlite-vec==0.1.7a10` or build from source.
- **Recommended config:** Set `memory.max_ram_mb: 2000` and use `bge-base` (not
  `bge-large`) on 2GB instances.

### Memory by Configuration

| Configuration | RAM Usage |
|---------------|-----------|
| Daemon only (no models) | ~50MB |
| Small embedder (bge-small) | ~180MB |
| Base embedder (bge-base) | ~500MB |
| Base embedder + reranker | ~600MB |
| Large embedder (bge-large) | ~1.3GB |

---

## Debian/Ubuntu VPS Setup

### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y python3.13 python3.13-venv build-essential

# Create fastsearch user
sudo useradd -r -m -s /bin/bash fastsearch

# Create directories
sudo mkdir -p /opt/fastsearch
sudo mkdir -p /var/lib/fastsearch
sudo mkdir -p /etc/fastsearch
sudo mkdir -p /var/log/fastsearch
sudo mkdir -p /run/fastsearch

# Set ownership
sudo chown -R fastsearch:fastsearch /opt/fastsearch
sudo chown -R fastsearch:fastsearch /var/lib/fastsearch
sudo chown -R fastsearch:fastsearch /var/log/fastsearch
sudo chown -R fastsearch:fastsearch /run/fastsearch
```

### 2. Install VPS-FastSearch

```bash
# Switch to fastsearch user
sudo -u fastsearch -i

# Create virtual environment
cd /opt/fastsearch
python3.13 -m venv venv
source venv/bin/activate

# Install VPS-FastSearch
pip install --upgrade pip
pip install "vps-fastsearch[rerank]"

# Verify installation
vps-fastsearch --help
```

### 3. Configure

```bash
# Create config file
sudo -u fastsearch tee /etc/fastsearch/config.yaml << 'EOF'
daemon:
  socket_path: /run/fastsearch/fastsearch.sock
  pid_path: /run/fastsearch/fastsearch.pid
  log_level: INFO

models:
  embedder:
    name: "BAAI/bge-base-en-v1.5"
    keep_loaded: always

  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    keep_loaded: on_demand
    idle_timeout_seconds: 300

memory:
  max_ram_mb: 2000
  eviction_policy: lru
EOF
```

---

## Systemd Service

### Service File

Create `/etc/systemd/system/fastsearch.service`:

```ini
[Unit]
Description=VPS-FastSearch Daemon
After=network.target
Documentation=https://github.com/NarlySoftware/VPS-fastsearch

[Service]
Type=simple
User=fastsearch
Group=fastsearch

# Environment
Environment=FASTSEARCH_CONFIG=/etc/fastsearch/config.yaml
Environment=FASTSEARCH_DB=/var/lib/fastsearch/fastsearch.db

# Working directory
WorkingDirectory=/var/lib/fastsearch

# Command
ExecStart=/opt/fastsearch/venv/bin/vps-fastsearch daemon start
ExecReload=/bin/kill -HUP $MAINPID

# Restart policy
Restart=on-failure
RestartSec=5

# Runtime directory
RuntimeDirectory=fastsearch
RuntimeDirectoryMode=0755

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
# ProtectHome=false required: fastembed caches models in ~/.cache/fastembed/
ProtectHome=false
PrivateTmp=true
ReadWritePaths=/var/lib/fastsearch /var/log/fastsearch /run/fastsearch

# Resource limits
MemoryMax=2500M
MemoryHigh=2000M

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fastsearch

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable at boot
sudo systemctl enable fastsearch

# Start service
sudo systemctl start fastsearch

# Check status
sudo systemctl status fastsearch
```

### Service Commands

```bash
# Start/stop/restart
sudo systemctl start fastsearch
sudo systemctl stop fastsearch
sudo systemctl restart fastsearch

# View logs
sudo journalctl -u fastsearch -f

# View recent logs
sudo journalctl -u fastsearch --since "1 hour ago"

# Reload config (no restart)
sudo systemctl reload fastsearch
```

---

## Scheduled Indexing

VPS-FastSearch includes systemd timer units for automated indexing. These are the
**only** supported scheduling mechanism — do not use crontab or other schedulers
to avoid duplicate indexing.

### Timer/Service Units

The repo root contains four unit files:

| File | Purpose |
|------|---------|
| `fastsearch-index-incremental.service` | Oneshot: runs indexer in `--mode incremental` |
| `fastsearch-index-incremental.timer` | Triggers incremental every 15 minutes (+ 3 min after boot) |
| `fastsearch-index-full.service` | Oneshot: runs indexer in `--mode full` |
| `fastsearch-index-full.timer` | Triggers full reindex nightly at 02:17 |

### User-Level Installation

For user-level deployments (e.g., `~/.config/systemd/user/`):

```bash
# Copy timer and service units
cp fastsearch-index-*.service fastsearch-index-*.timer \
   ~/.config/systemd/user/

# Edit the service files to set correct paths:
#   ExecStart= — path to your Python and indexer script
#   FASTSEARCH_DB= — path to your database
#   FASTSEARCH_CONFIG= — path to your config file

# Reload and enable
systemctl --user daemon-reload
systemctl --user enable --now fastsearch-index-incremental.timer
systemctl --user enable --now fastsearch-index-full.timer

# Verify timers are active
systemctl --user list-timers

# IMPORTANT: Enable linger so services survive logout and start at boot
sudo loginctl enable-linger $(whoami)
```

### System-Level Installation

For system-level deployments (dedicated `fastsearch` user):

```bash
# Copy to system directory
sudo cp fastsearch-index-*.service fastsearch-index-*.timer \
   /etc/systemd/system/

# Edit service files: set User=, ExecStart=, Environment= paths
sudo systemctl daemon-reload
sudo systemctl enable --now fastsearch-index-incremental.timer
sudo systemctl enable --now fastsearch-index-full.timer
```

### Monitoring

```bash
# Check timer schedule and last run
systemctl --user list-timers

# View indexer logs
journalctl --user -u fastsearch-index-incremental.service
journalctl --user -u fastsearch-index-full.service

# Manually trigger an incremental run
systemctl --user start fastsearch-index-incremental.service
```

### Indexer Script

The service units expect an indexer script (see `examples/incremental_indexer.py`
for a generic version). The script should:

1. Discover files to index (by glob pattern or explicit list)
2. Track modification times to skip unchanged files
3. Call `vps-fastsearch index <file> --reindex` for changed files
4. Call `vps-fastsearch delete --source <file>` for removed files

### Important

- **Use systemd timers only** — do not add crontab entries or other schedulers
  for FastSearch indexing. Duplicate scheduling causes redundant work and
  potential race conditions.
- The timer services have `After=vps-fastsearch.service` to ensure the daemon
  is running before indexing starts.

---

## Pre-Download Models

Models are downloaded on first use. For faster deployments, pre-download:

```bash
sudo -u fastsearch -i
source /opt/fastsearch/venv/bin/activate

# Pre-download embedder
python3 -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5')"

# Pre-download reranker (if using)
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

---

## Logs and Monitoring

### Log Locations

| Log | Location |
|-----|----------|
| Service logs | `journalctl -u fastsearch` |
| Application logs | `/var/log/fastsearch/` (if configured) |

### Monitoring Script

Create `/opt/fastsearch/monitor.sh`:

```bash
#!/bin/bash
# VPS-FastSearch monitoring script

SOCKET="/run/fastsearch/fastsearch.sock"
CONFIG="/etc/fastsearch/config.yaml"

# Check if socket exists
if [ ! -S "$SOCKET" ]; then
    echo "CRITICAL: Socket not found"
    exit 2
fi

# Get status via CLI
STATUS=$(/opt/fastsearch/venv/bin/vps-fastsearch --config "$CONFIG" daemon status --json 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "CRITICAL: Daemon not responding"
    exit 2
fi

# Parse status
UPTIME=$(echo "$STATUS" | jq -r '.uptime_seconds // 0')
MEMORY=$(echo "$STATUS" | jq -r '.total_memory_mb // 0')
REQUESTS=$(echo "$STATUS" | jq -r '.request_count // 0')

echo "OK: uptime=${UPTIME}s, memory=${MEMORY}MB, requests=${REQUESTS}"
exit 0
```

### Health Check Endpoint

For load balancers, create a simple HTTP wrapper:

```python
#!/usr/bin/env python3
"""Simple HTTP health check for VPS-FastSearch."""

from http.server import HTTPServer, BaseHTTPRequestHandler
from vps_fastsearch import FastSearchClient
import json

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            try:
                client = FastSearchClient(timeout=5.0)
                if client.ping():
                    status = client.status()
                    client.close()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", **status}).encode())
                else:
                    raise Exception("Ping failed")
            except Exception as e:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()
```

---

## Security Considerations

### Socket Permissions

```bash
# Socket is created with 0600 permissions by default
# Only the fastsearch user can access it

# For multi-user access, create a group:
sudo groupadd fastsearch-clients
sudo usermod -a -G fastsearch-clients yourapp

# Modify socket permissions in config:
# (or use a wrapper that sets umask)
```

### Firewall

VPS-FastSearch uses Unix sockets by default (no network exposure). If you create an HTTP wrapper:

```bash
# Allow only localhost
sudo ufw allow from 127.0.0.1 to any port 8080
```

### File Permissions

```bash
# Config file (contains no secrets, but restrict anyway)
sudo chmod 640 /etc/fastsearch/config.yaml
sudo chown root:fastsearch /etc/fastsearch/config.yaml

# Database files
sudo chmod 600 /var/lib/fastsearch/*.db
```

---

## Backup and Restore

### Backup Database

```bash
#!/bin/bash
# /opt/fastsearch/backup.sh

BACKUP_DIR="/var/backups/fastsearch"
DB_PATH="/var/lib/fastsearch/fastsearch.db"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# SQLite safe backup
sqlite3 "$DB_PATH" ".backup '$BACKUP_DIR/fastsearch_$DATE.db'"

# Compress
gzip "$BACKUP_DIR/fastsearch_$DATE.db"

# Keep last 7 days
find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete
```

Add to cron:
```bash
# Daily backup at 2 AM
0 2 * * * /opt/fastsearch/backup.sh
```

### Restore

```bash
# Stop service
sudo systemctl stop fastsearch

# Restore
gunzip -c /var/backups/fastsearch/fastsearch_20240115_020000.db.gz > /var/lib/fastsearch/fastsearch.db
chown fastsearch:fastsearch /var/lib/fastsearch/fastsearch.db

# Start service
sudo systemctl start fastsearch
```

---

## Scaling

### Multiple Databases

For large deployments, split by domain:

```bash
# Separate databases
/var/lib/fastsearch/docs.db      # Documentation
/var/lib/fastsearch/code.db      # Code files
/var/lib/fastsearch/support.db   # Support tickets

# Search specific database
fastsearch --db /var/lib/fastsearch/docs.db search "query"
```

### Multiple Instances

For isolation between projects:

```bash
# Create instance-specific configs
/etc/fastsearch/project-a.yaml
/etc/fastsearch/project-b.yaml

# Create instance-specific services
/etc/systemd/system/fastsearch-project-a.service
/etc/systemd/system/fastsearch-project-b.service
```

### Horizontal Scaling (Future)

For very high load, consider:
- Load balancer distributing to multiple daemons
- Shared database (PostgreSQL with pgvector)
- Distributed cache for embeddings

---

## Upgrade Process

### Minor Updates

```bash
# Stop service
sudo systemctl stop fastsearch

# Upgrade
sudo -u fastsearch /opt/fastsearch/venv/bin/pip install --upgrade vps-fastsearch

# Restart
sudo systemctl start fastsearch
```

### Major Updates

```bash
# Backup database
/opt/fastsearch/backup.sh

# Stop service
sudo systemctl stop fastsearch

# Create new venv (optional, safer)
sudo -u fastsearch mv /opt/fastsearch/venv /opt/fastsearch/venv.bak
sudo -u fastsearch python3.13 -m venv /opt/fastsearch/venv
sudo -u fastsearch /opt/fastsearch/venv/bin/pip install "vps-fastsearch[rerank]"

# Test
sudo -u fastsearch /opt/fastsearch/venv/bin/vps-fastsearch --version

# Restart
sudo systemctl start fastsearch

# Verify
sudo systemctl status fastsearch
vps-fastsearch search "test query"

# Remove old venv after confirmation
sudo rm -rf /opt/fastsearch/venv.bak
```

---

## Troubleshooting Deployment

### Service Won't Start

```bash
# Check logs
sudo journalctl -u fastsearch -n 50

# Common issues:
# 1. Socket already exists (stale from crash)
sudo rm /run/fastsearch/fastsearch.sock

# 2. Permission denied
sudo chown -R fastsearch:fastsearch /var/lib/fastsearch

# 3. Out of memory
# Reduce model size in config
```

### High Memory Usage

```bash
# Check current usage
vps-fastsearch daemon status

# Reduce memory budget
sudo vim /etc/fastsearch/config.yaml
# Set: memory.max_ram_mb: 1500

# Reload config
sudo systemctl reload fastsearch
```

### Slow Response Times

```bash
# Check if model is loaded
vps-fastsearch daemon status

# If embedder shows as not loaded, pre-warm:
vps-fastsearch search "warmup" > /dev/null

# Or set keep_loaded: always in config
```
