# Deployment Guide

This guide covers deploying FastSearch in production environments, including VPS setup, systemd configuration, and operational best practices.

## System Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 1GB | 2GB+ |
| CPU | 1 core | 2+ cores |
| Disk | 100MB + data | 1GB+ |
| Python | 3.10+ | 3.11+ |
| OS | Linux (glibc) | Debian 12/13, Ubuntu 22.04+ |

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
sudo apt install -y python3.11 python3.11-venv python3-pip build-essential

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

### 2. Install FastSearch

```bash
# Switch to fastsearch user
sudo -u fastsearch -i

# Create virtual environment
cd /opt/fastsearch
python3.11 -m venv venv
source venv/bin/activate

# Install FastSearch
pip install --upgrade pip
pip install "vps-fastsearch[rerank]"

# Verify installation
fastsearch --help
```

### 3. Configure

```bash
# Create config file
sudo -u fastsearch tee /etc/vps_fastsearch/config.yaml << 'EOF'
daemon:
  socket_path: /run/vps_fastsearch/vps_fastsearch.sock
  pid_path: /run/vps_fastsearch/fastsearch.pid
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
Description=FastSearch Daemon
After=network.target
Documentation=https://github.com/your-username/fastsearch

[Service]
Type=simple
User=fastsearch
Group=fastsearch

# Environment
Environment=FASTSEARCH_CONFIG=/etc/vps_fastsearch/config.yaml
Environment=FASTSEARCH_DB=/var/lib/vps_fastsearch/main.db

# Working directory
WorkingDirectory=/var/lib/fastsearch

# Command
ExecStart=/opt/vps_fastsearch/venv/bin/vps-fastsearch daemon start
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
ProtectHome=true
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

## Pre-Download Models

Models are downloaded on first use. For faster deployments, pre-download:

```bash
sudo -u fastsearch -i
source /opt/vps_fastsearch/venv/bin/activate

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
| Application logs | `/var/log/vps_fastsearch/` (if configured) |

### Monitoring Script

Create `/opt/vps_fastsearch/monitor.sh`:

```bash
#!/bin/bash
# FastSearch monitoring script

SOCKET="/run/vps_fastsearch/vps_fastsearch.sock"
CONFIG="/etc/vps_fastsearch/config.yaml"

# Check if socket exists
if [ ! -S "$SOCKET" ]; then
    echo "CRITICAL: Socket not found"
    exit 2
fi

# Get status via CLI
STATUS=$(/opt/vps_fastsearch/venv/bin/fastsearch --config "$CONFIG" daemon status --json 2>/dev/null)

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
"""Simple HTTP health check for FastSearch."""

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

FastSearch uses Unix sockets by default (no network exposure). If you create an HTTP wrapper:

```bash
# Allow only localhost
sudo ufw allow from 127.0.0.1 to any port 8080
```

### File Permissions

```bash
# Config file (contains no secrets, but restrict anyway)
sudo chmod 640 /etc/vps_fastsearch/config.yaml
sudo chown root:fastsearch /etc/vps_fastsearch/config.yaml

# Database files
sudo chmod 600 /var/lib/vps_fastsearch/*.db
```

---

## Backup and Restore

### Backup Database

```bash
#!/bin/bash
# /opt/vps_fastsearch/backup.sh

BACKUP_DIR="/var/backups/fastsearch"
DB_PATH="/var/lib/vps_fastsearch/main.db"
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
0 2 * * * /opt/vps_fastsearch/backup.sh
```

### Restore

```bash
# Stop service
sudo systemctl stop fastsearch

# Restore
gunzip -c /var/backups/vps_fastsearch/fastsearch_20240115_020000.db.gz > /var/lib/vps_fastsearch/main.db
chown fastsearch:fastsearch /var/lib/vps_fastsearch/main.db

# Start service
sudo systemctl start fastsearch
```

---

## Scaling

### Multiple Databases

For large deployments, split by domain:

```bash
# Separate databases
/var/lib/vps_fastsearch/docs.db      # Documentation
/var/lib/vps_fastsearch/code.db      # Code files
/var/lib/vps_fastsearch/support.db   # Support tickets

# Search specific database
fastsearch --db /var/lib/vps_fastsearch/docs.db search "query"
```

### Multiple Instances

For isolation between projects:

```bash
# Create instance-specific configs
/etc/vps_fastsearch/project-a.yaml
/etc/vps_fastsearch/project-b.yaml

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
sudo -u fastsearch /opt/vps_fastsearch/venv/bin/pip install --upgrade fastsearch

# Restart
sudo systemctl start fastsearch
```

### Major Updates

```bash
# Backup database
/opt/vps_fastsearch/backup.sh

# Stop service
sudo systemctl stop fastsearch

# Create new venv (optional, safer)
sudo -u fastsearch mv /opt/vps_fastsearch/venv /opt/vps_fastsearch/venv.bak
sudo -u fastsearch python3.11 -m venv /opt/vps_fastsearch/venv
sudo -u fastsearch /opt/vps_fastsearch/venv/bin/pip install "vps-fastsearch[rerank]"

# Test
sudo -u fastsearch /opt/vps_fastsearch/venv/bin/fastsearch --version

# Restart
sudo systemctl start fastsearch

# Verify
sudo systemctl status fastsearch
vps-fastsearch search "test query"

# Remove old venv after confirmation
sudo rm -rf /opt/vps_fastsearch/venv.bak
```

---

## Troubleshooting Deployment

### Service Won't Start

```bash
# Check logs
sudo journalctl -u fastsearch -n 50

# Common issues:
# 1. Socket already exists (stale from crash)
sudo rm /run/vps_fastsearch/vps_fastsearch.sock

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
sudo vim /etc/vps_fastsearch/config.yaml
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
