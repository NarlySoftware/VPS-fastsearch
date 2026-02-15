# Configuration Reference

FastSearch uses a YAML configuration file for all settings.

## Config File Location

Default path: `~/.config/fastsearch/config.yaml`

**Priority order:**
1. Explicit path via `--config` flag
2. `FASTSEARCH_CONFIG` environment variable
3. `~/.config/fastsearch/config.yaml`
4. Built-in defaults

## Quick Setup

```bash
# Create default config
fastsearch config init

# View current config
fastsearch config show

# Show config path
fastsearch config path
```

---

## Complete Configuration Reference

```yaml
# FastSearch Configuration
# ~/.config/fastsearch/config.yaml

# =============================================================================
# Daemon Settings
# =============================================================================
daemon:
  # Unix socket path for IPC
  # Default: /tmp/fastsearch.sock
  socket_path: /tmp/fastsearch.sock
  
  # PID file path for process management
  # Default: /tmp/fastsearch.pid
  pid_path: /tmp/fastsearch.pid
  
  # Logging level: DEBUG, INFO, WARNING, ERROR
  # Default: INFO
  log_level: INFO

# =============================================================================
# Model Slots
# =============================================================================
models:
  # Embedding model (required for vector search)
  embedder:
    # Model name (HuggingFace identifier)
    # Default: BAAI/bge-base-en-v1.5 (768 dimensions, ~450MB)
    # Alternatives:
    #   - BAAI/bge-small-en-v1.5 (384 dims, ~130MB, faster)
    #   - BAAI/bge-large-en-v1.5 (1024 dims, ~1.2GB, better quality)
    name: "BAAI/bge-base-en-v1.5"
    
    # Loading strategy:
    #   - always: Load at daemon start, never unload
    #   - on_demand: Load when needed, unload after idle timeout
    #   - never: Disable this model slot
    # Default: always
    keep_loaded: always
    
    # Seconds of idle time before auto-unload (on_demand only)
    # 0 = never auto-unload
    # Default: 0
    idle_timeout_seconds: 0

  # Reranker model (optional, for improved accuracy)
  reranker:
    # Model name
    # Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB)
    # Alternatives:
    #   - cross-encoder/ms-marco-TinyBERT-L-2-v2 (~20MB, faster)
    #   - cross-encoder/ms-marco-MiniLM-L-12-v2 (~130MB, better)
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Reranker is on-demand by default (only loaded when --rerank used)
    keep_loaded: on_demand
    
    # Auto-unload after 5 minutes idle
    # Default: 300
    idle_timeout_seconds: 300

# =============================================================================
# Memory Management
# =============================================================================
memory:
  # Maximum RAM budget in MB
  # Models are evicted when budget exceeded
  # Default: 4000 (4GB)
  max_ram_mb: 4000
  
  # Eviction policy when memory limit reached:
  #   - lru: Evict least recently used first
  #   - fifo: Evict oldest loaded first
  # Default: lru
  eviction_policy: lru
```

---

## Configuration Sections

### daemon

Controls the daemon server.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `socket_path` | string | `/tmp/fastsearch.sock` | Unix socket path |
| `pid_path` | string | `/tmp/fastsearch.pid` | PID file path |
| `log_level` | string | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |

### models

Defines model slots and their behavior.

**Embedder slot:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `BAAI/bge-base-en-v1.5` | HuggingFace model name |
| `keep_loaded` | string | `always` | Loading strategy |
| `idle_timeout_seconds` | int | `0` | Auto-unload timeout |

**Reranker slot:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model name |
| `keep_loaded` | string | `on_demand` | Loading strategy |
| `idle_timeout_seconds` | int | `300` | Auto-unload timeout |

### memory

Controls memory management.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_ram_mb` | int | `4000` | Memory budget in MB |
| `eviction_policy` | string | `lru` | Eviction strategy: `lru` or `fifo` |

---

## Model Loading Modes

### `always`

Model is loaded when daemon starts and never unloaded.

**Best for:**
- Embedder (frequently used)
- Production servers with consistent load
- When cold-start latency is unacceptable

```yaml
models:
  embedder:
    name: "BAAI/bge-base-en-v1.5"
    keep_loaded: always
```

### `on_demand`

Model is loaded when first needed, unloaded after idle timeout.

**Best for:**
- Reranker (used occasionally)
- Memory-constrained environments
- Development/testing

```yaml
models:
  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    keep_loaded: on_demand
    idle_timeout_seconds: 300  # Unload after 5 min idle
```

### `never`

Model slot is disabled. Requests using it will fail.

**Best for:**
- Disabling reranking entirely
- Minimal memory footprint

```yaml
models:
  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    keep_loaded: never
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FASTSEARCH_DB` | Default database path |
| `FASTSEARCH_CONFIG` | Config file path |

```bash
export FASTSEARCH_DB="/var/lib/fastsearch/main.db"
export FASTSEARCH_CONFIG="/etc/fastsearch/config.yaml"
```

---

## Example Configurations

### Development (Minimal Memory)

```yaml
# Minimal config for development
daemon:
  socket_path: /tmp/fastsearch-dev.sock
  log_level: DEBUG

models:
  embedder:
    name: "BAAI/bge-small-en-v1.5"  # Smaller model
    keep_loaded: on_demand
    idle_timeout_seconds: 60

  reranker:
    keep_loaded: never  # Disabled

memory:
  max_ram_mb: 1000
```

### Production (Standard)

```yaml
# Standard production config
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
  max_ram_mb: 4000
  eviction_policy: lru
```

### High Accuracy (Large Model)

```yaml
# Maximum quality config
daemon:
  socket_path: /tmp/fastsearch.sock
  log_level: INFO

models:
  embedder:
    name: "BAAI/bge-large-en-v1.5"  # Larger model
    keep_loaded: always

  reranker:
    name: "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Larger reranker
    keep_loaded: always  # Keep loaded for consistent latency

memory:
  max_ram_mb: 8000
```

### Memory-Constrained VPS (2GB RAM)

```yaml
# Config for small VPS
daemon:
  socket_path: /tmp/fastsearch.sock
  log_level: WARNING  # Reduce logging

models:
  embedder:
    name: "BAAI/bge-small-en-v1.5"  # Smallest model (~130MB)
    keep_loaded: on_demand
    idle_timeout_seconds: 60  # Aggressive unload

  reranker:
    keep_loaded: never  # Disable reranking

memory:
  max_ram_mb: 1500
  eviction_policy: lru
```

### Multiple Instances

Run separate instances for different projects:

```yaml
# /etc/fastsearch/project-a.yaml
daemon:
  socket_path: /tmp/fastsearch-project-a.sock
  pid_path: /tmp/fastsearch-project-a.pid

# /etc/fastsearch/project-b.yaml
daemon:
  socket_path: /tmp/fastsearch-project-b.sock
  pid_path: /tmp/fastsearch-project-b.pid
```

```bash
# Start separate instances
fastsearch --config /etc/fastsearch/project-a.yaml daemon start --detach
fastsearch --config /etc/fastsearch/project-b.yaml daemon start --detach

# Search specific instance
fastsearch --config /etc/fastsearch/project-a.yaml search "query"
```

---

## Available Models

### Embedding Models

| Model | Dimensions | Memory | Speed | Quality |
|-------|------------|--------|-------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | ★★★★★ | ★★★ |
| `BAAI/bge-base-en-v1.5` | 768 | ~450MB | ★★★★ | ★★★★ |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.2GB | ★★★ | ★★★★★ |

### Reranker Models

| Model | Memory | Speed | Quality |
|-------|--------|-------|---------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | ~20MB | ★★★★★ | ★★★ |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90MB | ★★★★ | ★★★★ |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130MB | ★★★ | ★★★★★ |

---

## Reload Configuration

Apply config changes without restarting the daemon:

```bash
fastsearch daemon reload
```

**What can be reloaded:**
- Memory limits
- Idle timeouts
- Log level

**What requires restart:**
- Socket path changes
- Model name changes
