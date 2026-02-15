# Troubleshooting Guide

Common issues and their solutions.

## Quick Diagnostics

```bash
# Check daemon status
vps-fastsearch daemon status

# Check socket exists
ls -la /tmp/vps_fastsearch.sock

# Check PID file
cat /tmp/fastsearch.pid

# Check database
fastsearch stats

# Test search
vps-fastsearch search "test" --json
```

---

## Daemon Issues

### Daemon Won't Start

**Symptom:** `vps-fastsearch daemon start` hangs or fails.

**Solutions:**

1. **Stale socket file**
   ```bash
   # Remove stale socket
   rm /tmp/vps_fastsearch.sock
   rm /tmp/fastsearch.pid
   
   # Try again
   vps-fastsearch daemon start
   ```

2. **Process already running**
   ```bash
   # Check for existing process
   ps aux | grep fastsearch
   
   # Kill it
   kill $(cat /tmp/fastsearch.pid)
   
   # Or force kill
   pkill -f "vps-fastsearch daemon"
   ```

3. **Port/socket permission denied**
   ```bash
   # Check ownership
   ls -la /tmp/vps_fastsearch.sock
   
   # If owned by another user, remove or change socket path
   rm /tmp/vps_fastsearch.sock
   
   # Or use different path in config
   # daemon.socket_path: /tmp/fastsearch-$(whoami).sock
   ```

4. **Missing dependencies**
   ```bash
   # Reinstall with all dependencies
   pip install --force-reinstall "fastsearch[rerank]"
   ```

### Daemon Crashes on Start

**Symptom:** Daemon starts but immediately exits.

**Solutions:**

1. **Check logs**
   ```bash
   # If using systemd
   journalctl -u fastsearch -n 100
   
   # Run in foreground to see errors
   vps-fastsearch daemon start  # Without --detach
   ```

2. **Out of memory**
   ```bash
   # Check available memory
   free -h
   
   # Use smaller model in config
   # models.embedder.name: BAAI/bge-small-en-v1.5
   ```

3. **Model download failed**
   ```bash
   # Clear model cache and redownload
   rm -rf ~/.cache/huggingface/hub/models--BAAI*
   
   # Try starting again
   vps-fastsearch daemon start
   ```

---

## Socket Connection Errors

### "Socket not found"

**Symptom:** `DaemonNotRunningError: Daemon socket not found`

**Solutions:**

1. **Start the daemon**
   ```bash
   vps-fastsearch daemon start --detach
   ```

2. **Check socket path**
   ```bash
   # Show config
   fastsearch config show | grep socket_path
   
   # Verify socket exists
   ls -la /tmp/vps_fastsearch.sock
   ```

3. **Config mismatch**
   ```bash
   # Ensure client and daemon use same config
   FASTSEARCH_CONFIG=/path/to/config.yaml vps-fastsearch daemon start --detach
   FASTSEARCH_CONFIG=/path/to/config.yaml vps-fastsearch search "test"
   ```

### "Connection refused"

**Symptom:** `Connection refused` when connecting.

**Solutions:**

1. **Daemon crashed**
   ```bash
   # Check status
   vps-fastsearch daemon status
   
   # Restart
   vps-fastsearch daemon stop
   vps-fastsearch daemon start --detach
   ```

2. **Socket permissions**
   ```bash
   # Check permissions
   ls -la /tmp/vps_fastsearch.sock
   # Should be: srw------- (600)
   
   # If wrong user, restart daemon as correct user
   ```

### "Timeout"

**Symptom:** Search hangs and times out.

**Solutions:**

1. **Daemon overloaded**
   ```bash
   # Check status
   vps-fastsearch daemon status
   
   # Restart daemon
   vps-fastsearch daemon stop
   vps-fastsearch daemon start --detach
   ```

2. **Model loading**
   ```bash
   # First query after start loads model (~10s)
   # Pre-warm with dummy query
   vps-fastsearch search "warmup" > /dev/null
   ```

3. **Increase timeout**
   ```python
   client = FastSearchClient(timeout=60.0)
   ```

---

## Model Loading Failures

### "Model not found"

**Symptom:** `Model not found` or download errors.

**Solutions:**

1. **Clear cache and retry**
   ```bash
   # Clear HuggingFace cache
   rm -rf ~/.cache/huggingface/
   
   # Restart daemon
   vps-fastsearch daemon stop
   vps-fastsearch daemon start --detach
   ```

2. **Network issues**
   ```bash
   # Test connectivity
   curl -I https://huggingface.co
   
   # If behind proxy, set environment
   export HTTP_PROXY=http://proxy:8080
   export HTTPS_PROXY=http://proxy:8080
   ```

3. **Pre-download models**
   ```bash
   python3 -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5')"
   ```

### "ONNX Runtime error"

**Symptom:** ONNX-related errors.

**Solutions:**

1. **Reinstall onnxruntime**
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime
   ```

2. **Check CPU compatibility**
   ```bash
   # Some older CPUs lack required instructions
   # Try the generic build
   pip install onnxruntime --no-binary onnxruntime
   ```

### "Out of memory loading model"

**Symptom:** Memory error when loading embedder.

**Solutions:**

1. **Use smaller model**
   ```yaml
   # config.yaml
   models:
     embedder:
       name: "BAAI/bge-small-en-v1.5"  # 130MB vs 450MB
   ```

2. **Disable reranker**
   ```yaml
   models:
     reranker:
       keep_loaded: never
   ```

3. **Increase swap**
   ```bash
   # Add swap (Linux)
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Memory Issues

### High Memory Usage

**Symptom:** Daemon using more memory than expected.

**Solutions:**

1. **Check what's loaded**
   ```bash
   vps-fastsearch daemon status
   # Shows loaded models and memory
   ```

2. **Configure memory budget**
   ```yaml
   memory:
     max_ram_mb: 2000  # Lower limit
   ```

3. **Use on-demand loading**
   ```yaml
   models:
     embedder:
       keep_loaded: on_demand
       idle_timeout_seconds: 60
   ```

4. **Force garbage collection**
   ```bash
   # Reload config triggers cleanup
   vps-fastsearch daemon reload
   ```

### Memory Leak

**Symptom:** Memory grows over time.

**Solutions:**

1. **Restart daemon periodically**
   ```bash
   # Cron job to restart nightly
   0 3 * * * systemctl restart fastsearch
   ```

2. **Monitor memory**
   ```bash
   # Watch memory usage
   watch -n 5 'vps-fastsearch daemon status --json | jq .total_memory_mb'
   ```

---

## Search Issues

### No Results

**Symptom:** Search returns empty results.

**Solutions:**

1. **Check database has content**
   ```bash
   fastsearch stats
   # Should show total_chunks > 0
   ```

2. **Index files first**
   ```bash
   vps-fastsearch index ./docs --glob "*.md"
   ```

3. **Try different mode**
   ```bash
   # Try BM25 (keyword match)
   vps-fastsearch search "exact words" --mode bm25
   
   # Try vector (semantic)
   vps-fastsearch search "concept meaning" --mode vector
   ```

4. **Check database path**
   ```bash
   # Ensure using correct database
   fastsearch --db /path/to/correct.db search "query"
   ```

### Wrong Results

**Symptom:** Results are irrelevant.

**Solutions:**

1. **Use reranking**
   ```bash
   vps-fastsearch search "query" --rerank
   ```

2. **Try BM25 for keywords**
   ```bash
   vps-fastsearch search "specific_function_name" --mode bm25
   ```

3. **Adjust chunk size**
   ```python
   # Smaller chunks may help
   from vps_fastsearch import chunk_text
   chunks = list(chunk_text(content, target_chars=1000))
   ```

### Slow Search

**Symptom:** Search takes too long.

**Solutions:**

1. **Use daemon mode**
   ```bash
   vps-fastsearch daemon start --detach
   # First search loads model, subsequent are fast
   ```

2. **Skip reranking**
   ```bash
   vps-fastsearch search "query"  # Without --rerank
   ```

3. **Reduce limit**
   ```bash
   vps-fastsearch search "query" --limit 5  # Instead of 20
   ```

4. **Use BM25 for speed**
   ```bash
   vps-fastsearch search "keyword" --mode bm25  # 2ms vs 4ms
   ```

---

## Debug Mode and Logging

### Enable Debug Logging

```yaml
# config.yaml
daemon:
  log_level: DEBUG
```

```bash
# Restart daemon
vps-fastsearch daemon stop
vps-fastsearch daemon start --detach

# Watch logs (systemd)
journalctl -u fastsearch -f
```

### Verbose CLI Output

```bash
# Use --json for detailed output
vps-fastsearch search "query" --json | jq .

# Check search timings
vps-fastsearch search "query" --json | jq '.search_time_ms'
```

### Python Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from vps_fastsearch import FastSearchClient

client = FastSearchClient()
# Now shows connection details
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Socket not found` | Daemon not running | Start daemon |
| `Connection refused` | Daemon crashed | Restart daemon |
| `Model not found` | Download failed | Clear cache, retry |
| `Out of memory` | Model too large | Use smaller model |
| `Permission denied` | Socket permissions | Run as correct user |
| `Timeout` | Model loading | Wait or pre-warm |
| `Invalid JSON` | Protocol error | Update client/daemon |

---

## Getting Help

### Collect Diagnostic Info

```bash
# Create diagnostic report
{
    echo "=== System ==="
    uname -a
    python3 --version
    
    echo "=== VPS-FastSearch ==="
    fastsearch --version 2>/dev/null || pip show fastsearch
    
    echo "=== Config ==="
    fastsearch config show
    
    echo "=== Status ==="
    vps-fastsearch daemon status --json 2>/dev/null || echo "Daemon not running"
    
    echo "=== Database ==="
    fastsearch stats 2>/dev/null || echo "No database"
    
    echo "=== Memory ==="
    free -h
    
    echo "=== Disk ==="
    df -h /tmp ~/.cache
} > fastsearch-diagnostic.txt
```

### Report Issues

When reporting issues, include:
- OS and Python version
- VPS-FastSearch version
- Config file (sanitized)
- Error message (full traceback)
- Steps to reproduce
- Diagnostic report output
