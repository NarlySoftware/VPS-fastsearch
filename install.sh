#!/bin/bash
set -euo pipefail

# =============================================================================
# VPS-FastSearch Install Script
# Installs VPS-FastSearch into ~/fastsearch with Python 3.13 venv
# Safe to re-run (idempotent)
# =============================================================================

INSTALL_DIR="$HOME/fastsearch"
VENV_DIR="$INSTALL_DIR/.venv"
CONFIG_DIR="$HOME/.config/fastsearch"
LOCAL_BIN="$HOME/.local/bin"

echo "=== VPS-FastSearch Installer ==="
echo ""

# ---- Step 1: Ensure Python 3.13 is available ----
echo "[1/8] Installing Python 3.13 and dependencies..."

sudo apt-get update -qq
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev python3-full build-essential libsqlite3-dev
echo "  Python 3.13 ready: $(python3.13 --version)"

# Raise socket buffer limits for large embed batches (2MB per socket)
if [ -f /proc/sys/net/core/rmem_max ]; then
    CURRENT_MAX=$(cat /proc/sys/net/core/rmem_max)
    if [ "$CURRENT_MAX" -lt 2097152 ]; then
        echo "  Raising socket buffer limits to 2MB..."
        sudo sysctl -w net.core.rmem_max=2097152 > /dev/null
        sudo sysctl -w net.core.wmem_max=2097152 > /dev/null
        # Persist across reboots
        if ! grep -q "net.core.rmem_max" /etc/sysctl.d/99-fastsearch.conf 2>/dev/null; then
            echo -e "net.core.rmem_max=2097152\nnet.core.wmem_max=2097152" | sudo tee /etc/sysctl.d/99-fastsearch.conf > /dev/null
        fi
        echo "  Socket buffer limits raised to 2MB"
    else
        echo "  Socket buffer limits already >= 2MB"
    fi
fi

# ---- Step 2: Create virtual environment ----
echo "[2/8] Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    # Verify the existing venv uses Python 3.13; rebuild if mismatched
    if "$VENV_DIR/bin/python" --version 2>/dev/null | grep -q "3.13"; then
        echo "  Venv already exists at $VENV_DIR (Python 3.13 confirmed)"
    else
        OLD_PY=$("$VENV_DIR/bin/python" --version 2>/dev/null || echo "unknown")
        echo "  Venv Python version mismatch ($OLD_PY), recreating with 3.13..."
        rm -rf "$VENV_DIR"
        python3.13 -m venv "$VENV_DIR"
        echo "  Recreated venv at $VENV_DIR"
    fi
else
    python3.13 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

# Upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet

# ---- Step 3: Install VPS-FastSearch ----
echo "[3/8] Installing VPS-FastSearch..."

"$VENV_DIR/bin/pip" install --timeout 300 --retries 10 "$INSTALL_DIR[all]" --quiet
echo "  Installed vps-fastsearch with all extras"

# ---- Step 4: Verify native extensions ----
echo "[4/8] Verifying native extensions..."

NATIVE_OK=true
if ! "$VENV_DIR/bin/python" -c "import apsw; import onnxruntime; print('  apsw + onnxruntime OK')" 2>/dev/null; then
    echo "  ERROR: apsw or onnxruntime failed to load"
    exit 1
fi
if ! "$VENV_DIR/bin/python" -c "import sqlite_vec; print('  sqlite_vec OK')" 2>/dev/null; then
    echo "  WARNING: sqlite_vec failed to load (may be fixed in step 4b)"
    NATIVE_OK=false
fi

# ---- Step 4b: Fix sqlite-vec ELFCLASS32 on ARM64 (upstream bug #211) ----
# See: https://github.com/asg017/sqlite-vec/issues/211
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    # Locate vec0.so even if import fails — find it via pip package location
    VEC_PKG_DIR=$("$VENV_DIR/bin/python" -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('sqlite_vec')
if spec and spec.origin:
    print(pathlib.Path(spec.origin).parent)
" 2>/dev/null || true)
    VEC_SO="${VEC_PKG_DIR:+$VEC_PKG_DIR/vec0.so}"

    if [ -n "$VEC_SO" ] && [ -f "$VEC_SO" ] && file "$VEC_SO" | grep -q "ELF 32-bit"; then
        echo "  Detected 32-bit vec0.so on ARM64 — rebuilding from source..."
        BUILD_DIR=$(mktemp -d /tmp/sqlite-vec-build.XXXXXX)
        trap "rm -rf '$BUILD_DIR'" EXIT
        git clone --depth 1 --branch v0.1.6 https://github.com/asg017/sqlite-vec.git "$BUILD_DIR"
        (cd "$BUILD_DIR" && make loadable)
        cp "$BUILD_DIR/dist/vec0.so" "$VEC_SO"
        echo "  Replaced vec0.so with native ARM64 build"
        # Re-verify after rebuild
        if "$VENV_DIR/bin/python" -c "import sqlite_vec" 2>/dev/null; then
            echo "  sqlite-vec now loads correctly"
            NATIVE_OK=true
        else
            echo "  ERROR: sqlite-vec still fails to load after rebuild"
        fi
    elif [ "$NATIVE_OK" = true ]; then
        echo "  sqlite-vec vec0.so is native ARM64, no fix needed"
    fi
fi

if [ "$NATIVE_OK" != true ]; then
    echo "  ERROR: Native extensions verification failed"
    exit 1
fi

# ---- Step 5: Symlink CLI to ~/.local/bin ----
echo "[5/8] Setting up CLI symlink..."

mkdir -p "$LOCAL_BIN"

# Remove stale symlink if it exists
if [ -L "$LOCAL_BIN/vps-fastsearch" ]; then
    rm "$LOCAL_BIN/vps-fastsearch"
fi

ln -s "$VENV_DIR/bin/vps-fastsearch" "$LOCAL_BIN/vps-fastsearch"
echo "  Symlinked vps-fastsearch to $LOCAL_BIN/vps-fastsearch"

# ---- Step 6: Install default config ----
echo "[6/8] Setting up configuration..."

if [ -f "$CONFIG_DIR/config.yaml" ]; then
    echo "  Config already exists at $CONFIG_DIR/config.yaml (not overwriting)"
else
    mkdir -p "$CONFIG_DIR"
    if [ -f "$INSTALL_DIR/config.yaml.example" ]; then
        cp "$INSTALL_DIR/config.yaml.example" "$CONFIG_DIR/config.yaml"
        echo "  Copied default config to $CONFIG_DIR/config.yaml"
    else
        echo "  No config.yaml.example found, skipping"
    fi
fi

# ---- Step 7: Smoke test ----
echo "[7/8] Running smoke test..."

if "$VENV_DIR/bin/vps-fastsearch" --help > /dev/null 2>&1; then
    echo "  vps-fastsearch --help: OK"
else
    echo "  ERROR: vps-fastsearch --help failed"
    exit 1
fi

# ---- Step 7b: Pre-download embedding model ----
echo "[7b/8] Pre-downloading embedding model..."
"$VENV_DIR/bin/python" -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5')" && \
    echo "  Model downloaded successfully" || \
    echo "  WARNING: Model download failed. It will be downloaded on first use."

# ---- Step 8: Install systemd user service ----
echo "[8/8] Installing systemd user service..."

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

if [ -f "$INSTALL_DIR/vps-fastsearch.service" ]; then
    cp "$INSTALL_DIR/vps-fastsearch.service" "$SYSTEMD_USER_DIR/vps-fastsearch.service"
    echo "  Installed service to $SYSTEMD_USER_DIR/vps-fastsearch.service"

    # Enable and start the service if systemctl --user is available
    if systemctl --user daemon-reload 2>/dev/null; then
        systemctl --user enable vps-fastsearch.service 2>/dev/null && \
            echo "  Enabled vps-fastsearch service" || \
            echo "  Could not enable service (systemd user session may not be available)"
        echo "  Start with: systemctl --user start vps-fastsearch"
    else
        echo "  systemctl --user not available (enable manually after login)"
    fi
else
    echo "  Service file not found, skipping"
fi

# ---- Done ----
echo ""
echo "=== VPS-FastSearch installed successfully ==="
echo ""
echo "  Install dir:  $INSTALL_DIR"
echo "  Venv:         $VENV_DIR"
echo "  CLI:          $LOCAL_BIN/vps-fastsearch"
echo "  Config:       $CONFIG_DIR/config.yaml"
echo ""

# Add ~/.local/bin to PATH if not already there
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    SHELL_RC="$HOME/.bashrc"
    if [ -n "${ZSH_VERSION:-}" ] || [[ "$SHELL" == */zsh ]]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    echo "  Added ~/.local/bin to PATH in $SHELL_RC"
    echo "  Run: source $SHELL_RC (or open a new terminal)"
    echo ""
fi

echo "Quick start:"
echo "  vps-fastsearch index README.md"
echo "  vps-fastsearch search \"your query\""
echo "  vps-fastsearch daemon start --detach"
