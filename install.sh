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
echo "[1/7] Checking for Python 3.13..."

if command -v python3.13 &> /dev/null; then
    echo "  Found python3.13 at $(command -v python3.13)"
else
    echo "  Python 3.13 not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
    echo "  Python 3.13 installed: $(python3.13 --version)"
fi

# ---- Step 2: Create virtual environment ----
echo "[2/7] Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists at $VENV_DIR"
else
    python3.13 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

# Upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet

# ---- Step 3: Install VPS-FastSearch ----
echo "[3/7] Installing VPS-FastSearch..."

"$VENV_DIR/bin/pip" install "$INSTALL_DIR[all]" --quiet
echo "  Installed vps-fastsearch with all extras"

# ---- Step 4: Verify native extensions ----
echo "[4/7] Verifying native extensions..."

if "$VENV_DIR/bin/python" -c "import sqlite_vec; import apsw; import onnxruntime; print('Native extensions OK')"; then
    echo "  Native extensions loaded successfully"
else
    echo "  ERROR: One or more native extensions failed to load"
    exit 1
fi

# ---- Step 5: Symlink CLI to ~/.local/bin ----
echo "[5/7] Setting up CLI symlink..."

mkdir -p "$LOCAL_BIN"

# Remove stale symlink if it exists
if [ -L "$LOCAL_BIN/vps-fastsearch" ]; then
    rm "$LOCAL_BIN/vps-fastsearch"
fi

ln -s "$VENV_DIR/bin/vps-fastsearch" "$LOCAL_BIN/vps-fastsearch"
echo "  Symlinked vps-fastsearch to $LOCAL_BIN/vps-fastsearch"

# ---- Step 6: Install default config ----
echo "[6/7] Setting up configuration..."

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
echo "[7/7] Running smoke test..."

if "$VENV_DIR/bin/vps-fastsearch" --help > /dev/null 2>&1; then
    echo "  vps-fastsearch --help: OK"
else
    echo "  ERROR: vps-fastsearch --help failed"
    exit 1
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

# Check if ~/.local/bin is on PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo "NOTE: Add ~/.local/bin to your PATH:"
    echo '  echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.bashrc'
    echo '  source ~/.bashrc'
    echo ""
fi

echo "Quick start:"
echo "  vps-fastsearch index README.md"
echo "  vps-fastsearch search \"your query\""
echo "  vps-fastsearch daemon start --detach"
