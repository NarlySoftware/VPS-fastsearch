#!/bin/bash
set -euo pipefail

# =============================================================================
# VPS-FastSearch Packaging Script
# Creates a distributable tarball (no git required)
# =============================================================================

VERSION=$(grep '^version =' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
PACKAGE_NAME="vps-fastsearch-${VERSION}"
STAGING_DIR=$(mktemp -d)

echo "Packaging ${PACKAGE_NAME}..."

# Copy files into a staging directory with the package name prefix
mkdir -p "$STAGING_DIR/$PACKAGE_NAME"

cp -R vps_fastsearch/ "$STAGING_DIR/$PACKAGE_NAME/vps_fastsearch/"
cp -R tests/ "$STAGING_DIR/$PACKAGE_NAME/tests/"
cp -R docs/ "$STAGING_DIR/$PACKAGE_NAME/docs/"
cp pyproject.toml README.md LICENSE INSTALL_GUIDE.md install.sh config.yaml.example \
   vps-fastsearch.service run_tests.py benchmark_reranker.py CLAUDE.md \
   "$STAGING_DIR/$PACKAGE_NAME/"

# Clean unwanted files from staging
find "$STAGING_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGING_DIR" -name '*.pyc' -delete 2>/dev/null || true
find "$STAGING_DIR" -name '*.db' -delete 2>/dev/null || true
find "$STAGING_DIR" -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true

# Create tarball
tar czf "${PACKAGE_NAME}.tar.gz" -C "$STAGING_DIR" "$PACKAGE_NAME"

# Clean up
rm -rf "$STAGING_DIR"

SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)
echo "Created ${PACKAGE_NAME}.tar.gz (${SIZE})"
