#!/usr/bin/env python3
"""Incremental indexer for VPS-FastSearch.

Tracks file modification times in a JSON state file and only re-indexes files
that have changed since the last run. Handles deleted files by removing them
from the search index.

Usage:
    # Index all .md files in a workspace (first run does full index)
    python incremental_indexer.py --workspace ~/my-project --glob "*.md"

    # Force a full re-index of everything
    python incremental_indexer.py --workspace ~/my-project --glob "*.md" --mode full

    # Incremental update (default — only changed/new files)
    python incremental_indexer.py --workspace ~/my-project --glob "*.md" --mode incremental

    # Custom CLI path and state file location
    python incremental_indexer.py --workspace ~/my-project --glob "*.md" \
        --cli ~/fastsearch/.venv/bin/vps-fastsearch \
        --state-file ~/.cache/fastsearch/index_state.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_state(state_file: Path) -> dict[str, float]:
    """Load the mtime state file. Returns {relative_path: mtime}."""
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


def save_state(state_file: Path, state: dict[str, float]) -> None:
    """Save the mtime state file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def discover_files(workspace: Path, glob_pattern: str) -> dict[str, float]:
    """Discover files matching the glob pattern and return {relative_path: mtime}."""
    found: dict[str, float] = {}
    for path in workspace.rglob(glob_pattern):
        if path.is_file():
            rel = str(path.relative_to(workspace))
            found[rel] = path.stat().st_mtime
    return found


def run_cli(cli: str, args: list[str]) -> bool:
    """Run a vps-fastsearch CLI command. Returns True on success."""
    cmd = [cli, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {' '.join(cmd)}", file=sys.stderr)
        if result.stderr:
            print(f"  {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def index_files(
    cli: str,
    workspace: Path,
    files: list[str],
    reindex: bool = False,
    base_dir: Path | None = None,
) -> int:
    """Index a list of files. Returns the number successfully indexed."""
    indexed = 0
    for rel_path in files:
        full_path = workspace / rel_path
        args = ["index", str(full_path)]
        if reindex:
            args.append("--reindex")
        if base_dir is not None:
            args.extend(["--base-dir", str(base_dir)])
        if run_cli(cli, args):
            indexed += 1
            print(f"  indexed: {rel_path}")
        else:
            print(f"  FAILED:  {rel_path}", file=sys.stderr)
    return indexed


def delete_sources(cli: str, sources: list[str]) -> int:
    """Delete sources from the index. Returns the number successfully deleted."""
    deleted = 0
    for source in sources:
        if run_cli(cli, ["delete", "--source", source]):
            deleted += 1
            print(f"  deleted: {source}")
        else:
            print(f"  FAILED:  {source}", file=sys.stderr)
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental indexer for VPS-FastSearch",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Root directory to scan for files",
    )
    parser.add_argument(
        "--glob",
        default="*.md",
        help="Glob pattern for files to index (default: *.md)",
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="Index mode: incremental (default) or full re-index",
    )
    parser.add_argument(
        "--cli",
        default="vps-fastsearch",
        help="Path to vps-fastsearch CLI (default: vps-fastsearch)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path.home() / ".cache" / "fastsearch" / "index_state.json",
        help="Path to mtime state file (default: ~/.cache/fastsearch/index_state.json)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for relative source paths (passed to vps-fastsearch index)",
    )
    args = parser.parse_args()

    workspace: Path = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)

    # Discover current files
    current = discover_files(workspace, args.glob)
    print(f"Found {len(current)} files matching '{args.glob}' in {workspace}")

    # Load previous state
    prev_state = load_state(args.state_file)

    if args.mode == "full":
        # Full mode: re-index everything
        changed = list(current.keys())
        removed: list[str] = []
        reindex = True
        print(f"Full mode: re-indexing all {len(changed)} files")
    else:
        # Incremental mode: find changed, new, and deleted files
        changed = [
            rel
            for rel, mtime in current.items()
            if rel not in prev_state or mtime > prev_state[rel]
        ]
        removed = [rel for rel in prev_state if rel not in current]
        reindex = True  # Always use --reindex for changed files to avoid duplicates
        print(
            f"Incremental: {len(changed)} changed/new, "
            f"{len(removed)} deleted, "
            f"{len(current) - len(changed)} unchanged"
        )

    # Process deletions
    if removed:
        print(f"\nDeleting {len(removed)} removed files from index...")
        delete_sources(args.cli, removed)

    # Process changed/new files
    if changed:
        print(f"\nIndexing {len(changed)} files...")
        index_files(args.cli, workspace, changed, reindex=reindex, base_dir=args.base_dir)

    if not changed and not removed:
        print("\nNothing to do — all files up to date.")

    # Save updated state
    save_state(args.state_file, current)
    print(f"\nState saved to {args.state_file}")


if __name__ == "__main__":
    main()
