#!/usr/bin/env python3
"""
QMD-compatible wrapper for VPS-FastSearch.
Translates QMD CLI commands into VPS-FastSearch calls.
OpenClaw calls this as if it were QMD.

Install:
  curl -fsSL https://raw.githubusercontent.com/NarlySoftware/VPS-fastsearch/main/docs/qmd-wrapper.py \
    -o ~/.openclaw/bin/fastsearch
  chmod +x ~/.openclaw/bin/fastsearch

See: https://github.com/NarlySoftware/VPS-fastsearch/blob/main/docs/OpenClaw-Integration.md
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
