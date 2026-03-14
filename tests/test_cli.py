"""CLI smoke tests for vps_fastsearch.cli using Click's CliRunner."""

from pathlib import Path

from click.testing import CliRunner

from tests.conftest import DUMMY_EMBEDDING
from vps_fastsearch.cli import cli
from vps_fastsearch.core import SearchDB


def test_cli_version() -> None:
    """--version should print the version string."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "vps-fastsearch" in result.output
    assert "0.3.2" in result.output


def test_cli_help() -> None:
    """--help should print the help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "VPS-FastSearch" in result.output


def test_cli_stats_no_db(tmp_path) -> None:
    """stats on a nonexistent DB should print 'not found'."""
    runner = CliRunner()
    db_path = str(tmp_path / "nonexistent.db")
    result = runner.invoke(cli, ["--db", db_path, "stats"])
    assert "not found" in result.output.lower() or result.exit_code != 0


def test_cli_list_empty(tmp_path) -> None:
    """list on an empty DB should show no sources."""
    db_path = str(tmp_path / "empty.db")
    # Create the empty database
    db = SearchDB(db_path)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "list"])
    assert result.exit_code == 0
    assert "No indexed sources" in result.output


def test_cli_migrate_paths_dry_run(tmp_path) -> None:
    """migrate-paths --dry-run should show plan without modifying DB."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    db.set_base_dir(str(tmp_path))
    abs_source = str(tmp_path / "docs" / "readme.md")
    db.index_document(abs_source, 0, "Migrate test", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output.lower()
    assert "To convert: 1" in result.output

    # Verify DB was NOT modified
    db2 = SearchDB(db_path)
    sources = [r[0] for r in db2._execute("SELECT DISTINCT source FROM docs")]
    db2.close()
    assert Path(sources[0]).is_absolute()


def test_cli_migrate_paths_actual(tmp_path) -> None:
    """migrate-paths without --dry-run should convert paths."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    db.set_base_dir(str(tmp_path))
    abs_source = str(tmp_path / "docs" / "readme.md")
    db.index_document(abs_source, 0, "Migrate test", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths"])
    assert result.exit_code == 0
    assert "Migrated 1 source" in result.output

    # Verify DB WAS modified
    db2 = SearchDB(db_path)
    sources = [r[0] for r in db2._execute("SELECT DISTINCT source FROM docs")]
    db2.close()
    assert not Path(sources[0]).is_absolute()


def test_cli_migrate_paths_collision(tmp_path) -> None:
    """migrate-paths should report collisions and skip them."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    db.set_base_dir(str(tmp_path))
    # Insert relative + absolute that would collide
    db.index_document("docs/readme.md", 0, "Relative", DUMMY_EMBEDDING)
    abs_source = str(tmp_path / "docs" / "readme.md")
    db.index_document(abs_source, 1, "Absolute", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths", "--dry-run"])
    assert result.exit_code == 0
    assert "Collisions (skipped): 1" in result.output
    assert "SKIPPED" in result.output


def test_cli_migrate_paths_outside_base_dir(tmp_path) -> None:
    """migrate-paths should abort when paths fall outside base_dir."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    # Set base_dir to a subdirectory so the indexed path is outside it
    sub = tmp_path / "narrow"
    sub.mkdir()
    db.set_base_dir(str(sub))
    abs_source = str(tmp_path / "outside" / "file.md")
    db.index_document(abs_source, 0, "Outside base_dir", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    # Without --force, should fail
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths", "--dry-run"])
    assert result.exit_code != 0
    assert "fall outside base directory" in result.output or "fall outside base directory" in (
        result.output + (result.stderr if hasattr(result, "stderr") else "")
    )

    # With --force, should succeed
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths", "--dry-run", "--force"])
    assert result.exit_code == 0
    assert "To convert: 1" in result.output


def test_cli_migrate_paths_rebase_old_base_dir(tmp_path) -> None:
    """--old-base-dir should rebase ../  relative paths against the correct base."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    # Simulate the VM scenario: base_dir is workspace, but paths were computed
    # relative to a different directory (e.g. the DB location 3 levels deep)
    workspace = tmp_path / "home" / "eva" / ".openclaw" / "workspace"
    workspace.mkdir(parents=True)
    old_base = tmp_path / "home" / "eva" / ".local" / "share" / "fastsearch"
    old_base.mkdir(parents=True)
    db.set_base_dir(str(workspace))

    # The file is at workspace/MEMORY.md, but was stored as a relative path
    # computed from old_base (the DB directory), giving ../../../.openclaw/workspace/MEMORY.md
    import os

    bad_rel = os.path.relpath(
        str(workspace / "MEMORY.md"), str(old_base)
    )
    assert bad_rel.startswith("..")  # Sanity check: it's a ../ path
    db.index_document(bad_rel, 0, "Misaligned relative", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    # Dry-run should show the rebase plan
    result = runner.invoke(
        cli,
        ["--db", db_path, "migrate-paths", "--dry-run", "--old-base-dir", str(old_base)],
    )
    assert result.exit_code == 0
    assert "To convert: 1" in result.output
    assert "MEMORY.md" in result.output
    assert "dry-run" in result.output.lower()

    # Actual run should fix the path
    result = runner.invoke(
        cli,
        ["--db", db_path, "migrate-paths", "--old-base-dir", str(old_base)],
    )
    assert result.exit_code == 0
    assert "Migrated 1 source" in result.output

    # Verify the path is now clean
    db2 = SearchDB(db_path)
    sources = [r[0] for r in db2._execute("SELECT DISTINCT source FROM docs")]
    db2.close()
    assert sources == ["MEMORY.md"]


def test_cli_migrate_paths_rebase_no_change(tmp_path) -> None:
    """--old-base-dir should skip paths that are already correct."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    db.set_base_dir(str(tmp_path))
    # Already clean relative path
    db.index_document("docs/file.md", 0, "Clean path", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db", db_path, "migrate-paths", "--old-base-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "already relative" in result.output.lower()


def test_cli_migrate_paths_all_relative(tmp_path) -> None:
    """migrate-paths on a DB with only relative paths should say nothing to do."""
    db_path = str(tmp_path / "migrate.db")
    db = SearchDB(db_path)
    db.index_document("relative/path.md", 0, "Already relative", DUMMY_EMBEDDING)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "migrate-paths"])
    assert result.exit_code == 0
    assert "already relative" in result.output.lower()
