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
    assert "0.3.1" in result.output


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
