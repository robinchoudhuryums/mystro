from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from masterbench import db
from masterbench.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and result.stdout.strip() == "0.1.0"


def test_init_creates_data_root_and_database(data_root: Path) -> None:
    first = runner.invoke(app, ["init"])
    assert first.exit_code == 0, first.output
    assert "001_init.sql" in first.stdout
    assert (data_root / "Inbox" / "electronic").is_dir()

    conn = db.connect(data_root / "masterbench.db")
    assert db.applied_versions(conn) == [m.version for m in db.load_migrations()]
    conn.close()

    second = runner.invoke(app, ["init"])
    assert second.exit_code == 0 and "up to date" in second.stdout


def test_fixtures_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fixtures", str(tmp_path / "fx")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "fx" / "loop.wav").is_file()
    assert (tmp_path / "fx" / "truth.json").is_file()
