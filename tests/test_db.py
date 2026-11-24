from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from masterbench import db

EXPECTED_TABLES = {
    "analyses",
    "band_deltas",
    "blind_tests",
    "flags",
    "notes",
    "reference_tracks",
    "runs",
    "schema_migrations",
    "sessions",
    "vocal_reports",
    "tracks",
}


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row["name"] for row in rows}


def test_migrations_are_contiguous_from_one() -> None:
    migrations = db.load_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "001_init.sql"


def test_migrate_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "nested" / "index.db")
    assert [m.version for m in db.migrate(conn)] == [m.version for m in db.load_migrations()]
    assert tables(conn) == EXPECTED_TABLES
    assert db.migrate(conn) == []
    assert db.applied_versions(conn) == [m.version for m in db.load_migrations()]


def test_foreign_keys_cascade() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.execute("INSERT INTO tracks (slug, title, genre) VALUES ('song', 'Song', 'hip-hop')")
    conn.execute(
        "INSERT INTO runs (id, track_id, kind, settings_json, path, created_at)"
        " VALUES ('20260101-abc', 1, 'master', '{}', '/x', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO analyses (run_id, role, lufs_i) VALUES ('20260101-abc', 'master', -14)"
    )
    conn.execute(
        "INSERT INTO band_deltas (analysis_id, band, delta_db, vs)"
        " VALUES (1, 'sub', -2.0, 'reference')"
    )
    conn.execute("DELETE FROM tracks WHERE slug = 'song'")
    for table in ("runs", "analyses", "band_deltas"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_constraints_reject_bad_rows() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.execute("INSERT INTO tracks (slug, title) VALUES ('song', 'Song')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO runs (id, track_id, kind, settings_json, path, created_at)"
            " VALUES ('r', 1, 'remix', '{}', '/x', '2026-01-01T00:00:00Z')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO analyses (role) VALUES ('mix')")


def test_failed_migration_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    good = db.load_migrations()
    broken = db.Migration(len(good) + 1, "999_broken.sql", "CREATE TABLE half (id INTEGER); NOPE;")
    monkeypatch.setattr(db, "load_migrations", lambda: [*good, broken])
    conn = db.connect(":memory:")
    with pytest.raises(db.MigrationError, match="999_broken.sql"):
        db.migrate(conn)
    assert "half" not in tables(conn)
    assert db.applied_versions(conn) == [m.version for m in good]


def test_database_newer_than_code_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.execute("INSERT INTO schema_migrations (version, name) VALUES (99, '099_future.sql')")
    with pytest.raises(db.MigrationError, match="doesn't know"):
        db.migrate(conn)
