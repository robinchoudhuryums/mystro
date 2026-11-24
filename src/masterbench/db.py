"""SQLite index: connection handling and numbered SQL migrations."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


class MigrationError(RuntimeError):
    pass


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the index DB. Transactions are managed explicitly (autocommit mode)."""
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def load_migrations() -> list[Migration]:
    migrations = []
    for entry in resources.files("masterbench.migrations").iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _MIGRATION_NAME.match(entry.name)
        if match is None:
            raise MigrationError(f"bad migration file name: {entry.name}")
        migrations.append(
            Migration(int(match.group(1)), entry.name, entry.read_text(encoding="utf-8"))
        )
    migrations.sort(key=lambda m: m.version)
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration versions must be contiguous from 001, got {versions}")
    return migrations


def applied_versions(conn: sqlite3.Connection) -> list[int]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))"
    )
    return [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]


def migrate(conn: sqlite3.Connection) -> list[Migration]:
    """Apply pending migrations in order, each in its own transaction. Returns those applied."""
    migrations = load_migrations()
    done = applied_versions(conn)
    known = {m.version for m in migrations}
    unknown = [v for v in done if v not in known]
    if unknown:
        raise MigrationError(f"database has migrations this code doesn't know: {unknown}")

    applied = []
    for migration in migrations:
        if migration.version in done:
            continue
        # executescript() commits implicitly, so the transaction lives inside the script:
        # a failing statement aborts before COMMIT and the ROLLBACK below undoes the rest.
        record = (
            "INSERT INTO schema_migrations (version, name) "
            f"VALUES ({migration.version}, '{migration.name}');"
        )
        try:
            conn.executescript(f"BEGIN;\n{migration.sql}\n{record}\nCOMMIT;")
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise MigrationError(f"{migration.name} failed: {exc}") from exc
        applied.append(migration)
    return applied
