"""Writes run manifests, analyses and references into the SQLite index, and rebuilds it."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from masterbench import blind, db, projects
from masterbench.analysis.report import AnalysisReport
from masterbench.config import Paths
from masterbench.projects import RunManifest, Track
from masterbench.references import library


@contextmanager
def open_index(paths: Paths) -> Iterator[sqlite3.Connection]:
    """Open the index, creating it if needed. A brand-new index is filled from disk, so a
    deleted database, or a synced data root opened on another machine, heals itself."""
    conn = db.connect(paths.db)
    try:
        created = any(migration.version == 1 for migration in db.migrate(conn))
        if created and paths.projects.is_dir():
            reindex(conn, paths)
        yield conn
    finally:
        conn.close()


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _upsert_track(conn: sqlite3.Connection, track: Track) -> int:
    row = conn.execute(
        "INSERT INTO tracks (slug, title, genre, created_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (slug) DO UPDATE SET title = excluded.title, genre = excluded.genre"
        " RETURNING id",
        (track.slug, track.title, track.genre, track.created_at),
    ).fetchone()
    return int(row[0])


def _insert_analysis(
    conn: sqlite3.Connection, report: AnalysisReport, run_id: str | None, created_at: str
) -> int:
    loud, info = report.loudness, report.info
    cursor = conn.execute(
        "INSERT INTO analyses (run_id, source_path, role, sha256, sample_rate, channels, bit_depth,"
        " duration_s, lufs_i, lufs_s_max, lufs_m_max, true_peak_dbtp, sample_peak_dbfs, lra, plr,"
        " crest_db, correlation, low_mono_corr, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            report.source_path,
            report.role,
            report.sha256,
            info.sample_rate,
            info.channels,
            info.bit_depth,
            info.duration_s,
            loud.lufs_i,
            loud.lufs_s_max,
            loud.lufs_m_max,
            loud.true_peak_dbtp,
            loud.sample_peak_dbfs,
            loud.lra,
            loud.plr,
            loud.crest_db,
            report.stereo.correlation if report.stereo else None,
            report.stereo.low_mono_corr if report.stereo else None,
            created_at,
        ),
    )
    analysis_id = cursor.lastrowid
    assert analysis_id is not None
    conn.executemany(
        "INSERT INTO flags (analysis_id, code, severity, message, value) VALUES (?, ?, ?, ?, ?)",
        [(analysis_id, f.code, f.severity, f.message, f.value) for f in report.flags],
    )
    conn.executemany(
        "INSERT INTO band_deltas (analysis_id, band, delta_db, vs) VALUES (?, ?, ?, 'reference')",
        [(analysis_id, band, delta) for band, delta in (report.band_deltas or {}).items()],
    )
    return analysis_id


def _index_run(
    conn: sqlite3.Connection,
    manifest: RunManifest,
    run_dir: Path,
    reports: dict[str, AnalysisReport],
) -> None:
    track_id = _upsert_track(conn, manifest.track)
    conn.execute("DELETE FROM runs WHERE id = ?", (manifest.id,))
    conn.execute(
        "INSERT INTO runs (id, track_id, kind, settings_json, path, lib_versions_json, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            manifest.id,
            track_id,
            manifest.kind,
            json.dumps(manifest.settings, sort_keys=True),
            str(run_dir),
            json.dumps(manifest.lib_versions, sort_keys=True),
            manifest.created_at,
        ),
    )
    for name in sorted(reports):
        _insert_analysis(conn, reports[name], manifest.id, manifest.created_at)


def index_run(
    conn: sqlite3.Connection,
    manifest: RunManifest,
    run_dir: Path,
    reports: dict[str, AnalysisReport],
) -> None:
    with _transaction(conn):
        _index_run(conn, manifest, run_dir, reports)


def _sync_references(conn: sqlite3.Connection, paths: Paths) -> int:
    conn.execute(
        "DELETE FROM analyses WHERE run_id IS NULL"
        " AND source_path IN (SELECT path FROM reference_tracks)"
    )
    conn.execute("DELETE FROM reference_tracks")
    references = library.list_references(paths)
    for ref in references:
        conn.execute(
            "INSERT INTO reference_tracks (genre, name, path, is_default, notes)"
            " VALUES (?, ?, ?, ?, ?)",
            (ref.genre, ref.name, str(ref.path), int(ref.is_default), ref.notes),
        )
        _insert_analysis(conn, ref.report, None, ref.added_at or projects.now_iso())
    return len(references)


def sync_references(conn: sqlite3.Connection, paths: Paths) -> int:
    """Make the reference tables mirror the library on disk."""
    with _transaction(conn):
        return _sync_references(conn, paths)


def _sync_blind_tests(conn: sqlite3.Connection, paths: Paths) -> int:
    conn.execute("DELETE FROM blind_tests")
    tests = blind.load_all(paths)
    for test in tests:
        conn.execute(
            "INSERT INTO blind_tests (track_id, candidates_json, ranking_json, revealed_json,"
            " created_at) SELECT id, ?, ?, ?, ? FROM tracks WHERE slug = ?",
            (
                json.dumps(test.candidates),
                json.dumps(test.ranking),
                json.dumps(test.revealed),
                test.created_at,
                test.track_slug,
            ),
        )
    return len(tests)


def sync_blind_tests(conn: sqlite3.Connection, paths: Paths) -> int:
    """Make the blind_tests table mirror the result files on disk."""
    with _transaction(conn):
        return _sync_blind_tests(conn, paths)


def _sync_notes(conn: sqlite3.Connection, paths: Paths) -> int:
    """AI notes live in each run's notes.json; the table mirrors them for spend tracking."""
    conn.execute("DELETE FROM notes")
    files = sorted(paths.projects.glob("*/runs/*/notes.json"))
    for file in files:
        note = json.loads(file.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO notes (run_id, model, cost_usd, payload_hash, body_json, created_at)"
            " SELECT id, ?, ?, ?, ?, ? FROM runs WHERE id = ?",
            (
                note["model"],
                note["cost_usd"],
                note["payload_hash"],
                json.dumps(note["body"], sort_keys=True),
                note["created_at"],
                file.parent.name,
            ),
        )
    return len(files)


def sync_notes(conn: sqlite3.Connection, paths: Paths) -> int:
    with _transaction(conn):
        return _sync_notes(conn, paths)


def reindex(conn: sqlite3.Connection, paths: Paths) -> dict[str, int]:
    """Rebuild every row from the files on disk: tracks, runs, analyses, references, tests."""
    with _transaction(conn):
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM analyses")
        on_disk = []
        for track_file in sorted(paths.projects.glob("*/track.json")):
            track = Track(**json.loads(track_file.read_text(encoding="utf-8")))
            on_disk.append(track.slug)
            _upsert_track(conn, track)
        marks = ", ".join("?" * len(on_disk))
        conn.execute(f"DELETE FROM tracks WHERE slug NOT IN ({marks})", on_disk)
        run_dirs = projects.iter_run_dirs(paths)
        for run_dir in run_dirs:
            manifest, reports = projects.read_run(run_dir)
            _index_run(conn, manifest, run_dir, reports)
        references = _sync_references(conn, paths)
        blind_tests = _sync_blind_tests(conn, paths)
        notes = _sync_notes(conn, paths)
    tracks = conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
    return {
        "tracks": tracks,
        "runs": len(run_dirs),
        "references": references,
        "blind tests": blind_tests,
        "notes": notes,
    }


def input_already_run(conn: sqlite3.Connection, sha256: str, kind: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM analyses a JOIN runs r ON r.id = a.run_id"
        " WHERE a.sha256 = ? AND a.role = 'mix' AND r.kind = ? LIMIT 1",
        (sha256, kind),
    ).fetchone()
    return row is not None


def recent_runs(
    conn: sqlite3.Connection, limit: int = 50, kind: str | None = None
) -> list[sqlite3.Row]:
    where, args = ("WHERE r.kind = ?", [kind]) if kind else ("", [])
    return conn.execute(
        "SELECT r.id, r.kind, r.path, r.created_at, t.slug, t.title, t.genre"
        f" FROM runs r JOIN tracks t ON t.id = r.track_id {where}"
        " ORDER BY r.created_at DESC, r.id DESC LIMIT ?",
        [*args, limit],
    ).fetchall()
