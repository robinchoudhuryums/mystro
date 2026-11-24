"""Progress tracker: how the mixes measure up from track to track."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPoint:
    """The mix of one track, as measured in its latest run."""

    slug: str
    title: str
    genre: str | None
    created_at: str
    run_id: str
    lufs_i: float
    true_peak_dbtp: float
    lra: float
    crest_db: float
    correlation: float | None
    band_deltas: dict[str, float]
    flags: list[str]


@dataclass(frozen=True)
class FlagCount:
    code: str
    tracks: int
    share: float


@dataclass(frozen=True)
class BlindResult:
    created_at: str
    title: str
    winner: str
    ranking: list[str]


def track_points(conn: sqlite3.Connection) -> list[TrackPoint]:
    """One point per track, oldest track first, from the newest run that analyzed its mix."""
    rows = conn.execute(
        "SELECT t.slug, t.title, t.genre, r.id AS run_id, r.created_at, a.id AS analysis_id,"
        " a.lufs_i, a.true_peak_dbtp, a.lra, a.crest_db, a.correlation"
        " FROM tracks t JOIN runs r ON r.track_id = t.id"
        " JOIN analyses a ON a.run_id = r.id AND a.role = 'mix'"
        " WHERE r.id = (SELECT r2.id FROM runs r2 JOIN analyses a2 ON a2.run_id = r2.id"
        "   WHERE r2.track_id = t.id AND a2.role = 'mix'"
        "   ORDER BY r2.created_at DESC, r2.id DESC LIMIT 1)"
        " ORDER BY t.created_at, t.slug"
    ).fetchall()
    points = []
    for row in rows:
        # The genre curve is the steadier yardstick; fall back to the single reference.
        deltas = conn.execute(
            "SELECT band, delta_db FROM band_deltas WHERE analysis_id = ? AND section = 'full'"
            " AND vs = (SELECT vs FROM band_deltas WHERE analysis_id = ? AND section = 'full'"
            "   ORDER BY vs = 'target_curve' DESC LIMIT 1)",
            (row["analysis_id"], row["analysis_id"]),
        ).fetchall()
        flags = conn.execute(
            "SELECT code FROM flags WHERE analysis_id = ? ORDER BY code", (row["analysis_id"],)
        ).fetchall()
        points.append(
            TrackPoint(
                slug=row["slug"],
                title=row["title"],
                genre=row["genre"],
                created_at=row["created_at"],
                run_id=row["run_id"],
                lufs_i=row["lufs_i"],
                true_peak_dbtp=row["true_peak_dbtp"],
                lra=row["lra"],
                crest_db=row["crest_db"],
                correlation=row["correlation"],
                band_deltas={d["band"]: d["delta_db"] for d in deltas},
                flags=[f["code"] for f in flags],
            )
        )
    return points


def flag_counts(points: list[TrackPoint]) -> list[FlagCount]:
    """How many tracks raise each flag, most common first."""
    counts: dict[str, int] = {}
    for point in points:
        for code in set(point.flags):
            counts[code] = counts.get(code, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [FlagCount(code, n, n / len(points)) for code, n in ordered]


def blind_results(conn: sqlite3.Connection) -> list[BlindResult]:
    """Blind tests, newest first, with labels restored."""
    rows = conn.execute(
        "SELECT b.created_at, t.title, b.candidates_json, b.ranking_json, b.revealed_json"
        " FROM blind_tests b JOIN tracks t ON t.id = b.track_id"
        " ORDER BY b.created_at DESC, b.id DESC"
    ).fetchall()
    results = []
    for row in rows:
        labels = {c["id"]: c["label"] for c in json.loads(row["candidates_json"])}
        revealed = json.loads(row["revealed_json"])
        ranking = [labels[revealed[letter]] for letter in json.loads(row["ranking_json"])]
        results.append(BlindResult(row["created_at"], row["title"], ranking[0], ranking))
    return results
