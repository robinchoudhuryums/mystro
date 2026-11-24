"""Progress tracker: how the mixes measure up from track to track."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

from masterbench.analysis.compare import band_label
from masterbench.config import Thresholds


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
    delta_vs: str | None = None
    hook_contrast_lu: float | None = None
    bass_in_tune_share: float | None = None
    vocal_in_tune_share: float | None = None
    vocal_mean_cents: float | None = None


@dataclass(frozen=True)
class Habit:
    """Something that keeps happening across tracks."""

    code: str
    statement: str
    tracks: int
    considered: int
    mean_value: float | None
    confidence: float
    examples: list[str]

    @property
    def share(self) -> float:
        return self.tracks / self.considered


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
        " a.lufs_i, a.true_peak_dbtp, a.lra, a.crest_db, a.correlation, a.hook_contrast_lu,"
        " a.bass_in_tune_share"
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
            "SELECT band, delta_db, vs FROM band_deltas WHERE analysis_id = ? AND section = 'full'"
            " AND vs = (SELECT vs FROM band_deltas WHERE analysis_id = ? AND section = 'full'"
            "   ORDER BY vs = 'target_curve' DESC LIMIT 1)",
            (row["analysis_id"], row["analysis_id"]),
        ).fetchall()
        flags = conn.execute(
            "SELECT code FROM flags WHERE analysis_id = ? ORDER BY code", (row["analysis_id"],)
        ).fetchall()
        vocal = conn.execute(
            "SELECT v.in_tune_share, v.mean_cents FROM vocal_reports v JOIN tracks t"
            " ON t.id = v.track_id WHERE t.slug = ? ORDER BY v.created_at DESC, v.id LIMIT 1",
            (row["slug"],),
        ).fetchone()
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
                delta_vs=deltas[0]["vs"] if deltas else None,
                hook_contrast_lu=row["hook_contrast_lu"],
                bass_in_tune_share=row["bass_in_tune_share"],
                vocal_in_tune_share=vocal["in_tune_share"] if vocal else None,
                vocal_mean_cents=vocal["mean_cents"] if vocal else None,
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


HABIT_WINDOW = 8
MIN_HABIT_TRACKS = 3
# A vocal this far off on average, the same way on most tracks, is a singing habit.
VOCAL_BIAS_CENTS = 8.0
# Flags that describe one moment or one file's condition, in the words a habit is told in.
_FLAG_HABITS = {
    "bass_detuned": "808 or bass notes are out of tune",
    "bass_out_of_key": "bass notes fall outside the song's key",
    "bass_small_speakers": "the bass has too few harmonics to carry on small speakers",
    "weak_hook_contrast": "the hook barely lifts over the verse",
    "mud": "the low-mids (200 to 500 Hz) are crowded",
    "harsh": "2 to 5 kHz is pushed",
    "harsh_at": "there are harsh moments in the upper mids",
    "resonance": "a narrow resonance stands out",
    "low_end_not_mono": "the low end is not mono",
    "wide_stereo": "the stereo image is wide enough to change in mono",
    "phase_negative": "the mix partly cancels in mono",
    "phase_at": "stretches of the mix cancel in mono",
    "clipping": "the bounce is clipped",
    "already_mastered": "the bounce arrives already limited or very loud",
    "dc_offset": "the bounce has a DC offset",
}
_YARDSTICKS = {"target_curve": "your genre curve", "reference": "the reference"}


def wilson_lower(hits: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the 95% interval on a proportion: high only with enough tracks.

    Three out of three could be chance; fifteen out of sixteen could not. This is what turns
    "it happened every time" into a confidence that grows with the number of tracks.
    """
    if total == 0:
        return 0.0
    share = hits / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt((share * (1 - share) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / (1 + z * z / total))


def habits(points: list[TrackPoint], limits: Thresholds, window: int = HABIT_WINDOW) -> list[Habit]:
    """Consistent biases across the most recent tracks, most certain first.

    Two kinds: a band that sits the same side of the yardstick on most tracks, and a flag
    that keeps being raised. A habit needs at least three tracks and 60% of those considered.
    """
    recent = points[-window:]
    found: list[Habit] = []

    compared = [p for p in recent if p.band_deltas]
    bands = list(compared[0].band_deltas) if compared else []
    for band in bands:
        for sign, word in ((1, "over"), (-1, "under")):
            hits = [
                p for p in compared if sign * p.band_deltas.get(band, 0.0) >= limits.habit_band_db
            ]
            if len(hits) < MIN_HABIT_TRACKS or len(hits) / len(compared) < limits.habit_share:
                continue
            mean = sum(p.band_deltas[band] for p in hits) / len(hits)
            kinds = {p.delta_vs for p in hits}
            only = next(iter(kinds)) if len(kinds) == 1 else None
            yardstick = _YARDSTICKS.get(only or "", "the comparison")
            statement = (
                f"{band_label(band)} is {word} {yardstick} on {len(hits)} of the last "
                f"{len(compared)} tracks, by {abs(mean):.1f} dB on average."
            )
            found.append(
                Habit(
                    code=f"band_{band}_{word}",
                    statement=statement,
                    tracks=len(hits),
                    considered=len(compared),
                    mean_value=round(mean, 2),
                    confidence=round(wilson_lower(len(hits), len(compared)), 2),
                    examples=[p.title for p in hits[-3:]],
                )
            )

    for code, phrase in _FLAG_HABITS.items():
        hits = [p for p in recent if code in p.flags]
        if len(hits) < MIN_HABIT_TRACKS or len(hits) / len(recent) < limits.habit_share:
            continue
        statement = f"On {len(hits)} of the last {len(recent)} tracks, {phrase}."
        found.append(
            Habit(
                code=code,
                statement=statement,
                tracks=len(hits),
                considered=len(recent),
                mean_value=None,
                confidence=round(wilson_lower(len(hits), len(recent)), 2),
                examples=[p.title for p in hits[-3:]],
            )
        )
    sung = [p for p in recent if p.vocal_mean_cents is not None]
    for sign, word in ((1, "sharp"), (-1, "flat")):
        hits = [p for p in sung if sign * (p.vocal_mean_cents or 0.0) >= VOCAL_BIAS_CENTS]
        if len(hits) < MIN_HABIT_TRACKS or len(hits) / len(sung) < limits.habit_share:
            continue
        mean = sum(p.vocal_mean_cents or 0.0 for p in hits) / len(hits)
        statement = (
            f"The vocal is {word} on {len(hits)} of the last {len(sung)} tracks with a pitch "
            f"report, by {abs(mean):.0f} cents on average."
        )
        found.append(
            Habit(
                code=f"vocal_{word}",
                statement=statement,
                tracks=len(hits),
                considered=len(sung),
                mean_value=round(mean, 1),
                confidence=round(wilson_lower(len(hits), len(sung)), 2),
                examples=[p.title for p in hits[-3:]],
            )
        )
    return sorted(found, key=lambda h: (-h.confidence, -h.tracks, h.code))


def blind_summary(results: list[BlindResult]) -> dict[str, int]:
    """How often each kind of version was preferred in blind tests, most often first."""
    wins: dict[str, int] = {}
    for result in results:
        kind = "Master" if result.winner.startswith("Master") else result.winner
        wins[kind] = wins.get(kind, 0) + 1
    return dict(sorted(wins.items(), key=lambda item: (-item[1], item[0])))
