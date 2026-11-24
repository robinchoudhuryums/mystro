"""MCP server: lets Claude Code or the Claude app drive the engine over stdio.

Tools return compact JSON with file paths, never audio. The server is read-only unless
`mcp_allow_writes = true` in settings.toml, and reads audio only from the data root and the
folders listed in `mcp_allowed_dirs`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from masterbench import audio_io, batch, index, progress
from masterbench.analysis.report import AnalysisReport
from masterbench.config import BANDS, Paths, Settings
from masterbench.master import pipeline
from masterbench.master.matchering_engine import MasteringError
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.references import library, suggest
from masterbench.vocal import pitch_report as vocal_report

INSTRUCTIONS = (
    "MasterBench masters and analyzes the user's own music on their machine. Runs are "
    "identified by ids such as 20261124-185126-7f55cf; list_runs finds them. Levels are in "
    "LUFS and dBTP. Band figures are in dB relative to each file's own loudness, so "
    "differences between files compare tonal balance at equal loudness. Results are "
    "measurements: nothing here has heard the audio."
)
WRITES_OFF = (
    "this server is read-only. To let it create runs, set mcp_allow_writes = true in "
    "settings.toml in the MasterBench data folder and restart it"
)


def _round(values: dict[str, float] | None) -> dict[str, float] | None:
    return {key: round(value, 2) for key, value in values.items()} if values else None


def summarize(report: AnalysisReport, detail: bool = True) -> dict[str, Any]:
    """The parts of an analysis worth an assistant's context, without the long curves."""
    loud = report.loudness
    out: dict[str, Any] = {
        "role": report.role,
        "file": report.source_path,
        "duration_s": round(report.info.duration_s, 1),
        "lufs_i": round(loud.lufs_i, 2),
        "true_peak_dbtp": round(loud.true_peak_dbtp, 2),
        "lra": round(loud.lra, 2),
        "crest_db": round(loud.crest_db, 2),
        "flags": [
            {"code": f.code, "severity": f.severity, "message": f.message, "at_s": f.start_s}
            for f in report.flags
        ],
    }
    if not detail:
        return out
    out["bands_db"] = _round(report.bands)
    out["band_deltas_vs_reference_db"] = _round(report.band_deltas)
    out["band_deltas_vs_genre_curve_db"] = _round(report.curve_deltas)
    if report.stereo:
        out["stereo"] = {
            "correlation": report.stereo.correlation,
            "side_vs_mid_db": report.stereo.side_mid_db,
            "correlation_below_120hz": report.stereo.low_mono_corr,
        }
    if report.musical:
        out["key"] = report.musical.key
        out["key_confidence"] = report.musical.key_confidence
        out["tempo_bpm"] = report.musical.tempo_bpm
    if report.sections:
        out["sections"] = {
            "source": report.sections.source,
            "hook_minus_verse_lu": report.sections.contrast_lu,
            "items": [
                {"label": s.label, "start_s": s.start_s, "end_s": s.end_s, "lufs": s.lufs}
                for s in report.sections.items
            ],
        }
    if report.lowend:
        low = report.lowend
        out["low_end"] = {
            "bass_in_tune_share": low.in_tune_share,
            "notes_off": [
                {"at_s": n.start_s, "note": n.note, "cents": n.cents, "in_key": n.in_key}
                for n in low.notes
                if abs(n.cents) > 15 or n.in_key is False
            ][:12],
            "harmonics_db_below_fundamental": low.harmonics_db,
        }
    return out


def run_summary(run: RunResult, detail: bool = False) -> dict[str, Any]:
    used = run.manifest.settings
    return {
        "run_id": run.manifest.id,
        "track": run.manifest.track.slug,
        "title": run.manifest.track.title,
        "genre": run.manifest.track.genre,
        "kind": run.manifest.kind,
        "created_at": run.manifest.created_at,
        "mode": used.get("mode"),
        "reference": (used.get("reference") or {}).get("spec"),
        "target_lufs": used.get("target_lufs"),
        "ceiling_dbtp": used.get("ceiling_dbtp"),
        "folder": str(run.run_dir),
        "files": {name: str(run.output(name)) for name in run.manifest.outputs},
        "analyses": {role: summarize(r, detail) for role, r in sorted(run.reports.items())},
    }


def compare(a: RunResult, b: RunResult, role: str) -> dict[str, Any]:
    """What changed from run a to run b, for the mix or the master."""
    if role not in a.reports or role not in b.reports:
        have = sorted(set(a.reports) & set(b.reports))
        raise ValueError(f"both runs need a '{role}' analysis; they share: {', '.join(have)}")
    first, second = a.reports[role], b.reports[role]
    loud_a, loud_b = first.loudness, second.loudness
    out: dict[str, Any] = {
        "role": role,
        "a": {"run_id": a.manifest.id, "track": a.manifest.track.slug, "file": first.source_path},
        "b": {"run_id": b.manifest.id, "track": b.manifest.track.slug, "file": second.source_path},
        "b_minus_a": {
            "lufs_i": round(loud_b.lufs_i - loud_a.lufs_i, 2),
            "true_peak_db": round(loud_b.true_peak_dbtp - loud_a.true_peak_dbtp, 2),
            "lra": round(loud_b.lra - loud_a.lra, 2),
            "crest_db": round(loud_b.crest_db - loud_a.crest_db, 2),
        },
    }
    if first.bands and second.bands:
        out["band_level_b_minus_a_db"] = {
            band: round(second.bands[band] - first.bands[band], 2) for band in BANDS
        }
    if first.sections and second.sections:
        theirs = {item.label: item for item in first.sections.items}
        out["section_lufs_b_minus_a"] = {
            item.label: round(item.lufs - theirs[item.label].lufs, 2)
            for item in second.sections.items
            if item.label in theirs
        }
        out["hook_minus_verse_lu"] = {
            "a": first.sections.contrast_lu,
            "b": second.sections.contrast_lu,
        }
    codes_a, codes_b = {f.code for f in first.flags}, {f.code for f in second.flags}
    out["flags"] = {
        "only_in_a": sorted(codes_a - codes_b),
        "only_in_b": sorted(codes_b - codes_a),
        "in_both": sorted(codes_a & codes_b),
    }
    return out


# Mistakes a caller can fix (a wrong id, a path outside the allowed folders, writes turned
# off). Their message goes back to the caller; anything else is a crash and stays generic.
EXPECTED_ERRORS = (ValueError, LookupError, PermissionError, MasteringError, audio_io.AudioError)


def explained(tool: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Turn the engine's own errors into tool errors that carry their message."""

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return tool(*args, **kwargs)
        except EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def build_server(paths: Paths, settings: Settings) -> MCPServer:
    server = MCPServer("masterbench", instructions=INSTRUCTIONS)
    allowed = [
        paths.root.resolve(),
        *[Path(d).expanduser().resolve() for d in settings.mcp_allowed_dirs],
    ]

    def readable(path: str) -> Path:
        file = Path(path).expanduser().resolve()
        if not any(file.is_relative_to(folder) for folder in allowed):
            folders = ", ".join(str(folder) for folder in allowed)
            raise ValueError(
                f"{file} is outside the folders this server may read ({folders}). Add its folder "
                "to mcp_allowed_dirs in settings.toml to allow it."
            )
        if not file.is_file():
            raise ValueError(f"no such file: {file}")
        return file

    def need_writes() -> None:
        if not settings.mcp_allow_writes:
            raise PermissionError(WRITES_OFF)

    def load(run_id: str) -> RunResult:
        return pipeline.load_run(paths, run_id)

    @server.tool()
    @explained
    def list_tracks() -> dict[str, Any]:
        """Every track MasterBench knows, with how many runs it has and its latest run id."""
        with index.open_index(paths) as conn:
            rows = conn.execute(
                "SELECT t.slug, t.title, t.genre, count(r.id) AS runs, max(r.id) AS latest"
                " FROM tracks t LEFT JOIN runs r ON r.track_id = t.id"
                " GROUP BY t.id ORDER BY max(r.created_at) DESC"
            ).fetchall()
        return {"tracks": [dict(row) for row in rows]}

    @server.tool()
    @explained
    def list_runs(
        track: str | None = None, kind: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Recent runs, newest first. track is a slug; kind is 'master' or 'analysis_only'."""
        with index.open_index(paths) as conn:
            rows = index.recent_runs(conn, limit=200, kind=kind)
        runs = [
            {
                "run_id": r["id"],
                "track": r["slug"],
                "title": r["title"],
                "kind": r["kind"],
                "genre": r["genre"],
                "created_at": r["created_at"],
            }
            for r in rows
            if track is None or r["slug"] == track
        ]
        return {"runs": runs[: max(1, limit)]}

    @server.tool()
    @explained
    def get_analysis(run_id: str) -> dict[str, Any]:
        """A run in full: settings, output files, and the analysis of its mix, master and
        reference (loudness, band balance, stereo, key, tempo, sections, low end, flags)."""
        return run_summary(load(run_id), detail=True)

    @server.tool()
    @explained
    def compare_runs(run_a: str, run_b: str, role: str = "master") -> dict[str, Any]:
        """What changed between two runs: loudness, band balance at equal loudness, section
        loudness and flags. role is 'master' (default) or 'mix'."""
        return compare(load(run_a), load(run_b), role)

    @server.tool()
    @explained
    def analyze_file(
        path: str, genre: str | None = None, title: str | None = None
    ) -> dict[str, Any]:
        """Analyze a mix. Compares it with the genre's default reference and target curve when
        they exist. Stored as a run only when the server allows writes."""
        source = readable(path)
        if settings.mcp_allow_writes:
            return {
                "saved": True,
                **run_summary(pipeline.analyze(source, paths, settings, genre, title), True),
            }
        reports = pipeline.measure(source, paths, settings, genre)
        analyses = {role: summarize(r, True) for role, r in sorted(reports.items())}
        return {"saved": False, "analyses": analyses}

    @server.tool()
    @explained
    def pitch_report(path: str, key: str | None = None, scale: str | None = None) -> dict[str, Any]:
        """How in tune a solo vocal is against a scale: time within ±20 cents, the average
        distance, the notes furthest out and the worst phrases. Give the song's key (e.g.
        'F minor') when known; otherwise it is guessed from the sung notes. Read-only."""
        source = readable(path)
        audio, info = audio_io.load(source)
        found = vocal_report.analyze(audio, info.sample_rate, key, scale)
        furthest = sorted(found.notes, key=lambda note: -abs(note.cents))[:12]
        return {
            "file": str(source),
            "key": found.key,
            "key_source": found.key_source,
            "key_confidence": found.key_confidence,
            "scale": found.scale,
            "notes": len(found.notes),
            "in_tune_share": found.in_tune_share,
            "mean_cents": found.mean_cents,
            "mean_abs_cents": found.mean_abs_cents,
            "out_of_key_notes": len(found.out_of_key),
            "furthest_notes": [
                {
                    "number": note.index + 1,
                    "at_s": note.start_s,
                    "sung": note.name,
                    "target": note.target_name,
                    "cents": note.cents,
                }
                for note in furthest
                if abs(note.cents) >= 10
            ],
            "worst_phrases": [
                {
                    "from_s": phrase.start_s,
                    "to_s": phrase.end_s,
                    "notes": phrase.notes,
                    "mean_abs_cents": phrase.mean_abs_cents,
                }
                for phrase in found.worst_phrases
            ],
            "flags": [
                {"code": f.code, "severity": f.severity, "message": f.message} for f in found.flags
            ],
        }

    @server.tool()
    @explained
    def suggest_reference(path: str, genre: str | None = None) -> dict[str, Any]:
        """Rank a genre's reference tracks by how close they sit to a mix."""
        source = readable(path)
        report, _ = library.analyze(source, settings)
        wanted = genre or settings.default_genre
        found = suggest.suggest(report, library.list_references(paths, wanted))
        return {
            "genre": wanted,
            "suggestions": [
                {"reference": s.reference.spec, "similarity": s.similarity, "why": s.why}
                for s in found
            ],
        }

    @server.tool()
    @explained
    def get_progress_trends(limit: int = 12) -> dict[str, Any]:
        """Each track's latest mix, oldest first: loudness, band differences from the genre
        curve or reference, hook lift, bass tuning and flags. For spotting trends."""
        with index.open_index(paths) as conn:
            points = progress.track_points(conn)
            wins = progress.blind_summary(progress.blind_results(conn))
        return {
            "tracks": [asdict(point) for point in points[-max(1, limit) :]],
            "blind_test_wins": wins,
        }

    @server.tool()
    @explained
    def get_habits() -> dict[str, Any]:
        """Recurring habits across the last eight tracks, most certain first. confidence is
        the lower bound of a 95% interval on the share of tracks, so it grows with the count."""
        with index.open_index(paths) as conn:
            points = progress.track_points(conn)
        found = progress.habits(points, settings.thresholds)
        considered = min(len(points), progress.HABIT_WINDOW)
        return {"tracks_considered": considered, "habits": [asdict(habit) for habit in found]}

    @server.tool()
    @explained
    def master_file(
        path: str,
        reference: str | None = None,
        genre: str | None = None,
        match: str = "reference",
        target_lufs: float | None = None,
        ceiling_dbtp: float | None = None,
        pre_chain: str | None = None,
        post_chain: str | None = None,
    ) -> dict[str, Any]:
        """Master a mix as a new run; the input is never changed. match is 'reference' (one
        track; reference is 'genre/name' or defaults to the genre's default) or 'curve' (the
        genre's target curve). Needs writes enabled."""
        need_writes()
        if match not in ("reference", "curve"):
            raise ValueError("match must be 'reference' or 'curve'")
        source = readable(path)
        options = MasterOptions(
            reference=reference or f"{genre or settings.default_genre}/default",
            target_lufs=settings.targets.lufs_i if target_lufs is None else target_lufs,
            ceiling_dbtp=settings.targets.true_peak_dbtp if ceiling_dbtp is None else ceiling_dbtp,
            genre=genre,
            pre_chain=pre_chain,
            post_chain=post_chain,
            mode="curve" if match == "curve" else "matchering",
        )
        return run_summary(pipeline.master(source, options, paths, settings))

    @server.tool()
    @explained
    def run_batch(
        file_paths: list[str],
        reference: str | None = None,
        genre: str | None = None,
        name: str = "",
    ) -> dict[str, Any]:
        """Master several mixes the same way and report how consistent the set is. Needs
        writes enabled."""
        need_writes()
        sources = [readable(p) for p in file_paths]
        options = MasterOptions.from_settings(
            settings, reference or f"{genre or settings.default_genre}/default", genre=genre
        )
        done = batch.run_batch(sources, options, paths, settings, name)
        return {
            "batch_id": done.id,
            "folder": str(done.dir),
            "run_ids": [run.manifest.id for run in done.runs],
            "failures": [asdict(failure) for failure in done.failures],
            "consistency": asdict(done.report) if done.report else None,
        }

    return server


def serve(paths: Paths, settings: Settings) -> None:
    paths.ensure()
    build_server(paths, settings).run("stdio")
