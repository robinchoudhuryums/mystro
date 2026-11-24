"""Batch / EP mode: master several tracks the same way and check that they belong together."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from masterbench import projects
from masterbench.analysis.compare import band_label
from masterbench.analysis.flags import Flag
from masterbench.config import BANDS, Paths, Settings, Thresholds
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult

BatchProgress = Callable[[str, float], object]


@dataclass(frozen=True)
class TrackSummary:
    title: str
    run_id: str
    lufs_i: float
    loudness_offset_lu: float
    true_peak_dbtp: float
    lra: float
    loudest_section_lufs: float
    band_offsets_db: dict[str, float]


@dataclass(frozen=True)
class ConsistencyReport:
    tracks: list[TrackSummary]
    loudness_spread_lu: float
    loudest_section_spread_lu: float
    max_true_peak_dbtp: float
    flags: list[Flag]


@dataclass(frozen=True)
class Failure:
    source: str
    error: str


@dataclass(frozen=True)
class BatchResult:
    id: str
    name: str
    dir: Path
    runs: list[RunResult]
    failures: list[Failure]
    report: ConsistencyReport | None


def consistency(runs: list[RunResult], limits: Thresholds) -> ConsistencyReport:
    """How far each master sits from the set's median, in loudness and in each band.

    Band levels are stored relative to each track's own loudness, so the offsets compare
    tonal balance, not level. The loudest section is the short-term loudness maximum.
    """
    masters = [(run, run.reports["master"]) for run in runs]
    median_lufs = statistics.median(m.loudness.lufs_i for _, m in masters)
    median_bands = {
        band: statistics.median(m.bands[band] for _, m in masters if m.bands) for band in BANDS
    }
    tracks, flags = [], []
    for run, master in masters:
        title = run.manifest.track.title
        offset = master.loudness.lufs_i - median_lufs
        bands = (
            {b: round(master.bands[b] - median_bands[b], 2) for b in BANDS} if master.bands else {}
        )
        tracks.append(
            TrackSummary(
                title=title,
                run_id=run.manifest.id,
                lufs_i=round(master.loudness.lufs_i, 2),
                loudness_offset_lu=round(offset, 2),
                true_peak_dbtp=round(master.loudness.true_peak_dbtp, 2),
                lra=round(master.loudness.lra, 2),
                loudest_section_lufs=round(master.loudness.lufs_s_max, 2),
                band_offsets_db=bands,
            )
        )
        if abs(offset) > limits.ep_loudness_lu:
            word = "louder" if offset > 0 else "quieter"
            message = f"{title} is {abs(offset):.1f} LU {word} than the set's median."
            flags.append(Flag("ep_loudness_outlier", "warn", message, round(offset, 2)))
        for band, delta in bands.items():
            if abs(delta) > limits.ep_band_db:
                word = "more" if delta > 0 else "less"
                message = (
                    f"{title} has {abs(delta):.1f} dB {word} {band_label(band)} than the set's "
                    "median, at equal loudness."
                )
                flags.append(Flag(f"ep_band_{band}_outlier", "warn", message, delta))

    sections = [t.loudest_section_lufs for t in tracks]
    section_median = statistics.median(sections)
    for track in tracks:
        delta = track.loudest_section_lufs - section_median
        if abs(delta) > 2 * limits.ep_loudness_lu:
            word = "louder" if delta > 0 else "quieter"
            message = (
                f"{track.title}'s loudest section is {abs(delta):.1f} LU {word} than the others' "
                f"({track.loudest_section_lufs:.1f} LUFS short-term)."
            )
            flags.append(Flag("ep_section_outlier", "info", message, round(delta, 2)))
    levels = [t.lufs_i for t in tracks]
    return ConsistencyReport(
        tracks=tracks,
        loudness_spread_lu=round(max(levels) - min(levels), 2),
        loudest_section_spread_lu=round(max(sections) - min(sections), 2),
        max_true_peak_dbtp=max(t.true_peak_dbtp for t in tracks),
        flags=flags,
    )


def run_batch(
    sources: list[Path],
    options: MasterOptions,
    paths: Paths,
    settings: Settings,
    name: str = "",
    references: dict[str, str] | None = None,
    progress: BatchProgress = lambda stage, fraction: None,
) -> BatchResult:
    """Master every source with the same options. One failure does not stop the rest.

    references maps a file name to its own reference spec, for sets that need one per track.
    """
    batch_id = projects.new_run_id()
    runs: list[RunResult] = []
    failures: list[Failure] = []
    for number, source in enumerate(sources):
        done = number / len(sources)
        own = (references or {}).get(source.name)
        track_options = replace(options, reference=own or options.reference, title=None)

        def report(stage: str, fraction: float, done=done, source=source) -> None:
            progress(f"{source.name}: {stage}", done + fraction / len(sources))

        try:
            runs.append(pipeline.master(source, track_options, paths, settings, report))
        except Exception as exc:  # an unattended batch records the failure and moves on
            failures.append(Failure(str(source), f"{type(exc).__name__}: {exc}"))
    report_ = consistency(runs, settings.thresholds) if len(runs) >= 2 else None
    folder = paths.root / "batches" / batch_id
    projects.write_json(
        folder / "batch.json",
        {
            "id": batch_id,
            "name": name,
            "created_at": projects.now_iso(),
            "options": asdict(options),
            "run_ids": [run.manifest.id for run in runs],
            "failures": [asdict(failure) for failure in failures],
            "report": asdict(report_) if report_ else None,
        },
    )
    progress("Done", 1.0)
    return BatchResult(batch_id, name, folder, runs, failures, report_)
