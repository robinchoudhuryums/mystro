"""The mastering pipeline: validate -> match -> limit -> export -> analyze -> index."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from masterbench import audio_io, index, projects
from masterbench.analysis import report as analysis
from masterbench.analysis.flags import Flag
from masterbench.analysis.report import AnalysisReport, Overrides
from masterbench.config import Paths, Settings
from masterbench.master import chain, curve_match, limiter, matchering_engine, presets
from masterbench.master.matchering_engine import MasteringError
from masterbench.projects import RunManifest
from masterbench.references import library, target_curve
from masterbench.references.target_curve import TargetCurve

ProgressFn = Callable[[str, float], object]
HEAVY_LIMITING_DB = 3.0


@dataclass(frozen=True)
class MasterOptions:
    reference: str
    target_lufs: float | None
    ceiling_dbtp: float
    genre: str | None = None
    title: str | None = None
    pre_chain: str | None = None
    post_chain: str | None = None
    mode: str = "matchering"
    curve: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings, reference: str, **overrides) -> MasterOptions:
        targets = settings.targets
        return cls(reference, targets.lufs_i, targets.true_peak_dbtp, **overrides)


@dataclass(frozen=True)
class RunResult:
    manifest: RunManifest
    run_dir: Path
    reports: dict[str, AnalysisReport]

    def output(self, name: str) -> Path:
        return self.run_dir / self.manifest.outputs[name]

    @property
    def flags(self) -> list[Flag]:
        return [flag for name in sorted(self.reports) for flag in self.reports[name].flags]

    @property
    def mix_path(self) -> Path:
        return self.run_dir.parent.parent / self.manifest.input_file


def load_run(paths: Paths, run_id: str) -> RunResult:
    """Read a stored run back from disk."""
    found = list(paths.projects.glob(f"*/runs/{run_id}/run.json"))
    if len(found) != 1 or not run_id.replace("-", "").isalnum():
        raise LookupError(f"no run with id {run_id}")
    manifest, reports = projects.read_run(found[0].parent)
    return RunResult(manifest, found[0].parent, reports)


def _noop(stage: str, fraction: float) -> None:
    pass


def overrides_for(project: projects.Project) -> Overrides:
    data = projects.read_overrides(project.dir)
    return Overrides(data.get("key"), data.get("markers"), data.get("envelope"))


def _genre_default(paths: Paths, genre: str | None) -> library.Reference | None:
    if not genre:
        return None
    try:
        return library.get(paths, genre, "default")
    except library.ReferenceError:
        return None


def _reference_settings(reference: library.Reference) -> dict[str, str]:
    return {
        "spec": reference.spec,
        "path": str(reference.path),
        "sha256": reference.report.sha256,
    }


def _curve_settings(curve: TargetCurve) -> dict[str, object]:
    return {
        "genre": curve.genre,
        "created_at": curve.created_at,
        "references": curve.references,
    }


def _compare(
    report: AnalysisReport,
    reference: library.Reference | None,
    curve: TargetCurve | None,
    settings: Settings,
    genre: str | None,
) -> AnalysisReport:
    """Measure a report against the reference and the genre curve.

    When a curve exists it judges the tonal balance, since the range several references
    cover is steadier than any one song; the reference's deltas are still recorded.
    """
    if reference is not None:
        report = analysis.against_reference(
            report, reference.report, settings, genre, judge_balance=curve is None
        )
    if curve is not None:
        report = analysis.against_curve(report, curve, settings, genre)
    if report.role == "master":
        # How the hook compares with the verse is decided in the mix; say it once, there.
        kept = [f for f in report.flags if f.code != "weak_hook_contrast" and f.code[:5] != "hook_"]
        report = replace(report, flags=kept)
    return report


def analyze(
    source: Path,
    paths: Paths,
    settings: Settings,
    genre: str | None = None,
    title: str | None = None,
    progress: ProgressFn = _noop,
) -> RunResult:
    """Validate and analyze a mix, stored as an analysis-only run."""
    progress("Validating the mix", 0.1)
    project = projects.open_project(paths, source, title, genre)
    audio, info = audio_io.load(project.input_path)
    progress("Analyzing the mix", 0.3)
    mix = analysis.analyze_audio(
        audio,
        info,
        project.input_path,
        "mix",
        settings,
        project.input_sha256,
        overrides_for(project),
    )
    track_genre = project.track.genre
    reference = _genre_default(paths, track_genre)
    curve = target_curve.load(paths, track_genre)
    settings_used: dict = {}
    reports = {"mix": _compare(mix, reference, curve, settings, track_genre)}
    if reference is not None:
        reports["reference"] = reference.report
        settings_used["reference"] = _reference_settings(reference)
    if curve is not None:
        settings_used["curve"] = _curve_settings(curve)
    manifest = RunManifest(
        id=projects.new_run_id(),
        track=project.track,
        kind="analysis_only",
        created_at=projects.now_iso(),
        input_file=str(project.input_path.relative_to(project.dir)),
        input_sha256=project.input_sha256,
        original_path=str(source),
        settings=settings_used,
        lib_versions=projects.lib_versions(),
    )
    return _finish(paths, project, manifest, reports, progress)


def master(
    source: Path,
    options: MasterOptions,
    paths: Paths,
    settings: Settings,
    progress: ProgressFn = _noop,
) -> RunResult:
    """Master a mix. Always creates a new run; the input is never changed.

    Two ways to match: "matchering" follows one reference track; "curve" applies a static
    matching EQ toward the genre's target curve. Either way the order is pre-chain, match,
    post-chain, then loudness and the true-peak limiter. The final limiter always runs last,
    so nothing is limited twice by accident and the ceiling holds whatever the chains do.
    """
    started = time.perf_counter()
    progress("Validating the mix", 0.05)
    by_curve = options.mode == "curve"
    if options.mode not in ("matchering", "curve"):
        raise MasteringError(f"unknown mode '{options.mode}'; use 'matchering' or 'curve'")
    match_curve = None
    if by_curve:
        wanted = options.curve or options.genre or settings.default_genre
        match_curve = target_curve.get(paths, wanted)
        reference = _genre_default(paths, match_curve.genre)
        genre = options.genre or match_curve.genre
    else:
        reference = library.resolve(paths, options.reference, settings)
        genre = options.genre or reference.genre or None
    project = projects.open_project(paths, source, options.title, genre)
    if reference is not None and project.input_sha256 == reference.report.sha256:
        if not by_curve:
            raise MasteringError("the mix and the reference are the same file")
        reference = None
    audio, info = audio_io.load(project.input_path)
    progress("Analyzing the mix", 0.08)
    mix = analysis.analyze_audio(
        audio,
        info,
        project.input_path,
        "mix",
        settings,
        project.input_sha256,
        overrides_for(project),
    )

    pre = presets.get(paths, options.pre_chain) if options.pre_chain else None
    post = presets.get(paths, options.post_chain) if options.post_chain else None
    eq = None
    if pre is not None:
        progress(f"Running the pre-chain ({pre.name})", 0.3)
        audio = chain.run(pre, audio, info.sample_rate)
    if match_curve is not None:
        progress(f"Matching to the {match_curve.genre} curve", 0.35)
        stereo = audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)
        matched, eq = curve_match.match(
            stereo, info.sample_rate, match_curve.freqs_hz, match_curve.median_db
        )
    else:
        assert reference is not None
        progress("Matching to the reference", 0.35)
        with tempfile.TemporaryDirectory(prefix="masterbench-") as tmp:
            target = project.input_path
            if pre is not None:
                target = audio_io.save_wav(Path(tmp) / "pre_chain.wav", audio, info.sample_rate, 32)
            matched = matchering_engine.match(target, reference.path, info.sample_rate)
    if post is not None:
        progress(f"Running the post-chain ({post.name})", 0.5)
        matched = chain.run(post, matched, info.sample_rate)

    progress("Setting loudness and limiting", 0.55)
    final = limiter.finalize(matched, info.sample_rate, options.target_lufs, options.ceiling_dbtp)

    progress("Exporting", 0.75)
    track_genre = project.track.genre
    curve = match_curve or target_curve.load(paths, track_genre)
    used: dict = {
        "mode": options.mode,
        "reference": _reference_settings(reference) if reference else None,
        "target_lufs": options.target_lufs,
        "ceiling_dbtp": options.ceiling_dbtp,
        "pre_chain": pre.to_dict() if pre else None,
        "post_chain": post.to_dict() if post else None,
    }
    if curve is not None:
        used["curve"] = _curve_settings(curve)
    if eq is not None:
        used["match_eq"] = asdict(eq)
    manifest = RunManifest(
        id=projects.new_run_id(),
        track=project.track,
        kind="master",
        created_at=projects.now_iso(),
        input_file=str(project.input_path.relative_to(project.dir)),
        input_sha256=project.input_sha256,
        original_path=str(source),
        settings=used,
        outputs={"master_24": "master_24.wav", "master_16": "master_16.wav"},
        lib_versions=projects.lib_versions(),
    )
    run_dir = project.run_dir(manifest.id)
    audio_io.save_wav(run_dir / "master_24.wav", final.audio, info.sample_rate, 24)
    audio_io.save_wav(run_dir / "master_16.wav", final.audio, info.sample_rate, 16)

    progress("Analyzing the master", 0.85)
    master_path = run_dir / "master_24.wav"
    master_audio, master_info = audio_io.load(master_path)
    mastered = analysis.analyze_audio(
        master_audio, master_info, master_path, "master", settings, like=mix
    )
    flags = analysis.master_flags(mastered.loudness, options.target_lufs, options.ceiling_dbtp)
    if final.max_reduction_db > HEAVY_LIMITING_DB:
        message = (
            f"The limiter took off up to {final.max_reduction_db:.1f} dB to reach this loudness. "
            "Expect audible pumping or distortion; a lower target will sound cleaner."
        )
        flags.append(Flag("heavy_limiting", "warn", message, final.max_reduction_db))
    if eq is not None and eq.max_abs_db >= curve_match.MAX_DB - 0.01:
        message = (
            f"The matching EQ hit its ±{curve_match.MAX_DB:.0f} dB limit, so the master does not "
            "reach the curve everywhere. The remaining difference is better fixed in the mix."
        )
        flags.append(Flag("match_eq_limited", "info", message, eq.max_abs_db))
    mastered = replace(mastered, flags=[*mastered.flags, *flags])
    mix = _compare(mix, reference, curve, settings, track_genre)
    mastered = _compare(mastered, reference, curve, settings, track_genre)

    stats = {
        "gain_db": round(final.gain_db, 3),
        "max_reduction_db": round(final.max_reduction_db, 3),
        "limiter_passes": final.iterations,
        "seconds": round(time.perf_counter() - started, 2),
    }
    manifest = replace(manifest, stats=stats)
    reports = {"mix": mix, "master": mastered}
    if reference is not None:
        reports["reference"] = reference.report
    return _finish(paths, project, manifest, reports, progress)


def _finish(
    paths: Paths,
    project: projects.Project,
    manifest: RunManifest,
    reports: dict[str, AnalysisReport],
    progress: ProgressFn,
) -> RunResult:
    progress("Saving the run", 0.95)
    run_dir = project.run_dir(manifest.id)
    projects.write_run(run_dir, manifest, reports)
    with index.open_index(paths) as conn:
        index.index_run(conn, manifest, run_dir, reports)
    progress("Done", 1.0)
    return RunResult(manifest, run_dir, reports)
