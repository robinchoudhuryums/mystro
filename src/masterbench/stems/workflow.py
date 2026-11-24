"""Stem operations as the interfaces use them: separate, import, diagnose, rebalance."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from masterbench import audio_io, projects
from masterbench.analysis import detectors
from masterbench.analysis.flags import Flag
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.projects import Project
from masterbench.references import library
from masterbench.stems import rebalance as rebalancing
from masterbench.stems import separate, stem_analysis
from masterbench.stems.rebalance import Rebalanced
from masterbench.stems.separate import Runner, StemError, StemSet
from masterbench.stems.stem_analysis import StemBalance


@dataclass(frozen=True)
class Diagnosis:
    stem_set: str
    balance: StemBalance
    reference: StemBalance | None
    flags: list[Flag]
    warning: str | None


def project_stems_dir(project_dir: Path) -> Path:
    return project_dir / "stems"


def separate_mix(
    project: Project,
    model_key: str = separate.DEFAULT_MODEL,
    runner: Runner = separate.audio_separator_runner,
) -> StemSet:
    return separate.separate(project.input_path, project_stems_dir(project.dir), model_key, runner)


def separate_reference(
    reference: library.Reference,
    model_key: str = separate.DEFAULT_MODEL,
    runner: Runner = separate.audio_separator_runner,
) -> StemSet:
    """Stems of a reference track, cached beside it."""
    return separate.separate(reference.path, reference.path.parent / ".stems", model_key, runner)


def import_logic_stems(
    paths: Paths,
    folder: Path,
    title: str,
    genre: str | None = None,
    mix: Path | None = None,
) -> tuple[Project, StemSet]:
    """Bring a folder of Logic stems into a track's project.

    Without the bounce, the stems' sum stands in as the mix.
    """
    project_dir = paths.projects / projects.slugify(title)
    stem_set = separate.import_stems(folder, project_stems_dir(project_dir))
    if mix is not None:
        return projects.open_project(paths, mix, title, genre), stem_set
    with tempfile.TemporaryDirectory(prefix="masterbench-") as tmp:
        summed = Path(tmp) / f"{title} (stem sum).wav"
        audio_io.save_wav(summed, separate.mixdown(stem_set), stem_set.sample_rate, 32)
        return projects.open_project(paths, summed, title, genre), stem_set


def diagnose(
    stem_set: StemSet,
    mix_path: Path,
    settings: Settings,
    reference: library.Reference | None = None,
    model_key: str | None = None,
    runner: Runner = separate.audio_separator_runner,
) -> Diagnosis:
    """Measure the stem balance and, given a reference, compare with its separated stems."""
    mix, _ = audio_io.load(mix_path)
    mine = stem_analysis.balance(stem_set, mix)
    theirs, flags = None, []
    if reference is not None:
        four = any(role in stem_set.roles.values() for role in ("drums", "bass"))
        fallback = "demucs-4-fast" if four else separate.DEFAULT_MODEL
        key = model_key or (stem_set.source if stem_set.separated else fallback)
        reference_set = separate_reference(reference, key, runner)
        reference_mix, _ = audio_io.load(reference.path)
        theirs = stem_analysis.balance(reference_set, reference_mix)
        flags = stem_analysis.compare(mine, theirs, settings.thresholds)
    flags += vocal_flags(stem_set, settings)
    diagnosis = Diagnosis(stem_set.name, mine, theirs, flags, stem_analysis.artifact_warning(mine))
    projects.write_json(stem_set.dir / "diagnosis.json", asdict(diagnosis))
    return diagnosis


def vocal_flags(stem_set: StemSet, settings: Settings) -> list[Flag]:
    """Sibilance and uneven phrases, measured on the vocal stem against everything else."""
    names = stem_set.by_role("vocals")
    if not names:
        return []
    vocal = np.sum([stem_set.load(name) for name in names], axis=0)
    beat = separate.mixdown(stem_set) - vocal
    rate, limits = stem_set.sample_rate, settings.thresholds
    return [
        *detectors.sibilance(vocal.mean(axis=1), rate, limits, is_vocal=True),
        *detectors.vocal_consistency(vocal, beat, rate, limits),
    ]


def find_set(paths: Paths, track_slug: str, name: str | None = None) -> StemSet:
    """A track's stem set by name, or its preferred one (real stems, else the newest)."""
    sets = separate.list_sets(paths.projects / track_slug)
    if not sets:
        raise StemError(f"track '{track_slug}' has no stems yet; separate or import some first")
    if name is None:
        real = [s for s in sets if not s.separated]
        return real[-1] if real else sets[-1]
    match = next((s for s in sets if s.name == name), None)
    if match is None:
        raise StemError(f"no stem set '{name}'; have {', '.join(s.name for s in sets)}")
    return match


def rebalance_and_master(
    paths: Paths,
    settings: Settings,
    stem_set: StemSet,
    gains_db: dict[str, float],
    options: MasterOptions,
    chains: dict[str, str] | None = None,
    progress: pipeline.ProgressFn = lambda stage, fraction: None,
) -> tuple[Rebalanced, RunResult]:
    """Rebalance the stems and master the result as a new run of the same track."""
    progress("Rebalancing the stems", 0.02)
    rebalanced = rebalancing.rebalance(stem_set, gains_db, paths, chains)
    result = pipeline.master(rebalanced.path, options, paths, settings, progress)
    return rebalanced, result


def sets_for_run(run: RunResult) -> list[StemSet]:
    """Stem sets that belong to a run's input: Logic stems, and separations of that file."""
    sets = separate.list_sets(run.run_dir.parent.parent)
    return [s for s in sets if not s.separated or s.input_sha256 == run.manifest.input_sha256]
