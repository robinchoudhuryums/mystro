"""Per-track tools the interfaces share: hand-set key and sections, and the low-end report."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from masterbench import audio_io, projects
from masterbench.analysis import lowend, musical, sections
from masterbench.analysis.lowend import LowEnd
from masterbench.analysis.sections import Sections
from masterbench.config import Paths, Settings
from masterbench.master.pipeline import RunResult
from masterbench.stems import workflow

_MARKER = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*[:=\s]\s*([a-z]+)\s*$")
# Words people use for the kinds the analysis knows.
_KIND_WORDS = {"chorus": "hook", "drop": "hook", "break": "verse", "build": "verse"}


class TrackError(ValueError):
    pass


def project_dir(run: RunResult) -> Path:
    return run.run_dir.parent.parent


def parse_markers(text: str) -> list[dict[str, float | str]]:
    """'0 intro, 6.9 verse, 0:20.6 hook' -> markers. Times are seconds or m:ss."""
    markers = []
    for part in re.split(r"[,\n;]+", text.lower()):
        if not part.strip():
            continue
        match = _MARKER.match(part)
        if match is None:
            raise TrackError(f"cannot read marker {part.strip()!r}; write e.g. '0:20.5 hook'")
        minutes, seconds, word = match.groups()
        kind = _KIND_WORDS.get(word, word)
        if kind not in sections.KINDS:
            raise TrackError(f"unknown section kind {word!r}; use {', '.join(sections.KINDS[:-1])}")
        markers.append({"start_s": int(minutes or 0) * 60 + float(seconds), "kind": kind})
    if not markers:
        raise TrackError("no markers given")
    markers.sort(key=lambda m: float(m["start_s"]))
    return markers


def markers_text(found: Sections) -> str:
    """Sections as editable marker text."""
    return ", ".join(f"{item.start_s:g} {item.kind}" for item in found.items)


def set_sections(run: RunResult, text: str) -> Sections:
    """Save hand-set sections for a run's track. They apply to every later analysis of it;
    the loudness envelope saved with them lets a re-bounce that starts earlier or later line up."""
    markers = parse_markers(text)
    audio, info = audio_io.load(run.mix_path)
    if float(markers[-1]["start_s"]) >= info.duration_s:
        raise TrackError(f"the track is only {info.duration_s:.1f} s long")
    envelope = [round(float(v), 1) for v in sections.envelope(audio, info.sample_rate)]
    projects.write_overrides(project_dir(run), markers=markers, envelope=envelope)
    starts = [float(m["start_s"]) for m in markers]
    kinds = [str(m["kind"]) for m in markers]
    if starts[0] > 0.05:
        starts, kinds = [0.0, *starts], ["intro", *kinds]
    return sections.build(audio, info.sample_rate, starts, kinds, source="manual")


def clear_sections(run: RunResult) -> None:
    projects.write_overrides(project_dir(run), markers=None, envelope=None)


def set_key(paths: Paths, track_slug: str, key: str | None) -> str | None:
    """Set a track's key by hand ('F minor'), or clear it with None."""
    folder = paths.projects / track_slug
    if not (folder / "track.json").exists():
        raise TrackError(f"no track '{track_slug}'")
    if key is not None:
        tonic, _, mode = key.strip().partition(" ")
        key = f"{tonic[:1].upper()}{tonic[1:]} {mode.lower()}"
        try:
            musical.parse_key(key)
        except ValueError as exc:
            raise TrackError(str(exc)) from exc
    projects.write_overrides(folder, key=key)
    return key


def low_end(run: RunResult, paths: Paths, settings: Settings) -> LowEnd:
    """The low-end report for a run. With drum and bass stems (from Logic or a four-stem
    separation) the pitches come from the bass alone and the kick clash is measured too."""
    mix = run.reports["mix"]
    key = mix.musical.key if mix.musical else None
    audio, info = audio_io.load(run.mix_path)
    bass = drums = None
    for stem_set in workflow.sets_for_run(run):
        bass_names, drum_names = stem_set.by_role("bass"), stem_set.by_role("drums")
        if bass_names and drum_names and stem_set.frames == len(audio):
            bass = np.sum([stem_set.load(name) for name in bass_names], axis=0)
            drums = np.sum([stem_set.load(name) for name in drum_names], axis=0)
            break
    found = lowend.analyze(audio, info.sample_rate, key, settings.thresholds, bass, drums)
    projects.write_json(run.run_dir / "lowend.json", asdict(found))
    return found
