"""Vocal tuning on disk: where a vocal, its pitch report and its tuned versions are kept.

projects/<track>/vocal/<vocal-hash>/
    vocal.json                 what the vocal is and where it came from
    <name>.wav                 a dry vocal brought in from outside (stems stay in place)
    pitch.npy, pitch.json      the pitch track (cache) and the report
    edits.json                 notes bypassed or retargeted by hand
    tuned_<settings>.wav/.json a tuned version and how it was made
    preview_<settings>.wav     full-mix path only: the tuned vocal back in the mix
    external/                  the same vocal tuned elsewhere, for comparison
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from masterbench import audio_io, index, projects
from masterbench.analysis import loudness
from masterbench.analysis.flags import Flag
from masterbench.config import Paths
from masterbench.projects import Track
from masterbench.stems import separate
from masterbench.stems import workflow as stem_workflow
from masterbench.stems.separate import Runner, StemSet
from masterbench.vocal import pitch_report, pitch_track, presets, tuning
from masterbench.vocal.pitch_report import PitchReport
from masterbench.vocal.pitch_track import PitchTrack
from masterbench.vocal.tuning import Edits, NoteEdit, TuningSettings

VOCAL_DIR = "vocal"
MANIFEST = "vocal.json"
REPORT = "pitch.json"
EDITS = "edits.json"
EXTERNAL = "external"
KINDS = ("dry", "stem", "separated")
BENCHMARK_PRESETS = ("natural", "hard-tune")
# A note counts as corrected when most of it moved by at least this much.
NOTICEABLE_CENTS = 2.0
PREVIEW_NOTE = (
    "Preview quality: the vocal was separated from the mix, and tuning makes what separation "
    "left behind (reverb, bleed, artifacts) easier to hear. Use it to audition; for release, "
    "export the dry vocal from Logic and tune that."
)


class VocalError(ValueError):
    """A vocal that cannot be found or used."""


@dataclass(frozen=True)
class VocalSource:
    track: Track
    folder: Path
    path: Path
    sha256: str
    kind: str
    label: str
    created_at: str = ""
    mix: Path | None = None

    @property
    def id(self) -> str:
        return f"{self.track.slug}/{self.folder.name}"

    @property
    def separated(self) -> bool:
        return self.kind == "separated"


@dataclass(frozen=True)
class Tuned:
    id: str
    file: str
    label: str
    created_at: str
    settings: TuningSettings
    edits: Edits
    seconds: float
    notes: int
    corrected: int
    largest_cents: float
    before: dict[str, float | None]
    after: dict[str, float | None]
    lufs: float
    preview_file: str | None = None
    flags: list[Flag] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["settings"] = self.settings.to_dict()
        data["edits"] = {str(index): asdict(edit) for index, edit in self.edits.items()}
        del data["cached"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tuned:
        return cls(
            **{
                **data,
                "settings": TuningSettings.from_dict(data["settings"]),
                "edits": {int(i): NoteEdit(**edit) for i, edit in data["edits"].items()},
                "flags": [Flag(**flag) for flag in data["flags"]],
            }
        )


def _relative(path: Path, project_dir: Path) -> str:
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def _write_manifest(source: VocalSource, project_dir: Path, original: Path) -> None:
    stored = _relative(source.path, project_dir)
    projects.write_json(
        source.folder / MANIFEST,
        {
            "track": source.track.slug,
            "file": stored,
            "sha256": source.sha256,
            "kind": source.kind,
            "label": source.label,
            "original_path": str(original),
            "created_at": source.created_at,
            "mix": _relative(source.mix, project_dir) if source.mix else None,
        },
    )


def _open(
    paths: Paths,
    track: Track,
    project_dir: Path,
    audio: Path,
    kind: str,
    label: str,
    copy: bool,
    mix: Path | None = None,
) -> VocalSource:
    sha256 = audio_io.file_sha256(audio)
    folder = project_dir / VOCAL_DIR / sha256[:8]
    existing = load_source(paths, f"{track.slug}/{sha256[:8]}", missing_ok=True)
    if existing is not None:
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    stored = audio
    if copy:
        stored = folder / audio.name
        shutil.copy2(audio, stored)
    source = VocalSource(track, folder, stored, sha256, kind, label, projects.now_iso(), mix)
    _write_manifest(source, project_dir, audio)
    return source


def open_vocal(
    paths: Paths, file: Path, track: str | None = None, genre: str | None = None
) -> VocalSource:
    """Bring a dry vocal file into a track's project. The file itself is never changed."""
    if not file.is_file():
        raise VocalError(f"no such file: {file}")
    title = track or projects.title_from_filename(file)
    existing = paths.projects / projects.slugify(title) / "track.json"
    if track and existing.is_file():
        title = json.loads(existing.read_text(encoding="utf-8"))["title"]
    found, project_dir = projects.ensure_track(paths, title, genre)
    return _open(paths, found, project_dir, file, "dry", file.stem, copy=True)


def from_stems(
    paths: Paths, track_slug: str, set_name: str | None = None, stem: str | None = None
) -> VocalSource:
    """Use the vocal stem of a track's stem set."""
    stem_set = stem_workflow.find_set(paths, track_slug, set_name)
    return _from_set(paths, track_slug, stem_set, stem)


def _from_set(paths: Paths, track_slug: str, stem_set: StemSet, stem: str | None) -> VocalSource:
    vocals = stem_set.by_role("vocals")
    if stem is None and len(vocals) > 1:
        raise VocalError(
            f"stem set '{stem_set.name}' has {len(vocals)} vocal stems "
            f"({', '.join(vocals)}); name the one to tune"
        )
    chosen = stem or (vocals[0] if vocals else None)
    if chosen is None or chosen not in stem_set.stems:
        raise VocalError(f"stem set '{stem_set.name}' has no vocal stem to tune")
    project_dir = paths.projects / track_slug
    found = Track(**json.loads((project_dir / "track.json").read_text(encoding="utf-8")))
    kind = "separated" if stem_set.separated else "stem"
    label = f"{chosen} ({'separated' if stem_set.separated else 'Logic stem'})"
    mix = _mix_of(project_dir, stem_set) if stem_set.separated else None
    return _open(paths, found, project_dir, stem_set.path(chosen), kind, label, False, mix)


def _mix_of(project_dir: Path, stem_set: StemSet) -> Path | None:
    """The bounce a stem set was separated from, if it is still in the project."""
    for file in sorted((project_dir / "input").glob("*")):
        if file.is_file() and audio_io.file_sha256(file) == stem_set.input_sha256:
            return file
    return None


def load_source(paths: Paths, source_id: str, missing_ok: bool = False) -> VocalSource | None:
    """A vocal by its id, '<track>/<hash>'."""
    slug, _, name = source_id.partition("/")
    project_dir = paths.projects / slug
    manifest = project_dir / VOCAL_DIR / name / MANIFEST
    safe = all(part.replace("-", "").replace("_", "").isalnum() for part in (slug, name))
    if not safe or not manifest.is_file():
        if missing_ok:
            return None
        raise VocalError(f"no vocal '{source_id}'")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    found = Track(**json.loads((project_dir / "track.json").read_text(encoding="utf-8")))
    audio = Path(data["file"])
    audio = audio if audio.is_absolute() else project_dir / audio
    if not audio.is_file():
        if missing_ok:
            return None
        raise VocalError(f"the audio for vocal '{source_id}' is gone: {audio}")
    mix = data.get("mix")
    mix = None if mix is None else Path(mix) if Path(mix).is_absolute() else project_dir / mix
    return VocalSource(
        found,
        manifest.parent,
        audio,
        data["sha256"],
        data["kind"],
        data["label"],
        data["created_at"],
        mix if mix is not None and mix.is_file() else None,
    )


def list_sources(paths: Paths, track_slug: str | None = None) -> list[VocalSource]:
    """Every vocal brought in so far, newest first."""
    pattern = f"{track_slug or '*'}/{VOCAL_DIR}/*/{MANIFEST}"
    found = []
    for manifest in paths.projects.glob(pattern):
        source = load_source(
            paths, f"{manifest.parent.parent.parent.name}/{manifest.parent.name}", missing_ok=True
        )
        if source is not None:
            found.append(source)
    return sorted(found, key=lambda s: s.created_at, reverse=True)


def pitch_of(source: VocalSource) -> PitchTrack:
    """The vocal's pitch track, computed once and kept beside it."""
    cache = source.folder / "pitch.npy"
    info = audio_io.load(source.path)[1] if cache.is_file() else None
    if info is not None:
        return PitchTrack(info.sample_rate, np.load(cache))
    audio, info = audio_io.load(source.path)
    pitch = pitch_track.track(audio, info.sample_rate)
    np.save(cache, pitch.f0_hz)
    return pitch


def track_key(source: VocalSource) -> tuple[str, float | None] | None:
    """The song's key: set by hand for the track, else from its most recent analysis."""
    project_dir = source.folder.parent.parent
    by_hand = projects.read_overrides(project_dir).get("key")
    if by_hand:
        return by_hand, 1.0
    for run_dir in sorted((project_dir / "runs").glob("*/"), reverse=True):
        if not (run_dir / "run.json").is_file():
            continue
        mix = projects.read_run(run_dir)[1].get("mix")
        if mix is not None and mix.musical is not None:
            return mix.musical.key, mix.musical.key_confidence
    return None


def report(
    source: VocalSource,
    paths: Paths,
    key: str | None = None,
    scale_kind: str | None = None,
    custom_notes: tuple[str, ...] = (),
) -> PitchReport:
    """The pitch report for a vocal, stored beside it and indexed for the progress view.

    The key is the one given, else the track's, else a guess from the sung notes.
    """
    audio, info = audio_io.load(source.path)
    key_source, confidence = "given", None
    if key is None:
        known = track_key(source)
        if known is not None:
            key, confidence = known
            key_source = "track"
    found = pitch_report.analyze(
        audio,
        info.sample_rate,
        key,
        scale_kind,
        custom_notes,
        key_source,
        source.separated,
        pitch_of(source),
    )
    if key_source == "track":
        found = replace(found, key_confidence=confidence)
    projects.write_json(source.folder / REPORT, found.to_dict())
    with index.open_index(paths) as conn:
        index.sync_vocal_reports(conn, paths)
    return found


def stored_report(source: VocalSource) -> PitchReport | None:
    file = source.folder / REPORT
    if not file.is_file():
        return None
    return PitchReport.from_dict(json.loads(file.read_text(encoding="utf-8")))


def read_edits(source: VocalSource) -> Edits:
    file = source.folder / EDITS
    if not file.is_file():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return {int(index): NoteEdit(**edit) for index, edit in data.items()}


def write_edits(source: VocalSource, edits: Edits) -> Edits:
    """Store hand edits; an edit that changes nothing is dropped."""
    kept = {i: e for i, e in sorted(edits.items()) if e.bypass or e.target is not None}
    projects.write_json(source.folder / EDITS, {str(i): asdict(e) for i, e in kept.items()})
    return kept


def _summary(found: PitchReport) -> dict[str, float | None]:
    return {
        "in_tune_share": found.in_tune_share,
        "mean_abs_cents": found.mean_abs_cents,
        "mean_cents": found.mean_cents,
    }


def _bit_depth(info: audio_io.AudioInfo) -> int:
    return 32 if info.subtype in ("FLOAT", "DOUBLE") else 24


def _label(name: str | None, settings: TuningSettings, edits: Edits) -> str:
    """What a tuned version is called in the player: enough to tell two of them apart."""
    parts = [settings.engine, f"{settings.amount:.0%}"]
    if edits:
        parts.append(f"{len(edits)} edit{'s' if len(edits) != 1 else ''}")
    return f"{name or 'Tuned'} ({', '.join(parts)})"


def tuned_versions(source: VocalSource) -> list[Tuned]:
    """Tuned versions of a vocal whose audio is still there, oldest first."""
    found = []
    for file in sorted(source.folder.glob("tuned_*.json")):
        version = Tuned.from_dict(json.loads(file.read_text(encoding="utf-8")))
        if (source.folder / version.file).is_file():
            found.append(version)
    return sorted(found, key=lambda v: v.created_at)


def tune(
    source: VocalSource,
    settings: TuningSettings,
    edits: Edits | None = None,
    label: str | None = None,
) -> Tuned:
    """Tune a vocal and store the result as tuned_<settings-hash>.wav beside it.

    The output has the input's length, sample rate and start, so it drops back into the
    session in place of the original. The same settings and edits give the stored file.
    """
    edits = read_edits(source) if edits is None else edits
    tuned_id = tuning.settings_hash(settings, edits)
    manifest = source.folder / f"tuned_{tuned_id}.json"
    if manifest.is_file() and (source.folder / f"tuned_{tuned_id}.wav").is_file():
        stored = Tuned.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
        return replace(stored, cached=True)

    started = time.perf_counter()
    audio, info = audio_io.load(source.path)
    pitch = pitch_of(source)
    result = tuning.tune(audio, info.sample_rate, settings, edits, pitch)
    scale = settings.resolve_scale()
    before = pitch_report.build(pitch, result.notes, scale, settings.key)
    after = pitch_report.analyze(
        result.audio, info.sample_rate, settings.key, settings.scale, settings.custom_notes
    )
    file = audio_io.save_wav(
        source.folder / f"tuned_{tuned_id}.wav", result.audio, info.sample_rate, _bit_depth(info)
    )
    moves = [np.abs(result.plan.correction[note.frames]) * 100.0 for note in result.notes]
    moved = [float(np.max(cents)) for cents in moves]
    typical = [float(np.median(cents)) for cents in moves]
    flags = list(before.flags)
    preview = None
    if source.separated:
        flags = [f for f in flags if f.code != "separated_vocal"]
        flags.insert(0, Flag("preview_quality", "warn", PREVIEW_NOTE))
    if source.separated and source.mix is not None:
        # The mix with its separated vocal swapped for the tuned one. Everything the
        # separation did not call vocal stays exactly as it was.
        full, mix_info = audio_io.load(source.mix)
        if full.shape[0] != audio.shape[0] or mix_info.sample_rate != info.sample_rate:
            raise VocalError("the separated vocal does not line up with its mix")
        preview = audio_io.save_wav(
            source.folder / f"preview_{tuned_id}.wav",
            full + (result.audio - audio),
            info.sample_rate,
            32,
        ).name
    version = Tuned(
        id=tuned_id,
        file=file.name,
        label=_label(label, settings, edits),
        created_at=projects.now_iso(),
        settings=settings,
        edits=dict(edits),
        seconds=round(time.perf_counter() - started, 2),
        notes=len(result.notes),
        corrected=sum(1 for cents in typical if cents >= NOTICEABLE_CENTS),
        largest_cents=round(max(moved, default=0.0), 1),
        before=_summary(before),
        after=_summary(after),
        lufs=round(loudness.integrated(result.audio, info.sample_rate), 2),
        preview_file=preview,
        flags=flags,
    )
    projects.write_json(manifest, version.to_dict())
    return version


def separate_vocal(
    paths: Paths,
    mix: Path,
    title: str | None = None,
    genre: str | None = None,
    model_key: str = separate.DEFAULT_MODEL,
    runner: Runner = separate.audio_separator_runner,
) -> VocalSource:
    """First step of the full-mix path: separate a mix and take its vocal.

    The beat cannot be pitch-corrected along with the voice, so the vocal has to come out
    first. Tuning this source also writes the mix with the tuned vocal put back.
    """
    project = projects.open_project(paths, mix, title, genre)
    stem_set = stem_workflow.separate_mix(project, model_key, runner)
    return _from_set(paths, project.track.slug, stem_set, None)


def tune_mix(
    paths: Paths,
    mix: Path,
    settings: TuningSettings,
    title: str | None = None,
    genre: str | None = None,
    model_key: str = separate.DEFAULT_MODEL,
    runner: Runner = separate.audio_separator_runner,
) -> tuple[VocalSource, Tuned]:
    """Full-mix path in one go: separate the vocal, tune it, put it back in the mix."""
    source = separate_vocal(paths, mix, title, genre, model_key, runner)
    return source, tune(source, settings)


def add_external(source: VocalSource, file: Path, label: str) -> Path:
    """Add the same vocal tuned in another tool (Waves, Flex Pitch) for a blind comparison."""
    theirs = audio_io.load(file)[1]
    ours = audio_io.load(source.path)[1]
    if theirs.sample_rate != ours.sample_rate:
        raise VocalError(
            f"{file.name} is at {theirs.sample_rate} Hz and the vocal at {ours.sample_rate} Hz; "
            "export it at the same rate"
        )
    if abs(theirs.duration_s - ours.duration_s) > 1.0:
        raise VocalError(
            f"{file.name} is {theirs.duration_s:.1f} s long and the vocal {ours.duration_s:.1f} s; "
            "export the same region so they play in step"
        )
    folder = source.folder / EXTERNAL
    folder.mkdir(exist_ok=True)
    dest = folder / f"{projects.slugify(label)}{file.suffix.lower()}"
    shutil.copy2(file, dest)
    listed = [e for e in externals(source) if e["file"] != dest.name]
    audio, info = audio_io.load(dest)
    level = round(loudness.integrated(audio, info.sample_rate), 2)
    listed.append({"label": label, "file": dest.name, "lufs": level})
    projects.write_json(folder / "external.json", listed)
    return dest


def externals(source: VocalSource) -> list[dict[str, Any]]:
    file = source.folder / EXTERNAL / "external.json"
    if not file.is_file():
        return []
    listed = json.loads(file.read_text(encoding="utf-8"))
    return [e for e in listed if (source.folder / EXTERNAL / e["file"]).is_file()]


def candidates(source: VocalSource) -> list[dict[str, Any]]:
    """Everything of this vocal there is to listen to: original, tuned versions, outside ones."""
    audio, info = audio_io.load(source.path)
    level = round(loudness.integrated(audio, info.sample_rate), 2)
    found = [{"id": "original", "label": "Original", "path": source.path, "lufs": level}]
    for version in tuned_versions(source):
        path = source.folder / version.file
        found.append(
            {
                "id": f"tuned_{version.id}",
                "label": version.label,
                "path": path,
                "lufs": version.lufs,
            }
        )
    for number, other in enumerate(externals(source)):
        path = source.folder / EXTERNAL / other["file"]
        found.append(
            {
                "id": f"external_{number}",
                "label": other["label"],
                "path": path,
                "lufs": other["lufs"],
            }
        )
    return found


def mix_candidates(source: VocalSource) -> list[dict[str, Any]]:
    """For a vocal separated from a mix: the mix, and the mix with each tuned vocal put back."""
    if source.mix is None:
        return []
    full, info = audio_io.load(source.mix)
    level = round(loudness.integrated(full, info.sample_rate), 2)
    found = [{"id": "mix", "label": "Mix", "path": source.mix, "lufs": level}]
    for version in tuned_versions(source):
        if version.preview_file and (source.folder / version.preview_file).is_file():
            found.append(
                {
                    "id": f"preview_{version.id}",
                    "label": f"Mix, vocal tuned ({version.settings.engine}, preview)",
                    "path": source.folder / version.preview_file,
                    # Swapping the vocal for a tuned one does not change the mix's loudness.
                    "lufs": level,
                }
            )
    return found


@dataclass(frozen=True)
class BenchmarkRow:
    preset: str
    engine: str
    tuned_id: str
    seconds: float
    realtime_factor: float
    in_tune_share: float | None
    mean_abs_cents: float | None
    vibrato_kept: float | None
    timbre_change_db: float


def _vibrato(pitch: PitchTrack, notes: list[pitch_track.Note]) -> float:
    """How much the pitch moves around each note's sustained level, in cents (RMS).

    Measured inside the given notes, away from their first 150 ms and their last frames,
    so the same stretches can be compared before and after tuning."""
    sustain = max(3, int(round(pitch_track.SUSTAIN_S / pitch.frame_s)))
    skip, tail = int(round(0.15 / pitch.frame_s)), int(round(0.03 / pitch.frame_s))
    moved = []
    for note in notes:
        frames = pitch.midi[note.start + skip : note.end - tail]
        frames = frames[np.isfinite(frames)]
        if len(frames) > sustain:
            moved.append(frames - pitch_track.sustained(frames, sustain))
    return float(np.sqrt(np.mean(np.concatenate(moved) ** 2)) * 100.0) if moved else 0.0


def _timbre_change_db(before: np.ndarray, after: np.ndarray, sample_rate: int) -> float:
    """Mean difference between the long-term spectra, third-octave smoothed, 200 Hz to 8 kHz."""
    levels = []
    for audio in (before, after):
        freqs, power = signal.welch(pitch_track.to_mono(audio), sample_rate, nperseg=4096)
        edges = 200.0 * 2.0 ** (np.arange(0, 17) / 3.0)
        bands = [
            power[(freqs >= lo) & (freqs < hi)].mean()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        levels.append(10.0 * np.log10(np.maximum(bands, 1e-20)))
    return float(np.mean(np.abs(levels[1] - levels[0])))


def benchmark(
    source: VocalSource,
    paths: Paths,
    key: str,
    scale_kind: str | None = None,
    preset_names: tuple[str, ...] = BENCHMARK_PRESETS,
) -> list[BenchmarkRow]:
    """Tune one vocal with both engines under each preset and measure what can be measured.

    The numbers say whether an engine hit the notes, kept the vibrato and left the tone
    alone. They do not say which sounds better; the versions are stored for a blind test.
    """
    audio, info = audio_io.load(source.path)
    pitch = pitch_of(source)
    notes = pitch_track.segment(pitch)
    vibrato_before = _vibrato(pitch, notes)
    rows = []
    for name in preset_names:
        preset = presets.get(paths, name)
        for engine in tuning.ENGINES:
            settings = preset.settings(key, scale_kind, engine=engine)
            version = tune(source, settings, edits={}, label=name.capitalize())
            tuned = audio_io.load(source.folder / version.file)[0]
            after = pitch_track.track(tuned, info.sample_rate)
            kept = None
            if vibrato_before > 1.0:
                kept = round(_vibrato(after, notes) / vibrato_before, 2)
            rows.append(
                BenchmarkRow(
                    preset=name,
                    engine=engine,
                    tuned_id=version.id,
                    seconds=version.seconds,
                    realtime_factor=round(info.duration_s / max(version.seconds, 1e-3), 1),
                    in_tune_share=version.after["in_tune_share"],
                    mean_abs_cents=version.after["mean_abs_cents"],
                    vibrato_kept=kept,
                    timbre_change_db=round(_timbre_change_db(audio, tuned, info.sample_rate), 2),
                )
            )
    projects.write_json(source.folder / "benchmark.json", [asdict(row) for row in rows])
    return rows
