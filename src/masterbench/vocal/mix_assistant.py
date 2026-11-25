"""Vocal mix assistant: a dry vocal and an instrumental in, a mixed and mastered track out.

The lead goes through a vocal chain (high-pass, cuts at its own resonances, de-esser,
compression, tone, saturation, an optional chain of your own), is ridden against the beat,
and gets reverb and a tempo-synced delay as parallel sends. The beat gets a small dip where
the vocal lives, while the vocal is there. The sum goes to the mastering pipeline, and the
processed vocal and instrumental are written separately too.

It is a fast way from a rough to a release candidate. It decides nothing creative.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import anthropic
import numpy as np
import pedalboard as pb

from masterbench import audio_io, index, projects
from masterbench.analysis import detectors, loudness
from masterbench.analysis import report as analysis
from masterbench.analysis.flags import Flag
from masterbench.config import Paths, Settings, Thresholds
from masterbench.master import chain, pipeline
from masterbench.master import presets as chain_presets
from masterbench.master.chain import ChainPreset, Module
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.notes import mix_notes
from masterbench.notes.mix_notes import MixNotes, NotesError
from masterbench.references import library
from masterbench.stems import workflow as stem_workflow
from masterbench.stems.separate import StemSet
from masterbench.vocal import dynamics
from masterbench.vocal.dynamics import SectionRatio

MIX_DIR = "vocal_mix"
DEFAULT_CHAIN = "modern-rap"
# Where the lead sits against the beat, in LU, when nothing better is known. Starting
# points chosen for this project, not measured from records.
GENRE_TARGET_DB = {"hip-hop": -2.0, "house": -6.0, "electronic": -6.0}
FALLBACK_TARGET_DB = -3.0
INPUT_LUFS = -20.0
HEADROOM_DBFS = -1.0
MAX_CORRECTIVE_CUT_DB = 6.0
NOTE_BEATS = {"1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/8d": 0.75, "1/16": 0.25}
ADLIB_PRESETS: dict[str, dict[str, float]] = {
    "tucked": {"level_db": -8.0, "pan": 0.0, "width": 0.0},
    "wide": {"level_db": -6.0, "pan": 0.0, "width": 1.0},
    "left": {"level_db": -6.0, "pan": -0.6, "width": 0.0},
    "right": {"level_db": -6.0, "pan": 0.6, "width": 0.0},
}
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ProgressFn = Callable[[str, float], object]


class MixError(ValueError):
    """Inputs or settings the assistant cannot work with."""


@dataclass(frozen=True)
class VocalChain:
    """A vocal chain preset. Thresholds assume the vocal has been brought to -20 LUFS."""

    name: str
    description: str = ""
    highpass_hz: float = 90.0
    corrective_eq: bool = True
    deess_threshold_db: float = -12.0
    deess_max_db: float = 8.0
    compressor: dict[str, float] | None = None
    tone: tuple[dict[str, Any], ...] = ()
    saturation: dict[str, float] | None = None
    insert: str | None = None
    reverb_db: float | None = None
    reverb_size: float = 0.3
    delay_db: float | None = None
    delay_note: str = "1/8"
    delay_feedback: float = 0.25

    def __post_init__(self) -> None:
        if self.delay_note not in NOTE_BEATS:
            raise MixError(f"delay_note must be one of {', '.join(NOTE_BEATS)}")
        if not 0.0 <= self.reverb_size <= 1.0 or not 0.0 <= self.delay_feedback < 0.9:
            raise MixError("reverb_size is 0 to 1 and delay_feedback 0 to 0.9")
        for level in (self.reverb_db, self.delay_db):
            if level is not None and not -60.0 <= level <= 0.0:
                raise MixError("send levels are in dB below the vocal, between -60 and 0")
        chain.validate(
            ChainPreset(self.name, "", (*self.before_deesser([]), *self.after_deesser()))
        )

    def before_deesser(self, cuts: list[Module]) -> tuple[Module, ...]:
        return (Module("highpass", {"cutoff_hz": self.highpass_hz}), *cuts)

    def after_deesser(self) -> tuple[Module, ...]:
        modules = []
        if self.compressor:
            modules.append(Module("compressor", dict(self.compressor)))
        for entry in self.tone:
            params = {key: value for key, value in entry.items() if key != "type"}
            modules.append(Module(str(entry.get("type", "")), params))
        if self.saturation:
            modules.append(Module("saturation", dict(self.saturation)))
        return tuple(modules)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tone"] = [dict(entry) for entry in self.tone]
        return data


def chain_from_toml(text: str, name: str) -> VocalChain:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise MixError(f"vocal chain '{name}' is not valid TOML: {exc}") from exc
    unknown = sorted(set(data) - set(VocalChain.__dataclass_fields__))
    if unknown:
        raise MixError(f"vocal chain '{name}' has unknown settings: {', '.join(unknown)}")
    data["tone"] = tuple(data.get("tone", ()))
    data["insert"] = data.get("insert") or None
    try:
        return VocalChain(**{**data, "name": name})
    except chain.ChainError as exc:
        raise MixError(f"vocal chain '{name}': {exc}") from exc


def _packaged() -> dict[str, str]:
    folder = resources.files("masterbench.vocal").joinpath("chains")
    return {
        entry.name.removesuffix(".toml"): entry.read_text(encoding="utf-8")
        for entry in folder.iterdir()
        if entry.name.endswith(".toml")
    }


def chain_names(paths: Paths) -> list[str]:
    """Vocal chain presets, the packaged three first; your own shadow them by name."""
    folder = paths.vocal_chains
    own = {p.stem for p in folder.glob("*.toml")} if folder.is_dir() else set()
    packaged = ["gentle", "modern-rap", "aggressive"]
    return packaged + sorted(own - set(packaged))


def get_chain(paths: Paths, name: str) -> VocalChain:
    if not _NAME.match(name):
        raise MixError(f"preset names use lowercase letters, digits, '-' and '_': {name!r}")
    own = paths.vocal_chains / f"{name}.toml"
    if own.is_file():
        return chain_from_toml(own.read_text(encoding="utf-8"), name)
    packaged = _packaged()
    if name not in packaged:
        raise MixError(f"no vocal chain named '{name}'; have {', '.join(chain_names(paths))}")
    return chain_from_toml(packaged[name], name)


@dataclass(frozen=True)
class MixOptions:
    """What the assistant should do. None means "decide from the genre and the preset"."""

    chain: str = DEFAULT_CHAIN
    title: str | None = None
    genre: str | None = None
    target_db: float | None = None
    target_from_reference: bool = False
    max_ride_db: float = 6.0
    carve_db: float = 3.0
    duck_db: float = 0.0
    reverb_db: float | None = None
    delay_db: float | None = None
    adlib_preset: str = "tucked"
    reference: str | None = None
    by_curve: bool = False
    target_lufs: float | None = None
    ceiling_dbtp: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_ride_db <= 12.0:
            raise MixError("the rider's range must be between 0 and 12 dB")
        if not 0.0 <= self.carve_db <= 6.0 or not 0.0 <= self.duck_db <= 6.0:
            raise MixError("the dip and the duck are limited to 6 dB")
        if self.adlib_preset not in ADLIB_PRESETS:
            raise MixError(f"ad-lib preset must be one of {', '.join(ADLIB_PRESETS)}")


@dataclass(frozen=True)
class VocalMix:
    """What was made, and the numbers that say how it went."""

    id: str
    folder: Path
    files: dict[str, str]
    run: RunResult
    mastered: bool
    target_db: float
    target_source: str
    fader_db: float
    ride_range_db: tuple[float, float]
    sections: list[SectionRatio]
    final_ratio_db: float | None
    reference_ratio_db: float | None
    input_trim_db: float
    corrective_cuts: list[dict[str, float]]
    deess: dict[str, float]
    carve: dict[str, float]
    tempo_bpm: float | None
    delay_ms: float | None
    trim_db: float
    preview_quality: bool = False
    flags: list[Flag] = field(default_factory=list)
    notes: MixNotes | None = None
    notes_error: str | None = None

    def path(self, name: str) -> Path:
        return self.folder / self.files[name]

    def summary(self) -> dict[str, Any]:
        return summarize(asdict(self))


def summarize(fields: dict[str, Any]) -> dict[str, Any]:
    """The numbers of a vocal mix, as stored in run.json and sent with AI notes."""
    sections = [s if isinstance(s, dict) else asdict(s) for s in fields["sections"]]
    return {
        "target_db": fields["target_db"],
        "target_source": fields["target_source"],
        "fader_db": fields["fader_db"],
        "ride_range_db": list(fields["ride_range_db"]),
        "sections": sections,
        "final_vocal_to_instrumental_db": fields["final_ratio_db"],
        "reference_vocal_to_instrumental_db": fields["reference_ratio_db"],
        "input_trim_db": fields["input_trim_db"],
        "corrective_cuts": fields["corrective_cuts"],
        "deess": fields["deess"],
        "carve": fields["carve"],
        "tempo_bpm": fields["tempo_bpm"],
        "delay_ms": fields["delay_ms"],
        "headroom_trim_db": fields["trim_db"],
        "separated_stems": fields["preview_quality"],
    }


def stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    return audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)


def _fit(audio: np.ndarray, frames: int) -> np.ndarray:
    if len(audio) >= frames:
        return audio[:frames]
    return np.concatenate([audio, np.zeros((frames - len(audio), audio.shape[1]))])


def corrective_cuts(vocal: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Module]:
    """Narrow cuts at the vocal's own resonances, half as deep as each one stands out."""
    found = detectors.resonance_peaks(dynamics.mono(vocal), sample_rate, limits)
    return [
        Module(
            "peak",
            {
                "freq_hz": round(peak.freq_hz, 1),
                "gain_db": -round(min(MAX_CORRECTIVE_CUT_DB, peak.prominence_db / 2.0), 1),
                "q": round(float(np.clip(peak.q, 2.0, 10.0)), 1),
            },
        )
        for peak in found
    ]


@dataclass(frozen=True)
class Processed:
    audio: np.ndarray
    input_trim_db: float
    cuts: list[Module]
    deess: dynamics.Deessed


def process_vocal(
    vocal: np.ndarray, sample_rate: int, preset: VocalChain, paths: Paths, limits: Thresholds
) -> Processed:
    """Run a vocal through the insert part of the chain. The result is stereo and dry."""
    level = loudness.integrated(stereo(vocal), sample_rate)
    if level <= loudness.FLOOR_DB + 1:
        raise MixError("the vocal is silent")
    trim = INPUT_LUFS - level
    audio = stereo(vocal) * 10.0 ** (trim / 20.0)
    cuts = corrective_cuts(audio, sample_rate, limits) if preset.corrective_eq else []
    audio = chain.run(ChainPreset("vocal", "", preset.before_deesser(cuts)), audio, sample_rate)
    deessed = dynamics.deess(audio, sample_rate, preset.deess_threshold_db, preset.deess_max_db)
    audio = deessed.audio
    after = preset.after_deesser()
    if after:
        audio = chain.run(ChainPreset("vocal", "", after), audio, sample_rate)
    if preset.insert:
        audio = chain.run(chain_presets.get(paths, preset.insert), audio, sample_rate)
    return Processed(audio, round(trim, 2), cuts, deessed)


def _at_level(wet: np.ndarray, dry: np.ndarray, sample_rate: int, below_db: float) -> np.ndarray:
    """Scale a send so it sits a set number of LU under the dry vocal."""
    wet_level = loudness.integrated(wet, sample_rate)
    if wet_level <= loudness.FLOOR_DB + 1:
        return np.zeros_like(wet)
    wanted = loudness.integrated(dry, sample_rate) + below_db
    return wet * 10.0 ** ((wanted - wet_level) / 20.0)


def sends(
    dry: np.ndarray, sample_rate: int, preset: VocalChain, tempo_bpm: float | None
) -> tuple[np.ndarray, float | None]:
    """Reverb and delay returns for a vocal, each on its own and summed. Tails are cut at the end.

    Returns the wet signal and the delay time in ms (None without a delay).
    """
    source = np.ascontiguousarray(dry.T, dtype=np.float32)
    wet = np.zeros_like(dry)
    if preset.reverb_db is not None:
        board = pb.Pedalboard(  # pyright: ignore[reportPrivateImportUsage]
            [
                pb.HighpassFilter(250.0),  # pyright: ignore[reportPrivateImportUsage]
                pb.Reverb(  # pyright: ignore[reportPrivateImportUsage]
                    room_size=preset.reverb_size, damping=0.5, wet_level=1.0, dry_level=0.0
                ),
            ]
        )
        room = np.asarray(board(source, sample_rate), dtype=np.float64).T
        wet += _at_level(room, dry, sample_rate, preset.reverb_db)
    delay_ms = None
    if preset.delay_db is not None and tempo_bpm:
        seconds = NOTE_BEATS[preset.delay_note] * 60.0 / tempo_bpm
        delay_ms = round(seconds * 1000.0, 1)
        board = pb.Pedalboard(  # pyright: ignore[reportPrivateImportUsage]
            [
                pb.Delay(  # pyright: ignore[reportPrivateImportUsage]
                    delay_seconds=seconds, feedback=preset.delay_feedback, mix=1.0
                ),
                pb.HighpassFilter(300.0),  # pyright: ignore[reportPrivateImportUsage]
                pb.LowpassFilter(5000.0),  # pyright: ignore[reportPrivateImportUsage]
            ]
        )
        echo = np.asarray(board(source, sample_rate), dtype=np.float64).T
        wet += _at_level(echo, dry, sample_rate, preset.delay_db)
    return wet, delay_ms


def place(audio: np.ndarray, sample_rate: int, pan: float, width: float) -> np.ndarray:
    """Pan (-1 left to 1 right, constant power) and widen (0 to 1) a centred stereo signal.

    Width delays the right side by up to 12 ms. It is wide on speakers and combs in mono,
    which is the usual price of this trick; keep it for parts that may thin out in mono.
    """
    out = audio.copy()
    if width > 0:
        shift = int(width * 0.012 * sample_rate)
        out[shift:, 1] = audio[: len(audio) - shift, 1]
        out[:shift, 1] = 0.0
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return out * np.array([np.cos(angle), np.sin(angle)]) * np.sqrt(2.0)


def target_for(
    paths: Paths, settings: Settings, options: MixOptions, genre: str | None
) -> tuple[float, str, float | None]:
    """The vocal-to-beat target, where it came from, and the reference's own figure if known."""
    reference_db = None
    reference = _reference(paths, settings, options, genre)
    if reference is not None:
        cached = reference.path.parent / ".stems"
        stem_set = _newest_set(cached) if cached.is_dir() else None
        if stem_set is None and options.target_from_reference:
            stem_set = stem_workflow.separate_reference(reference)
        if stem_set is not None:
            reference_db = _stem_ratio(stem_set)
    if options.target_db is not None:
        return options.target_db, "set by hand", reference_db
    if options.target_from_reference:
        if reference_db is None:
            raise MixError(
                "no reference with a vocal to take the target from; name one with --ref "
                "('auto' is only picked once the mix exists) or give a number"
            )
        return round(reference_db, 1), "the reference's separated stems", reference_db
    if genre and genre in GENRE_TARGET_DB:
        return GENRE_TARGET_DB[genre], f"the default for {genre}", reference_db
    return FALLBACK_TARGET_DB, "the general default", reference_db


def _newest_set(folder: Path) -> StemSet | None:
    from masterbench.stems import separate

    sets = [s for s in map(separate.load_set, folder.glob("*/")) if s is not None]
    return max(sets, key=lambda s: s.created_at) if sets else None


def _stem_ratio(stem_set: StemSet) -> float | None:
    vocals = stem_set.by_role("vocals")
    if not vocals:
        return None
    voice = sum((stem_set.load(name) for name in vocals), np.zeros((stem_set.frames, 2)))
    rest = [name for name in stem_set.stems if name not in vocals]
    beat = sum((stem_set.load(name) for name in rest), np.zeros((stem_set.frames, 2)))
    found = dynamics.ratio_db(voice, beat, stem_set.sample_rate)
    return None if found is None else round(found, 2)


def _reference_spec(settings: Settings, options: MixOptions, genre: str | None) -> str | None:
    return options.reference or (library.default_spec(settings, genre) if genre else None)


def _reference(
    paths: Paths, settings: Settings, options: MixOptions, genre: str | None
) -> library.Reference | None:
    spec = _reference_spec(settings, options, genre)
    if spec is None or library.auto_genre(spec) is not None:
        # The closest reference is only known once there is a mix to compare with.
        return None
    try:
        return library.resolve(paths, spec, settings)
    except library.ReferenceError:
        if options.reference:
            raise
        return None


def mix_id(inputs: list[str], preset: VocalChain, options: MixOptions) -> str:
    text = json.dumps(
        {"inputs": inputs, "chain": preset.to_dict(), "options": asdict(options)}, sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def mix(
    paths: Paths,
    settings: Settings,
    vocal: Path,
    instrumental: Path,
    options: MixOptions | None = None,
    adlibs: list[Path] | None = None,
    progress: ProgressFn = lambda stage, fraction: None,
    notes: bool = False,
    intent: str = "",
    client: anthropic.Anthropic | None = None,
    separated: bool = False,
) -> VocalMix:
    """Mix a dry vocal into an instrumental and master the result as a run of the track.

    Vocal and instrumental must share a sample rate and start at the same moment, as two
    bounces of one Logic project do. With notes on, AI mix notes are asked for at the end;
    a failure there (no key, over the cap) is reported and does not undo the mix.
    """
    options = options or MixOptions()
    preset = get_chain(paths, options.chain)
    overrides = {"reverb_db": options.reverb_db, "delay_db": options.delay_db}
    preset = replace(preset, **{k: v for k, v in overrides.items() if v is not None})
    limits = settings.thresholds
    flags: list[Flag] = []

    progress("Reading the vocal and the instrumental", 0.02)
    beat, beat_info = audio_io.load(instrumental)
    lead, lead_info = audio_io.load(vocal)
    extra = [audio_io.load(file) for file in adlibs or []]
    for name, info in [
        (vocal.name, lead_info),
        *[(f.name, e[1]) for f, e in zip(adlibs or [], extra, strict=True)],
    ]:
        if info.sample_rate != beat_info.sample_rate:
            raise MixError(
                f"{name} is at {info.sample_rate} Hz and the instrumental at "
                f"{beat_info.sample_rate} Hz; bounce them at the same rate"
            )
    rate = beat_info.sample_rate
    frames = max(len(beat), len(lead))
    if abs(len(beat) - len(lead)) > rate:
        message = (
            f"The vocal is {len(lead) / rate:.1f} s long and the instrumental "
            f"{len(beat) / rate:.1f} s. They are lined up from the start; if they were not "
            "bounced from the same point, the vocal will be out of time."
        )
        flags.append(Flag("lengths_differ", "warn", message))
    beat = _fit(stereo(beat), frames)
    lead = _fit(stereo(lead), frames)
    if separated:
        message = (
            "Preview quality: these stems were separated from a mix, so each still holds some "
            "of the other, and processing them makes that easier to hear. For release, bounce "
            "the vocal and the instrumental from Logic."
        )
        flags.append(Flag("preview_quality", "warn", message))

    title = options.title or projects.title_from_filename(instrumental)
    track, project_dir = projects.ensure_track(paths, title, options.genre)
    genre = track.genre
    target_db, target_source, reference_db = target_for(paths, settings, options, genre)

    progress("Running the vocal chain", 0.1)
    processed = process_vocal(lead, rate, preset, paths, limits)

    progress("Finding the song's sections and tempo", 0.25)
    rough = beat + processed.audio * 10.0 ** (
        (loudness.integrated(beat, rate) + target_db - INPUT_LUFS) / 20.0
    )
    rough_info = replace(beat_info, frames=frames, duration_s=frames / rate, channels=2)
    project = projects.Project(track, project_dir, instrumental, "")
    musical, parts = analysis.song_shape(
        rough, rough_info, settings, pipeline.overrides_for(project)
    )
    tempo = musical.tempo_bpm if musical else None
    listed = [(item.label, item.start_s, item.end_s) for item in parts.items]

    progress("Making room in the beat", 0.4)
    hooks = [(item.start_s, item.end_s) for item in parts.of_kind("hook")]
    carved = dynamics.carve(
        beat, processed.audio, rate, options.carve_db, options.duck_db, hooks or None
    )

    progress("Riding the vocal", 0.5)
    ridden = dynamics.ride(
        processed.audio, carved.audio, rate, target_db, listed, options.max_ride_db
    )
    if any(section.limited for section in ridden.sections):
        short = [s.label for s in ridden.sections if s.limited]
        message = (
            f"The rider ran out of range (±{options.max_ride_db:g} dB) in {', '.join(short)}, "
            "so the vocal is not on target there. Even the take out in Logic, or allow more."
        )
        flags.append(Flag("rider_at_limit", "warn", message, options.max_ride_db))
    wet, delay_ms = sends(ridden.audio, rate, preset, tempo)
    if preset.delay_db is not None and delay_ms is None:
        message = "No tempo could be detected, so the tempo-synced delay was left out."
        flags.append(Flag("delay_skipped", "info", message))
    lead_out = ridden.audio + wet

    backing = np.zeros_like(beat)
    place_as = ADLIB_PRESETS[options.adlib_preset]
    lead_level = loudness.integrated(ridden.audio, rate)
    for number, (audio, _) in enumerate(extra):
        progress(f"Placing ad-lib {number + 1}", 0.6)
        part = process_vocal(_fit(stereo(audio), frames), rate, preset, paths, limits).audio
        part *= 10.0 ** (
            (lead_level + place_as["level_db"] - loudness.integrated(part, rate)) / 20.0
        )
        # Several ad-lib files under one "left" or "right" preset take alternate sides.
        pan = place_as["pan"] * (-1.0 if number % 2 else 1.0)
        part = place(part, rate, pan, place_as["width"])
        backing += part + sends(part, rate, replace(preset, delay_db=None), tempo)[0]

    progress("Summing", 0.7)
    total = carved.audio + lead_out + backing
    peak = float(np.max(np.abs(total)))
    ceiling = 10.0 ** (HEADROOM_DBFS / 20.0)
    trim_db = min(0.0, 20.0 * np.log10(ceiling / peak)) if peak > 0 else 0.0
    scale = 10.0 ** (trim_db / 20.0)

    inputs = [audio_io.file_sha256(file) for file in (vocal, instrumental, *(adlibs or []))]
    made = mix_id(inputs, preset, options)
    folder = project_dir / MIX_DIR / made
    files = {"mix": f"vocal-mix-{made}.wav", "lead_vocal": "lead_vocal.wav"}
    files["instrumental"] = "instrumental.wav"
    audio_io.save_wav(folder / files["mix"], total * scale, rate, 24)
    audio_io.save_wav(folder / files["lead_vocal"], lead_out * scale, rate, 24)
    audio_io.save_wav(folder / files["instrumental"], carved.audio * scale, rate, 24)
    if extra:
        files["adlibs"] = "adlibs.wav"
        audio_io.save_wav(folder / files["adlibs"], backing * scale, rate, 24)

    final_ratio = dynamics.ratio_db(lead_out, carved.audio, rate)
    sections = [
        replace(
            section,
            after_db=_section_ratio(lead_out, carved.audio, rate, section),
        )
        for section in ridden.sections
    ]
    flags += detectors.sibilance(dynamics.mono(lead_out), rate, limits, is_vocal=True)
    # Where the beat itself steps from one section to the next, a one-second window reads a
    # jump in the vocal that is not there; only moments inside a section are reported.
    edges = [edge for _, start_s, end_s in listed for edge in (start_s, end_s)]
    flags += [
        flag
        for flag in detectors.vocal_consistency(ridden.audio, carved.audio, rate, limits)
        if all(abs((flag.start_s or 0.0) - edge) > 1.5 for edge in edges)
    ]

    fields: dict[str, Any] = {
        "id": made,
        "folder": folder,
        "files": files,
        "target_db": target_db,
        "target_source": target_source,
        "fader_db": ridden.fader_db,
        "ride_range_db": ridden.range_db,
        "sections": sections,
        "final_ratio_db": None if final_ratio is None else round(final_ratio, 2),
        "reference_ratio_db": reference_db,
        "input_trim_db": processed.input_trim_db,
        "corrective_cuts": [dict(cut.params) for cut in processed.cuts],
        "deess": {
            "events": processed.deess.events,
            "max_reduction_db": processed.deess.max_reduction_db,
            "mean_reduction_db": processed.deess.mean_reduction_db,
        },
        "carve": {
            "depth_db": options.carve_db,
            "mean_dip_db": carved.mean_dip_db,
            "max_duck_db": carved.max_duck_db,
        },
        "tempo_bpm": tempo,
        "delay_ms": delay_ms,
        "trim_db": round(float(trim_db), 2),
        "preview_quality": separated,
    }
    made_from = {
        "kind": "vocal_mix",
        "id": made,
        "vocal": vocal.name,
        "instrumental": instrumental.name,
        "adlibs": [file.name for file in adlibs or []],
        "chain": preset.to_dict(),
        "options": asdict(options),
        "result": summarize(fields),
    }
    run, mastered = _master(
        paths, settings, folder / files["mix"], options, title, genre, made_from, progress
    )
    if not mastered:
        message = (
            f"There is no reference or target curve for {genre or 'this track'}, so the mix was "
            "analyzed but not mastered. Add a reference (masterbench ref add) and run it again."
        )
        flags.append(Flag("not_mastered", "warn", message))
    result = VocalMix(run=run, mastered=mastered, flags=flags, **fields)
    projects.write_json(folder / "vocal_mix.json", {**made_from, "run_id": run.manifest.id})
    if notes:
        progress("Asking for AI mix notes", 0.97)
        try:
            found = mix_notes.generate(run, paths, settings, intent, client=client)
            with index.open_index(paths) as conn:
                index.sync_notes(conn, paths)
            result = replace(result, notes=found)
        except NotesError as exc:
            result = replace(result, notes_error=str(exc))
    return result


def _section_ratio(
    vocal: np.ndarray, beat: np.ndarray, sample_rate: int, section: SectionRatio
) -> float | None:
    if section.after_db is None:
        return None
    a, b = int(section.start_s * sample_rate), int(section.end_s * sample_rate)
    found = dynamics.ratio_db(vocal[a:b], beat[a:b], sample_rate)
    return None if found is None else round(found, 2)


def _master(
    paths: Paths,
    settings: Settings,
    source: Path,
    options: MixOptions,
    title: str,
    genre: str | None,
    made_from: dict[str, Any],
    progress: ProgressFn,
) -> tuple[RunResult, bool]:
    """Master the mix against the genre's reference or curve; without either, analyze it."""
    by_curve = options.by_curve
    spec = _reference_spec(settings, options, genre)
    bank = None if spec is None else library.auto_genre(spec)
    if bank is None:
        have_reference = _reference(paths, settings, options, genre) is not None
    else:
        have_reference = bool(library.list_references(paths, bank or genre))
    if not have_reference and not by_curve:
        from masterbench.references import target_curve

        by_curve = target_curve.load(paths, genre) is not None
    if not have_reference and not by_curve:
        return pipeline.analyze(source, paths, settings, genre, title, progress), False
    targets = settings.targets
    wanted = MasterOptions(
        reference=spec or "",
        target_lufs=options.target_lufs if options.target_lufs is not None else targets.lufs_i,
        ceiling_dbtp=(
            options.ceiling_dbtp if options.ceiling_dbtp is not None else targets.true_peak_dbtp
        ),
        genre=genre,
        title=title,
        mode="curve" if by_curve else "matchering",
        made_from=made_from,
    )
    return pipeline.master(source, wanted, paths, settings, progress), True


def from_stems(
    paths: Paths,
    settings: Settings,
    track_slug: str,
    options: MixOptions | None = None,
    **kwargs: Any,
) -> VocalMix:
    """Mix from a track's stem set: its vocal stem against the sum of the others."""
    stem_set = stem_workflow.find_set(paths, track_slug)
    vocals = stem_set.by_role("vocals")
    if len(vocals) != 1:
        raise MixError(f"stem set '{stem_set.name}' needs exactly one vocal stem to mix from")
    rest = [name for name in stem_set.stems if name != vocals[0]]
    beat = sum((stem_set.load(name) for name in rest), np.zeros((stem_set.frames, 2)))
    beat_file = audio_io.save_wav(
        stem_set.dir / "instrumental-sum.wav", beat, stem_set.sample_rate, 32
    )
    title = json.loads((paths.projects / track_slug / "track.json").read_text("utf-8"))["title"]
    wanted = replace(options or MixOptions(), title=title)
    return mix(
        paths,
        settings,
        stem_set.path(vocals[0]),
        beat_file,
        wanted,
        separated=stem_set.separated,
        **kwargs,
    )
