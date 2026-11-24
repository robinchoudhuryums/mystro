"""Pitch correction: what each frame of a vocal should be moved by, and the tuned audio.

The correction is planned in semitones per 5 ms frame and then rendered by an engine. Three
controls shape it:

- amount: how much of the distance to the scale note is closed.
- humanize: 1 moves each note as a whole, keeping vibrato, drift and slides; 0 flattens
  every frame onto its target.
- retune speed: how long the correction takes to arrive. At 0 ms it is instant, which with
  humanize at 0 is the hard-tune effect.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from masterbench.vocal import engines, pitch_track, scales
from masterbench.vocal.pitch_track import Note, PitchTrack
from masterbench.vocal.scales import Scale

ENGINES = ("rubberband", "world")
# A staircase target only changes once the pitch is this much nearer to another scale note.
HYSTERESIS_SEMITONES = 0.1
MAX_FORMANT_SHIFT = 12.0
GLITCH_SEMITONES = 0.7


class TuningError(ValueError):
    """Settings or edits that cannot be applied."""


@dataclass(frozen=True)
class TuningSettings:
    key: str = "C major"
    scale: str | None = None
    custom_notes: tuple[str, ...] = ()
    retune_ms: float = 100.0
    humanize: float = 1.0
    amount: float = 0.85
    preserve_formants: bool = True
    formant_shift: float = 0.0
    note_hold: bool = False
    engine: str = "rubberband"

    def __post_init__(self) -> None:
        # 100 and 100.0 are one setting and must give one file name.
        for name in ("retune_ms", "humanize", "amount", "formant_shift"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not 0.0 <= self.amount <= 1.0:
            raise TuningError("amount must be between 0 and 1")
        if not 0.0 <= self.humanize <= 1.0:
            raise TuningError("humanize must be between 0 and 1")
        if not 0.0 <= self.retune_ms <= 1000.0:
            raise TuningError("retune speed must be between 0 and 1000 ms")
        if abs(self.formant_shift) > MAX_FORMANT_SHIFT:
            raise TuningError(f"formant shift must be within ±{MAX_FORMANT_SHIFT:g} semitones")
        if self.engine not in ENGINES:
            raise TuningError(f"unknown engine '{self.engine}'; use {' or '.join(ENGINES)}")
        self.resolve_scale()

    def resolve_scale(self) -> Scale:
        if self.custom_notes:
            return scales.custom(list(self.custom_notes))
        return scales.scale(self.key, self.scale)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["custom_notes"] = list(self.custom_notes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuningSettings:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        if "custom_notes" in known:
            known["custom_notes"] = tuple(known["custom_notes"])
        return cls(**known)


@dataclass(frozen=True)
class NoteEdit:
    """A change to one note made by hand: leave it alone, or send it somewhere else."""

    bypass: bool = False
    target: float | None = None


Edits = dict[int, NoteEdit]


@dataclass(frozen=True)
class Plan:
    """The correction for every frame, in semitones, and where each note is sent."""

    targets: list[float | None]
    correction: np.ndarray


@dataclass(frozen=True)
class TuneResult:
    audio: np.ndarray
    pitch: PitchTrack
    notes: list[Note]
    plan: Plan
    settings: TuningSettings
    edits: Edits = field(default_factory=dict)


def settings_hash(settings: TuningSettings, edits: Edits | None = None) -> str:
    """Short id of everything that decides the result, for the output's file name."""
    listed = {str(index): asdict(edit) for index, edit in sorted((edits or {}).items())}
    text = json.dumps({"settings": settings.to_dict(), "edits": listed}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def note_targets(notes: list[Note], scale: Scale, edits: Edits | None = None) -> list[float | None]:
    """The pitch each note is sent to; None for a bypassed note."""
    edits = edits or {}
    unknown = sorted(set(edits) - {note.index for note in notes})
    if unknown:
        raise TuningError(f"no note number {unknown[0] + 1}; this vocal has {len(notes)} notes")
    targets: list[float | None] = []
    for note in notes:
        edit = edits.get(note.index, NoteEdit())
        if edit.bypass:
            targets.append(None)
        elif edit.target is not None:
            targets.append(float(edit.target))
        else:
            targets.append(float(scale.nearest(note.center)[0]))
    return targets


def _fill(values: np.ndarray, fallback: float) -> np.ndarray:
    """Replace NaN (a frame or two of lost voicing inside a note) by neighbours."""
    known = np.isfinite(values)
    if known.all():
        return values
    if not known.any():
        return np.full(len(values), fallback)
    index = np.arange(len(values))
    return np.interp(index, index[known], values[known])


def _staircase(midi: np.ndarray, scale: Scale) -> np.ndarray:
    """The nearest scale note for each frame, held until the pitch clearly favours another."""
    nearest = scale.nearest(midi)
    out = np.empty(len(midi))
    current = nearest[0]
    for i, pitch in enumerate(midi):
        if abs(pitch - nearest[i]) + HYSTERESIS_SEMITONES < abs(pitch - current):
            current = nearest[i]
        out[i] = current
    return out


def _between(
    midi: np.ndarray, left: float | None, right: float | None, scale: Scale, hold: bool
) -> np.ndarray:
    """Where the frames of a slide are snapped to when they are flattened."""
    anchors = [target for target in (left, right) if target is not None]
    if not hold or not anchors:
        return _staircase(midi, scale)
    choices = np.array(anchors)
    return choices[np.argmin(np.abs(midi[:, None] - choices[None, :]), axis=1)]


def plan(
    pitch: PitchTrack, notes: list[Note], settings: TuningSettings, edits: Edits | None = None
) -> Plan:
    """Work out the correction for every frame."""
    scale = settings.resolve_scale()
    targets = note_targets(notes, scale, edits)
    midi = pitch.midi
    frame_s = pitch.frame_s
    keep = settings.humanize
    sustain = max(3, int(round(pitch_track.SUSTAIN_S / frame_s)))
    desired = np.zeros(len(midi))
    in_note = np.zeros(len(midi), dtype=bool)
    # The sustained correction at each note's first and last frame, for the slides around it.
    edge: dict[int, tuple[float, float]] = {}
    owner = np.full(len(midi), -1)

    for note, target in zip(notes, targets, strict=True):
        in_note[note.frames] = True
        owner[note.frames] = note.index
        if target is None:
            edge[note.index] = (0.0, 0.0)
            continue
        deviation = _fill(midi[note.frames], note.center) - target
        slow = pitch_track.sustained(deviation, sustain)
        fast = deviation - slow
        # A frame far from its note is a tracking slip, not singing: do not chase it.
        fast[np.abs(fast) > GLITCH_SEMITONES] = 0.0
        desired[note.frames] = -(slow + (1.0 - keep) * fast)
        edge[note.index] = (-float(slow[0]), -float(slow[-1]))

    correction = np.zeros(len(midi))
    step = 1.0
    if settings.retune_ms > 0:
        # Retune speed is the time taken to cover 90% of the way.
        step = 1.0 - math.exp(-frame_s * math.log(10.0) / (settings.retune_ms / 1000.0))
    for start, end in pitch_track.voiced_runs(pitch.voiced):
        i = start
        while i < end:
            if in_note[i]:
                i += 1
                continue
            stop = i
            while stop < end and not in_note[stop]:
                stop += 1
            before = int(owner[i - 1]) if i > start else -1
            after = int(owner[stop]) if stop < end else -1
            sides = [edge[before][1]] if before >= 0 else []
            sides += [edge[after][0]] if after >= 0 else []
            # A slide with a note on one side only moves with that note; with none, not at all.
            ramp = np.linspace(sides[0], sides[-1], stop - i) if sides else np.zeros(stop - i)
            slide = _fill(midi[i:stop], float(np.nanmedian(midi[start:end])))
            snapped = _between(
                slide,
                targets[before] if before >= 0 else None,
                targets[after] if after >= 0 else None,
                scale,
                settings.note_hold,
            )
            desired[i:stop] = keep * ramp + (1.0 - keep) * (snapped - slide)
            i = stop
        wanted = desired[start:end] * settings.amount
        if step >= 1.0:
            correction[start:end] = wanted
            continue
        value = 0.0
        for offset, goal in enumerate(wanted):
            value += (goal - value) * step
            correction[start + offset] = value
    correction[~pitch.voiced] = 0.0
    return Plan(targets, correction)


def tune(
    audio: np.ndarray,
    sample_rate: int,
    settings: TuningSettings,
    edits: Edits | None = None,
    pitch: PitchTrack | None = None,
) -> TuneResult:
    """Tune a solo vocal. The result has the input's shape; unvoiced audio is untouched."""
    pitch = pitch or pitch_track.track(audio, sample_rate)
    notes = pitch_track.segment(pitch)
    planned = plan(pitch, notes, settings, edits)
    tuned = engines.render(audio, sample_rate, pitch, planned.correction, settings)
    return TuneResult(tuned, pitch, notes, planned, settings, dict(edits or {}))


def with_amount(settings: TuningSettings, amount: float | None) -> TuningSettings:
    return settings if amount is None else replace(settings, amount=amount)
