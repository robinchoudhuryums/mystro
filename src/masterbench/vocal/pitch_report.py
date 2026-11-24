"""The pitch report: how in tune a vocal is, note by note. Reads only; changes nothing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from masterbench.analysis.flags import Flag
from masterbench.vocal import pitch_track, scales
from masterbench.vocal.pitch_track import Note, PitchTrack
from masterbench.vocal.scales import Scale

IN_TUNE_CENTS = 20.0
PHRASE_GAP_S = 0.35
WORST_PHRASES = 3
WORTH_MENTIONING_CENTS = 10.0
CURVE_STEP = 2
MIN_VOICED_SHARE = 0.05


@dataclass(frozen=True)
class NoteReport:
    index: int
    start_s: float
    end_s: float
    center: float
    name: str
    target: float
    target_name: str
    cents: float
    in_key: bool
    spread_cents: float


@dataclass(frozen=True)
class Phrase:
    start_s: float
    end_s: float
    notes: int
    mean_abs_cents: float
    worst_note: int
    worst_cents: float


@dataclass(frozen=True)
class PitchReport:
    key: str
    key_source: str
    key_confidence: float | None
    scale: str
    scale_notes: list[str]
    duration_s: float
    voiced_share: float
    notes: list[NoteReport]
    in_tune_share: float | None
    mean_cents: float | None
    mean_abs_cents: float | None
    out_of_key: list[int]
    worst_phrases: list[Phrase]
    flags: list[Flag] = field(default_factory=list)
    curve_step_s: float = 0.01
    curve: list[float | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PitchReport:
        return cls(
            **{
                **data,
                "notes": [NoteReport(**note) for note in data["notes"]],
                "worst_phrases": [Phrase(**phrase) for phrase in data["worst_phrases"]],
                "flags": [Flag(**flag) for flag in data["flags"]],
            }
        )

    def headline(self) -> str:
        if self.in_tune_share is None:
            return "no sung notes found"
        return (
            f"{len(self.notes)} notes in {self.scale}, {self.in_tune_share:.0%} of the time "
            f"within ±{IN_TUNE_CENTS:.0f} cents, {self.mean_abs_cents:.0f} cents off on average"
        )


def _phrases(notes: list[NoteReport]) -> list[Phrase]:
    groups: list[list[NoteReport]] = []
    for note in notes:
        if groups and note.start_s - groups[-1][-1].end_s < PHRASE_GAP_S:
            groups[-1].append(note)
        else:
            groups.append([note])
    phrases = []
    for group in groups:
        weights = np.array([n.end_s - n.start_s for n in group])
        off = np.array([abs(n.cents) for n in group])
        worst = group[int(np.argmax(off))]
        phrases.append(
            Phrase(
                start_s=round(group[0].start_s, 2),
                end_s=round(group[-1].end_s, 2),
                notes=len(group),
                mean_abs_cents=round(float(np.average(off, weights=weights)), 1),
                worst_note=worst.index,
                worst_cents=worst.cents,
            )
        )
    return phrases


def _flags(
    report_notes: list[NoteReport],
    voiced_share: float,
    mean_cents: float | None,
    out_of_key: list[int],
    scale: Scale,
    key_source: str,
    key_confidence: float | None,
    separated: bool,
) -> list[Flag]:
    flags = []
    if separated:
        message = (
            "This vocal was separated from a mix. Leftover reverb and instruments confuse "
            "pitch tracking, so expect some wrong notes here; a dry vocal from Logic is better."
        )
        flags.append(Flag("separated_vocal", "warn", message))
    if len(report_notes) < 2 or voiced_share < MIN_VOICED_SHARE:
        message = (
            "Hardly any sung pitch was found. Is this a solo vocal? A full mix, a rap verse "
            "or a heavily effected track will not give a usable report."
        )
        flags.append(Flag("vocal_no_pitch", "problem", message, round(voiced_share, 3)))
        return flags
    if key_source == "detected" and (key_confidence or 0.0) < 0.5:
        message = (
            f"The key ({scale.label}) was guessed from the sung notes alone and is uncertain. "
            "Set the key by hand if the targets look wrong."
        )
        flags.append(Flag("key_uncertain", "info", message, key_confidence))
    if out_of_key:
        count = len(out_of_key)
        first = report_notes[out_of_key[0]]
        message = (
            f"{count} note{'s' if count != 1 else ''} sit on a pitch outside {scale.label} "
            f"(first: {first.name} at {first.start_s:.1f} s). Tuning moves them onto the scale; "
            "bypass them if they are meant, or use the chromatic scale."
        )
        flags.append(Flag("out_of_key_notes", "info", message, float(count), first.start_s))
    if mean_cents is not None and abs(mean_cents) >= WORTH_MENTIONING_CENTS:
        side = "sharp" if mean_cents > 0 else "flat"
        message = f"On average the vocal is {abs(mean_cents):.0f} cents {side}."
        flags.append(Flag(f"vocal_{side}", "info", message, round(mean_cents, 1)))
    return flags


def build(
    pitch: PitchTrack,
    notes: list[Note],
    scale: Scale,
    key: str,
    key_source: str = "given",
    key_confidence: float | None = None,
    separated: bool = False,
) -> PitchReport:
    """Measure detected notes against a scale."""
    frame_s = pitch.frame_s
    midi = pitch.midi
    sustain = max(3, int(round(pitch_track.SUSTAIN_S / frame_s)))
    report_notes, inside, total = [], 0, 0
    for note in notes:
        target = float(scale.nearest(note.center)[0])
        frames = midi[note.frames]
        frames = frames[np.isfinite(frames)]
        # Vibrato is not being out of tune: judge where the note sits, not each wobble.
        slow = pitch_track.sustained(frames - target, sustain)
        inside += int(np.count_nonzero(np.abs(slow) * 100.0 <= IN_TUNE_CENTS))
        total += len(slow)
        report_notes.append(
            NoteReport(
                index=note.index,
                start_s=round(note.start_s(frame_s), 3),
                end_s=round(note.end_s(frame_s), 3),
                center=note.center,
                name=scales.note_name(note.center),
                target=target,
                target_name=scales.note_name(target),
                cents=round((note.center - target) * 100.0, 1),
                in_key=scale.contains(note.center),
                spread_cents=note.spread_cents,
            )
        )
    mean_cents = mean_abs = share = None
    if report_notes:
        weights = np.array([n.end_s - n.start_s for n in report_notes])
        cents = np.array([n.cents for n in report_notes])
        mean_cents = round(float(np.average(cents, weights=weights)), 1)
        mean_abs = round(float(np.average(np.abs(cents), weights=weights)), 1)
        share = round(inside / total, 3)
    out_of_key = [n.index for n in report_notes if not n.in_key]
    phrases = sorted(_phrases(report_notes), key=lambda p: -p.mean_abs_cents)
    worst = [p for p in phrases if p.mean_abs_cents >= WORTH_MENTIONING_CENTS][:WORST_PHRASES]
    curve = [None if not np.isfinite(v) else round(float(v), 2) for v in midi[::CURVE_STEP]]
    return PitchReport(
        key=key,
        key_source=key_source,
        key_confidence=key_confidence,
        scale=scale.label,
        scale_notes=scale.note_names(),
        duration_s=round(len(midi) * frame_s, 2),
        voiced_share=round(pitch.voiced_share, 3),
        notes=report_notes,
        in_tune_share=share,
        mean_cents=mean_cents,
        mean_abs_cents=mean_abs,
        out_of_key=out_of_key,
        worst_phrases=worst,
        flags=_flags(
            report_notes,
            pitch.voiced_share,
            mean_cents,
            out_of_key,
            scale,
            key_source,
            key_confidence,
            separated,
        ),
        curve_step_s=round(frame_s * CURVE_STEP, 4),
        curve=curve,
    )


def analyze(
    audio: np.ndarray,
    sample_rate: int,
    key: str | None = None,
    scale_kind: str | None = None,
    custom_notes: tuple[str, ...] = (),
    key_source: str = "given",
    separated: bool = False,
    pitch: PitchTrack | None = None,
) -> PitchReport:
    """Track a vocal's pitch and report on it. Without a key, one is guessed from the notes."""
    pitch = pitch or pitch_track.track(audio, sample_rate)
    notes = pitch_track.segment(pitch)
    confidence = None
    if key is None:
        key, confidence, _ = scales.detect_key(
            [note.center for note in notes], [float(note.end - note.start) for note in notes]
        )
        key_source = "detected"
    chosen = scales.custom(list(custom_notes)) if custom_notes else scales.scale(key, scale_kind)
    return build(pitch, notes, chosen, key, key_source, confidence, separated)
