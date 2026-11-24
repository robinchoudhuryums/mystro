"""Scales a vocal is tuned to: a set of pitch classes, and the nearest allowed note."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from masterbench.analysis.musical import NOTE_NAMES, rank_keys

SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "major pentatonic": (0, 2, 4, 7, 9),
    "minor pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "chromatic": tuple(range(12)),
}
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


class ScaleError(ValueError):
    """A key, scale or note name that cannot be read."""


def pitch_class(name: str) -> int:
    """'F', 'G#' or 'Ab' -> 0..11."""
    cleaned = name.strip()
    cleaned = cleaned[:1].upper() + cleaned[1:]
    cleaned = _FLATS.get(cleaned, cleaned)
    if cleaned not in NOTE_NAMES:
        raise ScaleError(f"not a note name: {name!r} (use e.g. F, G# or Ab)")
    return NOTE_NAMES.index(cleaned)


def note_name(midi: float) -> str:
    """60 -> 'C4'."""
    nearest = int(round(midi))
    return f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"


def parse_note(name: str) -> int:
    """'G#3' -> 56."""
    text = name.strip()
    split = len(text.rstrip("-0123456789"))
    if split == len(text):
        raise ScaleError(f"not a note: {name!r} (use e.g. G#3)")
    return pitch_class(text[:split]) + 12 * (int(text[split:]) + 1)


@dataclass(frozen=True)
class Scale:
    """The notes a vocal may land on."""

    label: str
    pitch_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.pitch_classes:
            raise ScaleError("a scale needs at least one note")

    def contains(self, midi: float) -> bool:
        return int(round(midi)) % 12 in self.pitch_classes

    def nearest(self, midi: np.ndarray | list[float] | float) -> np.ndarray:
        """The closest scale note to each pitch. Exactly between two, the lower one wins."""
        pitches = np.atleast_1d(np.asarray(midi, dtype=float))
        base = np.floor(pitches / 12.0) * 12.0
        classes = np.array(sorted(self.pitch_classes), dtype=float)
        # One octave below to one above covers every nearest neighbour.
        ladder = np.concatenate([classes - 12.0, classes, classes + 12.0])
        candidates = base[:, None] + ladder[None, :]
        best = np.argmin(np.abs(candidates - pitches[:, None]), axis=1)
        return candidates[np.arange(len(pitches)), best]

    def note_names(self) -> list[str]:
        return [NOTE_NAMES[pc] for pc in self.pitch_classes]


def scale(key: str, kind: str | None = None) -> Scale:
    """A scale from a key like 'F minor', optionally with another scale kind on its tonic."""
    tonic_name, _, mode = key.strip().partition(" ")
    tonic = pitch_class(tonic_name)
    chosen = (kind or mode or "major").strip().lower().replace("_", " ")
    if chosen not in SCALES:
        raise ScaleError(f"unknown scale '{chosen}'; have {', '.join(SCALES)}")
    if chosen == "chromatic":
        return Scale("chromatic", SCALES["chromatic"])
    steps = tuple(sorted((tonic + step) % 12 for step in SCALES[chosen]))
    return Scale(f"{NOTE_NAMES[tonic]} {chosen}", steps)


def custom(notes: list[str] | str) -> Scale:
    """A scale from note names: 'F G# A# C' or ['F', 'Ab', 'Bb', 'C']."""
    names = notes.replace(",", " ").split() if isinstance(notes, str) else list(notes)
    classes = tuple(sorted({pitch_class(name) for name in names}))
    if not classes:
        raise ScaleError("a custom scale needs at least one note")
    return Scale("custom: " + " ".join(NOTE_NAMES[pc] for pc in classes), classes)


def detect_key(centers: list[float], durations: list[float]) -> tuple[str, float, list[str]]:
    """Key from sung notes: the key profile that best fits the time spent on each pitch class.

    Returns the key, a confidence between 0 and 1, and the next two candidates. A melody
    uses few notes, so relative major and minor are easily confused; the track's own key,
    when known, is the better source.
    """
    chroma = np.zeros(12)
    for center, duration in zip(centers, durations, strict=True):
        chroma[int(round(center)) % 12] += duration
    if not np.any(chroma) or np.count_nonzero(chroma) < 3:
        return "C major", 0.0, []
    ranked = rank_keys(chroma)
    margin = ranked[0][0] - ranked[1][0]
    confidence = float(np.clip(margin / 0.2, 0.0, 1.0) * np.clip(ranked[0][0] / 0.7, 0.0, 1.0))
    return ranked[0][1], round(confidence, 2), [name for _, name in ranked[1:3]]
