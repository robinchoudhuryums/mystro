"""Pitch tracking for a solo vocal, and segmentation of the pitch curve into notes."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from masterbench.vocal.scales import note_name

FRAME_MS = 5.0
F0_FLOOR_HZ = 65.0
F0_CEIL_HZ = 1000.0
# A note ends when the pitch has left its centre by this much for this long.
SPLIT_SEMITONES = 0.6
SPLIT_S = 0.035
# Frames at a note's edges further than this from its centre are the glide into or out of it.
CORE_SEMITONES = 0.45
MIN_NOTE_S = 0.06
BRIDGE_FRAMES = 2
# Movement slower than this is where the note "is"; faster movement is vibrato.
SUSTAIN_S = 0.35


@dataclass(frozen=True)
class PitchTrack:
    """f0 on a fixed 5 ms grid. Unvoiced frames are 0 Hz."""

    sample_rate: int
    f0_hz: np.ndarray

    @property
    def frame_s(self) -> float:
        return FRAME_MS / 1000.0

    @property
    def times_s(self) -> np.ndarray:
        return np.arange(len(self.f0_hz)) * self.frame_s

    @property
    def voiced(self) -> np.ndarray:
        return self.f0_hz > 0

    @property
    def midi(self) -> np.ndarray:
        """Pitch in semitones (MIDI numbers); NaN where unvoiced."""
        out = np.full(len(self.f0_hz), np.nan)
        voiced = self.voiced
        out[voiced] = 69.0 + 12.0 * np.log2(self.f0_hz[voiced] / 440.0)
        return out

    @property
    def voiced_share(self) -> float:
        return float(self.voiced.mean()) if len(self.f0_hz) else 0.0


@dataclass(frozen=True)
class Note:
    """One sung note: the stable stretch of pitch, without the glide into or out of it."""

    index: int
    start: int
    end: int
    center: float
    spread_cents: float

    def start_s(self, frame_s: float = FRAME_MS / 1000.0) -> float:
        return self.start * frame_s

    def end_s(self, frame_s: float = FRAME_MS / 1000.0) -> float:
        return self.end * frame_s

    @property
    def frames(self) -> slice:
        return slice(self.start, self.end)

    @property
    def name(self) -> str:
        return note_name(self.center)


def world() -> Any:
    """The pyworld module. It warns about pkg_resources on import; that is not ours to fix."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("pyworld")


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def track(audio: np.ndarray, sample_rate: int) -> PitchTrack:
    """f0 of a monophonic source with WORLD's Harvest, refined by StoneMask."""
    pyworld = world()
    mono = np.ascontiguousarray(to_mono(audio), dtype=np.float64)
    if not np.any(np.abs(mono) > 1e-5):
        return PitchTrack(sample_rate, np.zeros(int(len(mono) / sample_rate * 1000 / FRAME_MS) + 1))
    f0, times = pyworld.harvest(
        mono, sample_rate, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ, frame_period=FRAME_MS
    )
    f0 = pyworld.stonemask(mono, f0, times, sample_rate)
    f0[~np.isfinite(f0) | (f0 < F0_FLOOR_HZ / 2)] = 0.0
    return PitchTrack(sample_rate, f0)


def voiced_runs(voiced: np.ndarray, bridge: int = BRIDGE_FRAMES) -> list[tuple[int, int]]:
    """[start, end) of each voiced stretch; gaps of a frame or two are bridged."""
    filled = ndimage.binary_closing(voiced, structure=np.ones(2 * bridge + 1)) | voiced
    edges = np.flatnonzero(np.diff(np.concatenate([[0], filled.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _smooth(values: np.ndarray, frames: int) -> np.ndarray:
    """Moving average that ignores NaN."""
    known = np.isfinite(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(known, values, 0.0), kernel, mode="same")
    count = np.convolve(known.astype(float), kernel, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def sustained(values: np.ndarray, frames: int) -> np.ndarray:
    """The slowly moving part of a curve inside one note: where the note "is".

    Half-way between the curve's upper and lower envelope over about two vibrato cycles.
    A vibrato swings evenly about that line whatever part of its cycle the window starts
    or ends on, which a running mean or median does not manage near the ends of a note.
    The 10th and 90th percentiles stand in for the extremes so one stray frame cannot
    pull the line.
    """
    if len(values) < 5:
        return np.full(len(values), float(np.median(values)))
    width = min(frames | 1, (len(values) - 1) | 1)
    low = ndimage.percentile_filter(values, 10, size=width, mode="reflect")
    high = ndimage.percentile_filter(values, 90, size=width, mode="reflect")
    return ndimage.uniform_filter1d((low + high) / 2.0, size=max(3, width // 2), mode="nearest")


def _split(smooth: np.ndarray, hold: int) -> list[tuple[int, int]]:
    """Cut one voiced run where the pitch leaves the running note centre and stays away."""
    pieces, start, away = [], 0, 0
    for i in range(len(smooth)):
        center = np.nanmedian(smooth[start : max(start + 1, i - away)])
        if abs(smooth[i] - center) > SPLIT_SEMITONES:
            away += 1
            if away >= hold:
                pieces.append((start, i - away + 1))
                start, away = i - away + 1, 0
        else:
            away = 0
    pieces.append((start, len(smooth)))
    return [(a, b) for a, b in pieces if b > a]


def segment(pitch: PitchTrack) -> list[Note]:
    """Notes in a pitch track.

    Each voiced run is cut where the pitch moves to a new level and stays there. A note is
    then the part of a piece that sits near its centre; the slide at either end belongs to
    no note, so it can be left alone or stepped, as the tuning asks.

    A note's centre is the average of its sustained pitch. A plain median would do for a
    steady note, but with vibrato the pitch spends its time at the two extremes, and the
    median then jumps about with a few frames more on one side.
    """
    midi = pitch.midi
    frame_s = pitch.frame_s
    hold = max(2, int(round(SPLIT_S / frame_s)))
    minimum = max(2, int(round(MIN_NOTE_S / frame_s)))
    sustain = max(3, int(round(SUSTAIN_S / frame_s)))
    notes: list[Note] = []
    for run_start, run_end in voiced_runs(pitch.voiced):
        raw = midi[run_start:run_end]
        if np.count_nonzero(np.isfinite(raw)) < minimum:
            continue
        # A frame or two an octave out is the tracker slipping; it must not split a note.
        known = np.where(np.isfinite(raw), raw, np.nanmedian(raw))
        steady = np.where(
            np.isfinite(raw), ndimage.median_filter(known, size=5, mode="nearest"), np.nan
        )
        smooth = _smooth(steady, max(3, int(round(0.05 / frame_s))))
        filled = np.where(np.isfinite(smooth), smooth, np.nanmedian(smooth))
        for a, b in _split(filled, hold):
            inner = filled[a + (b - a) * 15 // 100 : b - (b - a) * 15 // 100]
            center = float(np.median(inner if len(inner) else filled[a:b]))
            near = np.abs(filled[a:b] - center) <= CORE_SEMITONES
            if not near.any():
                continue
            first = a + int(np.argmax(near))
            last = a + len(near) - int(np.argmax(near[::-1]))
            if last - first < minimum:
                continue
            core = raw[first:last]
            core = core[np.isfinite(core)]
            if len(core) < minimum:
                continue
            notes.append(
                Note(
                    index=len(notes),
                    start=run_start + first,
                    end=run_start + last,
                    center=round(float(np.mean(sustained(core, sustain))), 4),
                    spread_cents=round(float(np.std(core) * 100.0), 1),
                )
            )
    return notes
