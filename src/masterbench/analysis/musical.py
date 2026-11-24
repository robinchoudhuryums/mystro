"""Key, tempo and beat grid. Shared by the low-end tools, sections and reference matching."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import librosa
import numpy as np
from scipy import signal

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Krumhansl-Kessler key profiles.
_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
ANALYSIS_RATE = 22050
# The correlation margin over the runner-up that counts as full confidence.
_FULL_CONFIDENCE_MARGIN = 0.2
_GOOD_FIT = 0.7


@dataclass(frozen=True)
class Musical:
    key: str
    key_confidence: float
    key_alternatives: list[str]
    tuning_cents: float
    tempo_bpm: float
    tempo_detected_bpm: float
    beat_offset_s: float

    @property
    def beat_s(self) -> float:
        return 60.0 / self.tempo_bpm


def parse_key(key: str) -> tuple[int, str]:
    """'F minor' -> (5, 'minor')."""
    tonic, _, mode = key.partition(" ")
    if tonic not in NOTE_NAMES or mode not in SCALES:
        raise ValueError(f"not a key: {key!r} (use e.g. 'F minor' or 'C# major')")
    return NOTE_NAMES.index(tonic), mode


def scale_pitch_classes(key: str) -> frozenset[int]:
    tonic, mode = parse_key(key)
    return frozenset((tonic + step) % 12 for step in SCALES[mode])


def to_mono(audio: np.ndarray, sample_rate: int, rate: int = ANALYSIS_RATE) -> np.ndarray:
    """Mono mix at the analysis rate."""
    mono = audio.mean(axis=1)
    if sample_rate == rate:
        return mono.astype(np.float32)
    ratio = np.gcd(rate, sample_rate)
    resampled = signal.resample_poly(mono, rate // ratio, sample_rate // ratio)
    return np.asarray(resampled, dtype=np.float32)


def rank_keys(chroma: np.ndarray) -> list[tuple[float, str]]:
    """Every key with its correlation to the chroma profile, best first."""
    scores = []
    for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            score = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            scores.append((score, f"{NOTE_NAMES[tonic]} {mode}"))
    return sorted(scores, reverse=True)


def fold_tempo(bpm: float, low: float, high: float) -> float:
    """Move a tempo by octaves into [low, high). Beat trackers often lock to half or double."""
    while bpm < low:
        bpm *= 2
    while bpm >= high:
        bpm /= 2
    return bpm


def analyze(
    audio: np.ndarray, sample_rate: int, tempo_range: tuple[float, float] = (80.0, 160.0)
) -> Musical | None:
    """Key, tempo and beat phase of float audio (frames, channels). None for silence."""
    y = to_mono(audio, sample_rate)
    if len(y) < ANALYSIS_RATE or not np.any(np.abs(y) > 1e-4):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuning = float(librosa.estimate_tuning(y=y, sr=ANALYSIS_RATE))
        chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_RATE, tuning=tuning)
        onsets = librosa.onset.onset_strength(y=y, sr=ANALYSIS_RATE)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onsets, sr=ANALYSIS_RATE)
    ranked = rank_keys(chroma.mean(axis=1))
    margin = ranked[0][0] - ranked[1][0]
    # Noise correlates weakly with every profile; a real key fits its profile well.
    fit = float(np.clip(ranked[0][0] / _GOOD_FIT, 0.0, 1.0))

    detected = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=ANALYSIS_RATE)
    if len(beat_times) >= 8:
        # A straight-line fit through the beat times is finer than the tracker's tempo bins.
        period = float(np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0])
        detected = 60.0 / period
    if detected <= 0:
        return None
    bpm = fold_tempo(detected, *tempo_range)
    beat = 60.0 / bpm
    phases = np.mod(beat_times, beat) if len(beat_times) else np.zeros(1)
    # Circular mean, so beats that land just either side of a wrap do not average to mid-beat.
    angle = np.angle(np.mean(np.exp(2j * np.pi * phases / beat)))
    return Musical(
        key=ranked[0][1],
        key_confidence=round(float(np.clip(margin / _FULL_CONFIDENCE_MARGIN, 0, 1) * fit), 2),
        key_alternatives=[name for _, name in ranked[1:3]],
        tuning_cents=round(tuning * 100, 1),
        tempo_bpm=round(bpm, 2),
        tempo_detected_bpm=round(detected, 2),
        beat_offset_s=round(float(np.mod(angle / (2 * np.pi) * beat, beat)), 3),
    )
