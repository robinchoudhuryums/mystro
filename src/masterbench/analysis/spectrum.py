"""Frequency balance: smoothed spectrum and named-band levels, comparable at equal loudness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from masterbench.analysis.loudness import FLOOR_DB
from masterbench.config import BANDS, REGIONS

SMOOTHING_FRACTION = 6
_POINTS_PER_OCTAVE = 12
_SEGMENT = 32768


@dataclass(frozen=True)
class CrossSpectra:
    """Welch auto- and cross-spectra of the left and right channels."""

    freqs: np.ndarray
    left: np.ndarray
    right: np.ndarray
    cross: np.ndarray

    @property
    def power(self) -> np.ndarray:
        return (self.left + self.right) / 2

    def band(self, low_hz: float, high_hz: float) -> np.ndarray:
        return (self.freqs >= low_hz) & (self.freqs < high_hz)


@dataclass(frozen=True)
class Spectrum:
    """1/6-octave smoothed level per frequency, in dB relative to integrated loudness."""

    freqs_hz: list[float]
    level_db: list[float]


def cross_spectra(audio: np.ndarray, sample_rate: int) -> CrossSpectra:
    left, right = audio[:, 0], audio[:, -1]
    segment = min(_SEGMENT, len(audio))
    freqs, pxx = signal.welch(left, sample_rate, nperseg=segment)
    if audio.shape[1] == 1:
        return CrossSpectra(freqs, pxx, pxx, pxx.astype(complex))
    _, pyy = signal.welch(right, sample_rate, nperseg=segment)
    _, pxy = signal.csd(left, right, sample_rate, nperseg=segment)
    return CrossSpectra(freqs, pxx, pyy, pxy)


def _db(power: float) -> float:
    return max(FLOOR_DB, float(10 * np.log10(power))) if power > 0 else FLOOR_DB


def smoothed(spectra: CrossSpectra, lufs_i: float) -> Spectrum:
    """Average power in a 1/6-octave window around log-spaced centres from 20 Hz to 20 kHz."""
    nyquist = spectra.freqs[-1]
    count = int(np.floor(np.log2(min(20000.0, nyquist) / 20.0) * _POINTS_PER_OCTAVE)) + 1
    centres = 20.0 * 2 ** (np.arange(count) / _POINTS_PER_OCTAVE)
    half = 2 ** (1 / (2 * SMOOTHING_FRACTION))
    power = spectra.power
    levels = []
    for centre in centres:
        window = spectra.band(centre / half, centre * half)
        value = power[window].mean() if window.any() else np.interp(centre, spectra.freqs, power)
        levels.append(round(_db(float(value)) - lufs_i, 2))
    return Spectrum([round(float(c), 2) for c in centres], levels)


def band_levels(spectra: CrossSpectra, lufs_i: float) -> dict[str, float]:
    """Energy in each named band, in dB relative to integrated loudness."""
    step = float(spectra.freqs[1] - spectra.freqs[0])
    power = spectra.power
    return {
        name: round(_db(float(power[spectra.band(low, high)].sum() * step)) - lufs_i, 2)
        for name, (low, high) in BANDS.items()
    }


def region_levels(spectra: CrossSpectra, lufs_i: float) -> dict[str, float]:
    """Energy in the mud and harshness regions, in dB relative to integrated loudness."""
    step = float(spectra.freqs[1] - spectra.freqs[0])
    power = spectra.power
    return {
        name: round(_db(float(power[spectra.band(low, high)].sum() * step)) - lufs_i, 2)
        for name, (low, high) in REGIONS.items()
    }


def band_deltas(bands: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """How far each band sits from the reference once both play at the same loudness."""
    return {name: round(bands[name] - reference[name], 2) for name in BANDS}
