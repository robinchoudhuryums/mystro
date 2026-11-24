"""Curve-match mastering: a matching EQ toward a genre target curve.

Deterministic and static: one linear-phase FIR for the whole track, so the same curve can be
applied to an instrumental or a cappella and they still sum to the master.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from masterbench.analysis import loudness, spectrum

MAX_DB = 4.0
TAPS = 16383
_SMOOTH_POINTS = 5
_PASSES = 4
_CLOSE_ENOUGH_DB = 0.25


@dataclass(frozen=True)
class MatchEQ:
    freqs_hz: list[float]
    gain_db: list[float]

    @property
    def max_abs_db(self) -> float:
        return float(np.max(np.abs(self.gain_db)))


def _smooth(values: np.ndarray) -> np.ndarray:
    """About a third of an octave of smoothing, so the EQ follows balance, not single notes."""
    padded = np.pad(values, _SMOOTH_POINTS // 2, mode="edge")
    return np.convolve(padded, np.ones(_SMOOTH_POINTS) / _SMOOTH_POINTS, mode="valid")


def fir(eq: MatchEQ, sample_rate: int, taps: int = TAPS) -> np.ndarray:
    """Linear-phase FIR with the EQ's magnitude response."""
    nyquist = sample_rate / 2
    grid = np.linspace(0.0, nyquist, 4097)
    inside = [(f, g) for f, g in zip(eq.freqs_hz, eq.gain_db, strict=True) if f < nyquist]
    freqs, gains = np.array([f for f, _ in inside]), np.array([g for _, g in inside])
    # Interpolate on a log frequency axis; hold the end values below 20 Hz and above the top.
    shaped = np.interp(np.log2(np.maximum(grid, 1.0)), np.log2(freqs), gains)
    return np.asarray(signal.firwin2(taps, grid, 10 ** (shaped / 20), fs=sample_rate))


def apply(audio: np.ndarray, sample_rate: int, eq: MatchEQ) -> np.ndarray:
    """Filter audio shaped (frames, channels). Length and timing are unchanged."""
    kernel = fir(eq, sample_rate)
    delay = (len(kernel) - 1) // 2
    out = signal.fftconvolve(audio, kernel[:, None], mode="full")
    return out[delay : delay + len(audio)]


def _relative_spectrum(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    spectra = spectrum.cross_spectra(audio, sample_rate)
    return np.array(spectrum.smoothed(spectra, loudness.integrated(audio, sample_rate)).level_db)


def match(
    audio: np.ndarray,
    sample_rate: int,
    target_freqs_hz: list[float],
    target_db: list[float],
    max_db: float = MAX_DB,
) -> tuple[np.ndarray, MatchEQ]:
    """Equalize audio toward a target spectrum (dB relative to integrated loudness).

    The EQ is limited to ±max_db. Because EQ changes loudness, and levels are relative to
    loudness, the gain is refined over a few passes until the result sits on the target.
    """
    current = _relative_spectrum(audio, sample_rate)
    points = min(len(current), len(target_db))
    freqs, target = list(target_freqs_hz[:points]), np.array(target_db[:points])
    gain = np.zeros(points)
    out = audio
    for _ in range(_PASSES):
        residual = _smooth(target - current[:points])
        if np.max(np.abs(residual)) < _CLOSE_ENOUGH_DB:
            break
        updated = np.clip(gain + residual, -max_db, max_db)
        if np.allclose(updated, gain, atol=0.02):
            break
        gain = updated
        out = apply(audio, sample_rate, MatchEQ(freqs, [float(g) for g in gain]))
        current = _relative_spectrum(out, sample_rate)
    return out, MatchEQ(freqs, [round(float(g), 3) for g in gain])
