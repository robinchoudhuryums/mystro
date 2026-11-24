"""EBU R 128 loudness, true peak and dynamics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy import signal

FLOOR_DB = -120.0
MOMENTARY_S = 0.4
SHORT_TERM_S = 3.0
HOP_S = 0.1


@dataclass(frozen=True)
class Loudness:
    lufs_i: float
    lufs_s_max: float
    lufs_m_max: float
    lra: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    plr: float


def to_db(linear: float, power: bool = False) -> float:
    if linear <= 0:
        return FLOOR_DB
    return max(FLOOR_DB, float((10.0 if power else 20.0) * np.log10(linear)))


def k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """BS.1770 K-weighting: a high shelf followed by a high-pass."""
    shelf = pyln.IIRfilter(4.0, 1 / np.sqrt(2), 1500.0, sample_rate, "high_shelf")
    highpass = pyln.IIRfilter(0.0, 0.5, 38.0, sample_rate, "high_pass")
    weighted = signal.lfilter(shelf.b, shelf.a, audio, axis=0)
    return np.asarray(signal.lfilter(highpass.b, highpass.a, weighted, axis=0))


def windowed_loudness(
    weighted: np.ndarray, sample_rate: int, window_s: float, hop_s: float = HOP_S
) -> np.ndarray:
    """Ungated loudness (LUFS) of sliding windows over K-weighted audio."""
    window, hop = int(round(window_s * sample_rate)), int(round(hop_s * sample_rate))
    if len(weighted) < window:
        return np.empty(0)
    energy = np.concatenate([[0.0], np.cumsum(np.sum(weighted**2, axis=1))])
    starts = np.arange(0, len(weighted) - window + 1, hop)
    mean_square = (energy[starts + window] - energy[starts]) / window
    return -0.691 + 10 * np.log10(np.maximum(mean_square, 1e-12))


def short_term_curve(audio: np.ndarray, sample_rate: int, step_s: float = 1.0) -> list[float]:
    """Short-term loudness sampled once per step, for loudness-over-time plots."""
    values = windowed_loudness(k_weight(audio, sample_rate), sample_rate, SHORT_TERM_S, step_s)
    return [round(max(FLOOR_DB, float(v)), 2) for v in values]


def integrated(audio: np.ndarray, sample_rate: int) -> float:
    if len(audio) < int(MOMENTARY_S * sample_rate):
        return FLOOR_DB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    return max(FLOOR_DB, value) if np.isfinite(value) else FLOOR_DB


def loudness_range(short_term: np.ndarray) -> float:
    """EBU Tech 3342: 10th-95th percentile spread of gated short-term loudness."""
    gated = short_term[short_term > -70.0]
    if len(gated) == 0:
        return 0.0
    mean = 10 * np.log10(np.mean(10 ** (gated / 10)))
    gated = gated[gated > mean - 20.0]
    if len(gated) < 2:
        return 0.0
    low, high = np.percentile(gated, [10, 95])
    return float(high - low)


def true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    """Linear true peak, estimated on an oversampled copy."""
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(signal.resample_poly(audio, oversample, 1, axis=0))))


def measure(audio: np.ndarray, sample_rate: int) -> Loudness:
    """All loudness figures for float audio shaped (frames, channels)."""
    weighted = k_weight(audio, sample_rate)
    short_term = windowed_loudness(weighted, sample_rate, SHORT_TERM_S)
    momentary = windowed_loudness(weighted, sample_rate, MOMENTARY_S)
    lufs_i = integrated(audio, sample_rate)
    sample_peak = to_db(float(np.max(np.abs(audio))) if len(audio) else 0.0)
    rms = to_db(float(np.mean(audio**2)) if len(audio) else 0.0, power=True)
    true_peak_dbtp = max(to_db(true_peak(audio)), sample_peak)
    return Loudness(
        lufs_i=lufs_i,
        lufs_s_max=max(FLOOR_DB, float(short_term.max())) if len(short_term) else lufs_i,
        lufs_m_max=max(FLOOR_DB, float(momentary.max())) if len(momentary) else lufs_i,
        lra=loudness_range(short_term),
        true_peak_dbtp=true_peak_dbtp,
        sample_peak_dbfs=sample_peak,
        rms_dbfs=rms,
        crest_db=sample_peak - rms,
        plr=true_peak_dbtp - lufs_i,
    )
