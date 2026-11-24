"""Lookahead true-peak limiter.

Pedalboard's Limiter neither holds a ceiling nor sees inter-sample peaks (see NOTES.md),
so the final stage of every master goes through this instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from masterbench.analysis import loudness

OVERSAMPLE = 4
_SAFETY_DB = 0.05


@dataclass(frozen=True)
class LimitResult:
    audio: np.ndarray
    max_reduction_db: float
    trim_db: float
    true_peak_dbtp: float


def _required_gain(audio: np.ndarray, ceiling: float) -> np.ndarray:
    """Per-sample gain that would bring each sample's oversampled peak to the ceiling."""
    up = np.abs(signal.resample_poly(audio, OVERSAMPLE, 1, axis=0))
    peaks = up.reshape(len(audio), OVERSAMPLE, -1).max(axis=(1, 2))
    return np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))


def true_peak_limit(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_db_per_s: float = 40.0,
) -> LimitResult:
    """Limit float audio shaped (frames, channels) so its true peak stays under the ceiling.

    Gain is linked across channels. The gain curve is the required reduction, held over the
    lookahead window, released at a fixed dB/s rate, then smoothed so it arrives before the
    peak does. A final static trim absorbs whatever inter-sample overshoot the gain changes
    themselves introduce, so the ceiling is a guarantee rather than a target.
    """
    ceiling = 10 ** ((ceiling_dbtp - _SAFETY_DB) / 20)
    required = _required_gain(audio, ceiling)
    if required.min() >= 1.0:
        peak = loudness.to_db(loudness.true_peak(audio))
        return LimitResult(audio, 0.0, 0.0, peak)

    lookahead = max(1, int(lookahead_ms * sample_rate / 1000))
    held_db = 20 * np.log10(ndimage.minimum_filter1d(required, 2 * lookahead + 1, mode="nearest"))
    # Release: gain may recover no faster than the slope. env[n] = min_k(held[k] + slope*(n-k)).
    ramp = (release_db_per_s / sample_rate) * np.arange(len(held_db))
    envelope = 10 ** ((ramp + np.minimum.accumulate(held_db - ramp)) / 20)

    window = np.hanning(lookahead + 3)[1:-1]
    padded = np.pad(envelope, len(window) // 2, mode="edge")
    gain = np.convolve(padded, window / window.sum(), mode="valid")[: len(envelope)]

    limited = audio * gain[:, None]
    peak_db = loudness.to_db(loudness.true_peak(limited))
    trim_db = min(0.0, ceiling_dbtp - peak_db)
    if trim_db < 0:
        limited = limited * 10 ** (trim_db / 20)
    return LimitResult(
        audio=limited,
        max_reduction_db=float(-20 * np.log10(gain.min())),
        trim_db=float(trim_db),
        true_peak_dbtp=peak_db + trim_db,
    )


@dataclass(frozen=True)
class FinalizeResult:
    audio: np.ndarray
    gain_db: float
    max_reduction_db: float
    lufs_i: float
    true_peak_dbtp: float
    iterations: int


def finalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float | None,
    ceiling_dbtp: float,
    max_reduction_db: float = 6.0,
    tolerance_lu: float = 0.05,
    max_iterations: int = 6,
) -> FinalizeResult:
    """Bring audio to a loudness target under a true-peak ceiling.

    Limiting lowers loudness a little, so the make-up gain is corrected and the limiter re-run
    until the result lands on target. A target that would need more than max_reduction_db of
    limiting is not chased: the gain is backed off to that amount and the result comes out
    quieter than asked. With no target, the audio is only limited.
    """

    def run(gain_db: float) -> tuple[LimitResult, float]:
        result = true_peak_limit(audio * 10 ** (gain_db / 20), sample_rate, ceiling_dbtp)
        return result, loudness.integrated(result.audio, sample_rate)

    gain_db = 0.0 if target_lufs is None else target_lufs - loudness.integrated(audio, sample_rate)
    result, achieved = run(gain_db)
    iterations = 1
    while (
        target_lufs is not None
        and abs(target_lufs - achieved) > tolerance_lu
        and result.max_reduction_db <= max_reduction_db
        and iterations < max_iterations
    ):
        gain_db += target_lufs - achieved
        result, achieved = run(gain_db)
        iterations += 1
    if target_lufs is not None and result.max_reduction_db > max_reduction_db:
        gain_db -= result.max_reduction_db - max_reduction_db
        result, achieved = run(gain_db)
        iterations += 1
    return FinalizeResult(
        result.audio, gain_db, result.max_reduction_db, achieved, result.true_peak_dbtp, iterations
    )
