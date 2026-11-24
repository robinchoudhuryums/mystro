"""Stereo image: correlation, width, and whether the low end is mono."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from masterbench.analysis.loudness import FLOOR_DB
from masterbench.analysis.spectrum import CrossSpectra
from masterbench.config import BANDS

LOW_END_HZ = 120.0


@dataclass(frozen=True)
class Stereo:
    correlation: float
    side_mid_db: float
    low_mono_corr: float
    low_side_mid_db: float
    band_correlation: dict[str, float]


def _correlation(spectra: CrossSpectra, window: np.ndarray) -> float:
    """Zero-lag correlation of the two channels within a frequency range."""
    energy = float(np.sqrt(spectra.left[window].sum() * spectra.right[window].sum()))
    if energy <= 0:
        return 1.0
    return round(float(np.clip(np.real(spectra.cross[window]).sum() / energy, -1.0, 1.0)), 3)


def _side_mid_db(spectra: CrossSpectra, window: np.ndarray) -> float:
    total = spectra.left[window].sum() + spectra.right[window].sum()
    cross = 2 * np.real(spectra.cross[window]).sum()
    mid, side = total + cross, total - cross
    if side <= mid * 1e-12:
        return FLOOR_DB
    if mid <= side * 1e-12:
        return -FLOOR_DB
    return round(float(10 * np.log10(side / mid)), 2)


def analyze(spectra: CrossSpectra) -> Stereo:
    everything = spectra.freqs > 0
    low = spectra.band(0.0, LOW_END_HZ) & everything
    return Stereo(
        correlation=_correlation(spectra, everything),
        side_mid_db=_side_mid_db(spectra, everything),
        low_mono_corr=_correlation(spectra, low),
        low_side_mid_db=_side_mid_db(spectra, low),
        band_correlation={
            name: _correlation(spectra, spectra.band(lo, hi)) for name, (lo, hi) in BANDS.items()
        },
    )
