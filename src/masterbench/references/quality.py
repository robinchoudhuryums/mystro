"""Is a reference track fit to be measured against? Lossy rips, clipping, mono, snippets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from masterbench import validate
from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import Loudness
from masterbench.analysis.stereo import Stereo
from masterbench.audio_io import AudioInfo
from masterbench.config import Thresholds

BADGES = ("good", "caution", "poor")
_BAND_HZ = 500.0
_DROP_DB = 30.0


@dataclass(frozen=True)
class Quality:
    badge: str
    issues: list[Flag]
    cutoff_hz: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quality:
        return cls(data["badge"], [Flag(**f) for f in data["issues"]], data.get("cutoff_hz"))

    @property
    def usable_for_curves(self) -> bool:
        return self.badge != "poor"


def spectral_cutoff(audio: np.ndarray, sample_rate: int) -> float | None:
    """Frequency above which the spectrum falls off a cliff and stays there, if it does.

    Lossy encoders remove everything above a cutoff (about 16 kHz at 128 kbps). The level in
    500 Hz steps is compared with the 10-14 kHz region; a drop of 30 dB that lasts to the top
    of the spectrum is a cutoff.
    """
    if sample_rate < 40000 or len(audio) < 8192:
        return None
    freqs, power = signal.welch(audio.mean(axis=1), sample_rate, nperseg=8192)
    edges = np.arange(10000.0, sample_rate / 2 - _BAND_HZ, _BAND_HZ)
    levels = np.array(
        [
            10 * np.log10(power[(freqs >= lo) & (freqs < lo + _BAND_HZ)].mean() + 1e-30)
            for lo in edges
        ]
    )
    reference = float(np.median(levels[edges < 14000]))
    below = levels < reference - _DROP_DB
    for i in np.flatnonzero(edges >= 14000):
        if below[i:].all():
            return float(edges[i])
    return None


def assess(
    audio: np.ndarray, info: AudioInfo, loud: Loudness, image: Stereo, limits: Thresholds
) -> Quality:
    issues: list[Flag] = []
    cutoff = spectral_cutoff(audio, info.sample_rate)
    if cutoff is not None and cutoff < limits.lossy_cutoff_hz:
        message = (
            f"Nothing above {cutoff / 1000:.1f} kHz: this looks like a lossy rip (MP3, AAC or a "
            "stream capture). Its top end and stereo detail are not what the master had."
        )
        issues.append(Flag("lossy_rip", "problem", message, cutoff))
    elif info.lossy:
        message = (
            f"{info.format} is a lossy format; a WAV or FLAC of the track is a truer reference."
        )
        issues.append(Flag("lossy_format", "warn", message))

    runs = validate.clipped_runs(audio, limits.clip_run_samples)
    if runs > 10:
        message = f"Clipped: {runs} runs of consecutive full-scale samples."
        issues.append(Flag("clipped", "problem", message, float(runs)))
    elif loud.crest_db < limits.reference_low_crest_db:
        message = (
            f"Very heavily limited (crest factor {loud.crest_db:.1f} dB). Matching it will push "
            "your master toward the same density."
        )
        issues.append(Flag("brickwalled", "warn", message, loud.crest_db))

    if info.channels == 1 or image.side_mid_db < -40:
        issues.append(Flag("mono", "problem", "Mono: there is no stereo image to compare with."))
    elif image.correlation > 0.999:
        message = "Left and right are the same signal at different levels (panned mono)."
        issues.append(Flag("fake_stereo", "warn", message, image.correlation))

    if info.duration_s < limits.reference_min_s:
        message = f"Only {info.duration_s:.0f} s long: a snippet says little about a whole track."
        issues.append(Flag("short", "warn", message, info.duration_s))

    worst = {flag.severity for flag in issues}
    badge = "poor" if "problem" in worst else "caution" if "warn" in worst else "good"
    return Quality(badge, issues, cutoff)
