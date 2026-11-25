"""Level tools for a vocal against a beat: de-esser, rider, and room made in the beat.

Each one works out a gain over time and applies it; none of them adds delay, so what comes
out lines up with what went in. The gains are returned too, so a run can say what was done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

from masterbench.analysis import loudness

SIBILANCE_HZ = (5000.0, 10000.0)
BODY_HZ = (200.0, 4000.0)
PRESENCE_HZ = (1000.0, 4000.0)
# Windows that hold less singing than this say nothing about the vocal's level.
MIN_SUNG_SHARE = 0.2
SINGING_RANGE_DB = 25.0
RIDER_WINDOW_S = 3.0
RIDER_HOP_S = 0.25
RIDER_SMOOTH_S = 0.75
SECTION_FADE_S = 0.3
MIN_SECTION_SUNG_S = 2.0


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def band_split(audio: np.ndarray, sample_rate: int, band_hz: tuple[float, float]) -> np.ndarray:
    """The part of the audio inside a band, with no phase shift, so audio - band is the rest."""
    high = min(band_hz[1], 0.45 * sample_rate)
    sos = signal.butter(4, [band_hz[0], high], "bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def level_db(samples: np.ndarray, sample_rate: int, window_s: float) -> np.ndarray:
    """Running RMS level in dB, sample by sample."""
    width = max(3, int(window_s * sample_rate))
    power = ndimage.uniform_filter1d(samples**2, width, mode="nearest")
    return 10.0 * np.log10(np.maximum(power, 1e-14))


def singing(vocal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Where the vocal is sounding: within 25 dB of its own loud moments."""
    level = level_db(mono(vocal), sample_rate, 0.05)
    if level.max() < -100:
        return np.zeros(len(level), dtype=bool)
    return level > np.percentile(level, 95) - SINGING_RANGE_DB


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _apply(audio: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return audio * (gain[:, None] if audio.ndim == 2 else gain)


@dataclass(frozen=True)
class Deessed:
    audio: np.ndarray
    events: int
    max_reduction_db: float
    mean_reduction_db: float


def deess(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -12.0,
    max_reduction_db: float = 8.0,
    ratio: float = 4.0,
) -> Deessed:
    """Turn down 5-10 kHz while it stands out against the voice's body.

    The threshold is relative: how far below the vocal's usual body level (200 Hz to 4 kHz)
    the sibilance band may sit before it is reduced. That makes it independent of how hot
    the file is. Only the band is turned down; everything under 5 kHz passes untouched.
    """
    band = band_split(audio, sample_rate, SIBILANCE_HZ)
    voice = mono(audio)
    sung = singing(audio, sample_rate)
    if not sung.any() or max_reduction_db <= 0:
        return Deessed(audio.copy(), 0, 0.0, 0.0)
    body = level_db(mono(band_split(audio, sample_rate, BODY_HZ)), sample_rate, 0.05)
    usual = float(np.median(body[sung]))
    hiss = level_db(mono(band), sample_rate, 0.004)
    over = np.maximum(hiss - (usual + threshold_db), 0.0)
    reduction = np.minimum(over * (1.0 - 1.0 / ratio), max_reduction_db)
    # Reach full reduction just before the burst and let go just after: no late attack.
    hold = max(3, int(0.004 * sample_rate))
    reduction = ndimage.maximum_filter1d(reduction, hold, mode="nearest")
    reduction = ndimage.uniform_filter1d(reduction, hold, mode="nearest")
    reduction[np.abs(voice) < 1e-9] = 0.0
    # The running sums leave rounding dust where there was nothing to reduce; zero is zero.
    reduction[reduction < 1e-6] = 0.0
    # Added as a change to the input, so wherever the gain is 1 the samples are untouched.
    out = audio + _apply(band, 10.0 ** (-reduction / 20.0) - 1.0)
    active = reduction > 1.0
    events = [span for span in _spans(active) if span[1] - span[0] > 0.01 * sample_rate]
    return Deessed(
        audio=out,
        events=len(events),
        max_reduction_db=round(float(reduction.max()), 2),
        mean_reduction_db=round(float(reduction[active].mean()), 2) if active.any() else 0.0,
    )


def ratio_db(vocal: np.ndarray, beat: np.ndarray, sample_rate: int) -> float | None:
    """Vocal loudness minus beat loudness over a stretch (integrated, gated); None if silent."""
    voice = loudness.integrated(_stereo(vocal), sample_rate)
    backing = loudness.integrated(_stereo(beat), sample_rate)
    if voice <= loudness.FLOOR_DB + 1 or backing <= loudness.FLOOR_DB + 1:
        return None
    return voice - backing


def _stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    return audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)


@dataclass(frozen=True)
class SectionRatio:
    """The vocal against the beat over one section, before and after riding."""

    label: str
    start_s: float
    end_s: float
    before_db: float | None
    after_db: float | None
    gain_db: float | None
    limited: bool = False


@dataclass(frozen=True)
class Ridden:
    audio: np.ndarray
    gain_db: np.ndarray
    target_db: float
    fader_db: float
    max_gain_db: float
    sections: list[SectionRatio] = field(default_factory=list)

    @property
    def range_db(self) -> tuple[float, float]:
        """The lowest and highest the rider went, around the fader."""
        moves = self.gain_db - self.fader_db
        return round(float(moves.min()), 2), round(float(moves.max()), 2)


def _window_levels(
    vocal: np.ndarray, beat: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hop times, and the vocal-minus-beat level in each 3 s window (NaN with little singing)."""
    window, hop = int(RIDER_WINDOW_S * sample_rate), int(RIDER_HOP_S * sample_rate)
    sung = singing(vocal, sample_rate).astype(float)
    voice = np.mean(loudness.k_weight(_stereo(vocal), sample_rate) ** 2, axis=1) * sung
    backing = np.mean(loudness.k_weight(_stereo(beat), sample_rate) ** 2, axis=1)
    sums = [np.concatenate([[0.0], np.cumsum(x)]) for x in (voice, backing, sung)]
    centres = np.arange(0, len(sung), hop)
    lo = np.clip(centres - window // 2, 0, len(sung))
    hi = np.clip(centres + window // 2, 0, len(sung))
    voice_sum, backing_sum, sung_count = (s[hi] - s[lo] for s in sums)
    share = sung_count / np.maximum(hi - lo, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        voice_db = 10.0 * np.log10(voice_sum / np.maximum(sung_count, 1.0))
        backing_db = 10.0 * np.log10(backing_sum / np.maximum(hi - lo, 1))
    ratio = np.where(
        (share >= MIN_SUNG_SHARE) & np.isfinite(voice_db), voice_db - backing_db, np.nan
    )
    return centres, ratio, share


def _fill(values: np.ndarray) -> np.ndarray:
    known = np.isfinite(values)
    if not known.any():
        return np.zeros(len(values))
    index = np.arange(len(values))
    return np.interp(index, index[known], values[known])


def ride(
    vocal: np.ndarray,
    beat: np.ndarray,
    sample_rate: int,
    target_db: float,
    sections: list[tuple[str, float, float]] | None = None,
    max_gain_db: float = 6.0,
) -> Ridden:
    """Set the vocal's level against the beat, then ride it so it stays there.

    Three layers, as by hand. The fader: one gain that puts the vocal on target over the
    whole song. A slow rider that follows the vocal against the beat in 3 s windows and
    evens out phrases. And a trim per section so that, measured over each section, the
    vocal is on target: that is where a hook with a louder beat gets its lift. Rider and
    trim together never move more than max_gain_db from the fader, and they move slowly
    enough not to pump.
    """
    frames = len(vocal)
    overall = ratio_db(vocal, beat, sample_rate)
    sung = singing(vocal, sample_rate)
    if overall is None or not sung.any():
        return Ridden(vocal.copy(), np.zeros(frames), target_db, 0.0, max_gain_db)
    fader = target_db - overall
    centres, ratio, _ = _window_levels(vocal, beat, sample_rate)
    slow = np.clip(target_db - fader - _fill(ratio), -max_gain_db, max_gain_db)
    slow = ndimage.gaussian_filter1d(slow, RIDER_SMOOTH_S / RIDER_HOP_S, mode="nearest")
    moves = np.interp(np.arange(frames), centres, slow)

    def measured(a: int, b: int, gain: np.ndarray) -> float | None:
        return ratio_db(_apply(vocal[a:b], 10.0 ** (gain / 20.0)), beat[a:b], sample_rate)

    listed = sections or [("whole song", 0.0, frames / sample_rate)]
    spans = [(int(a * sample_rate), min(frames, int(b * sample_rate))) for _, a, b in listed]
    trims = np.zeros(frames)
    limited = []
    for a, b in spans:
        hit_limit = False
        if sung[a:b].sum() >= MIN_SECTION_SUNG_S * sample_rate:
            trim = 0.0
            for _ in range(3):
                now = measured(a, b, fader + moves[a:b] + trim)
                if now is None:
                    break
                trim += target_db - now
                # The trim may only use what the slow rider has left of the allowed range.
                low = min(0.0, -max_gain_db - float(moves[a:b].min()))
                high = max(0.0, max_gain_db - float(moves[a:b].max()))
                hit_limit = not low <= trim <= high
                trim = float(np.clip(trim, low, high))
            trims[a:b] = trim
        limited.append(hit_limit)
    fade = int(SECTION_FADE_S * sample_rate)
    trims = ndimage.uniform_filter1d(trims, max(3, fade), mode="nearest")
    total = fader + np.clip(moves + trims, -max_gain_db, max_gain_db)

    found = []
    for (label, start_s, end_s), (a, b), hit_limit in zip(listed, spans, limited, strict=True):
        enough = sung[a:b].sum() >= MIN_SECTION_SUNG_S * sample_rate
        before = measured(a, b, np.zeros(b - a)) if enough else None
        after = measured(a, b, total[a:b]) if enough else None
        found.append(
            SectionRatio(
                label,
                round(start_s, 2),
                round(end_s, 2),
                None if before is None else round(before, 2),
                None if after is None else round(after, 2),
                round(float(np.mean(total[a:b])), 2) if enough else None,
                hit_limit,
            )
        )
    return Ridden(
        _apply(vocal, 10.0 ** (total / 20.0)), total, target_db, round(fader, 2), max_gain_db, found
    )


@dataclass(frozen=True)
class Carved:
    audio: np.ndarray
    mean_dip_db: float
    max_dip_db: float
    max_duck_db: float


def _follow(level: np.ndarray, sample_rate: int, attack_s: float, release_s: float) -> np.ndarray:
    """0 to 1: how present the vocal is, rising fast and falling slowly."""
    if level.max() < -100:
        return np.zeros(len(level))
    top = float(np.percentile(level, 95))
    presence = np.clip((level - (top - 18.0)) / 12.0, 0.0, 1.0)
    held = ndimage.maximum_filter1d(presence, max(3, int(release_s * sample_rate)), mode="nearest")
    smooth = ndimage.uniform_filter1d(
        held, max(3, int(release_s * sample_rate / 2)), mode="nearest"
    )
    return ndimage.uniform_filter1d(smooth, max(3, int(attack_s * sample_rate)), mode="nearest")


def carve(
    beat: np.ndarray,
    vocal: np.ndarray,
    sample_rate: int,
    depth_db: float = 3.0,
    duck_db: float = 0.0,
    duck_spans: list[tuple[float, float]] | None = None,
) -> Carved:
    """Make room for the vocal in the beat while the vocal is there.

    A dip of up to depth_db in the beat's 1-4 kHz, following the vocal's own level in that
    range, and optionally a broadband duck of up to duck_db, in duck_spans only if given
    (the hooks). With no vocal the beat is returned as it was.
    """
    band = band_split(beat, sample_rate, PRESENCE_HZ)
    voice_band = level_db(mono(band_split(vocal, sample_rate, PRESENCE_HZ)), sample_rate, 0.02)
    dip = depth_db * _follow(voice_band, sample_rate, 0.01, 0.15)
    dip[dip < 1e-6] = 0.0
    out = beat + _apply(band, 10.0 ** (-dip / 20.0) - 1.0)
    duck = np.zeros(len(beat))
    if duck_db > 0:
        duck = duck_db * _follow(level_db(mono(vocal), sample_rate, 0.02), sample_rate, 0.02, 0.25)
        if duck_spans is not None:
            inside = np.zeros(len(beat))
            for start_s, end_s in duck_spans:
                inside[int(start_s * sample_rate) : int(end_s * sample_rate)] = 1.0
            duck *= ndimage.uniform_filter1d(inside, max(3, int(0.1 * sample_rate)), mode="nearest")
        out = _apply(out, 10.0 ** (-duck / 20.0))
    sung = singing(vocal, sample_rate)
    return Carved(
        audio=out,
        mean_dip_db=round(float(dip[sung].mean()), 2) if sung.any() else 0.0,
        max_dip_db=round(float(dip.max()), 2),
        max_duck_db=round(float(duck.max()), 2),
    )
