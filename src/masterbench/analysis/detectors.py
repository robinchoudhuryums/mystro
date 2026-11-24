"""Problem detectors: sibilance, resonances, harsh moments, uneven vocals, phase trouble.

Each returns flags; those that describe a moment carry its start and end time so the player
can jump there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from masterbench.analysis.flags import Flag
from masterbench.config import Thresholds

MAX_EVENTS = 6
_FRAME = 1024
_HOP = 256
_SIBILANT_HZ = (5000.0, 10000.0)
_BODY_HZ = (300.0, 3000.0)
_HARSH_HZ = (2000.0, 5000.0)
_RESONANCE_HZ = (150.0, 10000.0)
_ON_PITCH_CENTS = 20.0


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    value: float


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _band_levels(
    mono: np.ndarray, sample_rate: int, bands: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Frame times, each band's level per frame (dB), and the full-band level per frame."""
    freqs, times, spec = signal.stft(mono, sample_rate, nperseg=_FRAME, noverlap=_FRAME - _HOP)
    power = np.abs(spec) ** 2
    levels = {
        name: 10 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum(axis=0) + 1e-20)
        for name, (lo, hi) in bands.items()
    }
    return times, levels, 10 * np.log10(power.sum(axis=0) + 1e-20)


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask)  # pyright: ignore[reportGeneralTypeIssues]
    return np.asarray(labels), int(count)


def _events(
    times: np.ndarray, hot: np.ndarray, values: np.ndarray, min_s: float, join_s: float
) -> list[Event]:
    """Runs of hot frames, bridging gaps shorter than join_s and dropping runs under min_s."""
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    if step <= 0 or not hot.any():
        return []
    joined = ndimage.binary_closing(hot, structure=np.ones(max(1, int(join_s / step)) | 1))
    labels, count = _label(joined)
    found = []
    for index in range(1, count + 1):
        frames = np.flatnonzero(labels == index)
        start, end = float(times[frames[0]]), float(times[frames[-1]] + step)
        if end - start >= min_s:
            found.append(Event(round(start, 2), round(end, 2), float(values[frames].max())))
    return found


def sibilance_events(mono: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Event]:
    """Moments where 5-10 kHz stands close to, or over, the level of the voice's body.

    The value is the sibilant band's peak level minus the body's typical level, in dB.
    """
    times, levels, total = _band_levels(mono, sample_rate, {"s": _SIBILANT_HZ, "body": _BODY_HZ})
    active = total > total.max() - 40
    if not active.any():
        return []
    body = float(
        np.median(
            levels["body"][active & (levels["body"] > levels["s"])]
            if (active & (levels["body"] > levels["s"])).any()
            else levels["body"][active]
        )
    )
    relative = levels["s"] - body
    hot = active & (levels["s"] > levels["body"]) & (relative > limits.sibilance_db)
    return _events(times, hot, relative, min_s=0.03, join_s=0.04)


def sibilance(
    mono: np.ndarray, sample_rate: int, limits: Thresholds, is_vocal: bool = False
) -> list[Flag]:
    events = sorted(sibilance_events(mono, sample_rate, limits), key=lambda e: -e.value)
    if not events:
        return []
    what = "Sibilance" if is_vocal else "Sibilance or other bright bursts"
    caveat = (
        ""
        if is_vocal
        else " On a full mix this can also be hi-hats; a vocal stem tells them apart."
    )
    worst = events[0]
    summary = (
        f"{what}: {len(events)} moment(s) where 5 to 10 kHz stands out, the strongest at "
        f"{clock(worst.start_s)} ({worst.value:+.1f} dB against the body of the sound). "
        f"A de-esser on the vocal is the usual fix.{caveat}"
    )
    flags = [Flag("sibilance", "warn", summary, float(len(events)))]
    for event in sorted(events[:MAX_EVENTS], key=lambda e: e.start_s):
        message = (
            f"Bright burst at {clock(event.start_s)} ({event.value:+.1f} dB against the body)."
        )
        flags.append(
            Flag("sibilance_at", "info", message, round(event.value, 1), event.start_s, event.end_s)
        )
    return flags


def harsh_events(mono: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Event]:
    """Loud moments where 2-5 kHz carries more than half of all the energy."""
    times, levels, total = _band_levels(mono, sample_rate, {"harsh": _HARSH_HZ})
    loud = total > np.percentile(total, 95) - 12
    share = 10 ** ((levels["harsh"] - total) / 10)
    return _events(times, loud & (share > limits.harsh_share), share, min_s=0.1, join_s=0.15)


def harshness(mono: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Flag]:
    events = harsh_events(mono, sample_rate, limits)
    flags = []
    for event in sorted(
        sorted(events, key=lambda e: -e.value)[:MAX_EVENTS], key=lambda e: e.start_s
    ):
        message = (
            f"Harsh moment at {clock(event.start_s)}: 2 to 5 kHz carries {event.value:.0%} of the "
            f"energy for {event.end_s - event.start_s:.1f} s."
        )
        flags.append(
            Flag("harsh_at", "warn", message, round(event.value, 2), event.start_s, event.end_s)
        )
    return flags


def _note_offset_cents(freq_hz: float) -> float:
    midi = 69 + 12 * np.log2(freq_hz / 440.0)
    return float((midi - round(midi)) * 100)


def resonances(mono: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Flag]:
    """Narrow peaks that stand above their surroundings whenever that part of the spectrum sounds.

    A held note looks the same in a spectrum, so peaks that sit on a musical pitch are left
    out. That misses a resonance which happens to land on a note.
    """
    frame = 8192
    freqs, _, spec = signal.stft(mono, sample_rate, nperseg=frame, noverlap=frame // 2)
    power = np.abs(spec) ** 2
    if power.shape[1] < 8 or power.max() <= 0:
        return []
    # A light smoothing, so ripple on top of a broad peak is not read as several peaks.
    mean_power = np.convolve(power.mean(axis=1), np.ones(5) / 5, mode="same")
    mean_db = 10 * np.log10(mean_power + 1e-20)
    inside = np.flatnonzero((freqs >= _RESONANCE_HZ[0]) & (freqs <= _RESONANCE_HZ[1]))
    step = float(freqs[1] - freqs[0])

    def octave(index: int) -> slice:
        return slice(int(index / 2**0.5), int(index * 2**0.5) + 2)

    found = []
    peaks, _ = signal.find_peaks(mean_db[inside], prominence=limits.resonance_db / 2)
    for peak in inside[peaks]:
        around = octave(peak)
        prominence = mean_db[peak] - float(np.median(mean_db[around]))
        if prominence < limits.resonance_db:
            continue
        low, mid, high = mean_db[peak - 1 : peak + 2]
        freq = float(freqs[peak]) + 0.5 * (low - high) / (low - 2 * mid + high) * step
        if abs(_note_offset_cents(freq)) <= _ON_PITCH_CENTS:
            continue
        region = power[around].sum(axis=0)
        sounding = region > region.max() * 1e-3
        peak_db = 10 * np.log10(power[peak - 2 : peak + 3].mean(axis=0) + 1e-20)
        floor_db = 10 * np.log10(np.median(power[around], axis=0) + 1e-20)
        persistence = float(np.mean((peak_db - floor_db)[sounding] >= limits.resonance_db / 2))
        if sounding.sum() < 8 or persistence < 0.6:
            continue
        width_hz = max(int(np.sum(mean_db[around] >= mid - 3)), 1) * step
        found.append((prominence, freq, freq / width_hz, persistence))
    flags, reported = [], []
    for prominence, freq, q, persistence in sorted(found, reverse=True):
        if len(flags) == 3 or any(abs(np.log2(freq / other)) < 1 / 3 for other in reported):
            continue
        reported.append(freq)
        shown = f"{freq / 1000:.2f} kHz" if freq >= 1000 else f"{freq:.0f} Hz"
        message = (
            f"A narrow peak at {shown} stands {prominence:.0f} dB above its surroundings "
            f"{persistence:.0%} of the time that range sounds (Q about {q:.0f}). If it is not "
            "a held note, a narrow cut there may help."
        )
        flags.append(Flag("resonance", "info", message, round(freq, 1)))
    return flags


def phase_events(audio: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Flag]:
    """Stretches where left and right are out of phase enough to cancel in mono."""
    if audio.shape[1] < 2:
        return []
    window, hop = 2 * sample_rate, sample_rate // 2
    starts = np.arange(0, max(len(audio) - window, 0) + 1, hop)
    if len(starts) == 0:
        return []
    left, right = audio[:, 0], audio[:, 1]
    sums = [np.concatenate([[0.0], np.cumsum(x)]) for x in (left * right, left**2, right**2)]
    cross, ll, rr = ((c[starts + window] - c[starts]) for c in sums)
    energy = ll + rr
    correlation = cross / np.sqrt(np.maximum(ll * rr, 1e-20))
    hot = (correlation < limits.phase_correlation) & (energy > energy.max() * 1e-3)
    times = starts / sample_rate
    flags = []
    for event in _events(times, hot, -correlation, min_s=0.5, join_s=0.5)[:MAX_EVENTS]:
        end = event.end_s + window / sample_rate - hop / sample_rate
        message = (
            f"Stereo correlation falls to {-event.value:+.2f} between {clock(event.start_s)} and "
            f"{clock(end)}: this stretch partly cancels in mono."
        )
        flags.append(
            Flag("phase_at", "warn", message, round(-event.value, 2), event.start_s, round(end, 2))
        )
    return flags


def vocal_consistency(
    vocal: np.ndarray, instrumental: np.ndarray, sample_rate: int, limits: Thresholds
) -> list[Flag]:
    """Stretches where the vocal sits well above or below its usual level against the beat."""
    window, hop = sample_rate, sample_rate // 2
    count = (min(len(vocal), len(instrumental)) - window) // hop + 1
    if count < 6:
        return []
    starts = np.arange(count) * hop

    def levels(x: np.ndarray) -> np.ndarray:
        power = np.concatenate([[0.0], np.cumsum(np.mean(x**2, axis=1))])
        return 10 * np.log10((power[starts + window] - power[starts]) / window + 1e-20)

    voice, beat = levels(vocal), levels(instrumental)
    singing = voice > np.percentile(voice, 90) - 20
    if singing.sum() < 6:
        return []
    ratio = voice - beat
    delta = ratio - float(np.median(ratio[singing]))
    times = starts / sample_rate
    flags = []
    for sign, code, word in (
        (1, "vocal_phrase_loud", "louder"),
        (-1, "vocal_phrase_quiet", "quieter"),
    ):
        hot = singing & (sign * delta >= limits.vocal_jump_db)
        for event in _events(times, hot, sign * delta, min_s=0.5, join_s=0.5):
            end = round(event.end_s + (window - hop) / sample_rate, 2)
            message = (
                f"From {clock(event.start_s)} to {clock(end)} the vocal sits up to "
                f"{event.value:.1f} dB {word} against the beat than it usually does. Clip gain or "
                "a rider evens this out."
            )
            flags.append(
                Flag(code, "warn", message, round(sign * event.value, 1), event.start_s, end)
            )
    return sorted(flags, key=lambda f: f.start_s or 0.0)[: 2 * MAX_EVENTS]


def run(audio: np.ndarray, sample_rate: int, limits: Thresholds) -> list[Flag]:
    """The detectors that work on a full mix.

    Sibilance is left to vocal stems: on a full mix every hi-hat reads as an "s".
    """
    mono = audio.mean(axis=1)
    return [
        *harshness(mono, sample_rate, limits),
        *resonances(mono, sample_rate, limits),
        *phase_events(audio, sample_rate, limits),
    ]
