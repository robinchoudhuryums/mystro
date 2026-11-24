"""Hip-hop low-end tools: is the 808 in tune and in key, does it fight the kick, and how
much of it survives small speakers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from scipy import ndimage, signal

from masterbench.analysis import musical
from masterbench.analysis.detectors import clock
from masterbench.analysis.flags import Flag
from masterbench.config import Thresholds

BASS_HZ = (25.0, 150.0)
CLASH_HZ = (40.0, 120.0)
_CLASH_EDGES = (40.0, 50.0, 63.0, 80.0, 100.0, 120.0)
_BASS_RATE = 5500
_MIN_NOTE_S = 0.25
_GLIDE_S = 0.08
_REJOIN_S = 0.3
_MIN_TONALITY_DB = 32.0
MAX_FLAGS = 8


@dataclass(frozen=True)
class BassNote:
    start_s: float
    end_s: float
    freq_hz: float
    note: str
    cents: float
    in_key: bool | None


@dataclass(frozen=True)
class Clash:
    hits: int
    clashing: int
    share: float
    overlap_db: float
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class LowEnd:
    source: str
    key: str | None
    notes: list[BassNote]
    in_tune_share: float | None
    harmonic_hz: float | None
    harmonics_db: float | None
    clash: Clash | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LowEnd:
        return cls(
            **{
                **data,
                "notes": [BassNote(**note) for note in data["notes"]],
                "clash": Clash(**data["clash"]) if data.get("clash") else None,
            }
        )


def note_name(midi: int) -> str:
    return f"{musical.NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _isolate(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """The band below 150 Hz, decimated: all a bass fundamental needs, and quick to track."""
    lowpass = signal.butter(4, BASS_HZ[1], "lowpass", fs=sample_rate, output="sos")
    factor = max(1, sample_rate // _BASS_RATE)
    return signal.resample_poly(signal.sosfiltfilt(lowpass, mono), 1, factor), sample_rate / factor


def refine_pitch(segment: np.ndarray, rate: float, guess_hz: float) -> tuple[float, float]:
    """The spectral peak near a pitch estimate, or near its octave when that clearly dominates.

    pYIN alone is not enough this low: it lands an octave down on some notes and reads a
    few cents sharp on decaying 808s. Also returns how far the peak stands above the rest of
    the low band (dB), which separates a pitched note from rumble.
    """
    size = 1 << 18
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), size))
    freqs = np.fft.rfftfreq(size, 1 / rate)
    best_level, best_hz = 0.0, guess_hz
    for candidate in (guess_hz, 2 * guess_hz):
        window = np.flatnonzero(
            (freqs >= candidate * 2 ** (-1.5 / 12)) & (freqs <= candidate * 2 ** (1.5 / 12))
        )
        if len(window) < 3:
            continue
        k = window[np.argmax(spectrum[window])]
        if spectrum[k] <= 4 * best_level or k < 1 or k + 1 >= len(spectrum):
            continue
        a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-30)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
        best_level, best_hz = float(spectrum[k]), float(freqs[k] + shift * (freqs[1] - freqs[0]))
    band = spectrum[(freqs >= BASS_HZ[0]) & (freqs <= 2 * BASS_HZ[1])]
    tonality = 20 * np.log10(max(best_level, 1e-30) / max(float(np.median(band)), 1e-30))
    return best_hz, float(tonality)


def bass_notes(bass_mono: np.ndarray, sample_rate: int, key: str | None) -> list[BassNote]:
    """Sustained bass notes with their pitch, cents off equal temperament, and key membership."""
    low, rate = _isolate(bass_mono, sample_rate)
    if len(low) < 2048 or not np.any(np.abs(low) > 1e-5):
        return []
    hop = 256
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, _ = librosa.pyin(
            low, fmin=BASS_HZ[0], fmax=BASS_HZ[1], sr=rate, frame_length=1024
        )
    times = librosa.times_like(f0, sr=rate, hop_length=hop)
    level = librosa.feature.rms(y=low, frame_length=1024, hop_length=hop)[0][: len(f0)]
    sounding = voiced & (level > level.max() * 0.05)
    # Fold octaves, so an octave slip in the tracker does not split a note in two.
    chroma = np.where(
        sounding, np.mod(12 * np.log2(np.where(sounding, f0, 440.0) / 440.0), 12), np.nan
    )

    runs: list[tuple[int, int]] = []
    start = None
    for i in range(len(chroma) + 1):
        here = chroma[i] if i < len(chroma) else np.nan
        if start is not None:
            anchor = float(np.nanmedian(chroma[start:i]))
            jump = np.isnan(here) or min(abs(here - anchor), 12 - abs(here - anchor)) > 0.6
            if jump:
                runs.append((start, i))
                start = None
        if start is None and not np.isnan(here):
            start = i

    scale = musical.scale_pitch_classes(key) if key else None
    notes = []
    for first, last in runs:
        start_s, end_s = float(times[first]), float(times[last - 1]) + hop / rate
        if end_s - start_s < _MIN_NOTE_S:
            continue
        guess = float(np.nanmedian(f0[first:last]))
        body = low[int((start_s + _GLIDE_S) * rate) : int(end_s * rate)]
        if len(body) < 256:
            continue
        freq, tonality = refine_pitch(body, rate, guess)
        if tonality < _MIN_TONALITY_DB or not BASS_HZ[0] <= freq <= BASS_HZ[1] * 1.07:
            continue
        midi = 69 + 12 * np.log2(freq / 440.0)
        nearest = int(round(midi))
        note = BassNote(
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            freq_hz=round(freq, 2),
            note=note_name(nearest),
            cents=round(float((midi - nearest) * 100), 1),
            in_key=(nearest % 12 in scale) if scale is not None else None,
        )
        previous = notes[-1] if notes else None
        same = (
            previous is not None
            and previous.note == note.note
            and abs(previous.cents - note.cents) < 10
            and note.start_s - previous.end_s < _REJOIN_S
        )
        if previous is not None and same:
            # A kick landing on a held note breaks the pitch track; it is still one note.
            longer = max(previous, note, key=lambda n: n.end_s - n.start_s)
            notes[-1] = BassNote(
                previous.start_s,
                note.end_s,
                longer.freq_hz,
                longer.note,
                longer.cents,
                longer.in_key,
            )
        else:
            notes.append(note)
    return notes


def harmonics(
    mono: np.ndarray, sample_rate: int, notes: list[BassNote], above_hz: float
) -> tuple[float | None, float | None]:
    """The strongest harmonic above a small speaker's cutoff, relative to the fundamental.

    Returns (frequency, dB below the fundamental), the median over the longest notes.
    """
    found = []
    for note in sorted(notes, key=lambda n: n.start_s - n.end_s)[:8]:
        piece = mono[int((note.start_s + _GLIDE_S) * sample_rate) : int(note.end_s * sample_rate)]
        if len(piece) < sample_rate // 5:
            continue
        spectrum = np.abs(np.fft.rfft(piece * np.hanning(len(piece))))
        freqs = np.fft.rfftfreq(len(piece), 1 / sample_rate)

        def peak(freq: float, spectrum=spectrum, freqs=freqs) -> float:
            window = (freqs >= freq * 0.97) & (freqs <= freq * 1.03)
            return float(spectrum[window].max()) if window.any() else 0.0

        fundamental = peak(note.freq_hz)
        orders = range(2, int(1000 / note.freq_hz) + 1)
        upper = [
            (peak(k * note.freq_hz), k * note.freq_hz)
            for k in orders
            if k * note.freq_hz >= above_hz
        ]
        if fundamental > 0 and upper:
            level, freq = max(upper)
            found.append((20 * np.log10(max(level, 1e-12) / fundamental), freq))
    if not found:
        return None, None
    found.sort()
    level, freq = found[len(found) // 2]
    return round(float(freq), 1), round(float(level), 1)


def kick_times(drums_mono: np.ndarray, sample_rate: int) -> list[float]:
    """Kick hits: sharp rises in the drum stem's energy below 150 Hz."""
    band = signal.butter(4, [30.0, BASS_HZ[1]], "bandpass", fs=sample_rate, output="sos")
    hop = sample_rate // 200
    low = signal.sosfiltfilt(band, drums_mono)
    frames = len(low) // hop
    energy = (low[: frames * hop] ** 2).reshape(frames, hop).mean(axis=1)
    level = 10 * np.log10(ndimage.uniform_filter1d(energy, 3) + 1e-12)
    rise = np.diff(level, prepend=level.max() - 60)
    floor = level.max() - 20
    peaks, _ = signal.find_peaks(rise, height=3.0, distance=int(0.1 * sample_rate / hop))
    return [
        round(float(p) * hop / sample_rate, 3)
        for p in peaks
        if level[min(p + 3, frames - 1)] > floor
    ]


def kick_clash(
    drums: np.ndarray, bass: np.ndarray, sample_rate: int, limits: Thresholds
) -> Clash | None:
    """How often the bass is as loud as the kick in the kick's own range when the kick lands."""
    drums_mono, bass_mono = drums.mean(axis=1), bass.mean(axis=1)
    hits = kick_times(drums_mono, sample_rate)
    if not hits:
        return None
    window = int(0.12 * sample_rate)
    edges = list(zip(_CLASH_EDGES[:-1], _CLASH_EDGES[1:], strict=True))
    overlaps, shared = [], np.zeros(len(edges))
    for hit in hits:
        start = int(hit * sample_rate)
        kick, under = drums_mono[start : start + window], bass_mono[start : start + window]
        if len(kick) < window:
            continue
        taper = np.hanning(window)
        freqs = np.fft.rfftfreq(window * 4, 1 / sample_rate)
        kick_power = np.abs(np.fft.rfft(kick * taper, window * 4)) ** 2
        bass_power = np.abs(np.fft.rfft(under * taper, window * 4)) ** 2
        whole = (freqs >= CLASH_HZ[0]) & (freqs < CLASH_HZ[1])
        overlaps.append(
            10 * np.log10((bass_power[whole].sum() + 1e-20) / (kick_power[whole].sum() + 1e-20))
        )
        for i, (lo, hi) in enumerate(edges):
            part = (freqs >= lo) & (freqs < hi)
            shared[i] += min(kick_power[part].sum(), bass_power[part].sum())
    if not overlaps:
        return None
    clashing = int(np.sum(np.array(overlaps) > limits.kick_clash_db))
    lo, hi = edges[int(np.argmax(shared))]
    return Clash(
        hits=len(overlaps),
        clashing=clashing,
        share=round(clashing / len(overlaps), 2),
        overlap_db=round(float(np.median(overlaps)), 1),
        low_hz=lo,
        high_hz=hi,
    )


def analyze(
    audio: np.ndarray,
    sample_rate: int,
    key: str | None,
    limits: Thresholds,
    bass: np.ndarray | None = None,
    drums: np.ndarray | None = None,
) -> LowEnd:
    """Low-end findings for a mix. A bass stem makes the pitches and harmonics trustworthy;
    with a drum stem as well, the kick clash is measured too."""
    source = bass if bass is not None else audio
    mono = source.mean(axis=1)
    notes = bass_notes(mono, sample_rate, key)
    lengths = np.array([n.end_s - n.start_s for n in notes])
    good = np.array(
        [abs(n.cents) <= limits.bass_tune_cents and n.in_key is not False for n in notes]
    )
    share = round(float(lengths[good].sum() / lengths.sum()), 2) if len(notes) else None
    harmonic_hz, harmonics_db = harmonics(mono, sample_rate, notes, limits.small_speaker_hz)
    clash = None
    if bass is not None and drums is not None:
        clash = kick_clash(drums, bass, sample_rate, limits)
    return LowEnd(
        source="bass stem" if bass is not None else "mix",
        key=key,
        notes=notes,
        in_tune_share=share,
        harmonic_hz=harmonic_hz,
        harmonics_db=harmonics_db,
        clash=clash,
    )


def flags(found: LowEnd, limits: Thresholds) -> list[Flag]:
    out: list[Flag] = []
    off = [n for n in found.notes if abs(n.cents) > limits.bass_tune_cents or n.in_key is False]
    for note in off[:MAX_FLAGS]:
        if note.in_key is False:
            message = (
                f"The bass note at {clock(note.start_s)} is {note.note} ({note.freq_hz:.1f} Hz), "
                f"which is not in {found.key}."
            )
            code = "bass_out_of_key"
        else:
            message = (
                f"The bass note at {clock(note.start_s)} is {abs(note.cents):.0f} cents "
                f"{'sharp' if note.cents > 0 else 'flat'} of {note.note} ({note.freq_hz:.1f} Hz)."
            )
            code = "bass_detuned"
        out.append(Flag(code, "warn", message, note.cents, note.start_s, note.end_s))
    if found.harmonics_db is not None and found.harmonics_db < limits.bass_harmonics_db:
        message = (
            f"Above {limits.small_speaker_hz:.0f} Hz the bass's strongest harmonic is "
            f"{-found.harmonics_db:.0f} dB under its fundamental, so on phones and laptops the "
            "bass line mostly disappears. Saturation on the 808 adds harmonics that carry it."
        )
        out.append(Flag("bass_small_speakers", "warn", message, found.harmonics_db))
    clash = found.clash
    if clash is not None and clash.share >= 0.5:
        message = (
            f"On {clash.clashing} of {clash.hits} kick hits the bass is within "
            f"{-limits.kick_clash_db:.0f} dB of the kick between 40 and 120 Hz (strongest overlap "
            f"{clash.low_hz:.0f} to {clash.high_hz:.0f} Hz), so the kick loses its punch. "
            "Sidechain the 808 to the kick, carve that range out of one of them, or choose a kick "
            "that sits away from the 808's root."
        )
        out.append(Flag("kick_bass_clash", "warn", message, clash.overlap_db))
    return out
