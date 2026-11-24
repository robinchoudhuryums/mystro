"""Deterministic synthetic test audio with known ground truth.

Everything here is generated from seeds, so tests can assert against exact answers
(key, tempo, note pitches, planted problems) and no real music ever enters the repo.
Depends on numpy/scipy/soundfile only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def midi_to_hz(midi: float, cents: float = 0.0) -> float:
    return float(440.0 * 2.0 ** ((midi - 69 + cents / 100.0) / 12.0))


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class Note:
    start_s: float
    duration_s: float
    midi: int
    cents: float = 0.0

    @property
    def freq_hz(self) -> float:
        return midi_to_hz(self.midi, self.cents)

    @property
    def name(self) -> str:
        return midi_name(self.midi)


@dataclass(frozen=True)
class LoopTruth:
    key: str
    tempo_bpm: float
    bars: int
    notes_808: list[Note]
    kick_times_s: list[float]
    snare_times_s: list[float]


@dataclass(frozen=True)
class VocalTruth:
    key: str
    vibrato_hz: float
    vibrato_cents: float
    notes: list[Note]
    off_cents: float = field(default=30.0)


def _normalize_peak(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    return x * (db_to_lin(peak_dbfs) / np.max(np.abs(x)))


def _stereo(mid: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
    side = np.zeros_like(mid) if side is None else side
    return np.column_stack([mid + side, mid - side])


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum / np.sqrt(freqs), n)


def pink_noise(
    duration_s: float = 10.0, sr: int = SR, rms_dbfs: float = -20.0, seed: int = 0
) -> np.ndarray:
    """Stereo pink noise (-3 dB/octave) at a set RMS, with a correlated mid and a quiet side."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    audio = _stereo(_pink(n, rng), 0.3 * _pink(n, rng))
    return audio * (db_to_lin(rms_dbfs) / np.sqrt(np.mean(audio**2)))


def sine_sweep(
    duration_s: float = 10.0,
    sr: int = SR,
    f_start: float = 20.0,
    f_end: float = 20000.0,
    peak_dbfs: float = -6.0,
) -> np.ndarray:
    """Stereo logarithmic sine sweep with short fades."""
    t = np.arange(int(duration_s * sr)) / sr
    sweep = signal.chirp(t, f_start, duration_s, f_end, method="logarithmic", phi=-90)
    fade = int(0.01 * sr)
    sweep[:fade] *= np.linspace(0.0, 1.0, fade)
    sweep[-fade:] *= np.linspace(1.0, 0.0, fade)
    return _stereo(sweep * db_to_lin(peak_dbfs))


def _butter(x: np.ndarray, order: int, freq: float | list[float], kind: str, sr: int) -> np.ndarray:
    sos = signal.butter(order, freq, kind, fs=sr, output="sos")
    return np.asarray(signal.sosfilt(sos, x))


def _place(dest: np.ndarray, start_s: float, sound: np.ndarray, sr: int) -> None:
    start = int(round(start_s * sr))
    end = min(start + len(sound), len(dest))
    if start < len(dest):
        dest[start:end] += sound[: end - start]


def _kick(sr: int) -> np.ndarray:
    t = np.arange(int(0.25 * sr)) / sr
    freq = 48.0 + 110.0 * np.exp(-t / 0.018)
    body = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / 0.07)
    click = np.exp(-t / 0.002) * 0.4
    return body + click


def _snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.2 * sr)) / sr
    noise = _butter(rng.standard_normal(len(t)), 2, [1200, 8000], "bandpass", sr)
    noise *= np.exp(-t / 0.05)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t / 0.03)
    return 0.8 * noise + 0.5 * tone


def _hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.05 * sr)) / sr
    return _butter(rng.standard_normal(len(t)), 4, 7000, "highpass", sr) * np.exp(-t / 0.012)


def _bass_808(note: Note, sr: int) -> np.ndarray:
    """Sustained sine with a short downward glide into pitch and a little harmonic content."""
    t = np.arange(int(note.duration_s * sr)) / sr
    freq = note.freq_hz * (1.0 + 0.5 * np.exp(-t / 0.008))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / (note.duration_s * 0.9))
    release = int(0.02 * sr)
    env[-release:] *= np.linspace(1.0, 0.0, release)
    return np.tanh(1.6 * np.sin(phase)) * env


def _pad(midis: tuple[int, ...], duration_s: float, sr: int, detune_cents: float) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    out = np.zeros_like(t)
    for midi in midis:
        f = midi_to_hz(midi, detune_cents)
        for harmonic, level in ((1, 1.0), (2, 0.35), (3, 0.15)):
            out += level * np.sin(2 * np.pi * f * harmonic * t)
    attack = int(0.05 * sr)
    out[:attack] *= np.linspace(0.0, 1.0, attack)
    out[-attack:] *= np.linspace(1.0, 0.0, attack)
    return out / len(midis)


# F natural minor. i - iv - v - i, two bars per chord.
_LOOP_KEY = "F minor"
_LOOP_CHORDS = ((53, 56, 60), (58, 61, 65), (60, 63, 67), (53, 56, 60))
# One 808 note per bar (F1, F1, G#1, F1, A#1, C2, D#2, F1); bar 6 is planted out of tune.
_LOOP_808_MIDI = (29, 29, 32, 29, 34, 36, 39, 29)
DETUNED_BAR = 5
DETUNED_CENTS = 35.0


def _loop_parts(
    sr: int, tempo_bpm: float, seed: int, hat_gain: float, bass_gain: float, detune: bool
) -> tuple[dict[str, np.ndarray], LoopTruth]:
    """The loop's drums, bass and pad as separate stereo stems, before any normalization."""
    rng = np.random.default_rng(seed)
    bars = len(_LOOP_808_MIDI)
    beat = 60.0 / tempo_bpm
    n = int(round(bars * 4 * beat * sr))
    drums, bass, pad_l, pad_r = (np.zeros(n) for _ in range(4))
    hats_l, hats_r = np.zeros(n), np.zeros(n)

    kick, kick_times, snare_times, notes = _kick(sr), [], [], []
    for bar in range(bars):
        bar_start = bar * 4 * beat
        for kick_beat in (0.0, 2.5):
            kick_times.append(bar_start + kick_beat * beat)
            _place(drums, kick_times[-1], kick, sr)
        for snare_beat in (1.0, 3.0):
            snare_times.append(bar_start + snare_beat * beat)
            _place(drums, snare_times[-1], 0.6 * _snare(sr, rng), sr)
        for eighth in range(8):
            hat = hat_gain * (1.0 if eighth % 2 == 0 else 0.6) * _hat(sr, rng)
            pan = 0.35 if eighth % 2 == 0 else -0.35
            _place(hats_l, bar_start + eighth * beat / 2, hat * (1 - pan), sr)
            _place(hats_r, bar_start + eighth * beat / 2, hat * (1 + pan), sr)

        cents = DETUNED_CENTS if detune and bar == DETUNED_BAR else 0.0
        note = Note(bar_start, 3.5 * beat, _LOOP_808_MIDI[bar], cents)
        notes.append(note)
        _place(bass, note.start_s, bass_gain * _bass_808(note, sr), sr)

    for index, chord in enumerate(_LOOP_CHORDS):
        start = index * 8 * beat
        _place(pad_l, start, 0.22 * _pad(chord, 8 * beat, sr, -4.0), sr)
        _place(pad_r, start, 0.22 * _pad(chord, 8 * beat, sr, 4.0), sr)

    truth = LoopTruth(_LOOP_KEY, tempo_bpm, bars, notes, kick_times, snare_times)
    parts = {
        "drums": np.column_stack([drums + hats_l, drums + hats_r]),
        "bass": _stereo(bass),
        "other": np.column_stack([pad_l, pad_r]),
    }
    return parts, truth


def drum_808_loop(
    sr: int = SR,
    tempo_bpm: float = 140.0,
    seed: int = 1,
    peak_dbfs: float = -6.0,
    hat_gain: float = 0.12,
    bass_gain: float = 0.8,
    detune: bool = True,
) -> tuple[np.ndarray, LoopTruth]:
    """8-bar hip-hop style loop in F minor: kick, snare, hats, 808, pad."""
    parts, truth = _loop_parts(sr, tempo_bpm, seed, hat_gain, bass_gain, detune)
    return _normalize_peak(np.sum(list(parts.values()), axis=0), peak_dbfs), truth


def song_stems(
    sr: int = SR, vocal_db: float = -6.0, peak_dbfs: float = -6.0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The loop as drums/bass/other stems plus the synthetic vocal, and their exact sum.

    vocal_db sets the vocal's RMS relative to the instrumental's. Every stem is stereo and
    as long as the mix, the way stems exported from Logic are.
    """
    parts, _ = _loop_parts(sr, 140.0, 1, 0.12, 0.8, True)
    instrumental = np.sum(list(parts.values()), axis=0)
    take, _ = synthetic_vocal(sr=sr)
    vocal = np.zeros(len(instrumental))
    for start in range(0, len(vocal), len(take)):
        chunk = take[: len(vocal) - start]
        vocal[start : start + len(chunk)] = chunk
    rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
    vocal *= rms(instrumental) * db_to_lin(vocal_db) / rms(_stereo(vocal))
    stems = {**parts, "vocals": _stereo(vocal)}
    mix = np.sum(list(stems.values()), axis=0)
    scale = db_to_lin(peak_dbfs) / np.max(np.abs(mix))
    return {name: stem * scale for name, stem in stems.items()}, mix * scale


@dataclass(frozen=True)
class SectionTruth:
    label: str
    start_s: float
    end_s: float


def arranged_song(
    sr: int = SR, verse_db: float = -6.0, peak_dbfs: float = -3.0, seed: int = 1
) -> tuple[np.ndarray, list[SectionTruth]]:
    """A 40-bar song from the loop: intro, verse, hook, verse, hook, outro.

    Intro and outro are the pad alone (4 bars each). Verses are drums and 808 at verse_db;
    hooks are everything at full level, so the hook-to-verse contrast is set by verse_db.
    """
    parts, truth = _loop_parts(sr, 140.0, seed, 0.12, 0.8, True)
    bars8 = len(parts["drums"])
    bars4 = bars8 // 2
    rhythm = parts["drums"] + parts["bass"]
    blocks = [
        ("intro", 0.5 * parts["other"][:bars4]),
        ("verse", db_to_lin(verse_db) * rhythm),
        ("hook", rhythm + parts["other"]),
        ("verse", db_to_lin(verse_db) * rhythm),
        ("hook", rhythm + parts["other"]),
        ("outro", 0.5 * parts["other"][:bars4]),
    ]
    sections, start = [], 0
    for label, block in blocks:
        sections.append(SectionTruth(label, start / sr, (start + len(block)) / sr))
        start += len(block)
    audio = np.concatenate([block for _, block in blocks])
    assert truth.tempo_bpm == 140.0
    return _normalize_peak(audio, peak_dbfs), sections


def reference_master(sr: int = SR) -> np.ndarray:
    """A louder, brighter, bass-heavier take on the loop, standing in for a commercial master."""
    audio, _ = drum_808_loop(sr=sr, seed=7, hat_gain=0.25, bass_gain=1.1, detune=False)
    driven = np.tanh(audio * db_to_lin(14.0))
    return _normalize_peak(driven, -1.0)


def clipped_mix(sr: int = SR) -> np.ndarray:
    """The loop pushed 12 dB into hard clipping: runs of consecutive full-scale samples."""
    audio, _ = drum_808_loop(sr=sr, peak_dbfs=0.0)
    return np.clip(audio * db_to_lin(12.0), -1.0, 1.0)


# A melody in F minor (F3 G#3 C4 A#3 G#3 G3 F3 C4); every third note is sung sharp.
_VOCAL_MIDI = (53, 56, 60, 58, 56, 55, 53, 60)
_VOCAL_OFF = (2, 5)
_VOCAL_FORMANTS = ((700.0, 130.0, 1.0), (1220.0, 170.0, 0.5), (2600.0, 250.0, 0.25))


def synthetic_vocal(
    sr: int = SR,
    note_s: float = 0.8,
    gap_s: float = 0.15,
    off_cents: float = 30.0,
    vibrato_hz: float = 5.5,
    vibrato_cents: float = 20.0,
    peak_dbfs: float = -6.0,
) -> tuple[np.ndarray, VocalTruth]:
    """Mono band-limited sawtooth through fixed "ah" formants, with delayed vibrato."""
    notes = [
        Note(0.25 + i * (note_s + gap_s), note_s, midi, off_cents if i in _VOCAL_OFF else 0.0)
        for i, midi in enumerate(_VOCAL_MIDI)
    ]
    n = int((notes[-1].start_s + note_s + 0.25) * sr)
    out = np.zeros(n)
    t = np.arange(int(note_s * sr)) / sr
    vibrato_depth = vibrato_cents * np.clip((t - 0.15) / 0.2, 0.0, 1.0)
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    edge = int(0.03 * sr)
    env = np.ones_like(t)
    env[:edge] = np.linspace(0.0, 1.0, edge)
    env[-edge:] = np.linspace(1.0, 0.0, edge)

    for note in notes:
        freq = note.freq_hz * 2.0 ** (vibrato / 1200.0)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        harmonics = int((sr / 2) // freq.max())
        saw = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
        _place(out, note.start_s, saw * env, sr)

    shaped = 0.05 * out
    for centre, bandwidth, level in _VOCAL_FORMANTS:
        band = [centre - bandwidth / 2, centre + bandwidth / 2]
        shaped += level * _butter(out, 1, band, "bandpass", sr)
    truth = VocalTruth(_LOOP_KEY, vibrato_hz, vibrato_cents, notes, off_cents)
    return _normalize_peak(shaped, peak_dbfs), truth


def sibilant_vocal(
    sr: int = SR, burst_db: float = 0.0, burst_s: float = 0.12
) -> tuple[np.ndarray, list[float]]:
    """The synthetic vocal with a burst of 5-10 kHz noise ("s") just before notes 2, 4 and 7.

    burst_db sets the bursts' RMS relative to the vocal's. Returns the audio and burst times.
    """
    vocal, truth = synthetic_vocal(sr=sr)
    rng = np.random.default_rng(11)
    rms = float(np.sqrt(np.mean(vocal**2)))
    times = [truth.notes[i].start_s - burst_s - 0.01 for i in (1, 3, 6)]
    out = vocal.copy()
    for start in times:
        noise = _butter(rng.standard_normal(int(burst_s * sr)), 4, [5000, 10000], "bandpass", sr)
        noise *= np.hanning(len(noise)) ** 0.25
        noise *= rms * db_to_lin(burst_db) / float(np.sqrt(np.mean(noise**2)))
        _place(out, start, noise, sr)
    return out, [round(t, 3) for t in times]


def write_wav(path: Path, audio: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype=subtype)
    return path


def write_all(out_dir: Path, sr: int = SR) -> dict[str, Path]:
    """Write every fixture plus truth.json. Returns name -> path."""
    loop, loop_truth = drum_808_loop(sr=sr)
    vocal, vocal_truth = synthetic_vocal(sr=sr)
    audio = {
        "pink_noise": pink_noise(sr=sr),
        "sine_sweep": sine_sweep(sr=sr),
        "loop": loop,
        "reference": reference_master(sr=sr),
        "clipped": clipped_mix(sr=sr),
        "vocal": vocal,
    }
    paths = {name: write_wav(out_dir / f"{name}.wav", data, sr) for name, data in audio.items()}
    paths["truth"] = out_dir / "truth.json"
    truth = {"sample_rate": sr, "loop": asdict(loop_truth), "vocal": asdict(vocal_truth)}
    paths["truth"].write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return paths
