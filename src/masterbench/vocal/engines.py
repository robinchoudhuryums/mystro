"""Rendering a planned pitch correction: the WORLD vocoder or Rubber Band R3.

Both get the same per-frame correction and both keep the formants where they were. Only
the voiced stretches that are actually corrected are replaced; everything else in the
output is the input, sample for sample.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from scipy import signal

from masterbench.vocal import pitch_track
from masterbench.vocal.pitch_track import FRAME_MS, PitchTrack

if TYPE_CHECKING:
    from masterbench.vocal.tuning import TuningSettings

# Corrections smaller than this (half a cent) are not worth resynthesizing a phrase for.
MIN_CORRECTION = 0.005
SPLICE_PAD_S = 0.02
SPLICE_FADE_S = 0.008
SPLICE_MERGE_S = 0.06
MAX_LEVEL_FIX = 2.0


class EngineError(RuntimeError):
    """An engine that is missing or failed."""


def render(
    audio: np.ndarray,
    sample_rate: int,
    pitch: PitchTrack,
    correction: np.ndarray,
    settings: TuningSettings,
) -> np.ndarray:
    """Apply a correction (semitones per frame) and splice the result into the original."""
    spans = _spans(len(audio), sample_rate, pitch, correction, settings.formant_shift)
    if not spans:
        return audio.copy()
    stereo = audio if audio.ndim == 2 else audio[:, None]
    if settings.engine == "world":
        wet = _world(stereo, sample_rate, pitch, correction, settings)
    else:
        if settings.formant_shift:
            raise EngineError("formant shift needs the WORLD engine; Rubber Band only preserves")
        wet = _rubberband(stereo, sample_rate, correction, settings.preserve_formants)
    for a, b in spans:
        # Tuning must not move the vocal's level in the mix; a vocoder does not promise that.
        before = float(np.sum(stereo[a:b] ** 2))
        after = float(np.sum(wet[a:b] ** 2))
        if before > 0 and after > 0:
            wet[a:b] *= float(np.clip(np.sqrt(before / after), 1 / MAX_LEVEL_FIX, MAX_LEVEL_FIX))
    mask = _mask(len(audio), sample_rate, spans)
    out = stereo + mask[:, None] * (wet - stereo)
    return out if audio.ndim == 2 else out[:, 0]


def _spans(
    frames: int, sample_rate: int, pitch: PitchTrack, correction: np.ndarray, formant_shift: float
) -> list[tuple[int, int]]:
    """The stretches of samples to replace: each corrected voiced run, with a little room.

    The room puts the switch between original and engine in the consonant, breath or
    silence around the run, where two versions of the signal can be crossfaded safely.
    """
    hop = sample_rate * FRAME_MS / 1000.0
    pad = int(SPLICE_PAD_S * sample_rate)
    spans: list[list[int]] = []
    for start, end in pitch_track.voiced_runs(pitch.voiced):
        if not formant_shift and np.max(np.abs(correction[start:end])) < MIN_CORRECTION:
            continue
        a = max(0, int(start * hop) - pad)
        b = min(frames, int(end * hop) + pad)
        if spans and a - spans[-1][1] < SPLICE_MERGE_S * sample_rate:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans]


def _mask(frames: int, sample_rate: int, spans: list[tuple[int, int]]) -> np.ndarray:
    """1 where the engine's output is used, 0 where the original stays, with short fades."""
    fade = int(SPLICE_FADE_S * sample_rate)
    mask = np.zeros(frames)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
    for a, b in spans:
        mask[a:b] = 1.0
        rise = min(fade, b - a)
        if a > 0:
            mask[a : a + rise] = ramp[:rise]
        if b < frames:
            mask[b - rise : b] = np.minimum(mask[b - rise : b], ramp[:rise][::-1])
    return mask


def _fit(audio: np.ndarray, frames: int) -> np.ndarray:
    if len(audio) >= frames:
        return audio[:frames]
    return np.concatenate([audio, np.zeros((frames - len(audio), *audio.shape[1:]))])


def _warp(envelope: np.ndarray, semitones: np.ndarray) -> np.ndarray:
    """Move a spectral envelope up or down along the frequency axis, frame by frame."""
    bins = np.arange(envelope.shape[1], dtype=float)
    out = envelope.copy()
    for i in np.flatnonzero(np.abs(semitones) > 1e-6):
        out[i] = np.interp(bins / 2.0 ** (semitones[i] / 12.0), bins, envelope[i])
    return out


def _world(
    audio: np.ndarray,
    sample_rate: int,
    pitch: PitchTrack,
    correction: np.ndarray,
    settings: TuningSettings,
) -> np.ndarray:
    pyworld = pitch_track.world()
    f0 = np.ascontiguousarray(pitch.f0_hz, dtype=np.float64)
    times = np.arange(len(f0)) * FRAME_MS / 1000.0
    moved = np.where(f0 > 0, f0 * 2.0 ** (correction / 12.0), 0.0)
    # Without formant preservation the envelope travels with the pitch, as on tape.
    carried = settings.formant_shift + (0.0 if settings.preserve_formants else correction)
    carried = np.broadcast_to(carried, f0.shape)
    channels = []
    for channel in audio.T:
        samples = np.ascontiguousarray(channel, dtype=np.float64)
        envelope = pyworld.cheaptrick(samples, f0, times, sample_rate)
        aperiodicity = pyworld.d4c(samples, f0, times, sample_rate)
        if np.any(carried):
            envelope = np.ascontiguousarray(_warp(envelope, carried))
        voice = pyworld.synthesize(moved, envelope, aperiodicity, sample_rate, FRAME_MS)
        channels.append(_fit(voice, len(samples)))
    return np.stack(channels, axis=1)


def rubberband_path() -> str:
    found = shutil.which("rubberband")
    if found is None:
        raise EngineError(
            "the rubberband command is not installed (brew install rubberband); "
            "or use the WORLD engine"
        )
    return found


def _run_rubberband(
    audio: np.ndarray, sample_rate: int, pitch_map: list[tuple[int, float]], formants: bool
) -> np.ndarray:
    command = [rubberband_path(), "-q", "--fine"]
    if formants:
        command.append("--formant")
    with tempfile.TemporaryDirectory(prefix="masterbench-rb-") as tmp:
        source, result, mapping = (Path(tmp) / name for name in ("in.wav", "out.wav", "map.txt"))
        sf.write(source, audio, sample_rate, subtype="FLOAT")
        mapping.write_text(
            "".join(f"{frame} {semitones:.5f}\n" for frame, semitones in pitch_map),
            encoding="utf-8",
        )
        done = subprocess.run(
            [*command, "--pitchmap", str(mapping), str(source), str(result)],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0 or not result.is_file():
            raise EngineError(f"rubberband failed: {done.stderr.strip()[-300:]}")
        out, _ = sf.read(result, dtype="float64", always_2d=True)
    return _fit(out, len(audio))


@lru_cache(maxsize=8)
def rubberband_latency(sample_rate: int, formants: bool = True) -> int:
    """How many samples before its stated position a pitch-map entry takes effect.

    The command line applies the map against the frames it has read, while the frames it is
    writing lag behind by the engine's lookahead, so every change lands early. Measured here
    with a one-semitone step on a steady tone rather than assumed, since it depends on the
    engine version and the sample rate.
    """
    seconds, step_at = 3.0, 1.5
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sum([np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 9)], axis=0)
    step = int(step_at * sample_rate)
    out = _run_rubberband(tone[:, None], sample_rate, [(0, 0.0), (step, 1.0)], formants)[:, 0]
    band = signal.butter(4, [150.0, 300.0], "bandpass", fs=sample_rate, output="sos")
    analytic = np.asarray(signal.hilbert(signal.sosfiltfilt(band, out)))
    phase = np.unwrap(np.angle(analytic))
    semitones = 12.0 * np.log2(np.maximum(np.diff(phase) * sample_rate / (2 * np.pi), 1.0) / 220.0)
    width = max(8, sample_rate // 200)
    smooth = np.convolve(semitones, np.ones(width) / width, mode="same")
    search = slice(step - sample_rate // 2, step + sample_rate // 2)
    crossed = np.flatnonzero(smooth[search] > 0.5)
    if len(crossed) == 0:
        raise EngineError("could not measure rubberband's pitch-map latency")
    return int(sample_rate // 2 - crossed[0])


def _rubberband(
    audio: np.ndarray, sample_rate: int, correction: np.ndarray, formants: bool
) -> np.ndarray:
    latency = rubberband_latency(sample_rate, formants)
    hop = sample_rate * FRAME_MS / 1000.0
    pitch_map: list[tuple[int, float]] = [(0, 0.0)]
    for i, semitones in enumerate(correction):
        frame = int(round(i * hop)) + latency
        value = round(float(semitones), 5)
        if frame > pitch_map[-1][0] and value != pitch_map[-1][1]:
            pitch_map.append((frame, value))
    return _run_rubberband(audio, sample_rate, pitch_map, formants)
