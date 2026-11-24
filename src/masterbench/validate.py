"""Pre-flight checks on an input mix (spec 5.1)."""

from __future__ import annotations

import numpy as np

from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import Loudness
from masterbench.audio_io import AudioInfo
from masterbench.config import Thresholds

BYPASS_HINT = "Bounce from Logic with the master bus bypassed."
_FULL_SCALE = 0.9999
_SILENCE = 10 ** (-60 / 20)


def clipped_runs(audio: np.ndarray, min_run: int) -> int:
    """Number of runs of at least min_run consecutive full-scale samples, over all channels."""
    runs = 0
    for channel in audio.T:
        hot = np.concatenate([[False], np.abs(channel) >= _FULL_SCALE, [False]])
        edges = np.flatnonzero(np.diff(hot.astype(np.int8)))
        runs += int(np.sum(edges[1::2] - edges[::2] >= min_run))
    return runs


def edge_silence(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Seconds of silence (below -60 dBFS) at the head and tail."""
    loud = np.flatnonzero(np.max(np.abs(audio), axis=1) > _SILENCE)
    if len(loud) == 0:
        return len(audio) / sample_rate, len(audio) / sample_rate
    return loud[0] / sample_rate, (len(audio) - 1 - loud[-1]) / sample_rate


def check(audio: np.ndarray, info: AudioInfo, loud: Loudness, limits: Thresholds) -> list[Flag]:
    flags: list[Flag] = []
    if info.lossy:
        flags.append(
            Flag(
                "lossy_target",
                "warn",
                f"{info.format} is a lossy format. Mastering exaggerates its artifacts and "
                "streaming services encode the file again. Bounce a WAV or AIFF instead.",
            )
        )
    if info.channels == 1:
        flags.append(Flag("mono", "info", "Mono file; it is converted to stereo for mastering."))

    runs = clipped_runs(audio, limits.clip_run_samples)
    if runs:
        message = f"Clipping: {runs} run(s) of consecutive full-scale samples. {BYPASS_HINT}"
        flags.append(Flag("clipping", "problem", message, runs))

    mastered = []
    if loud.lufs_i > limits.already_mastered_lufs:
        mastered.append(f"integrated loudness {loud.lufs_i:.1f} LUFS")
    if loud.true_peak_dbtp > limits.near_full_scale_dbtp:
        mastered.append(f"true peak {loud.true_peak_dbtp:+.1f} dBTP")
    if loud.crest_db < limits.low_crest_db:
        mastered.append(f"crest factor {loud.crest_db:.1f} dB")
    if mastered:
        message = f"Looks already mastered or limited ({', '.join(mastered)}). {BYPASS_HINT}"
        flags.append(Flag("already_mastered", "warn", message, loud.lufs_i))

    offset = float(np.max(np.abs(audio.mean(axis=0)))) if len(audio) else 0.0
    if offset > limits.dc_offset:
        flags.append(Flag("dc_offset", "warn", f"DC offset of {offset:.4f} full scale.", offset))

    head, tail = edge_silence(audio, info.sample_rate)
    if head > limits.edge_silence_s:
        flags.append(Flag("silence_head", "info", f"{head:.1f} s of silence at the start.", head))
    if tail > limits.edge_silence_s:
        flags.append(Flag("silence_tail", "info", f"{tail:.1f} s of silence at the end.", tail))
    return flags
