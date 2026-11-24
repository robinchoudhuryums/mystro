from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from masterbench import audio_io
from masterbench.analysis import loudness

FIXTURES = ["pink_noise", "sine_sweep", "loop", "reference", "clipped", "vocal"]


def ffmpeg_ebur128(path: Path) -> dict[str, float]:
    cmd = ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true"]
    log = subprocess.run([*cmd, "-f", "null", "-"], capture_output=True, text=True).stderr
    frames, summary = log.split("Summary:")

    def last(pattern: str) -> float:
        return float(re.findall(pattern, summary)[0])

    return {
        "i": last(r"I:\s+(-?[\d.]+) LUFS"),
        "lra": last(r"LRA:\s+(-?[\d.]+) LU"),
        "peak": last(r"Peak:\s+(-?[\d.]+) dBFS"),
        "s_max": max(float(v) for v in re.findall(r" S:\s*(-?[\d.]+)", frames)),
        "m_max": max(float(v) for v in re.findall(r" M:\s*(-?[\d.]+)", frames)),
    }


@pytest.mark.parametrize("name", FIXTURES)
def test_matches_ffmpeg_ebur128(fx: dict[str, Path], name: str) -> None:
    """ffmpeg prints one decimal, so 0.1 LU here means agreement to its last digit."""
    audio, info = audio_io.load(fx[name])
    ours, theirs = loudness.measure(audio, info.sample_rate), ffmpeg_ebur128(fx[name])
    assert ours.lufs_i == pytest.approx(theirs["i"], abs=0.1)
    assert ours.lufs_s_max == pytest.approx(theirs["s_max"], abs=0.15)
    assert ours.lufs_m_max == pytest.approx(theirs["m_max"], abs=0.15)
    assert ours.lra == pytest.approx(theirs["lra"], abs=0.2)
    assert ours.true_peak_dbtp == pytest.approx(theirs["peak"], abs=0.15)


def test_derived_figures_are_consistent(fx: dict[str, Path]) -> None:
    audio, info = audio_io.load(fx["loop"])
    m = loudness.measure(audio, info.sample_rate)
    assert m.sample_peak_dbfs == pytest.approx(-6.0, abs=0.01)
    assert m.true_peak_dbtp >= m.sample_peak_dbfs
    assert m.crest_db == pytest.approx(m.sample_peak_dbfs - m.rms_dbfs)
    assert m.plr == pytest.approx(m.true_peak_dbtp - m.lufs_i)
    assert m.lufs_m_max >= m.lufs_s_max >= m.lufs_i - 0.5


def test_true_peak_sees_between_the_samples() -> None:
    # A sine at fs/4 sampled 45 degrees off its crests: every sample is 3 dB under the peak.
    n = np.arange(4800)
    audio = np.sin(np.pi / 2 * n + np.pi / 4)[:, None]
    assert loudness.to_db(float(np.abs(audio).max())) == pytest.approx(-3.01, abs=0.01)
    assert loudness.to_db(loudness.true_peak(audio)) == pytest.approx(0.0, abs=0.2)


def test_gain_moves_loudness_one_for_one(fx: dict[str, Path]) -> None:
    audio, info = audio_io.load(fx["pink_noise"])
    base = loudness.integrated(audio, info.sample_rate)
    assert loudness.integrated(audio * 10 ** (-6 / 20), info.sample_rate) == pytest.approx(base - 6)


def test_silence_and_short_clips_do_not_blow_up() -> None:
    silence = loudness.measure(np.zeros((48000 * 4, 2)), 48000)
    assert silence.lufs_i == loudness.FLOOR_DB and silence.true_peak_dbtp == loudness.FLOOR_DB
    assert silence.lra == 0.0
    blip = loudness.measure(np.full((1000, 2), 0.5), 48000)
    assert blip.lufs_i == loudness.FLOOR_DB and blip.sample_peak_dbfs == pytest.approx(
        -6.02, abs=0.01
    )
